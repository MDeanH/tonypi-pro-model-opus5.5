# TonyPi Pro in NVIDIA Isaac Sim

The import target is a URDF candidate derived from the twin's parameter ledger, `isaac_sim/generated/tonypi_pro_candidate.urdf`. The twin MJCF (`build/tonypi_pro_as_is.xml`, and the floating `build/tonypi_pro_contact_candidate.xml`) can be imported instead through NVIDIA's MJCF importer. Generated files stay in `isaac_sim/generated/`, which is git-ignored.

Commands below use `.venv/bin/python`. On Windows, use `.venv\Scripts\python.exe` in its place.

## Build the URDF candidate

`build_tonypi_urdf.py` takes every number from an active row of [`model/ledger.csv`](../model/ledger.csv): joint frames, axes, ranges, torque caps, speeds, passive damping and friction, link masses, centres of mass and full inertia tensors. It does not read the rounded numbers in `build/tonypi_pro_as_is.xml`. Structure and geometry come from the sha-pinned vendor model `vendor/tonypi/model/robot.xml`: the 21-link tree, the visual STL meshes (each geom's own mesh attribute, so the vendor's left/right file swaps are kept), and the collision primitives. URDF has no capsule, so each vendor capsule becomes a cylinder plus two end spheres, which is the same solid. The 63 vendor mass-only boxes are not exported; their masses are already in the ledger's link inertials.

The build refuses its inputs when:

- `tools/seed_ledger.ledger_problems` reports anything;
- a row's `unit` is not the unit its URDF field needs (m, rad, -, kg, kg·m², N·m, N·m·s/rad, N·m/rad, rad/s). For example, a speed in deg/s or s/60° is refused and never written mis-scaled;
- the ledger or vendor model lies outside this repository (checked before any work).

From the repository root:

```sh
.venv/bin/python -I isaac_sim/build_tonypi_urdf.py
.venv/bin/python -I isaac_sim/build_tonypi_urdf.py --check
.venv/bin/python -I model/build_twin.py      # once: the tests need the git-ignored build/tonypi_pro_as_is.mjb
.venv/bin/python -I isaac_sim/test_build_tonypi_urdf.py
```

The build also writes `tonypi_pro_candidate.build.json`. It records:

- the hashes of the URDF, the generator and every input it read;
- the joints whose speed is unknown, and the ledger's `scene.home.root` pose;
- per vendor collision class, the contact semantics (condim, friction, priority);
- per joint, the **importer terms**: armature, passive damping and friction, effort, speed, servo `kp` and `dampratio`, and the servo damping `kv` that MuJoCo derives from `dampratio`. That is `kv = 2·dampratio·sqrt(kp / (M⁻¹)ⱼⱼ)` at the zero pose, where M includes armature. The build computes it by compiling the URDF in MuJoCo, and it equals the twin's compiled value to 2e-16.

`--check` regenerates both files in memory and fails if either is stale.

### What the URDF does and does not match

The URDF matches the twin in **kinematics, masses, centres of mass and link inertia only**. It does not match the twin's dynamics, because URDF cannot carry joint armature (0.005 kg·m² for the head, 0.015 kg·m² for the bus servos, an upstream estimate), and armature is most of each joint's inertia. The test checks that the URDF's joint-space mass matrix equals the twin's minus diag(armature), to 1.8e-16 over 9 configurations. At the zero pose, armature is this share of the mass-matrix diagonal:

| Joint | Armature share |
| --- | ---: |
| hip_roll | 0.660 |
| hip_pitch | 0.783 |
| knee | 0.910 |
| shoulder_pitch, shoulder_roll | 0.92 |
| ankle_pitch | 0.979 |
| elbow | 0.984 |
| ankle_roll | 0.998 |
| head_pan, head_tilt | 0.998, 0.999 |
| gripper | 0.9992 |

Without armature, the twin's servo gains at its 4 ms step would be far stiffer relative to the inertia. The URDF also carries no servo gains. So use the URDF only together with the build record's importer terms; `import_scene.py` authors them.

**Speed placeholders.** When a joint's speed row has status `missing`, the URDF gets `velocity="0"` as an explicit unknown placeholder, as in the A2 `build_luka_urdf.py`. This is not a measured limit. Every importer reads it as a 0 rad/s limit, which freezes the joint. **Import the raw URDF only through `import_scene.py`**, which removes that limit. Right now 18 joints carry Hiwonder no-load speeds (source `docs`), and `r_gripper` and `l_gripper` carry the placeholder.

The ledger is inconsistent here: the gripper damping 0.3184 = 1.667 / 5.236 and the effort 1.667 already assume the LX-824HV servo, whose no-load speed is 5.236 rad/s. Reconcile it in the ledger, either by marking those values as LX-824HV assumptions or by adding that speed as an estimate row; `isaac_sim/` does not edit the ledger.

The 18 speeds are hard caps that the MuJoCo twin does not have.

## Offline equivalence check

`test_build_tonypi_urdf.py` runs without Isaac Sim. It loads the generated URDF in MuJoCo and compares it with the binary twin `build/tonypi_pro_as_is.mjb`. Run `model/build_twin.py` first: the tests skip with that reason if the `.mjb` is missing, and the `DEFAULT_OUTPUT` case skips explicitly if the URDF has not been built. The comparison is made in the torso frame, at the zero pose and 8 seeded random configurations. It covers:

- every link pose;
- every centre of mass and full inertia tensor, and link masses;
- joint axes, anchors and ranges;
- damping, friction and effort, the last both as MuJoCo's URDF parser maps it and as the URDF text states it, against the twin's servo force range;
- every visual mesh vertex;
- each of the 27 vendor collision primitives, matched to its URDF counterparts (63 elements).

Other tests check:

- the URDF text: every value equals the active ledger value exactly, 21 links and 20 revolute joints, all mesh paths relative and pinned, and a byte-identical rebuild;
- the mass matrix against the twin's minus armature, and the importer terms against the twin's compiled `kp`, `kv`, armature, damping, friction and effort;
- refusals: ledger problems, wrong units, an out-of-repository ledger, bad inertia, bad speeds, unpinned vendor files;
- the build record.

A negative control shows that a 1 µm frame change or a 1 ppm mass change is detected.

## Import and inspect in Isaac Sim 6.2

Use **Isaac Sim 6.2** on a machine that meets [NVIDIA's requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html). From this repository's root (Windows PowerShell shown; on Linux use Isaac Sim's `python.sh`):

```powershell
$isaacSimRoot = 'C:\path\to\isaac-sim'
& (Join-Path $isaacSimRoot 'python.bat') .\isaac_sim\source_integrity.py --source urdf
& (Join-Path $isaacSimRoot 'python.bat') .\isaac_sim\import_scene.py --source urdf --headless
& (Join-Path $isaacSimRoot 'python.bat') .\isaac_sim\run_scene.py --scene .\isaac_sim\generated\tonypi_urdf_fixed_scene.usda --headless --steps 240 --expected-dofs 20
```

Add `--floating` to the import for a floating base; it writes `tonypi_urdf_floating_scene.usda`. `--source mjcf` imports the fixed twin MJCF, and `--source mjcf --floating` imports the contact candidate; these write `tonypi_mjcf_fixed_scene.usda` and `tonypi_mjcf_floating_scene.usda`.

Before Isaac Sim starts, `import_scene.py` verifies its inputs with `source_integrity.py`:

- `SOURCE_MANIFEST.json` pins the ledger, the vendor model, scene and meshes, and the twin tools.
- The contact candidate must also reproduce byte-for-byte from `sim/build_contact_scene.py`.
- The URDF must match its build record.

**Transaction.** The importer writes into an empty staging folder. The 6.2 importer never overwrites; it would write `<name>_1`, `<name>_2` and so on next to an old asset. Then:

1. The importer's layer documentation, which holds the absolute staging path, is cleared. Any absolute path left in a generated file stops the import.
2. The scene is built and checked against the staged asset.
3. The inputs are verified again: every pinned file by hash. A regenerated `SOURCE_MANIFEST.json` whose pinned entries are unchanged no longer trips this check.
4. Only then do the asset folder `generated/usd/<variant>/`, the scene and the adjacent `.integrity.json` replace the previous ones.

On any failure, the previous files are restored and the staging folder is deleted.

The integrity report also records a content hash for each binary crate layer. The importer's `geometries.usd` can differ byte-for-byte between identical imports while its content is identical. All text layers and the scene are byte-identical across repeated imports (measured for the fixed URDF).

**Scene layer.** The imported asset is never edited. The scene selects the asset's `physx` Physics variant and refuses to save unless it has one articulation root, the twin's 20 revolute joints by name, and exactly one ground collider. It then authors the following, and lists all of it in the scene's `customLayerData`:

- **Joint dynamics.** Values come from the URDF build record for `--source urdf`, or from the MJCF as MuJoCo compiles it for `--source mjcf`.
  - drive stiffness = servo `kp` and drive damping = servo `kv`, in USD angular units (per degree);
  - `newton:armature` and `physxJoint:armature`;
  - `physxJoint:maxJointVelocity` (deg/s) for the 18 joints with a ledger speed, next to the importer's `newton:velocityLimit`.

  The importer's `maxForce`, `newton:damping`, `newton:friction` and `newton:velocityLimit` are checked against the same values. MJCF scenes get no speed limit, as in the twin.
- **Speed placeholders.** The importer writes the URDF's `velocity="0"` as `newton:velocityLimit = 0`, which Newton reads as 0 rad/s; this was measured. The scene blocks that authored zero on `r_gripper` and `l_gripper`, so no limit applies, as in the twin.
- **Self-collision off for PhysX too.** The importer's `allow_self_collision=False` only writes `newton:selfCollisionEnabled = 0`, and the PhysX schema defaults `physxArticulation:enabledSelfCollisions` to on. The scene sets it to false on the articulation root. The vendor model self-collides with 22 explicit excludes; the URDF cannot carry them, so self-collision is off in every scene.
- **Contact materials.** Each robot collider is bound to the vendor class's sliding friction:
  - the 19 self-collision guards (55 URDF elements) get 0, with PhysX combine mode `min`, because their `condim 1` makes them frictionless;
  - the soles and gripper pads (8) get 0.9;
  - the ground gets the vendor floor's 0.9 (`vendor/tonypi/model/scene.xml`).

  `condim`, `priority` and `solref` have no PhysX form.
- **World geometry.** World geometry imported with the robot is deactivated: the contact candidate's own 20 m floor plane, which `import_scene=False` does not skip. The scene's `/World/Ground` is the only ground.
- **Physics step.** `physxScene:timeStepsPerSecond = 250`, from the twin's `<option timestep="0.004">`. PhysX would otherwise use its default rate.

**Placement:**

- A floating URDF spawns at the ledger's home pose: torso at z = 0.000715 m, floor top at z = 0.
- Fixed-base scenes get an uncalibrated model-check floor 5 mm below the lowest collider.

`run_scene.py` checks that physics time advances and that the runtime DOF positions are finite (exactly 20 with `--expected-dofs 20`). It does not establish correct contact, balance, drive behaviour or safe hardware transfer.

## Without Kit, and Newton on this machine

`import_scene.py --no-kit` calls the same importer classes without `SimulationApp`. It works where the importer runs without Kit, such as the pip `isaacsim-asset-isolated` package in an Isaac Lab venv. It runs the same scene checks. Its output has not been compared with a Kit import, because Kit cannot start here, and there is no PhysX run.

`check_newton.py` steps a fixed-base model on the GPU with Newton's MuJoCo-Warp solver, using the twin's step, integrator, cone and impratio. It commands one joint (default `head_pan` to +15°) and holds the others at 0. It runs the same command on the twin itself with CPU MuJoCo as the reference. It fails unless all of these hold:

- all states are finite and there are 20 revolute DOFs;
- the joint has settled: every sample in the final 0.2 s is within 1° of the target, and the final speed is ≤ 0.05 rad/s;
- the trajectory stays within `--twin-tolerance-deg` (default 1.5°) of the twin's at every step.

The sources:

- `--source mjcf` uses the twin's own servos.
- `--source urdf` adds the build record's `kp`, `kv`, armature, passive damping and friction to Newton's URDF model.
- `--source usd` takes the scene exactly as authored, and only checks that Newton read the scene's gains and armature.

Floating scenes are refused. A trajectory check needs a fixed base: a free base adds contact and balance, which this check does not cover.

Run these from PowerShell in the repository root, using the WSL Isaac Lab venv on this machine:

```powershell
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/import_scene.py --source urdf --no-kit
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py --source mjcf
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py --source urdf
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py --source usd --scene isaac_sim/generated/tonypi_urdf_fixed_scene.usda
```

The WSL venv has MuJoCo 3.12, which rejects the `input="pos"` attribute that the project venv's MuJoCo 3.17 writes into the twin XML. `mujoco_compat.py` removes exactly that attribute in memory, and only after the unmodified file fails to compile. `input="pos"` is the default, so this changes nothing in the model, and the edit is reported.

## What was verified here (2026-11-19)

**Windows host** (project venv, MuJoCo 3.17.0):

- The URDF builds and `--check` passes.
- All 20 unit tests pass.
- Max differences from the binary twin, URDF in MuJoCo:
  - link positions 5.6e-17 m and orientations 0;
  - COM (world) 5.6e-17 m;
  - masses and inertia tensors 0 (relative);
  - joint axes and ranges 0, anchors 5.6e-17 m;
  - damping, friction and effort 0;
  - visual mesh vertices 1.1e-16 m;
  - collision primitives 5.6e-17 m;
  - total mass 1.800002 kg in both.
- Mass matrix = twin − armature to 1.8e-16. Importer `kv` = twin `kv` to 2e-16.
- `source_integrity.py` verified all three sources.

**WSL Ubuntu-24.04.** The Isaac Lab venv has `isaacsim-asset-isolated` 6.2.0.0, `urdf-usd-converter` 0.3.2, `mujoco-usd-converter` 0.6.1, newton 1.6.1, warp 1.17.0, and mujoco and mujoco_warp 3.12.0. The GPU is an RTX 2080 Ti.

`import_scene.py --no-kit` completed for all four variants. Each had:

- one articulation root and the 20 named revolute joints;
- `Physics=physx`, with the scene-layer terms above;
- exactly one ground collider. The URDF scenes have 63 robot colliders + ground; the MJCF scenes have 27 + ground, after the contact candidate's floor is deactivated;
- an integrity report;
- no absolute path in any generated file.

`check_newton.py` on `cuda:0`, 1 s at the twin's 4 ms step: all states finite, 21 Newton joints (fixed base + 20 revolute), 20 DOFs, 21 bodies. For the `usd` rows the drives are exactly as the scene authors them, and the floating-scene row is refused by design.

| Source | Command | Peak | At 0.1 s (twin 8.383°) | Final 0.2 s | Max deviation from twin |
| --- | --- | ---: | ---: | --- | ---: |
| twin MJCF | `head_pan` → 15° | 14.950° | 8.383° | 14.942–14.950° | 5e-6° |
| URDF + build-record terms | `head_pan` → 15° | 14.949° | 7.420° | 14.940–14.949° | 0.97° at 0.096 s |
| USD of the URDF | `head_pan` → 15° | 14.949° | 7.420° | 14.940–14.949° | 0.97° at 0.096 s |
| USD of the URDF | placeholder `r_gripper` → 15° | 14.956° | n/a | settled | 0.32° at 0.052 s |
| USD of the MJCF | `head_pan` → 15° | 14.949° | 7.420° | 14.940–14.949° | 0.97° at 0.096 s |

The remaining ≈1° deviation is in Newton 1.6.1's drive conversion, not in the model. Newton clamps drive torque with the joint's `actuatorfrcrange`, not the actuator's `forcerange`. Under the `implicitfast` integrator, MuJoCo then keeps the servo damping `kv` active while the torque is saturated, so the rise is slower.

This was measured on CPU MuJoCo with Newton's own saved model: at 0.1 s it reaches 7.420°, and 8.383° with the clamp moved to the actuator, which equals the twin. At 1 s it reaches 14.9488° and 14.9505° respectively.

Before the scene layer authored the servo, the MJCF USD was effectively undamped (ζ ≈ 0.06). It peaked at 23.3° and was still oscillating at 1 s; the 15.739° reported earlier was one sample of that oscillation. It now has no overshoot.

**Not run anywhere: Isaac Sim with Kit or PhysX.** `run_scene.py` and the Kit path of `import_scene.py` (`SimulationApp`, `enable_extension`) are written against the 6.2 APIs of the A2 scripts but untested here. The pip `isaacsim` in WSL cannot start headless. It fails with `ImportError: cannot import name 'SimulationApp' from 'isaacsim' (unknown location)`, because only `isaacsim-asset-isolated` and `isaacsim-robot-schema` are installed (no Kit). This host's RTX 2080 Ti (11 GB) is also below Isaac Sim 6.2's published GPU minimum.

The `physx*` attributes are written against the schema bundled with the importer (`PhysxJointAPI`, `PhysxArticulationAPI`) and the PhysX names `PhysxSceneAPI` and `PhysxMaterialAPI`. Whether PhysX honours each one is unverified.

## Limits

- **No unit calibration.** No number has been measured on a physical TonyPi Pro. Geometry, masses and inertials come from the third-party pab47/TonyPi reconstruction. Speeds are Hiwonder no-load figures; loaded speed is lower, and the head speed assumes 6 V.
- **Upstream estimates.** Armature is an upstream ESTIMATE, and it dominates joint inertia (table above). Joint frictionloss has no stated basis upstream. Servo `kp` and `dampratio` are simulation tuning.
- **PhysX passive terms.** The bundled PhysX schema has no passive viscous damping or Coulomb friction-torque attribute; `physxJoint:jointFriction` is a unitless coefficient with different meaning. Under PhysX, the scenes therefore carry the servo `kp`/`kv`, armature and torque cap, but not the passive damping (0.0168 or 0.3184 N·m·s/rad) or the frictionloss (0.01 or 0.02 N·m). Newton reads them from `newton:*`. For PhysX training, add them in an Isaac Lab actuator config. For example, an implicit actuator with damping kv + passive damping is an approximation, because the twin clamps only the servo part.
- **Contact semantics.** The scenes carry sliding friction per collider. They do not carry `condim`, `priority`, `solref`/`solimp`, the vendor's 22 contact excludes (self-collision is off instead) or the sole–sole pair.

  Newton's MuJoCo solver combines per-shape friction by its own rule; MuJoCo's rule is the maximum. So a 0-friction guard against the 0.9 ground may not be frictionless there. This was not verified, and it does not matter while self-collision is off and the robot is welded.

  The MJCF asset's `mujoco` Physics variant keeps the twin's `condim`, the servos (`MjcActuator` gain/bias, 20 of them) and the solver parameters; the `physx` variant that the scenes select does not. The URDF asset's `mujoco` variant has force ranges only. For a MuJoCo-Warp-only workflow on the MJCF asset, select `Physics=mujoco` instead. That path was not checked here.
- **Precision.** USD Physics stores mass, centre of mass, inertia and drive gains as 32-bit floats, about 1e-7 relative. The URDF and `newton:inertia` keep doubles.
- **Contact and balance.** Not validated in PhysX or Newton. There is no standing, gait or balance controller, and the fixed-base floor placement is uncalibrated.

The import uses NVIDIA's [URDF importer](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html) and [MJCF importer](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/importer_exporter/import_mjcf.html).
