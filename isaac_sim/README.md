# TonyPi Pro in NVIDIA Isaac Sim

The import target is a URDF candidate derived from the twin's parameter ledger, `isaac_sim/generated/tonypi_pro_candidate.urdf`. The twin MJCF (`build/tonypi_pro_as_is.xml`, and the floating `build/tonypi_pro_contact_candidate.xml`) can be imported instead through NVIDIA's MJCF importer. Generated files stay in `isaac_sim/generated/`, which is git-ignored.

## Build the URDF candidate

`build_tonypi_urdf.py` takes every number from an active row of [`model/ledger.csv`](../model/ledger.csv): joint frames, axes, ranges, torque caps, passive damping and friction, link masses, centres of mass and full inertia tensors. It does not read the rounded numbers in `build/tonypi_pro_as_is.xml`. Structure and geometry come from the sha-pinned vendor model `vendor/tonypi/model/robot.xml`: the 21-link tree, the visual STL meshes (each geom's own mesh attribute, so the vendor's left/right file swaps are kept), and the collision primitives. URDF has no capsule, so each vendor capsule becomes a cylinder plus two end spheres, which is the same solid. The 63 vendor mass-only boxes are not exported; their masses are already in the ledger's link inertials. The ledger is refused if `tools/seed_ledger.ledger_problems` reports anything. From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -I .\isaac_sim\build_tonypi_urdf.py
.\.venv\Scripts\python.exe -I .\isaac_sim\build_tonypi_urdf.py --check
.\.venv\Scripts\python.exe -I .\isaac_sim\test_build_tonypi_urdf.py
```

The build also writes `tonypi_pro_candidate.build.json`, which records the hashes of the URDF, the generator and every input it read, the joints whose speed is unknown, and the ledger's `scene.home.root` pose. `--check` regenerates both files in memory and fails if either is stale.

Servo speed comes from the ledger. When a joint's speed row has status `missing`, the URDF gets `velocity="0"` as an explicit unknown placeholder, as in the A2 `build_luka_urdf.py`. This is not a measured limit. The build prints these joints and lists them in the build record. A joint with neither an active nor a missing speed row stops the build. Right now 18 joints carry Hiwonder no-load speeds (source `docs`), and `r_gripper` and `l_gripper` carry the placeholder.

## Offline equivalence check

`test_build_tonypi_urdf.py` runs without Isaac Sim. It loads the generated URDF in MuJoCo and compares it with the exact binary twin `build/tonypi_pro_as_is.mjb`. The loader is a temporary copy with a `<mujoco><compiler>` block that resolves meshes and keeps every link. The comparison is made in the torso frame, at the zero pose and 8 seeded random joint configurations. It covers every link pose (forward kinematics), every centre of mass and full inertia tensor, link masses, joint axes, anchors and ranges, damping, friction and effort, and every visual mesh vertex. It also matches each of the 27 vendor collision primitives to its URDF counterparts (63 elements in total). Other tests check the URDF text: every value equals the active ledger value exactly, there are 21 links and 20 revolute joints, all mesh paths are relative and pinned, and a rebuild is byte-identical. Further tests check refusals and the build record. A negative control shows that a 1 µm frame change or a 1 ppm mass change is detected.

## Import and inspect in Isaac Sim 6.2

Use **Isaac Sim 6.2** on a machine that meets [NVIDIA's requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html). From this repository's root in Windows PowerShell:

```powershell
$isaacSimRoot = 'C:\path\to\isaac-sim'
& (Join-Path $isaacSimRoot 'python.bat') .\isaac_sim\source_integrity.py --source urdf
& (Join-Path $isaacSimRoot 'python.bat') .\isaac_sim\import_scene.py --source urdf --headless
& (Join-Path $isaacSimRoot 'python.bat') .\isaac_sim\run_scene.py --scene .\isaac_sim\generated\tonypi_urdf_fixed_scene.usda --headless --steps 240 --expected-dofs 20
```

On Linux, use Isaac Sim's `python.sh`. Add `--floating` to the import for a floating base; it writes `tonypi_urdf_floating_scene.usda`. `--source mjcf` imports the fixed twin MJCF, and `--source mjcf --floating` imports the contact candidate; these write `tonypi_mjcf_fixed_scene.usda` and `tonypi_mjcf_floating_scene.usda`.

Before Isaac Sim starts, `import_scene.py` verifies its inputs with `source_integrity.py`:

- `SOURCE_MANIFEST.json` pins the ledger, the vendor model and meshes, and the twin tools.
- The contact candidate must also reproduce byte-for-byte from `sim/build_contact_scene.py`.
- The URDF must match its build record.

The import writes the scene, a referenced USD asset under `generated/usd/`, and an adjacent `.integrity.json` with source and output hashes. It selects the asset's `physx` Physics variant (or `physics`) and refuses to save unless the scene has one articulation root and the twin's 20 revolute joints by name.

The scene is placed as follows:

- A floating URDF spawns at the ledger's home pose: torso at z = 0.000715 m, floor top at z = 0.
- Fixed-base scenes get an uncalibrated model-check floor 5 mm below the lowest collider.

The 6.2 importer turns the placeholder `velocity="0"` into `newton:velocityLimit = 0`. A Newton-based simulator reads that as a 0 rad/s limit; this was measured below. The scene layer therefore blocks that one authored value on the placeholder joints, so the schema fallback (no limit) applies, as in the MuJoCo twin. The asset itself is not edited, and the scene's `customLayerData` lists the blocked joints.

`run_scene.py` checks that physics time advances and that the runtime DOF positions are finite (exactly 20 with `--expected-dofs 20`). It does not establish correct contact, balance, drive behaviour or safe hardware transfer.

## Without Kit, and Newton on this machine

`import_scene.py --no-kit` calls the same importer classes without `SimulationApp`. It works where the importer runs without Kit, such as the pip `isaacsim-asset-isolated` package in an Isaac Lab venv. It produces the same USD, checks and integrity report, but no PhysX run. `check_newton.py` loads the twin into Newton and steps it on the GPU with the MuJoCo-Warp solver. It commands one joint (default `head_pan` to +15°) and reports the reached angle, joint counts and finiteness. These commands run from PowerShell in the repository root, using the WSL Isaac Lab venv on this machine:

```powershell
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/import_scene.py --source urdf --no-kit
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py --source mjcf
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py --source urdf
wsl.exe -d Ubuntu-24.04 -- /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py --source usd --scene isaac_sim/generated/tonypi_urdf_fixed_scene.usda
```

With `--source mjcf`, the twin's own 20 MuJoCo position servos do the work, with the ledger kp, dampratio and force caps. URDF and the importer's USD carry no servo gains. For those sources the check gives all 20 joints a Newton position drive with stiffness = ledger `actuator.*.kp`. This is a smoke-test servo, not the twin's servo.

## What was verified here (2026-11-19)

On this Windows host (project venv, MuJoCo 3.17.0):

- The URDF builds and `--check` passes.
- All 17 unit tests pass.
- Max differences from the binary twin, URDF in MuJoCo:
  - link positions 5.6e-17 m and orientations 0;
  - COM (world) 5.6e-17 m;
  - masses and inertia tensors 0 (relative);
  - joint axes and ranges 0, anchors 5.6e-17 m;
  - damping, friction and effort 0;
  - visual mesh vertices 1.1e-16 m;
  - collision primitives 5.6e-17 m;
  - total mass 1.800002 kg in both.
- `source_integrity.py` verified all three sources against the manifest of that time.

In WSL Ubuntu-24.04 (Isaac Lab venv: `isaacsim-asset-isolated` 6.2.0.0, `urdf-usd-converter` 0.3.2, `mujoco-usd-converter` 0.6.1, newton 1.6.1, warp 1.17.0, mujoco and mujoco_warp 3.12.0, RTX 2080 Ti):

- `import_scene.py --no-kit` completed for all four variants (URDF and MJCF, fixed and floating). Each had one articulation root, the 20 named revolute joints, `Physics=physx` selected and an integrity report written. 63 colliders were imported from the URDF and 27 from the MJCF.
- `check_newton.py` on `cuda:0`, 1 s at the twin's 4 ms step, all states finite, 21 Newton joints (fixed base + 20 revolute), 20 DOFs, 21 bodies:

  | Source | Commanded joint | Reached |
  | --- | --- | ---: |
  | twin MJCF | `head_pan` → 15° | 14.950° |
  | URDF | `head_pan` → 15° | 15.000° |
  | importer USD of the URDF | `head_pan` → 15° | 15.000° |
  | importer USD of the URDF | placeholder joint `r_gripper` → 15° | 15.007° |
  | importer USD of the MJCF | `head_pan` → 15° | 15.739° |

- Loaded without the scene-layer block, the raw importer asset gives Newton a 0.0 rad/s velocity limit for `r_gripper` and `l_gripper`.

Not run anywhere: Isaac Sim with Kit or PhysX. `run_scene.py` and the Kit path of `import_scene.py` (`SimulationApp`, `enable_extension`) are written against the 6.2 APIs of the A2 scripts but untested here. The pip `isaacsim` in WSL cannot start headless. It fails with `ImportError: cannot import name 'SimulationApp' from 'isaacsim' (unknown location)`, because only `isaacsim-asset-isolated` and `isaacsim-robot-schema` are installed (no Kit). This host's RTX 2080 Ti (11 GB) is also below Isaac Sim 6.2's published GPU minimum.

## Limits

- **No unit calibration.** No number has been measured on a physical TonyPi Pro. Geometry, masses and inertials come from the third-party pab47/TonyPi reconstruction. Speeds are Hiwonder no-load figures; loaded speed is lower, and the head speed assumes 6 V.
- **Upstream estimates.** Armature is an upstream ESTIMATE and cannot be expressed in URDF, so the URDF has none (the MJCF import keeps it). Joint frictionloss has no stated basis upstream. Servo kp and dampratio are simulation tuning.
- **Velocity placeholders.** `r_gripper` and `l_gripper` have no known speed. They carry `velocity="0"` in the URDF and no limit in the scenes.
- **Not in the URDF.** Servo gains, MuJoCo contact semantics (frictionless condim-1 guards, priority-2 soles and pads with friction 0.9), the parent/child contact excludes, the sole-sole pair, sites and sensors. Self-collision is off in both imports (`allow_self_collision=False`), whereas the vendor model self-collides with explicit excludes.
- **Importer mapping.** In the 6.2 conversion inspected in WSL, URDF/MJCF damping, friction and armature were authored only as `newton:*` joint attributes. Drives carried only `maxForce` (the ledger torque cap), with no stiffness or damping. The MJCF importer did not convert the twin's dampratio servos to drive gains (it reports that their gain/bias arrays are not in the expected format). It also reports that MuJoCo `pairs` and `sensors` are not supported. Whether Isaac Sim's PhysX reads the `newton:*` attributes was not verified, so set servo gains, damping, friction and armature explicitly before training (for example in an Isaac Lab actuator config).
- **Precision.** USD Physics stores mass, centre of mass and inertia as 32-bit floats, about 1e-7 relative. The URDF and `newton:inertia` keep doubles.
- **Contact and balance.** Not validated in PhysX. There is no standing, gait or balance controller, and the fixed-base floor placement is uncalibrated.

The import uses NVIDIA's [URDF importer](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html) and [MJCF importer](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/importer_exporter/import_mjcf.html).
