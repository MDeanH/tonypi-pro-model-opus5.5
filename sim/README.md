# TonyPi Pro in MuJoCo: simulation starting point

Use the model and action groups here as **simulation inputs**. This is a software setup, not a calibrated representation of your TonyPi Pro or a policy ready for the robot.

## Run it

From the repository root:

Python 3.12 or newer is required by the pinned requirements.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python -I sim/check_integration.py
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`. Open either XML file in MuJoCo's viewer (`python -m mujoco.viewer --mjcf=<file>`), or load it with `mujoco.MjModel.from_xml_path`:

| File | Purpose | Verified software behavior | Limit |
| --- | --- | --- | --- |
| [`build/tonypi_pro_as_is.xml`](../build/tonypi_pro_as_is.xml) | Ledger-built, fixed-base model | Loads with 20 joints and 20 position servos. It reproduces the vendor model's own simulation within 3e-9 rad over 2 s | Torso welded; no floor |
| [`build/tonypi_pro_contact_candidate.xml`](../build/tonypi_pro_contact_candidate.xml) | Floating base, vendor floor, `home`/`prone`/`supine` keyframes | Matches the vendor scene within 1e-5 rad over 2 s. Under its servos it stands at `home` with 9.2 N and 8.5 N on the soles, which carry the 17.66 N weight | No balance controller; servo parameters partly upstream estimates |

The contact scene is generated from the manifest-checked fixed-base file by [`build_contact_scene.py`](build_contact_scene.py). It adds exactly what the vendor's own `scene.xml` adds: floor, light, sky, ground material and the vendor's three free-root keyframes. Self-contact stays on as the vendor designed it: primitive guards with parent/child exclusions and a sole-to-sole pair, and the `home` pose starts with floor contacts only. Regenerate or check it with:

```sh
.venv/bin/python -I sim/build_contact_scene.py --check
```

The TonyPi Pro's servos are position servos. The twin models each one as a MuJoCo position actuator with the ledger's gain, damping ratio and torque cap, so a command of 0 holds the CAD home pose. This differs from the A2 Ultra twin, whose leg actuators take torques.

## Training-pipeline example

The fixed-base [`LegPostureEnv`](leg_posture_env.py) exposes `reset(seed)` and `step(action)`:

- **Observation:** 30 values: 10 leg angles, 10 velocities and 10 synthetic targets.
- **Action:** 10 normalized position commands. Full scale is ±0.5 rad, clipped to each servo's range.
- **Timing:** one action lasts 20 ms (5 physics steps).
- **Other servos:** the arm, hand and head servos hold home.

[`tune_posture_demo.py`](tune_posture_demo.py) searches one outer-loop gain `k` in `command = target + k * (target − angle)`. Larger gains cancel the legs' gravity sag but overshoot:

```sh
.venv/bin/python -I sim/tune_posture_demo.py
```

Over 2 episodes of 1 s, holding home scored −0.0332. Sending the target unchanged (`k` = 0) scored best at −0.0056, with a final RMS error of 2.6 mrad. `k` = 2 cut the final error to 0.66 mrad but scored −0.0137. `k` = 4 and 8 made the final error worse (8.7 and 10.4 mrad). This demonstration does not train walking, balance or a deployable policy; its targets are synthetic and the base is fixed.

## Play action groups

[`play_action.py`](play_action.py) plays a Hiwonder `.d6a` action group on either scene and reports tracking error, clamped targets, and for the floating scene: torso tilt from vertical (heading is not counted), falls (tilt over 45°) and sole contact.

```sh
.venv/bin/python -I sim/play_action.py data/action-groups/demo_wave_left.d6a --calibration reference --scene fixed
.venv/bin/python -I sim/play_action.py data/action-groups/demo_squat.d6a --calibration reference --scene floating --viewer
```

- **Servo model:** for each frame, every servo with a value moves its set-point in a straight line from its measured angle to the target over the frame's `Time`. That matches how Hiwonder's software drives the servos (constant speed, one call per row) and how the upstream simulator models it.
- **Hands:** IDs 17/18 have no published pulse-to-angle calibration, so populated hand cells are counted and the hands hold home.
- **Calibration:** `--calibration` is required. `unit` refuses while `model/servo_map.csv` has blank `unit_*` cells. `reference` borrows the upstream author's robot's servo zeros and signs.
- **macOS:** use `mjpython` for `--viewer`.

[`preview_action_pose.py`](preview_action_pose.py) validates a file and, with `--calibration`, poses one frame with `mj_forward`. It renders a PNG on request, and it refuses existing files and the data, vendor, build and model folders.

The integration check decodes Hiwonder's stock `stand` pose (16 pulse values) through the joint map and reference calibration. It gets a mirror-symmetric crouch: hips −44.9°/−44.2°, knees 60.7°/62.9°, ankles −21.4°/−22.8°. Ramped in over 1 s, the floating twin holds it with 2.1° of tilt and soles level within 0.3°. A one-sided sign error or a swapped pitch joint breaks that symmetry; a full left/right swap or a sign error shared by both sides on the roll, shoulder or elbow servos does not, so the side assignment rests on Hiwonder's diagram and action files. This is consistency evidence, not a measurement of your unit.

## Before task training or hardware transfer

Choose the first real task. For locomotion or balance, measure your unit first:

1. Weigh it and take the ruler measurements in `model/measurements.csv`.
2. Record its servo zeros and signs in `model/servo_map.csv`.
3. Identify the LX-824HV's response: the upstream armature is an estimate, and no system identification of any Hiwonder servo is published.
4. Check foot contact.

For action-group replay, compare a stock action on the robot and the twin, joint by joint. Simulation success alone does not verify behavior on the robot.
