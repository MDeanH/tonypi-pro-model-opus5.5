# Research log

## 2026-11-19 — sources for a TonyPi Pro twin

- **Vendor model.** No Hiwonder URDF, MJCF or CAD download was found. The public MuJoCo model in [`pab47/TonyPi`](https://github.com/pab47/TonyPi) (MIT, Pranav Bhounsule) was chosen. Its README says Salvador Echeveste built it from a Hiwonder STEP file. Pinned commit `97807f5e26049ce1c3d90b1c80d5d2a92606d40f` (2026-09-24) was still `main`'s head (compare API: `identical`). The 24 model, mesh and licence files were copied from a local snapshot of that commit, and all 24 git blob SHA-1s matched GitHub's recursive tree for the commit.
- **Hiwonder figures.** Each Hiwonder figure below is a `docs` row in the ledger with its URL.
  - [Product page](https://www.hiwonder.com/products/tonypi-pro): 373 × 186 × 106 mm, about 1800 g (the 18-DOF TonyPi page lists the same weight), 20 DOF, 11.1 V 2000 mAh battery, LX-824HV bus servos and LFD-01M head servos.
  - LX-824HV datasheet image: 17 kg·cm at 11.1 V, 0.20 s/60°, 0–1000 = 0–240°, 57 g.
  - LFD-01M datasheet image: 1.8 kg·cm and 0.10 s/60° at 6 V (1.5 kg·cm and 0.12 s/60° at 4.8 V), 500–2500 µs for 0–180°.
  - Dimension drawing: 113 mm foot length, 78.5 mm foot width and 169 mm stance. These are readings from an image, not stated specifications.
- **Servo IDs.**
  - Body: Hiwonder's 20-DOF ID diagram and stock action groups put IDs 1–8 on the robot's left (legs 1–5 from foot to hip, arm 6–8) and 9–16 on the right. The upstream executable simulator `control/TonyPi_sw/tonypi_sim.py` agrees. The vendor XML's actuator comments reverse the sides.
  - Hands: the wiki's PC-software page calls 17 and 18 the left and right hands. Hiwonder's grasp code drives the right hand with ID 17, and stock `grab_right`/`grab_left` move arm 14–16 with 17 and arm 6–8 with 18. The map uses 17 = right and records the conflict.
  - Head: PWM 1 is tilt and PWM 2 is pan, 1500 µs neutral.
- **Action groups.** A `.d6a` file is SQLite: `ActionGroup([Index], Time ms, Servo1…Servo18)`. Hiwonder's runtime sends each row with its move time and sleeps `Time`. The servos move at constant speed with no interpolation on the Pi. Per-servo deviations (±125 units, about ±30°) are stored inside each servo, and the Pi never adds them. Hiwonder's stock `stand.d6a` pulses for IDs 1–16 are 500, 390, 500, 600, 500, 575, 800, 725, 500, 610, 500, 400, 500, 425, 200, 275 ([source](https://github.com/Hiwonder/TonyPi/blob/main/ActionGroups/stand.d6a), decoded by a research agent; `stand_slow` differs by up to 6 units). Hiwonder's TonyPi repository has no licence, so no Hiwonder file is redistributed here.
- **Calibration.** The upstream `control/servo_map.py` gives zero pulses and signs for IDs 1–16, read with the author's robot standing in the model's zero pose. The scale is 0.24°/pulse, marked "verify later" upstream but matching the LX-824HV datasheet. The active signs for the four roll servos are flipped against a commented-out older block in the same file. The signs are physically consistent: pitch joints have opposite signs left and right, as servos facing outward on each side should, and roll joints share a sign. The older commented-out block satisfies the same rule, so this does not decide between them.
- **Vendor numbers that disagree with other sources:**
  - Expansion board: the vendor model has a 55 g box, while Hiwonder's board datasheet gives 31 g.
  - Hands: Hiwonder sells a 91 g robotic hand with an LX-824HV, but no source ties it to the TonyPi Pro.
  - Mass label: the 1800 g figure is printed for both the 18-DOF TonyPi and the Pro.
  - Servo dynamics: published identification of a similar small bus servo (STS3215) suggests more armature and damping than the upstream estimates.
  None of these are applied. They are reasons to weigh and identify a real unit.
- **Not found anywhere:** the hand servo model, link masses or inertias from Hiwonder, the LX-824HV gear ratio or rotor inertia, any system identification of a Hiwonder servo, the IMU's mounting axes, and the PWM servo supply voltage on the TonyPi board.

Status: desk research only. Nothing in this entry was measured on a robot.

## 2026-11-19 — first ledger-driven build

- `tools/seed_ledger.py` reads every number the build uses from the vendor MJCF. Frames, axes, ranges, gains and class defaults are copied as literal strings. Link masses, centres of mass and full inertia tensors are compiled by MuJoCo 3.17.0 from the vendor's per-part box masses, because the vendor file has no `<inertial>` elements. Each vendor row's `note` repeats the upstream label: ESTIMATE for armature, simulation tuning for gains, no stated basis for frictionloss. The ledger has 628 rows: 597 vendor, 29 Hiwonder and 2 missing (the hand-servo velocity limits).
- `model/build_twin.py` turns off MuJoCo's mass-from-geometry inference and inertia balancing, sets every frame, axis, limit, dynamics value, servo gain and inertial from active ledger rows, and welds the torso at the vendor `home` pose. The result reproduced the vendor model's own fixed-base simulation within 3e-9 rad over 500 steps (6e-15 rad for the exact `.mjb`). The rebuild is byte-identical.
- **Defect found and fixed:** MuJoCo converts a position servo's `dampratio` into a damping gain using the inertia of the model being compiled. Its XML writer then saved those fixed-base gains into the build output. The derived floating scene inherited them: 9% too much damping on the hip-roll servos, and 1.6e-3 rad of divergence from the vendor scene within 2 s. The build now writes the ledger's `dampratio` back into each servo, so every scene derives its own gain. Afterwards the floating scene matched the vendor's within 8e-7 rad. The integration check now guards this with a 1e-5 rad tolerance; the defect produced 1.4e-4 rad within 10 steps.
- The floating scene stands at `home` under its servos: 9.16 N and 8.50 N on the soles, summing to the 17.66 N weight, with 1.3° of tilt after 2 s and no self-contacts.
- Hiwonder's stock stand pulses, decoded with the reference calibration, give a mirror-symmetric crouch: hip pitch −44.9°/−44.2°, knee 60.7°/62.9°, ankle pitch −21.4°/−22.8°, arms mirrored. The floating twin holds it after a 1 s ramp with 2.1° of tilt, both soles level within 0.3°, and the torso 19 mm lower.
- The vendor's servo numbers agree with Hiwonder's datasheets: torque caps within 0.3%, and damping equal to stall torque divided by no-load speed within 0.35%.
- The synthetic demonstration action groups replay on both scenes. On the floor, the squat stays within 4.1° of upright and the wave within 4.0°, with soles loaded throughout.
- **Defect found and fixed:** the `.d6a` reader left its SQLite connection open, because Python's connection context manager does not close it. Windows then kept the file locked. The integration check's malformed-file cleanup exposed this. The reader now closes the connection explicitly.

Status: software checks only. No physical TonyPi Pro was measured, weighed or compared with this twin.

## 2026-11-19 — independent review and fixes

Four read-only reviewers (claims and physics, code, fresh-clone reproducibility, A2 parity) each had their findings challenged by a separate skeptic. Fixed after confirmation:

- **Tilt metric:** tilt was the torso's total rotation, so a robot that only turned could be reported as fallen. Tilt is now the angle between the torso's z axis and vertical.
- **Ledger guards:** the build silently used 0 for joint dynamics with no active row, and accepted rows nothing reads. Mistyped parameters, deprecated vendor rows without a replacement, and unread active rows now stop the build.
- **Build portability:** the build found the mesh folder by string-matching an absolute path, which fails in folders containing `&` or `'`. It now compiles with a relative mesh folder. Servo `kp` values with more than six significant digits stopped the build; the check now uses a tolerance.
- **Calibration-proof integration check:** the check now tests the blank-calibration refusal on in-memory rows, and reports vendor equivalence as not applicable once measured rows override vendor values. Before, both would fail permanently after you calibrated your unit.
- **Overclaim corrected:** the stand-pose check catches one-sided sign errors and pitch-chain mix-ups. A full left/right swap or a sign error shared by both sides passes it, so the side assignment rests on Hiwonder's diagram and action files.
- **Smaller fixes:** `.gitattributes` now forces LF on every text file (a Windows clone with `core.autocrlf` broke the manifest). The docs state the Python 3.12 requirement. The preview render is deterministic. The datasheet damping bound is 0.35%.

Open: measure a unit (`model/measurements.csv`, `model/servo_map.csv` unit columns). Identify LX-824HV dynamics (armature, friction) on hardware. Establish the IMU axes and the hand servo model. Compare a stock action group on the robot and the twin.
