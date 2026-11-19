# Action groups

This folder holds two **synthetic** demonstration action groups. They are not recordings from a robot. The A2 Ultra repository ships joint recordings copied from Luka's iPad app. No recordings from your own TonyPi Pro exist yet, so this folder holds test inputs for the parser and playback instead.

| File | Frames | Duration | Motion |
| --- | ---: | ---: | --- |
| `demo_wave_left.d6a` | 6 | 2.6 s | Raises the left arm sideways, waves the forearm twice, returns home |
| `demo_squat.d6a` | 3 | 2.2 s | Shallow squat with level soles, hold, stand up |

[`tools/make_demo_action.py`](../../tools/make_demo_action.py) writes each file from a few joint-angle keyframes. It converts the angles to bus-servo pulses with the **reference** calibration, which is the upstream author's robot's servo zeros and signs in [`model/servo_map.csv`](../../model/servo_map.csv). [`MANIFEST.json`](MANIFEST.json) records each file's bytes and SHA-256. The generator is deterministic, and `--check` confirms that the checked-in bytes match it.

## Format

A `.d6a` file is Hiwonder's action-group format: an SQLite database with one table, `ActionGroup`. Each row is one keyframe:

- `[Index]` gives the play order.
- `Time` is the move duration in milliseconds.
- `Servo1` to `Servo18` hold bus-servo positions from 0 to 1000 (240° of travel, 500 at mid-travel). The Pro's hands are IDs 17 and 18. A NULL cell leaves that servo where it is.

Hiwonder's software sends each row to the servos with that move time and waits `Time` before the next row. The servos move at constant speed. The head's PWM servos are not part of action groups.

```sh
.venv/bin/python -I tools/d6a_info.py data/action-groups/*.d6a
.venv/bin/python -I sim/play_action.py data/action-groups/demo_squat.d6a --calibration reference --scene floating
```

## Using your robot's own action groups

Copy `.d6a` files from your robot. Hiwonder's software keeps them in its `ActionGroups` folder; Hiwonder's GitHub repository uses that name. Then play them on the twin:

```sh
.venv/bin/python -I sim/play_action.py path/to/stand.d6a --calibration reference --scene floating
```

`--calibration unit` refuses to run until you measure your robot's servo zeros and signs and enter them in `model/servo_map.csv`. `--calibration reference` borrows the upstream author's numbers, which can be off by tens of pulses on another unit. Hiwonder also stores a factory deviation inside each servo, and its software never adds that to the action-group values. A unit with tuned deviations may therefore read close to the reference zeros, but this is unverified.

Hiwonder's stock action groups have no stated licence, so none are included here. The integration check uses only the 16 pulse values of Hiwonder's stock `stand` pose as a test input; the pose and its source are in [`RESEARCH_LOG.md`](../../RESEARCH_LOG.md).
