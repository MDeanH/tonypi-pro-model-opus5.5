"""Play a Hiwonder action group (.d6a) on the twin and report what the simulated robot did.

Usage:
  .venv/bin/python -I sim/play_action.py <file.d6a> --calibration unit|reference [--scene fixed|floating]
                                          [--settle SECONDS] [--viewer]

--calibration is required. `unit` uses your robot's zeros and signs from model/servo_map.csv and refuses while they
are blank; `reference` explicitly borrows the upstream author's robot's numbers (pab47/TonyPi), which can be tens
of pulses off for your unit.

Servo model (the upstream simulator's model of Board.setBusServoPulse): for each frame, every servo with a value
moves its position set-point in a straight line from its MEASURED angle to the frame's target over the frame's Time
(ms); the next frame starts when that time is up. An empty (NULL) cell keeps the servo's previous target. The
twin's position servos (kp, dampratio, torque cap from the ledger) do the rest. Targets are clamped to each servo's
ctrlrange, and the count is reported. Hand servos 17/18 have no calibration anywhere, so populated hand cells are
counted and the hands hold their home targets. The head (PWM) is not part of .d6a files.

--scene fixed welds the torso (servo tracking only); --scene floating stands the twin on the floor from the `home`
keyframe (balance included). Simulation only: nothing here talks to a robot.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import d6a_info  # noqa: E402
import servo_map  # noqa: E402

SCENES = {"fixed": ROOT / "build/tonypi_pro_as_is.xml",
          "floating": ROOT / "build/tonypi_pro_contact_candidate.xml"}
HAND_IDS = (17, 18)
FALL_TILT_DEG = 45.0


def load_targets(path, calibration, rows=None):
    """Frames as [(time_ms, {joint: target_rad or None})] plus counts of what was not converted."""
    action = d6a_info.read_action(path)
    ids = action["servo_ids"]
    unknown = [i for i in ids if i not in range(1, 19)]
    if unknown:
        raise d6a_info.D6AError(f"{path}: servo columns {unknown} are not TonyPi Pro bus servos 1-18")
    body = [i for i in ids if i not in HAND_IDS]
    cal = servo_map.calibration(calibration, body, rows)
    joint_of = {sid: r["joint"] for sid, r in servo_map.bus_channels(rows).items()}
    frames, hand_cells = [], 0
    for f in action["frames"]:
        targets = {}
        for sid, pulse in f["pulses"].items():
            if sid in HAND_IDS:
                hand_cells += pulse is not None
                continue
            targets[joint_of[sid]] = None if pulse is None else servo_map.pulse_to_rad(pulse, *cal[sid])
        frames.append((f["time_ms"], targets))
    return frames, {"servo_columns": ids, "hand_cells_ignored": hand_cells, "frames": len(frames),
                    "duration_s": action["total_ms"] / 1000.0}


def tilt_deg(quat):
    """Angle between the torso's z axis and world z (heading/yaw is not tilt)."""
    w, x, y, z = (float(v) for v in quat)
    cz = 1.0 - 2.0 * (x * x + y * y) / (w * w + x * x + y * y + z * z)
    return math.degrees(math.acos(max(-1.0, min(1.0, cz))))


def play(model, frames, settle_s=0.5, data=None, viewer=None):
    """Run the frames; returns a dict of measured outcomes. Pass `data` to share it with a viewer."""
    data = data if data is not None else mujoco.MjData(model)
    floating = model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    act = {model.actuator(i).name: i for i in range(model.nu)}
    qadr = {model.actuator(i).name: int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)}
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    plan = {name: (float(data.qpos[qadr[name]]),) * 2 + (0.0, 0.0) for name in act}  # q0, goal, t0, T
    goal = {name: float(data.ctrl[act[name]]) for name in act}
    clamped, frame_errors, max_tilt, min_z, sole_steps, steps = 0, [], 0.0, math.inf, 0, 0
    touch = [model.sensor(n).id for n in ("l_ankle_roll_touch", "r_ankle_roll_touch")] if floating else []

    def advance(n_steps):
        nonlocal max_tilt, min_z, sole_steps, steps
        for _ in range(n_steps):
            t = data.time
            for name, (q0, g, t0, T) in plan.items():
                a = 1.0 if T <= 0 else min(1.0, max(0.0, (t - t0) / T))
                data.ctrl[act[name]] = q0 + (g - q0) * a
            mujoco.mj_step(model, data)
            steps += 1
            if floating:
                max_tilt = max(max_tilt, tilt_deg(data.qpos[3:7]))
                min_z = min(min_z, float(data.qpos[2]))
                sole_steps += any(data.sensordata[model.sensor_adr[s]] > 0 for s in touch)
            if viewer is not None:
                viewer.sync()
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                raise FloatingPointError("simulation state became non-finite")

    for time_ms, targets in frames:
        for name, target in targets.items():
            if target is None:
                continue
            i = act[name]
            g = float(min(max(target, lo[i]), hi[i]))
            clamped += g != target
            goal[name] = g
            plan[name] = (float(data.qpos[qadr[name]]), g, float(data.time), time_ms / 1000.0)
        advance(max(1, round(time_ms / 1000.0 / dt)))
        frame_errors.append(max(abs(float(data.qpos[qadr[n]]) - goal[n]) for n in targets) if targets else 0.0)
    advance(round(settle_s / dt))
    out = {"simulated_s": round(steps * dt, 6), "targets_clamped": clamped,
           "max_frame_end_error_rad": max(frame_errors) if frame_errors else 0.0,
           "final_max_error_rad": max(abs(float(data.qpos[qadr[n]]) - goal[n]) for n in act)}
    if floating:
        out.update({"max_tilt_deg": max_tilt, "fell": max_tilt > FALL_TILT_DEG, "min_base_z_m": min_z,
                    "final_base_xyz_m": [float(x) for x in data.qpos[:3]],
                    "sole_contact_fraction": sole_steps / max(1, steps)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", type=Path)
    ap.add_argument("--calibration", choices=servo_map.CALIBRATIONS, required=True)
    ap.add_argument("--scene", choices=sorted(SCENES), default="fixed")
    ap.add_argument("--settle", type=float, default=0.5, help="seconds to hold after the last frame")
    ap.add_argument("--viewer", action="store_true", help="watch in MuJoCo's viewer at real time (mjpython on macOS)")
    a = ap.parse_args(argv)
    if not 0 <= a.settle <= 10:
        ap.error("--settle must be between 0 and 10 seconds")
    try:
        frames, info = load_targets(a.action, a.calibration)
    except (servo_map.CalibrationError, d6a_info.D6AError) as e:
        raise SystemExit(f"refused: {e}")
    model = mujoco.MjModel.from_xml_path(str(SCENES[a.scene]))
    if a.viewer:
        import time
        from mujoco import viewer as mj_viewer

        class Paced:
            def __init__(self, v):
                self.v, self.t0, self.n = v, time.perf_counter(), 0

            def sync(self):
                self.n += 1
                self.v.sync()
                lag = self.n * model.opt.timestep - (time.perf_counter() - self.t0)
                if lag > 0:
                    time.sleep(lag)
                if not self.v.is_running():
                    raise SystemExit("viewer closed")

        data = mujoco.MjData(model)
        with mj_viewer.launch_passive(model, data) as v:
            result = play(model, frames, a.settle, data=data, viewer=Paced(v))
    else:
        result = play(model, frames, a.settle)
    print(json.dumps({"status": "simulation only; calibration '" + a.calibration + "'"
                      + (" borrows another robot's servo zeros" if a.calibration == "reference" else ""),
                      "action": a.action.name, "scene": a.scene, **info, **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
