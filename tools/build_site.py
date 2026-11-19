"""Build the static web viewer in site/: the twin's visual meshes plus body poses simulated by MuJoCo.

Usage: .venv/bin/python -I tools/build_site.py

Every pose and animation in the page comes from this repository's MuJoCo scenes: the floating twin replaying the
synthetic demo action groups and Hiwonder's stock stand pose (reference calibration), sampled at 25 frames/s. The
browser only applies the recorded rigid-body transforms; it runs no physics.
"""
import base64
import json
import math
import os
import shutil
import sys

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim"))
import servo_map  # noqa: E402
from play_action import load_targets  # noqa: E402

SITE = os.path.join(REPO, "site")
SCENE = os.path.join(REPO, "build", "tonypi_pro_contact_candidate.xml")
FPS = 25
STAND = [500, 390, 500, 600, 500, 575, 800, 725, 500, 610, 500, 400, 500, 425, 200, 275]  # see check_integration


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def meshes(m):
    """Visual mesh geoms (group 2) in their body's frame."""
    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]
        out.append({"body": int(m.geom_bodyid[g]), "v": b64(v, "<f4"), "f": b64(f, "<u4")})
    return out


def record(m, frames, settle_s):
    """Run the playback servo model and sample body poses at FPS."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    act = {m.actuator(i).name: i for i in range(m.nu)}
    qadr = {n: int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for n, i in act.items()}
    plan = {n: (0.0, 0.0, 0.0, 0.0) for n in act}
    dt, every = m.opt.timestep, round(1 / FPS / m.opt.timestep)
    samples, step = [], 0

    def run(n):
        nonlocal step
        for _ in range(n):
            for name, (q0, g, t0, T) in plan.items():
                a = 1.0 if T <= 0 else min(1.0, max(0.0, (d.time - t0) / T))
                d.ctrl[act[name]] = q0 + (g - q0) * a
            if step % every == 0:
                samples.append(np.hstack([d.xpos[1:], d.xquat[1:]]).astype("<f4"))
            mujoco.mj_step(m, d)
            step += 1

    run(round(0.4 / dt))
    for time_ms, targets in frames:
        for n, t in targets.items():
            if t is not None:
                lo, hi = m.actuator_ctrlrange[act[n]]
                plan[n] = (float(d.qpos[qadr[n]]), float(min(max(t, lo), hi)), float(d.time), time_ms / 1000)
        run(max(1, round(time_ms / 1000 / dt)))
    run(round(settle_s / dt))
    return {"frames": len(samples), "data": b64(np.stack(samples), "<f4")}


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    cal = servo_map.calibration("reference", range(1, 17))
    joint = {i: r["joint"] for i, r in servo_map.bus_channels().items()}
    stand = {joint[i]: servo_map.pulse_to_rad(STAND[i - 1], *cal[i]) for i in range(1, 17)}
    data_dir = os.path.join(REPO, "data", "action-groups")
    clips = {
        "Hiwonder stand pose": record(m, [(1000, stand)], 1.5),
        "Wave (demo)": record(m, load_targets(os.path.join(data_dir, "demo_wave_left.d6a"), "reference")[0], 0.6),
        "Squat (demo)": record(m, load_targets(os.path.join(data_dir, "demo_squat.d6a"), "reference")[0], 0.6),
    }
    model = {"nbody": m.nbody - 1, "fps": FPS, "meshes": meshes(m), "clips": clips,
             "bodies": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]}
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "model.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(model, f, separators=(",", ":"))
    shutil.copyfile(os.path.join(REPO, "media", "preview.png"), os.path.join(SITE, "preview.png"))
    print(f"wrote site/model.json ({os.path.getsize(os.path.join(SITE, 'model.json')) // 1024} KB), "
          f"{len(model['meshes'])} meshes, clips {[(k, v['frames']) for k, v in clips.items()]}")


if __name__ == "__main__":
    main()
