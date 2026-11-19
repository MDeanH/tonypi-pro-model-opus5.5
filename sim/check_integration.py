"""Run structural and behavioral checks against this repo's model and simulation outputs.

Usage: .venv/bin/python -I sim/check_integration.py

Software-only: every check compares files and simulations with each other. None of them measures a physical
TonyPi Pro.
"""

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "model"))
from build_contact_scene import OUTPUT, generate  # noqa: E402
from leg_posture_env import LegPostureEnv  # noqa: E402
import seed_ledger  # noqa: E402

AS_IS = ROOT / "build/tonypi_pro_as_is.xml"
VENDOR_ROBOT = ROOT / "vendor/tonypi/model/robot.xml"
VENDOR_SCENE = ROOT / "vendor/tonypi/model/scene.xml"
# The checked-in XML is rounded by MuJoCo's writer (~6 significant figures). Fixed-base runs stay within ~1e-8 rad
# of the vendor; floor contact amplifies the rounding to ~1e-6 rad over 2 s. A structural error is far larger: a
# fixed-base servo damping wrongly carried into the floating scene diverged by 1.4e-4 rad within 10 steps.
EQUIVALENCE_TOL_RAD = {"fixed": 1e-6, "floating": 1e-5}


def check_vendor_inputs() -> int:
    vendor = ROOT / "vendor/tonypi"
    lines = [line for line in (vendor / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if line]
    for line in lines:
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((vendor / relative).read_bytes()).hexdigest()
        assert actual == expected, f"vendor input changed: {relative}"
    return len(lines)


def check_ledger() -> dict:
    rows = seed_ledger.read(seed_ledger.LEDGER)
    problems = seed_ledger.ledger_problems(rows)
    assert not problems, "ledger problems: " + "; ".join(problems[:5])
    by = {}
    for r in rows:
        key = r["source"] or "missing"
        by[key] = by.get(key, 0) + 1
    meas = seed_ledger.read(seed_ledger.MEASUREMENTS)
    want = {mid: v for mid, _, _, v, _ in seed_ledger.measurement_rows()}
    assert {r["id"]: r["vendor_value"] for r in meas} == want, "measurements.csv vendor values drifted"
    return {"rows": len(rows), "by_source": dict(sorted(by.items())), "measurement_rows": len(meas)}


def check_rebuild() -> dict:
    import build_twin
    out = ROOT / "build/rebuilt-check.xml"  # same folder, so the relative meshdir is the same
    try:
        build_twin.build(out=str(out))
        assert out.read_bytes() == AS_IS.read_bytes(), "rebuilt XML differs from build/tonypi_pro_as_is.xml"
        report = (ROOT / "build/tonypi_pro_as_is.report.md").read_text(encoding="utf-8").splitlines()[1:]
        assert out.with_suffix(".report.md").read_text(encoding="utf-8").splitlines()[1:] == report
    finally:
        for suffix in (".xml", ".mjb", ".report.md"):
            (ROOT / f"build/rebuilt-check{suffix}").unlink(missing_ok=True)
    return {"xml_byte_identical": True, "report_identical_except_title": True}


def _vendor_fixed() -> mujoco.MjModel:
    """The vendor robot with its free root removed and the torso welded at the vendor home placement."""
    spec = mujoco.MjSpec.from_file(str(VENDOR_ROBOT))
    spec.delete([j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE][0])
    home_z = float(seed_ledger.vendor_values()["scene.home.root.z"][0])
    for k in list(spec.keys):
        spec.delete(k)
    spec.body("torso").pos = [0.0, 0.0, home_z]
    return spec.compile()


def _trajectory(m: mujoco.MjModel, key: str | None, amp: float, freq: float, steps: int = 500) -> np.ndarray:
    d = mujoco.MjData(m)
    if key is not None:
        mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key))
    out = []
    for i in range(steps):
        d.ctrl[:] = amp * math.sin(i * freq)
        mujoco.mj_step(m, d)
        out.append(d.qpos.copy())
    return np.array(out)


def check_vendor_equivalence() -> dict:
    """The ledger-built twin must reproduce the vendor model's own simulation, fixed and floating."""
    fixed_v = _vendor_fixed()
    fixed_t = mujoco.MjModel.from_xml_path(str(AS_IS))
    e_fixed = float(np.abs(_trajectory(fixed_v, None, 0.3, 0.02) - _trajectory(fixed_t, None, 0.3, 0.02)).max())
    scene_v = mujoco.MjModel.from_xml_path(str(VENDOR_SCENE))
    scene_t = mujoco.MjModel.from_xml_path(str(OUTPUT))
    e_float = float(np.abs(_trajectory(scene_v, "home", 0.15, 0.01) - _trajectory(scene_t, "home", 0.15, 0.01)).max())
    assert e_fixed < EQUIVALENCE_TOL_RAD["fixed"], f"fixed-base twin diverges from vendor by {e_fixed:.3e}"
    assert e_float < EQUIVALENCE_TOL_RAD["floating"], f"floating twin diverges from vendor scene by {e_float:.3e}"
    assert abs(scene_t.body_mass.sum() - scene_v.body_mass.sum()) < 1e-9
    return {"steps": 500, "fixed_base_max_abs_qpos_diff": e_fixed, "floating_max_abs_qpos_diff": e_float,
            "tolerance": EQUIVALENCE_TOL_RAD}


def vendor_equivalence_or_skip() -> dict:
    """Once you enter measurements of your unit, the twin is meant to differ from the vendor model."""
    vendor = seed_ledger.vendor_values()
    overrides = sorted(r["param"] for r in seed_ledger.read(seed_ledger.LEDGER)
                       if r["status"] == "active" and r["param"] in vendor and r["source"] != "vendor")
    if overrides:
        return {"not_applicable": f"{len(overrides)} active rows override vendor values", "overrides": overrides[:20]}
    return check_vendor_equivalence()


def check_scene() -> dict:
    assert OUTPUT.read_bytes() == generate(), "checked-in scene has drifted"
    baseline = mujoco.MjModel.from_xml_path(str(AS_IS))
    scene = mujoco.MjModel.from_xml_path(str(OUTPUT))
    assert baseline.nq == 20 and baseline.nv == 20 and baseline.nu == 20
    assert scene.nq == 27 and scene.nv == 26 and scene.nu == 20
    floor = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor >= 0 and scene.geom_bodyid[floor] == 0
    data = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, data, mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(scene, data)
    initial = [(scene.geom(c.geom1).name, scene.geom(c.geom2).name) for c in data.contact[:data.ncon]]
    assert initial and all(floor in (c.geom1, c.geom2) for c in data.contact[:data.ncon]), \
        "home pose must start with floor-only contacts"
    z0 = float(data.qpos[2])
    for _ in range(round(2.0 / scene.opt.timestep)):  # servos hold the home targets (ctrl = 0)
        mujoco.mj_step(scene, data)
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    self_contacts = sum(1 for c in data.contact[:data.ncon] if floor not in (c.geom1, c.geom2))
    touch = {s: float(data.sensor(f"{s}_ankle_roll_touch").data[0]) for s in ("l", "r")}
    weight = float(scene.body_mass.sum()) * float(-scene.opt.gravity[2])
    from play_action import tilt_deg
    tilt = tilt_deg(data.qpos[3:7])
    assert tilt < 5.0, f"twin tipped {tilt:.2f} deg holding home"
    assert abs(sum(touch.values()) - weight) < 0.05 * weight, "sole loads do not carry the weight"
    assert self_contacts == 0
    return {"initial_contacts": len(initial), "base_z_start_m": z0, "base_z_after_2s_m": float(data.qpos[2]),
            "tilt_after_2s_deg": tilt, "sole_load_N": touch, "weight_N": weight}


def check_training_api() -> dict:
    env = LegPostureEnv(horizon=3)
    a, info = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    n = env.n
    assert info["target_source"] == "synthetic" and np.array_equal(a, b)
    assert a.shape == (3 * n,) and n == 10
    for _ in range(3):
        observation, reward, terminated, truncated, state = env.step(np.zeros(n))
        assert observation.shape == (3 * n,) and np.isfinite(reward) and state["finite"]
        assert not terminated
    assert truncated
    try:
        env.step(np.zeros(n))
    except RuntimeError:
        pass
    else:
        raise AssertionError("ended episode accepted another action")
    for bad in (np.zeros(n + 1), np.full(n, np.nan)):
        try:
            env.reset(seed=0)
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed action accepted")
    return {"observation_values": 3 * n, "actions": n, "horizon_steps": 3}


# Hiwonder's stock standing pose, bus IDs 1-16 (github.com/Hiwonder/TonyPi ActionGroups/stand.d6a, as decoded in
# RESEARCH_LOG.md; the TonyPi Pro wiki "initial posture" differs by 5-10 pulses). Sixteen numbers used as a test
# input only; Hiwonder's action files themselves are not redistributed here.
HIWONDER_STAND_PULSES = [500, 390, 500, 600, 500, 575, 800, 725, 500, 610, 500, 400, 500, 425, 200, 275]


def _make_d6a(path: Path, columns: str, rows: list) -> None:
    import sqlite3
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE ActionGroup ({columns})")
    con.executemany(f"INSERT INTO ActionGroup VALUES ({', '.join('?' * len(rows[0]))})", rows)
    con.commit()
    con.close()


def check_action_groups() -> dict:
    import d6a_info
    import servo_map
    from play_action import SCENES, load_targets, play
    from preview_action_pose import DATA, validate_render_output, verified_if_listed
    import contextlib
    import io
    import make_demo_action
    with contextlib.redirect_stdout(io.StringIO()):  # keep this script's stdout pure JSON
        make_demo_action.main(["--check"])  # raises SystemExit if the demo files drifted
    manifest = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
    out = {"demo_files": len(manifest["files"])}
    for entry in manifest["files"]:
        path = DATA / entry["path"]
        assert verified_if_listed(path) == entry
        action = d6a_info.read_action(path)
        assert len(action["frames"]) == entry["frames"] and action["total_ms"] == entry["duration_ms"]
    # Mapping + reference calibration + model, end to end: Hiwonder's stand pulses must decode to a mirror-symmetric
    # crouch with level soles, and the floating twin must hold it.
    cal = servo_map.calibration("reference", range(1, 17))
    joint = {i: r["joint"] for i, r in servo_map.bus_channels().items()}
    pose = {joint[i]: servo_map.pulse_to_rad(HIWONDER_STAND_PULSES[i - 1], *cal[i]) for i in range(1, 17)}
    for part in ("hip_pitch", "knee", "ankle_pitch"):
        assert abs(pose[f"l_{part}"] - pose[f"r_{part}"]) < math.radians(3), f"stand pose asymmetric at {part}"
    for part in ("elbow", "shoulder_roll"):
        assert abs(pose[f"l_{part}"] + pose[f"r_{part}"]) < math.radians(5), f"stand pose not mirrored at {part}"
    flat = [pose[f"{s}_hip_pitch"] + pose[f"{s}_knee"] + pose[f"{s}_ankle_pitch"] for s in "lr"]
    assert all(abs(f) < math.radians(6) for f in flat), "stand pose soles not level in the sagittal plane"
    scene = mujoco.MjModel.from_xml_path(str(SCENES["floating"]))
    stand = play(scene, [(1000, pose)], settle_s=3.0)
    assert not stand["fell"] and stand["max_tilt_deg"] < 10 and stand["sole_contact_fraction"] == 1.0
    out["hiwonder_stand_pose"] = {"knee_deg": [round(math.degrees(pose[f"{s}_knee"]), 1) for s in "lr"],
                                  "max_tilt_deg": round(stand["max_tilt_deg"], 2),
                                  "min_base_z_m": round(stand["min_base_z_m"], 4)}
    # The demos replay on both scenes without falling.
    for name in ("demo_wave_left.d6a", "demo_squat.d6a"):
        frames, info = load_targets(DATA / name, "reference")
        fixed = play(mujoco.MjModel.from_xml_path(str(SCENES["fixed"])), frames)
        floating = play(scene, frames)
        assert fixed["targets_clamped"] == 0 and fixed["final_max_error_rad"] < 0.01
        assert not floating["fell"] and floating["sole_contact_fraction"] == 1.0
        out[name] = {"fixed_final_error_rad": round(fixed["final_max_error_rad"], 4),
                     "floating_max_tilt_deg": round(floating["max_tilt_deg"], 2)}
    # Refusals: blank unit calibration, protected or existing render paths, malformed files.
    try:
        blank = [dict(r, unit_zero_pulse="", unit_sign="") for r in servo_map.read()]
        load_targets(DATA / "demo_squat.d6a", "unit", rows=blank)
    except servo_map.CalibrationError:
        pass
    else:
        raise AssertionError("blank unit calibration was accepted")
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        existing = tmp / "existing.png"
        existing.write_bytes(b"keep")
        for forbidden in (DATA / "x.png", ROOT / "vendor/x.png", existing, tmp / "x.jpg"):
            try:
                validate_render_output(forbidden)
            except (ValueError, FileExistsError):
                pass
            else:
                raise AssertionError(f"render accepted protected path: {forbidden}")
        assert existing.read_bytes() == b"keep"
        cols = "[Index] INTEGER PRIMARY KEY, Time INT, Servo1 INT"
        bad = {"pulse_over_1000.d6a": (cols, [(1, 500, 1001)]), "zero_time.d6a": (cols, [(1, 0, 500)]),
               "unknown_column.d6a": (cols + ", Speed INT", [(1, 500, 500, 3)]),
               "servo_19.d6a": ("[Index] INTEGER PRIMARY KEY, Time INT, Servo19 INT", [(1, 500, 500)])}
        for fname, (columns, rows) in bad.items():
            _make_d6a(tmp / fname, columns, rows)
            try:
                load_targets(tmp / fname, "reference")
            except d6a_info.D6AError:
                pass
            else:
                raise AssertionError(f"malformed action group accepted: {fname}")
        out["malformed_files_refused"] = len(bad)
    return out


def main() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    results = {"status": "software-only integration checks",
               "vendor_files_checked": check_vendor_inputs(),
               "ledger": check_ledger(),
               "rebuild": check_rebuild(),
               "vendor_equivalence": vendor_equivalence_or_skip(),
               "scene": check_scene(),
               "fixed_base_task": check_training_api(),
               "action_groups": check_action_groups()}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
