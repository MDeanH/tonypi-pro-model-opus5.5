"""Newton GPU check of the TonyPi Pro twin with the MuJoCo-Warp solver, against the twin's own transient.

Sources (fixed base only):
  --source mjcf (default)  build/tonypi_pro_as_is.xml through newton.ModelBuilder.add_mjcf: the twin with its own
                           20 MuJoCo position servos (ledger kp, dampratio, force caps) and armature.
  --source urdf            isaac_sim/generated/tonypi_pro_candidate.urdf through add_urdf. URDF carries no
                           armature or servo, and Newton 1.6.1 puts the URDF <dynamics damping> into the drive
                           damping (inside the torque clamp) instead of the passive joint damping. The check
                           restores the twin's terms from the URDF build record (ledger values): drive ke = kp,
                           kd = kv (MuJoCo's dampratio rule), passive damping, friction and armature. So this
                           tests "URDF kinematics and inertia + the build record's joint terms".
  --source usd             a fixed-base scene written by import_scene.py (default
                           isaac_sim/generated/tonypi_urdf_fixed_scene.usda) through add_usd, exactly as authored:
                           the scene layer carries the servo gains, armature and passive terms. The check only
                           verifies that Newton read the scene's drive gains (per-degree USD values converted
                           back) and armature; it sets nothing. Floating scenes are refused: a free base adds
                           contact and balance, which this check does not cover.

Every source simulates --seconds (default 1 s) at the twin's 4 ms step on --device (default cuda:0) with
newton.solvers.SolverMuJoCo (use_mujoco_cpu=False, i.e. mujoco_warp). One joint is commanded (default head_pan to
+15 deg), all others are held at 0. The same command is run on the twin with CPU MuJoCo as the reference. Reported:
the commanded joint's trajectory (peak, value at sampled times, min/max over the final --settle-s), its largest
deviation from the twin's trajectory, Newton's joint, DOF, body and shape counts, the gains, armature, passive
terms and velocity limits Newton read, and whether every state stayed finite.

Exit status 1 if a state is not finite, the joint counts are wrong, the joint has not settled (some sample in the
final --settle-s misses the target by more than --tolerance-deg, or the final speed exceeds --settle-speed), or
the trajectory deviates from the twin's by more than --twin-tolerance-deg anywhere.

This checks that the model loads, steps and follows the twin's single-joint servo response in Newton. It is not
a validation of contact, balance or the physical robot.

Usage (from the repository root inside WSL):
  /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py [--source mjcf|urdf|usd] [--scene PATH]
      [--joint head_pan] [--target-deg 15] [--seconds 1.0] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
TWIN = REPO / "build/tonypi_pro_as_is.xml"
URDF = REPO / "isaac_sim/generated/tonypi_pro_candidate.urdf"
RECORD = REPO / "isaac_sim/generated/tonypi_pro_candidate.build.json"
SCENE = REPO / "isaac_sim/generated/tonypi_urdf_fixed_scene.usda"
EXPECTED_REVOLUTE = 20
DEG = math.pi / 180.0
SAMPLE_TIMES = (0.1, 0.25, 0.5, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("mjcf", "urdf", "usd"), default="mjcf")
    parser.add_argument("--scene", type=Path, default=SCENE, help="USD scene for --source usd")
    parser.add_argument("--joint", default="head_pan", help="joint to command (default head_pan)")
    parser.add_argument("--target-deg", type=float, default=15.0)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance-deg", type=float, default=1.0,
                        help="largest accepted |q - target| over the final --settle-s (default 1 deg)")
    parser.add_argument("--settle-s", type=float, default=0.2, help="final window that must be settled (s)")
    parser.add_argument("--settle-speed", type=float, default=0.05,
                        help="largest accepted |qd| of the commanded joint at the end (rad/s)")
    parser.add_argument("--twin-tolerance-deg", type=float, default=1.5,
                        help="largest accepted deviation from the twin's trajectory at any step (default 1.5 deg; "
                             "Newton 1.6.1 clamps drive torque on the joint, not the actuator, which under "
                             "implicitfast keeps the servo damping active during saturation: about 1 deg on a "
                             "15 deg head_pan step)")
    args = parser.parse_args()
    if args.scene != SCENE and args.source != "usd":
        parser.error("--scene is only used with --source usd")
    if not (0 < args.seconds <= 10):
        parser.error("--seconds must be in (0, 10]: keep GPU jobs short")
    if not (0 < args.settle_s <= args.seconds):
        parser.error("--settle-s must be in (0, --seconds]")
    for name in ("target_deg", "tolerance_deg", "settle_speed", "twin_tolerance_deg"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.tolerance_deg <= 0 or args.settle_speed <= 0 or args.twin_tolerance_deg <= 0:
        parser.error("tolerances must be positive")
    return args


def shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def twin_options() -> tuple[float, dict[str, object]]:
    """The twin's step and the solver options it sets (integrator, cone, impratio), applied to every source."""
    option = ET.parse(TWIN).getroot().find("option")
    if option is None or option.get("timestep") is None:
        raise SystemExit(f"{TWIN}: no <option timestep>; refusing to guess the step size")
    solver_options: dict[str, object] = {}
    for key in ("integrator", "cone"):
        if option.get(key) is not None:
            solver_options[key] = option.get(key)
    if option.get("impratio") is not None:
        solver_options["impratio"] = float(option.get("impratio"))
    return float(option.get("timestep")), solver_options


def twin_reference(joint: str, target: float, dt: float, steps: int):
    """The commanded joint's trajectory in the twin itself (CPU MuJoCo), one sample per step."""
    import mujoco
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mujoco_compat import load_mjcf

    model, edits = load_mjcf(TWIN)
    for edit in edits:
        print(f"Twin reference: {edit}", file=sys.stderr)
    if abs(model.opt.timestep - dt) > 1e-15:
        raise SystemExit(f"twin timestep {model.opt.timestep} differs from {dt}")
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    actuators = [a for a in range(model.nu) if model.actuator_trnid[a, 0] == joint_id]
    if joint_id < 0 or len(actuators) != 1:
        raise SystemExit(f"twin has no single servo for {joint}")
    data.ctrl[actuators[0]] = target
    address = model.jnt_qposadr[joint_id]
    out = np.empty(steps)
    for step in range(steps):
        mujoco.mj_step(model, data)
        out[step] = data.qpos[address]
    return out, mujoco.__version__


def record_terms() -> dict[str, dict]:
    try:
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        return record["importer_joint_terms"]["joints"]
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"cannot read the URDF build record's joint terms ({RECORD}): {exc}; "
                         "run isaac_sim/build_tonypi_urdf.py") from exc


def scene_drive_gains(scene: Path) -> dict[str, dict[str, float]]:
    """Drive gains and armature as authored in the scene (USD angular gains are per degree)."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(scene))
    out: dict[str, dict[str, float]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        get = lambda name: prim.GetAttribute(name).Get() if prim.GetAttribute(name) else None  # noqa: E731
        stiffness, damping = get("drive:angular:physics:stiffness"), get("drive:angular:physics:damping")
        if stiffness is None or damping is None:
            raise SystemExit(f"{scene}: joint {prim.GetName()} has no drive gains; rerun isaac_sim/import_scene.py")
        out[prim.GetName()] = {"ke": float(stiffness) / DEG, "kd": float(damping) / DEG,
                               "armature": float(get("newton:armature") or 0.0)}
    return out


def main() -> int:
    args = parse_args()
    source = {"mjcf": TWIN, "urdf": URDF, "usd": args.scene.resolve()}[args.source]
    if not source.is_file():
        hint = {"urdf": "; run isaac_sim/build_tonypi_urdf.py on Windows first",
                "usd": "; run isaac_sim/import_scene.py first"}.get(args.source, "")
        raise SystemExit(f"missing {source}{hint}")
    import numpy as np
    import warp as wp
    import newton
    import newton.solvers

    dt, solver_options = twin_options()  # the twin's own step and solver options, used for every source
    steps = max(1, round(args.seconds / dt))
    target = math.radians(args.target_deg)
    terms = record_terms()
    if args.joint not in terms:
        raise SystemExit(f"unknown joint {args.joint}")

    builder = newton.ModelBuilder()
    t0 = time.perf_counter()
    if args.source == "mjcf":
        # The vendor's collision classes are "collision" (soles, hand pads) and "selfcol" (guards); "internal"
        # boxes only carried mass, which the twin already folds into explicit inertials.
        builder.add_mjcf(str(source), floating=False, collider_classes=("collision", "selfcol"),
                         visual_classes=("visual",), ignore_classes=("internal",))
    elif args.source == "urdf":
        # URDF cannot carry the vendor's contact excludes, so neighbouring guards (for example head_pan and the
        # shoulder brackets, which overlap at home) would push each other apart. Same choice as import_scene.py.
        builder.add_urdf(str(source), floating=False, enable_self_collisions=False)
    else:
        builder.add_usd(str(source), enable_self_collisions=False)
    free = [i for i, kind in enumerate(builder.joint_type) if int(kind) == int(newton.JointType.FREE)]
    if free:
        raise SystemExit(f"{shown(source)} has a floating base; this check covers fixed-base scenes only")
    reference, mujoco_version = twin_reference(args.joint, target, dt, steps)
    labels = list(builder.joint_label)
    matches = [i for i, label in enumerate(labels) if label.rsplit("/", 1)[-1] == args.joint]
    if len(matches) != 1:
        raise SystemExit(f"expected one Newton joint named {args.joint}, found {len(matches)} in {labels}")
    joint = matches[0]
    drives: dict[str, dict[str, float]] = {}
    expected_gains = scene_drive_gains(source) if args.source == "usd" else None
    revolute_dofs: list[int] = []
    for index, label in enumerate(labels):
        if int(builder.joint_type[index]) != int(newton.JointType.REVOLUTE):
            continue
        name = label.rsplit("/", 1)[-1]
        dof = builder.joint_qd_start[index]
        revolute_dofs.append(dof)
        if args.source == "urdf":
            if builder.joint_target_mode[dof] != int(newton.JointTargetMode.POSITION):
                raise SystemExit(f"Newton did not give URDF joint {name} a position drive; refusing to guess")
            t = terms[name]
            builder.joint_target_ke[dof] = float(t["kp"])
            builder.joint_target_kd[dof] = float(t["kv"])
            builder.joint_damping[dof] = float(t["damping"])
            builder.joint_friction[dof] = float(t["frictionloss"])
            builder.joint_armature[dof] = float(t["armature"])
        elif args.source == "usd":
            want = expected_gains.get(name)  # type: ignore[union-attr]
            got = (builder.joint_target_ke[dof], builder.joint_target_kd[dof], builder.joint_armature[dof])
            if want is None or any(abs(g - w) > 1e-5 * max(abs(w), 1e-9)
                                   for g, w in zip(got, (want["ke"], want["kd"], want["armature"]))):
                raise SystemExit(f"Newton read {name} ke/kd/armature {got}, the scene authors {want}")
        drives[name] = {"ke": builder.joint_target_ke[dof], "kd": builder.joint_target_kd[dof],
                        "armature": builder.joint_armature[dof], "damping": builder.joint_damping[dof]}
    model = builder.finalize(device=args.device)
    load_s = time.perf_counter() - t0

    joint_type = model.joint_type.numpy()
    revolute = int(np.sum(joint_type == int(newton.JointType.REVOLUTE)))
    q_index = int(model.joint_q_start.numpy()[joint])
    qd_index = int(model.joint_qd_start.numpy()[joint])
    state_a, state_b = model.state(), model.state()
    control = model.control()
    if args.source == "mjcf":
        actuators = list(model.mujoco.actuator_label)
        if actuators.count(args.joint) != 1:
            raise SystemExit(f"expected one MuJoCo actuator named {args.joint}, found {actuators}")
        ctrl = control.mujoco.ctrl.numpy()
        ctrl[actuators.index(args.joint)] = target
        control.mujoco.ctrl.assign(ctrl)
        servo = "twin MuJoCo position servos (ledger kp, dampratio, force cap)"
    else:
        targets = control.joint_target_q.numpy()  # Newton >= 1.5 (joint_target_pos was removed)
        targets[q_index] = target
        control.joint_target_q.assign(targets)
        servo = ("Newton position drives set from the URDF build record (ke = kp, kd = kv, + passive damping, "
                 "friction, armature)" if args.source == "urdf" else
                 "Newton position drives exactly as the import_scene.py scene authors them")
    # MuJoCo-Warp computes its own contacts. A URDF or USD source carries no MuJoCo <option>, so the twin's
    # integrator, cone and impratio are passed explicitly (an MJCF source sets the same values itself).
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, **solver_options)

    trajectory = np.empty(steps)
    t1 = time.perf_counter()
    for step in range(steps):
        solver.step(state_a, state_b, control, None, dt)
        state_a, state_b = state_b, state_a
        trajectory[step] = state_a.joint_q.numpy()[q_index]
    wp.synchronize_device(args.device)
    sim_s = time.perf_counter() - t1
    q = state_a.joint_q.numpy()
    qd = state_a.joint_qd.numpy()
    body_q = state_a.body_q.numpy()
    finite = bool(np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(body_q).all()
                  and np.isfinite(trajectory).all())
    deg = np.degrees(trajectory)
    twin_deg = np.degrees(reference)
    window = max(1, round(args.settle_s / dt))
    final = deg[-window:]
    others = [abs(math.degrees(float(q[d]))) for d in revolute_dofs if d != qd_index]
    deviation = np.abs(deg - twin_deg)
    samples = {f"{s:g}": round(float(deg[round(s / dt) - 1]), 4) for s in SAMPLE_TIMES + (steps * dt,)
               if round(s / dt) <= steps}
    twin_samples = {f"{s:g}": round(float(twin_deg[round(s / dt) - 1]), 4) for s in SAMPLE_TIMES + (steps * dt,)
                    if round(s / dt) <= steps}
    velocity_limits = model.joint_velocity_limit.numpy()
    commanded = drives.get(args.joint, {})
    result = {
        "source": shown(source),
        "device": str(wp.get_device(args.device)),
        "versions": {"newton": newton.__version__, "warp": wp.__version__, "mujoco_reference": mujoco_version},
        "solver": "newton.solvers.SolverMuJoCo(use_mujoco_cpu=False) (mujoco_warp)",
        "solver_options_from_twin": solver_options,
        "newton_joints": int(model.joint_count),
        "revolute_joints": revolute,
        "dofs": int(model.joint_dof_count),
        "bodies": int(model.body_count),
        "shapes": int(model.shape_count),
        "total_mass_kg": float(model.body_mass.numpy().sum()),
        "dt_s": dt,
        "steps": steps,
        "simulated_s": steps * dt,
        "commanded_joint": args.joint,
        "servo": servo,
        "commanded_joint_drive_read_by_newton": {k: round(float(v), 9) for k, v in commanded.items()},
        "target_deg": args.target_deg,
        "reached_deg": float(deg[-1]),
        "peak_deg": float(deg.max()),
        "final_window_s": window * dt,
        "final_window_min_max_deg": [float(final.min()), float(final.max())],
        "final_speed_rad_s": float(qd[qd_index]),
        "trajectory_deg_at_s": samples,
        "twin_cpu_mujoco": {"peak_deg": float(twin_deg.max()), "trajectory_deg_at_s": twin_samples},
        "max_deviation_from_twin_deg": float(deviation.max()),
        "max_deviation_at_s": float((int(deviation.argmax()) + 1) * dt),
        "max_abs_other_revolute_deg": max(others) if others else 0.0,
        "all_states_finite": finite,
        "velocity_limits_read_by_newton_rad_s": sorted({round(float(v), 6) for v in velocity_limits}),
        "passive_damping_read_by_newton": sorted({round(float(v), 6) for v in model.joint_damping.numpy()}),
        "friction_read_by_newton": sorted({round(float(v), 6) for v in model.joint_friction.numpy()}),
        "armature_read_by_newton": sorted({round(float(v), 6) for v in model.joint_armature.numpy()}),
        "load_s": round(load_s, 2),
        "sim_wall_s": round(sim_s, 2),
    }
    print(json.dumps(result, indent=2))
    problems = []
    if not finite:
        problems.append("non-finite joint or body state")
    if revolute != EXPECTED_REVOLUTE or int(model.joint_dof_count) != EXPECTED_REVOLUTE:
        problems.append(f"expected {EXPECTED_REVOLUTE} revolute joints and DOFs, found {revolute} / {model.joint_dof_count}")
    worst = float(np.abs(final - args.target_deg).max())
    if not worst <= args.tolerance_deg:
        problems.append(f"{args.joint} not settled: final {window * dt:g} s spans {final.min():.3f}..{final.max():.3f} "
                        f"deg, target {args.target_deg} +- {args.tolerance_deg}")
    if not abs(float(qd[qd_index])) <= args.settle_speed:
        problems.append(f"{args.joint} still moving at {float(qd[qd_index]):.4f} rad/s (limit {args.settle_speed})")
    if not float(deviation.max()) <= args.twin_tolerance_deg:
        problems.append(f"{args.joint} deviates from the twin's trajectory by {float(deviation.max()):.3f} deg at "
                        f"{result['max_deviation_at_s']:.3f} s (limit {args.twin_tolerance_deg})")
    for problem in problems:
        print(f"CHECK FAILED: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
