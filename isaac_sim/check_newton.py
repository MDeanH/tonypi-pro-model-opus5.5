"""Newton GPU smoke check of the TonyPi Pro twin with the MuJoCo-Warp solver (WSL Isaac Lab venv).

--source mjcf (default) loads build/tonypi_pro_as_is.xml, the fixed-base twin with its own 20 MuJoCo position
servos (ledger kp, dampratio and force caps), into newton.ModelBuilder.add_mjcf. --source urdf loads the generated
isaac_sim/generated/tonypi_pro_candidate.urdf with add_urdf instead, with self-collision off (URDF cannot carry
the vendor's contact excludes). URDF carries no servo: Newton 1.6.1 gives each URDF joint a position drive with
stiffness 0 and damping = the URDF <dynamics damping>, so this check sets every drive's stiffness to the ledger's
actuator.<joint>.kp and holds all joints but the commanded one at 0. Those drives are a smoke-test servo, not the
twin's servo (no armature, no dampratio). --source usd loads a scene written by import_scene.py (default
isaac_sim/generated/tonypi_urdf_fixed_scene.usda) with add_usd: this is the Isaac Sim 6.2 importer's USD as a
Newton-based simulator reads it. Its drives carry a force cap but no gains, so the same ledger stiffness is set
(damping 0: the USD's passive joint damping applies).

All sources simulate --seconds (default 1 s) on --device (default cuda:0) with newton.solvers.SolverMuJoCo
(use_mujoco_cpu=False, i.e. mujoco_warp), command one joint (default head_pan to +15 deg), and report the reached
angle, Newton's joint, DOF, body and shape counts, the velocity limits Newton read, and whether every joint and
body state stayed finite. Exit status 1 if a state is not finite, the joint counts are wrong or the joint misses
its target by more than --tolerance-deg.

This checks that the model loads and steps in Newton. It is not a validation of dynamics, contact or balance.

Usage (from the repository root inside WSL):
  /root/luka-isaac/IsaacLab/.venv/bin/python isaac_sim/check_newton.py [--source mjcf|urdf|usd] [--scene PATH]
      [--joint head_pan] [--target-deg 15] [--seconds 1.0] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
TWIN = REPO / "build/tonypi_pro_as_is.xml"
URDF = REPO / "isaac_sim/generated/tonypi_pro_candidate.urdf"
SCENE = REPO / "isaac_sim/generated/tonypi_urdf_fixed_scene.usda"
LEDGER = REPO / "model/ledger.csv"
EXPECTED_REVOLUTE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("mjcf", "urdf", "usd"), default="mjcf")
    parser.add_argument("--scene", type=Path, default=SCENE, help="USD scene for --source usd")
    parser.add_argument("--joint", default="head_pan", help="joint to command (default head_pan)")
    parser.add_argument("--target-deg", type=float, default=15.0)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance-deg", type=float, default=1.0,
                        help="largest accepted |reached - target| (default 1 deg)")
    args = parser.parse_args()
    if args.scene != SCENE and args.source != "usd":
        parser.error("--scene is only used with --source usd")
    if not (0 < args.seconds <= 10):
        parser.error("--seconds must be in (0, 10]: keep GPU jobs short")
    if not math.isfinite(args.target_deg) or not math.isfinite(args.tolerance_deg) or args.tolerance_deg <= 0:
        parser.error("--target-deg and --tolerance-deg must be finite (tolerance positive)")
    return args


def active_ledger() -> dict[str, str]:
    with LEDGER.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    active: dict[str, str] = {}
    for row in rows:
        if row["status"] == "active":
            if row["param"] in active:
                raise SystemExit(f"ledger has two active rows for {row['param']}")
            active[row["param"]] = row["value"]
    return active


def twin_timestep() -> float:
    option = ET.parse(TWIN).getroot().find("option")
    if option is None or option.get("timestep") is None:
        raise SystemExit(f"{TWIN}: no <option timestep>; refusing to guess the step size")
    return float(option.get("timestep"))


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

    ledger = active_ledger()
    dt = twin_timestep()  # the twin's own step (vendor option timestep), used for both sources
    steps = max(1, round(args.seconds / dt))
    target = math.radians(args.target_deg)
    lo, hi = float(ledger[f"joint.{args.joint}.limit.lower"]), float(ledger[f"joint.{args.joint}.limit.upper"])
    if not lo <= target <= hi:
        raise SystemExit(f"target {args.target_deg} deg is outside {args.joint}'s range [{lo}, {hi}] rad")

    builder = newton.ModelBuilder()
    t0 = time.perf_counter()
    if args.source == "mjcf":
        # The vendor's collision classes are "collision" (soles, hand pads) and "selfcol" (guards); "internal"
        # boxes only carried mass, which the twin already folds into explicit inertials.
        builder.add_mjcf(str(source), floating=False, collider_classes=("collision", "selfcol"),
                         visual_classes=("visual",), ignore_classes=("internal",))
    elif args.source == "urdf":
        # URDF cannot carry the vendor's contact excludes, so neighbouring guards (for example head_pan and the
        # shoulder brackets, which overlap at home) would push each other apart. Same choice as import_scene.py's
        # allow_self_collision=False.
        builder.add_urdf(str(source), floating=False, enable_self_collisions=False)
    else:
        builder.add_usd(str(source), enable_self_collisions=False)
    labels = list(builder.joint_label)
    matches = [i for i, label in enumerate(labels) if label.rsplit("/", 1)[-1] == args.joint]
    if len(matches) != 1:
        raise SystemExit(f"expected one Newton joint named {args.joint}, found {len(matches)} in {labels}")
    joint = matches[0]
    urdf_drives: dict[str, dict[str, float]] = {}
    if args.source in ("urdf", "usd"):
        # Newton's URDF parser gives every revolute joint a position drive with ke = 0 and kd = the URDF
        # <dynamics damping> (it does not fill joint_damping); the importer's USD drives have a force cap only and
        # its passive damping does land in joint_damping. Give each joint the twin's servo stiffness from the
        # ledger so all 20 hold their pose as the twin's 20 servos do.
        for index, label in enumerate(labels):
            name = label.rsplit("/", 1)[-1]
            if f"actuator.{name}.kp" not in ledger:
                continue
            dof = builder.joint_qd_start[index]
            if args.source == "urdf" and builder.joint_target_mode[dof] != int(newton.JointTargetMode.POSITION):
                raise SystemExit(f"Newton did not give URDF joint {name} a position drive; refusing to guess")
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
            builder.joint_target_ke[dof] = float(ledger[f"actuator.{name}.kp"])
            if args.source == "usd":
                builder.joint_target_kd[dof] = 0.0
            urdf_drives[name] = {"ke": builder.joint_target_ke[dof], "kd": builder.joint_target_kd[dof]}
        if len(urdf_drives) != EXPECTED_REVOLUTE:
            raise SystemExit(f"expected {EXPECTED_REVOLUTE} joints with a ledger kp, found {len(urdf_drives)}")
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
        servo = "twin MuJoCo position servo (ledger kp, dampratio, force cap)"
    else:
        targets = control.joint_target_q.numpy()  # Newton >= 1.5 (joint_target_pos was removed)
        targets[q_index] = target
        control.joint_target_q.assign(targets)
        drive = urdf_drives[args.joint]
        kd_text = ("URDF <dynamics damping> as read by Newton" if args.source == "urdf"
                   else "0 (passive USD joint damping applies)")
        servo = (f"Newton position drives on all 20 joints, ke = ledger actuator.*.kp, kd = {kd_text} "
                 f"({args.joint}: ke {drive['ke']}, kd {drive['kd']:.6g}); smoke test, not the twin's dampratio servo")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False)  # MuJoCo-Warp computes its own contacts
    contacts = None

    finite = True
    t1 = time.perf_counter()
    for _ in range(steps):
        solver.step(state_a, state_b, control, contacts, dt)
        state_a, state_b = state_b, state_a
    wp.synchronize_device(args.device)
    sim_s = time.perf_counter() - t1
    q = state_a.joint_q.numpy()
    qd = state_a.joint_qd.numpy()
    body_q = state_a.body_q.numpy()
    finite = bool(np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(body_q).all())
    reached = math.degrees(float(q[q_index]))
    others = [abs(math.degrees(float(v))) for i, v in enumerate(q) if i != q_index]
    velocity_limits = model.joint_velocity_limit.numpy()
    result = {
        "source": source.relative_to(REPO).as_posix(),
        "device": str(wp.get_device(args.device)),
        "versions": {"newton": newton.__version__, "warp": wp.__version__},
        "solver": "newton.solvers.SolverMuJoCo(use_mujoco_cpu=False) (mujoco_warp)",
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
        "target_deg": args.target_deg,
        "reached_deg": reached,
        "error_deg": reached - args.target_deg,
        "max_abs_other_joint_deg": max(others) if others else 0.0,
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
    if not abs(reached - args.target_deg) <= args.tolerance_deg:
        problems.append(f"{args.joint} reached {reached:.3f} deg, target {args.target_deg} +- {args.tolerance_deg}")
    for problem in problems:
        print(f"CHECK FAILED: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
