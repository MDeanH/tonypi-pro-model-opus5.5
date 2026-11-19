"""Seed model/ledger.csv and model/measurements.csv from the vendor MJCF (spine 2: every twin number has a source).

Usage: .venv/bin/python -I tools/seed_ledger.py [--fresh]

Merge-only: adds rows that don't exist yet and never changes or removes an existing row, so it's safe to re-run
after measurements of your own TonyPi Pro have been added. Anything the twin needs that no source provides gets an
explicit `missing` row, so what's unknown shows up in the ledger instead of being hidden.

Ledger rows: param, value, unit, source, status, date, evidence, note
  source  vendor | docs | estimate | measured-on-unit | verified-on-unit   (empty for a missing row)
          vendor   = the pinned pab47/TonyPi MJCF (a third-party model, MIT); `note` repeats what the upstream file
                     itself says about the number (e.g. "upstream: ESTIMATE") so a vendor row is never mistaken
                     for a Hiwonder specification
          docs     = a Hiwonder publication (URL in `evidence`)
          estimate = a stated-method guess, flagged in every build report
  status  active | deprecated | missing     (at most one active row per param; a unit value deprecates the vendor one)
"""
import csv
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import inspect_mjcf  # noqa: E402

VENDOR = inspect_mjcf.VENDOR
LEDGER = os.path.join(REPO, "model", "ledger.csv")
MEASUREMENTS = os.path.join(REPO, "model", "measurements.csv")
LEDGER_COLUMNS = ["param", "value", "unit", "source", "status", "date", "evidence", "note"]
MEASUREMENT_COLUMNS = ["id", "description", "unit", "vendor_value", "docs_value", "unit_left", "unit_right",
                       "unit_value", "date", "evidence"]
SOURCES = {"", "vendor", "docs", "estimate", "measured-on-unit", "verified-on-unit"}
STATUSES = {"active", "deprecated", "missing"}
UNIT_SOURCES = ("measured-on-unit", "verified-on-unit")
VENDOR_EVIDENCE = "vendor/tonypi/model/robot.xml @ pab47/TonyPi 97807f5"
COMPILED_EVIDENCE = VENDOR_EVIDENCE + " (MuJoCo 3.17.0 inertial from the file's per-geom masses and shapes)"
SEED_DATE = "2026-11-19"
COMPILED_RTOL = 1e-9  # compiled rows are re-derived on every check; allow float noise, nothing more
INERTIA_KEYS = inspect_mjcf.INERTIA_KEYS

# What the upstream file's own comments say about a class-level number (verbatim gist, see robot.xml <default>).
UPSTREAM_NOTE = {
    ("body_servo", "damping"): "upstream: LX-824HV torque-speed line, c = stall/no-load = 1.667/5.236 N*m*s/rad",
    ("body_servo", "frictionloss"): "upstream gives no basis for this value",
    ("body_servo", "armature"): "upstream: ESTIMATE, reflected rotor inertia assuming a ~250-300:1 reduction "
                                "(Hiwonder does not publish the LX-824HV gear ratio)",
    ("body_servo", "effort"): "upstream: flat torque cap at the LX-824HV stall torque",
    ("body_servo", "kp"): "upstream simulation PD gain; not a servo specification",
    ("body_servo", "dampratio"): "upstream simulation choice (it reports measured zeta 0.90-0.98 at 0.6)",
    ("head_servo", "damping"): "upstream: LFD-01M torque-speed line, c = 0.1765/10.472 N*m*s/rad",
    ("head_servo", "frictionloss"): "upstream gives no basis for this value",
    ("head_servo", "armature"): "upstream gives no basis for this value",
    ("head_servo", "effort"): "upstream: LFD-01M stall 1.8 kgf*cm = 0.1765 N*m",
    ("head_servo", "kp"): "upstream simulation PD gain; not a servo specification",
    ("head_servo", "dampratio"): "upstream simulation choice",
}
KP_OVERRIDE_NOTE = ("upstream ROLL-STIFFNESS FIX: kp raised 12 -> 64 on hip/ankle roll to widen simulated foot "
                    "clearance while walking; a simulation tuning, not a hardware property")
INERTIAL_NOTE = ("compiled from upstream per-part masses: LX-824HV cases 57 g, LFD-01M cases 14 g, battery 148 g, "
                 "Raspberry Pi 5 46 g, expansion board 55 g, plus structure/fastener/sole boxes")

# Hiwonder publications (source=docs). Each row: param, value, unit, evidence (URL + quote), note.
PRO_PAGE = "https://www.hiwonder.com/products/tonypi-pro"
DRAWING = "https://cdn.shopify.com/s/files/1/0084/2799/5187/files/18_41b7106a-0259-4d28-b035-b19d51cc560d.jpg?v=1747052133"
LX824HV_SHEET = "https://cdn.shopify.com/s/files/1/0084/2799/5187/files/19.2_2.jpg?v=1747052134"
LFD01M_SHEET = "https://cdn.shopify.com/s/files/1/0084/2799/5187/files/19.1_1.jpg?v=1747052133"
KGFCM = 0.0980665  # N*m per kgf*cm
LX_SPEED = (math.pi / 3) / 0.20  # 0.20 s per 60 deg
LFD_SPEED = (math.pi / 3) / 0.10  # 0.10 s per 60 deg at 6 V
BODY_JOINTS = [f"{s}_{j}" for s in ("r", "l") for j in ("hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll",
                                                       "shoulder_pitch", "shoulder_roll", "elbow")]
DOCS = [
    ("robot.mass_label", "1.8", "kg", f"{PRO_PAGE} 'Weight: About 1800g'",
     "the 18-DOF TonyPi page (no hands) lists the same 1800 g, so this may exclude the Pro hands"),
    ("robot.size_label.height", "0.373", "m", f"{PRO_PAGE} 'Size: 373* 186* 106mm'",
     "height x width x depth per Hiwonder's dimension drawing"),
    ("robot.size_label.width", "0.186", "m", f"{PRO_PAGE} 'Size: 373* 186* 106mm'",
     "height x width x depth per Hiwonder's dimension drawing"),
    ("robot.size_label.depth", "0.106", "m", f"{PRO_PAGE} 'Size: 373* 186* 106mm'",
     "torso/head depth: the same drawing shows a 113 mm foot"),
    ("robot.drawing.foot_length", "0.113", "m", f"{DRAWING} (side view, '113mm')",
     "read from Hiwonder's dimension-drawing image; reading, not a stated specification"),
    ("robot.drawing.foot_width", "0.0785", "m", f"{DRAWING} (front view, '78.5mm')",
     "read from Hiwonder's dimension-drawing image; reading, not a stated specification"),
    ("robot.drawing.stance_width", "0.169", "m", f"{DRAWING} (front view, '169mm', outer edge to outer edge)",
     "read from Hiwonder's dimension-drawing image; standing pose of the drawing, not necessarily the model zero"),
    ("servo.LX-824HV.stall_torque", repr(round(17 * KGFCM, 7)), "N*m", f"{LX824HV_SHEET} 'Torque 17kg.cm/11.1V(LX-824HV)'",
     "17 kgf*cm at 11.1 V, converted at 0.0980665 N*m per kgf*cm"),
    ("servo.LX-824HV.no_load_speed", repr(round(LX_SPEED, 9)), "rad/s", f"{LX824HV_SHEET} 'Speed 0.20sec/60deg(11.1V)'",
     "0.20 s per 60 degrees at 11.1 V"),
    ("servo.LFD-01M.stall_torque", repr(round(1.8 * KGFCM, 7)), "N*m", f"{LFD01M_SHEET} 'Locked torque 1.5KG.cm 4.8V 1.8KG.cm6V'",
     "1.8 kgf*cm at 6 V (1.5 at 4.8 V); the TonyPi board's PWM servo supply voltage is not established here"),
    ("servo.LFD-01M.no_load_speed", repr(round(LFD_SPEED, 9)), "rad/s", f"{LFD01M_SHEET} 'Rotate speed 0.12sec/60deg4.8V; 0.10sec/60deg6V'",
     "0.10 s per 60 degrees at 6 V (0.12 s at 4.8 V = 8.727 rad/s); supply voltage not established here"),
] + [(f"joint.{j}.limit.velocity", repr(round(LX_SPEED, 9)), "rad/s", f"{LX824HV_SHEET} 'Speed 0.20sec/60deg(11.1V)'",
      "LX-824HV no-load speed (Hiwonder's spec table names LX-824HV for the body servos); loaded speed is lower")
     for j in BODY_JOINTS] + [
    (f"joint.{j}.limit.velocity", repr(round(LFD_SPEED, 9)), "rad/s", f"{LFD01M_SHEET} 'Rotate speed ... 0.10sec/60deg6V'",
     "LFD-01M no-load speed at 6 V (head pan/tilt); 8.727 rad/s at 4.8 V; supply voltage not established here")
    for j in ("head_pan", "head_tilt")]


def _fmt(x):
    return repr(float(f"{x:.12g}"))


def vendor_rows(path=VENDOR):
    """Every number the build reads from the vendor model, as (param, value, unit, kind, note) in file order.
    kind is 'literal' (string copied from the file) or 'compiled' (computed by MuJoCo from the file)."""
    v = inspect_mjcf.Vendor(path)
    inert = inspect_mjcf.compiled_inertials(path)
    out = []
    home = v.keys["home"]["qpos"]
    for k, val in zip(("x", "y", "z", "qw", "qx", "qy", "qz"), home[:7]):
        out.append((f"scene.home.root.{k}", val, "m" if k in "xyz" else "-", "literal",
                    "vendor `home` keyframe free-root pose; the fixed-base twin is welded here"))
    for b in v.bodies:
        name = b["name"]
        mass, com, inertia = inert[name]
        out.append((f"link.{name}.mass", _fmt(mass), "kg", "compiled", INERTIAL_NOTE))
        out += [(f"link.{name}.com.{k}", _fmt(c), "m", "compiled", INERTIAL_NOTE) for k, c in zip("xyz", com)]
        out += [(f"link.{name}.inertia.{k}", _fmt(i), "kg*m^2", "compiled", INERTIAL_NOTE)
                for k, i in zip(INERTIA_KEYS, inertia)]
        j = b["joint"]
        if j is None:
            continue
        joint = v.joint(j)
        cls = joint["cls"]
        out += [(f"joint.{j}.origin.{k}", p, "m", "literal", "upstream joint frame (STEP-derived per its generator)")
                for k, p in zip("xyz", b["pos"])]
        if b["quat"] or b["euler"]:
            raise SystemExit(f"vendor body {name} has an orientation attribute; extend seed_ledger to read it")
        out += [(f"joint.{j}.origin.{k}", "0", "rad", "literal",
                 f"no quat/euler on vendor body {name}: MJCF default identity orientation")
                for k in ("roll", "pitch", "yaw")]
        out += [(f"joint.{j}.axis.{k}", a, "-", "literal", "") for k, a in zip("xyz", joint["axis"])]
        out += [(f"joint.{j}.limit.lower", joint["range"][0], "rad", "literal", ""),
                (f"joint.{j}.limit.upper", joint["range"][1], "rad", "literal", "")]
        for k, unit in (("damping", "N*m*s/rad"), ("frictionloss", "N*m"), ("armature", "kg*m^2")):
            out.append((f"joint.{j}.dynamics.{k}", v.joint_default(cls, k), unit, "literal",
                        UPSTREAM_NOTE[(cls, k)]))
        lo, hi = v.position_default(cls, "forcerange").split()
        if float(lo) != -float(hi):
            raise SystemExit(f"class {cls}: asymmetric forcerange {lo} {hi}; extend seed_ledger")
        out.append((f"joint.{j}.limit.effort", hi, "N*m", "literal", UPSTREAM_NOTE[(cls, "effort")]))
    for a in v.actuators:
        if a["tag"] != "position":
            raise SystemExit(f"actuator {a['name']} is <{a['tag']}>; extend seed_ledger")
        cls = a["cls"]
        out.append((f"actuator.{a['name']}.kp", v.kp(a), "N*m/rad", "literal",
                    KP_OVERRIDE_NOTE if a["kp"] is not None else UPSTREAM_NOTE[(cls, "kp")]))
        out.append((f"actuator.{a['name']}.dampratio", v.position_default(cls, "dampratio"), "-", "literal",
                    UPSTREAM_NOTE[(cls, "dampratio")]))
        out += [(f"actuator.{a['name']}.ctrlrange.lower", a["ctrlrange"][0], "rad", "literal", ""),
                (f"actuator.{a['name']}.ctrlrange.upper", a["ctrlrange"][1], "rad", "literal", "")]
    return out


def vendor_values(path=VENDOR):
    return {p: (val, kind) for p, val, _, kind, _ in vendor_rows(path)}


def missing_rows(path=VENDOR):
    """Params the twin should carry that neither the vendor file nor a Hiwonder publication provides."""
    v = inspect_mjcf.Vendor(path)
    docs = {d[0] for d in DOCS}
    out = []
    for j in v.joints:
        p = f"joint.{j['name']}.limit.velocity"
        if p not in docs:
            out.append((p, "rad/s", "no velocity limit in the vendor file, and Hiwonder does not publish the hand "
                                    "servo model; measure on your unit" if "gripper" in p else
                        "no velocity limit in the vendor file; need the servo's published no-load speed"))
    return out


def measurement_rows(path=VENDOR):
    """Ruler-and-scale table. Vendor values are computed from the vendor model at its home pose (no hand-typed
    numbers); docs values are Hiwonder's published figures where one exists."""
    poses, m, d = inspect_mjcf.zero_pose(path)
    lo, hi = inspect_mjcf.mesh_extents(m, d)
    p = lambda n: poses[n][1]
    dist = lambda a, b: math.dist(p(a), p(b))
    r4 = lambda x: f"{x:.4f}"
    soles = {s: inspect_mjcf.geom_box_world(m, d, f"{s}_ankle_roll_sole") for s in ("l", "r")}
    sole_bottom = min(c[2] - h[2] for c, h, _ in soles.values())
    (lc, lh, _), (rc, rh, _) = soles["l"], soles["r"]
    inner_gap = (lc[0] - lh[0]) - (rc[0] + rh[0])
    total = float(m.body_mass.sum())
    doc = {d[0]: d[1] for d in DOCS}
    return [
        ("M01", "Top of head to sole, legs straight, head level", "m", r4(hi[2] - sole_bottom), doc.get("robot.size_label.height", "")),
        ("M02", "Shoulder-pitch axis centre, left to right", "m", r4(dist("l_shoulder_pitch", "r_shoulder_pitch")), ""),
        ("M03", "Shoulder-roll centre to elbow centre (upper arm), per side", "m", r4(dist("r_shoulder_roll", "r_elbow")), ""),
        ("M04", "Elbow centre to hand-servo axis (forearm), per side", "m", r4(dist("r_elbow", "r_gripper")), ""),
        ("M05", "Hip-roll axis centre, left to right", "m", r4(dist("l_hip_roll", "r_hip_roll")), ""),
        ("M06", "Hip-pitch centre to knee centre (thigh), per side", "m", r4(dist("r_hip_pitch", "r_knee")), ""),
        ("M07", "Knee centre to ankle-pitch centre (shin), per side", "m", r4(dist("r_knee", "r_ankle_pitch")), ""),
        ("M08", "Ankle-roll centre to sole, per side", "m", r4(p("r_ankle_roll")[2] - sole_bottom), ""),
        ("M09", "Shoulder-pitch centre to sole", "m", r4(p("r_shoulder_pitch")[2] - sole_bottom), ""),
        ("M10", "Widest point across, arms hanging", "m", r4(hi[0] - lo[0]), doc.get("robot.size_label.width", "")),
        ("M11", "Front-to-back depth", "m", r4(hi[1] - lo[1]), doc.get("robot.size_label.depth", "")),
        ("M12", "Sole length, front to back, per foot", "m", r4(2 * rh[1]), doc.get("robot.drawing.foot_length", "")),
        ("M13", "Sole width, side to side, per foot", "m", r4(2 * rh[0]), doc.get("robot.drawing.foot_width", "")),
        ("M14", "Inner-edge gap between the feet, home pose", "m", r4(inner_gap), ""),
        ("M15", "Weight, battery fitted", "kg", f"{total:.3f}", doc.get("robot.mass_label", "")),
        ("M16", "Outer edge to outer edge of the soles, home pose", "m", r4((lc[0] + lh[0]) - (rc[0] - rh[0])),
         doc.get("robot.drawing.stance_width", "")),
    ]


def ledger_problems(rows, path=VENDOR):
    """Spine-2 checks: vocabulary, one active row per param, vendor rows still equal the vendor file."""
    problems = []
    vendor = vendor_values(path)
    docs = {d[0]: d[1] for d in DOCS}
    active = {}
    for r in rows:
        p = r["param"]
        if r["source"] not in SOURCES or r["status"] not in STATUSES:
            problems.append(f"{p}: bad source/status {r['source']!r}/{r['status']!r}")
        if r["status"] == "missing" and (r["value"] or r["source"]):
            problems.append(f"{p}: a missing row must have no value and no source")
        if r["status"] != "missing" and (r["value"] == "" or not r["source"] or not r["evidence"]):
            problems.append(f"{p}: a {r['status']} row needs a value, a source and evidence")
        if r["status"] != "missing" and r["value"] != "":
            try:
                if not math.isfinite(float(r["value"])):
                    problems.append(f"{p}: value {r['value']!r} is not finite")
            except ValueError:
                problems.append(f"{p}: value {r['value']!r} is not a number")
        if r["status"] == "active":
            if p in active:
                problems.append(f"{p}: more than one active row")
            active[p] = r
        if r["source"] == "vendor":
            if p not in vendor:
                problems.append(f"{p}: vendor row has no counterpart in the vendor file")
            else:
                want, kind = vendor[p]
                if kind == "literal" and r["value"] != want:
                    problems.append(f"{p}: vendor row {r['value']!r} != vendor file {want!r}")
                if kind == "compiled" and not math.isclose(float(r["value"]), float(want), rel_tol=COMPILED_RTOL,
                                                           abs_tol=1e-15):
                    problems.append(f"{p}: vendor row {r['value']!r} != compiled vendor value {want!r}")
        if r["source"] == "docs" and p in docs and r["value"] != docs[p]:
            problems.append(f"{p}: docs row {r['value']!r} != seed_ledger.DOCS {docs[p]!r}")
    for p in vendor:
        if not any(r["param"] == p and r["source"] == "vendor" for r in rows):
            problems.append(f"{p}: in the vendor file but not in the ledger")
        elif p not in active and not any(r["param"] == p and r["status"] == "missing" for r in rows):
            problems.append(f"{p}: vendor row deprecated without an active replacement")
    known = set(vendor) | set(docs) | {m[0] for m in missing_rows(path)}
    for r in rows:
        if r["param"] not in known:
            problems.append(f"{r['param']}: unknown parameter; nothing reads it (check the spelling)")
    for r in rows:
        if r["status"] == "missing" and r["param"] in active:
            problems.append(f"{r['param']}: marked missing but also has an active row")
    return problems


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    os.replace(tmp, path)


def seed(ledger_path=LEDGER, measurements_path=MEASUREMENTS, path=VENDOR, fresh=False):
    rows = read(ledger_path)
    if fresh:
        if any(r["source"] in UNIT_SOURCES for r in rows):
            raise SystemExit("refusing --fresh: the ledger holds measurements of your unit")
        rows = []
    have = {r["param"] for r in rows}
    active = {r["param"] for r in rows if r["status"] == "active"}
    have_vendor = {r["param"] for r in rows if r["source"] == "vendor"}
    have_docs = {r["param"] for r in rows if r["source"] == "docs"}
    added = 0
    for param, value, unit, kind, note in vendor_rows(path):
        if param in have_vendor:
            continue
        status = "deprecated" if param in active else "active"
        rows.append(dict(param=param, value=value, unit=unit, source="vendor", status=status, date=SEED_DATE,
                         evidence=COMPILED_EVIDENCE if kind == "compiled" else VENDOR_EVIDENCE, note=note))
        have.add(param)
        active.add(param)
        added += 1
    for param, value, unit, evidence, note in DOCS:
        if param in have_docs:
            continue
        status = "deprecated" if param in active else "active"
        rows = [r for r in rows if not (r["param"] == param and r["status"] == "missing")]
        rows.append(dict(param=param, value=value, unit=unit, source="docs", status=status, date=SEED_DATE,
                         evidence=evidence, note=note))
        have.add(param)
        active.add(param)
        added += 1
    for param, unit, note in missing_rows(path):
        if param not in active and not any(r["param"] == param and r["status"] == "missing" for r in rows):
            rows.append(dict(param=param, value="", unit=unit, source="", status="missing", date=SEED_DATE,
                             evidence="", note=note))
            added += 1
    write(ledger_path, LEDGER_COLUMNS, rows)

    meas = read(measurements_path)
    by_id = {r["id"]: r for r in meas}
    madd = 0
    for mid, desc, unit, vendor_value, docs_value in measurement_rows(path):
        if mid in by_id:
            if by_id[mid]["vendor_value"] != vendor_value:
                raise SystemExit(f"{mid}: vendor value changed {by_id[mid]['vendor_value']} -> {vendor_value} "
                                 f"(vendor edited?)")
            if not by_id[mid].get("docs_value") and docs_value:
                by_id[mid]["docs_value"] = docs_value
            continue
        meas.append(dict(id=mid, description=desc, unit=unit, vendor_value=vendor_value, docs_value=docs_value))
        madd += 1
    write(measurements_path, MEASUREMENT_COLUMNS, meas)
    return added, madd, len(rows), len(meas)


if __name__ == "__main__":
    # --fresh rebuilds from scratch, and refuses if any measurement of your unit is in the ledger.
    added, madd, total, mtotal = seed(fresh="--fresh" in sys.argv[1:])
    print(f"ledger: +{added} rows ({total} total) · measurements: +{madd} rows ({mtotal} total)")
