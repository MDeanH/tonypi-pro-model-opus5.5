"""Servo channels of the TonyPi Pro and the pulse <-> joint-angle conversion, from model/servo_map.csv.

Usage:
  .venv/bin/python -I tools/servo_map.py            print the map
  .venv/bin/python -I tools/servo_map.py --seed     create/merge model/servo_map.csv (never overwrites unit_* cells)

    angle_rad = sign * (pulse - zero_pulse) * deg_per_pulse * pi / 180

Two calibrations live side by side in the CSV:
  ref_*   the upstream author's robot (pab47/TonyPi control/servo_map.py at 97807f5): zero pulses read with that
          robot standing in the model's zero pose, and mounting signs. ANOTHER robot's numbers: servo horns are
          seated per unit, so your zeros can differ by tens of pulses.
  unit_*  YOUR TonyPi Pro, blank until measured (move one servo at a time; record the pulse at the model's zero
          pose and whether increasing pulse moves the joint in the model's positive direction).
Converters refuse to use a calibration that has blank cells for a channel they are asked to convert.
"""
import csv
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVO_MAP = os.path.join(REPO, "model", "servo_map.csv")
COLUMNS = ["channel", "interface", "id", "joint", "servo_model", "pulse_min", "pulse_max", "deg_per_pulse",
           "mapping_evidence", "ref_zero_pulse", "ref_sign", "ref_evidence", "unit_zero_pulse", "unit_sign",
           "date", "evidence", "note"]
CALIBRATIONS = ("unit", "reference")
UPSTREAM = "pab47/TonyPi 97807f5"
ID_DIAGRAM = "https://cdn.shopify.com/s/files/1/0084/2799/5187/files/1.1_1_979b4f2c-2d0d-425d-890c-9acd0a05c194.jpg?v=1747120276"
LFD01M_SHEET = "https://cdn.shopify.com/s/files/1/0084/2799/5187/files/19.1_1.jpg?v=1747052133"
MAPPING_EVIDENCE = (f"Hiwonder 20DOF ID diagram {ID_DIAGRAM} (robot's left = 1-8) and control/TonyPi_sw/tonypi_sim.py "
                    f"SERVO_JOINT @ {UPSTREAM}; the vendor XML comments beside the body actuators reverse 1-8 and 9-16")
GRIPPER_EVIDENCE = ("Hiwonder code: right_splay()/right_grasp() drive ID 17 "
                    "(wiki.hiwonder.com/projects/TonyPi_Pro/en/latest/docs/8_vision_gripping_lessons.html); stock "
                    "grab_right.d6a moves 14-16 + 17 and grab_left.d6a moves 6-8 + 18. The wiki PC-software page and "
                    "the vendor XML comments say the reverse (17 = left)")
PWM_EVIDENCE = (f"Hiwonder TonyPi Pro docs and code: PWM servo 1 = head tilt (nod), 2 = pan (shake), 1500 us neutral; "
                f"LFD-01M sheet {LFD01M_SHEET} 'PWM pulse 500~2500us(0-180deg)'; vendor XML comments agree")
REF_EVIDENCE = (f"control/servo_map.py ZERO/SIGN @ {UPSTREAM}: measured on the upstream author's robot (a TonyPi with "
                f"fixed hands) standing in the model zero pose; date and servo deviations not recorded")
BUS_SCALE_EVIDENCE = f"control/servo_map.py DEG_PER_PULSE = 240/1000 @ {UPSTREAM} ('Hiwonder LX-series spec; verify later')"
# Upstream servo_map.py (bus servos 1-16 only). Keys are bus IDs.
UPSTREAM_ZERO = {1: 498, 2: 479, 3: 247, 4: 413, 5: 508, 6: 500, 7: 870, 8: 664,
                 9: 495, 10: 515, 11: 762, 12: 584, 13: 500, 14: 516, 15: 129, 16: 337}
UPSTREAM_SIGN = {1: +1, 2: +1, 3: +1, 4: -1, 5: -1, 6: -1, 7: -1, 8: +1,
                 9: +1, 10: -1, 11: -1, 12: +1, 13: -1, 14: -1, 15: -1, 16: -1}
SIDE_ORDER = ("ankle_roll", "ankle_pitch", "knee", "hip_pitch", "hip_roll", "elbow", "shoulder_roll", "shoulder_pitch")


class CalibrationError(ValueError):
    pass


def default_rows():
    rows = []
    for i, part in enumerate(SIDE_ORDER):
        for side, base in (("l", 1), ("r", 9)):
            sid = base + i
            rows.append(dict(channel=f"bus{sid}", interface="bus", id=str(sid), joint=f"{side}_{part}",
                             servo_model="LX-824HV", pulse_min="0", pulse_max="1000", deg_per_pulse="0.24",
                             mapping_evidence=MAPPING_EVIDENCE, ref_zero_pulse=str(UPSTREAM_ZERO[sid]),
                             ref_sign=f"{UPSTREAM_SIGN[sid]:+d}", ref_evidence=REF_EVIDENCE,
                             note=BUS_SCALE_EVIDENCE))
    for sid, joint, opened in ((17, "r_gripper", 760), (18, "l_gripper", 240)):
        rows.append(dict(channel=f"bus{sid}", interface="bus", id=str(sid), joint=joint, servo_model="unpublished",
                         pulse_min="0", pulse_max="1000", deg_per_pulse="", mapping_evidence=GRIPPER_EVIDENCE,
                         note=f"hand servo model and travel unpublished; Hiwonder code uses {opened} = open, 500 = "
                              f"closed. No pulse-to-angle calibration exists; the upstream author's robot had fixed hands"))
    for sid, joint in ((1, "head_tilt"), (2, "head_pan")):
        rows.append(dict(channel=f"pwm{sid}", interface="pwm", id=str(sid), joint=joint, servo_model="LFD-01M",
                         pulse_min="500", pulse_max="2500", deg_per_pulse="0.09", mapping_evidence=PWM_EVIDENCE,
                         note="units are microseconds of PWM pulse (1500 = neutral); the head is not part of .d6a "
                              "files and its zero/sign against the model are not established"))
    rows.sort(key=lambda r: (r["interface"] != "bus", int(r["id"])))
    return rows


def read(path=SERVO_MAP):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or list(rows[0]) != COLUMNS:
        raise SystemExit(f"{path}: unexpected columns")
    return rows


def seed(path=SERVO_MAP):
    """Create the CSV, or add missing channels / fill blank reference cells. Unit cells are never touched."""
    rows = read(path) if os.path.exists(path) else []
    have = {r["channel"]: r for r in rows}
    added = 0
    for d in default_rows():
        r = have.get(d["channel"])
        if r is None:
            rows.append({c: d.get(c, "") for c in COLUMNS})
            added += 1
            continue
        if r["joint"] != d["joint"]:
            raise SystemExit(f"{d['channel']}: CSV says {r['joint']}, default map says {d['joint']}; resolve by hand")
    rows.sort(key=lambda r: (r["interface"] != "bus", int(r["id"])))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return added, len(rows)


def bus_channels(rows=None):
    """{bus id: row} for the bus servos."""
    rows = rows if rows is not None else read()
    return {int(r["id"]): r for r in rows if r["interface"] == "bus"}


def calibration(which, ids, rows=None):
    """{bus id: (zero_pulse, sign, deg_per_pulse)} for the requested ids, or CalibrationError naming the gaps."""
    if which not in CALIBRATIONS:
        raise CalibrationError(f"calibration must be one of {CALIBRATIONS}")
    prefix = "unit" if which == "unit" else "ref"
    bus = bus_channels(rows)
    out, gaps = {}, []
    for sid in ids:
        r = bus.get(sid)
        if r is None:
            raise CalibrationError(f"bus servo {sid} is not in model/servo_map.csv")
        zero, sign = r[f"{prefix}_zero_pulse"], r[f"{prefix}_sign"]
        if zero == "" or sign == "":
            gaps.append(f"bus{sid} ({r['joint']})")
            continue
        if sign not in ("+1", "-1", "1"):
            raise CalibrationError(f"bus{sid}: {prefix}_sign must be +1 or -1, not {sign!r}")
        out[sid] = (float(zero), float(sign), float(r["deg_per_pulse"]))
    if gaps:
        hint = ("measure your unit and fill unit_zero_pulse/unit_sign in model/servo_map.csv, or choose the "
                "upstream robot's numbers explicitly with --calibration reference")
        raise CalibrationError(f"{which} calibration is blank for {', '.join(gaps)}: {hint}")
    return out


def pulse_to_rad(pulse, zero, sign, deg_per_pulse):
    return sign * (pulse - zero) * math.radians(deg_per_pulse)


def rad_to_pulse(angle, zero, sign, deg_per_pulse):
    return zero + sign * angle / math.radians(deg_per_pulse)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--seed" in argv:
        added, total = seed()
        print(f"servo_map.csv: +{added} channels ({total} total)")
        return 0
    for r in read():
        print(f"{r['channel']:<6} {r['joint']:<17} {r['servo_model']:<9} ref zero {r['ref_zero_pulse'] or '-':>4} "
              f"sign {r['ref_sign'] or '-':>2} | unit zero {r['unit_zero_pulse'] or '-':>4} sign {r['unit_sign'] or '-':>2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
