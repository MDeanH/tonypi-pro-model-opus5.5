"""List the frames, duration and servo channels of Hiwonder action-group files (.d6a) without changing them.

Usage: .venv/bin/python -I tools/d6a_info.py <file.d6a> [more.d6a ...]

A .d6a file is an SQLite database with an `ActionGroup` table: one row per keyframe, an integer `Index` (play
order), a `Time` (how long the move to this keyframe takes, in milliseconds) and `ServoN` columns holding bus-servo
pulse targets (0-1000). Files are opened read-only and immutable, so SQLite creates no journal or lock file next to
them. Only the structure is checked here; whether the pulses mean the same joint angles on your robot depends on its
servo calibration (see model/servo_map.csv).
"""
import os
import re
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

MAX_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 10_000
MAX_TOTAL_MS = 10 * 60 * 1000
SERVO_COLUMN = re.compile(r"^Servo(\d+)$")


class D6AError(ValueError):
    pass


def connect_readonly(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise D6AError(f"{path}: not a file")
    if path.stat().st_size > MAX_BYTES:
        raise D6AError(f"{path}: larger than {MAX_BYTES} bytes")
    uri = "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def read_action(path):
    """Validated frames: {'servo_ids': [ids...], 'frames': [{'index', 'time_ms', 'pulses': {id: int|None}}]}."""
    # sqlite3's own context manager only ends transactions; closing() releases the file (Windows locks it otherwise).
    with closing(connect_readonly(path)) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "ActionGroup" not in tables:
            raise D6AError(f"{path}: no ActionGroup table (tables: {sorted(tables)})")
        cols = [r[1] for r in con.execute("PRAGMA table_info(ActionGroup)")]
        if "Index" not in cols or "Time" not in cols:
            raise D6AError(f"{path}: ActionGroup lacks Index/Time columns ({cols})")
        servo_cols = {int(m.group(1)): c for c in cols if (m := SERVO_COLUMN.match(c))}
        other = [c for c in cols if c not in ("Index", "Time") and c not in servo_cols.values()]
        if other:
            raise D6AError(f"{path}: unexpected ActionGroup columns {other}")
        ids = sorted(servo_cols)
        select = ", ".join(['"Index"', '"Time"'] + [f'"{servo_cols[i]}"' for i in ids])
        rows = con.execute(f"SELECT {select} FROM ActionGroup ORDER BY \"Index\"").fetchall()
    if not rows:
        raise D6AError(f"{path}: ActionGroup is empty")
    if len(rows) > MAX_FRAMES:
        raise D6AError(f"{path}: more than {MAX_FRAMES} frames")
    frames, seen, total = [], set(), 0
    for row in rows:
        index, time_ms, values = row[0], row[1], row[2:]
        if not isinstance(index, int) or index in seen:
            raise D6AError(f"{path}: Index {index!r} is not a unique integer")
        if not isinstance(time_ms, int) or time_ms <= 0:
            raise D6AError(f"{path}: frame {index} Time {time_ms!r} is not a positive integer (ms)")
        pulses = {}
        for sid, v in zip(ids, values):
            if v is not None and (not isinstance(v, int) or not 0 <= v <= 1000):
                raise D6AError(f"{path}: frame {index} Servo{sid} = {v!r} is not an integer pulse 0-1000")
            pulses[sid] = v
        seen.add(index)
        total += time_ms
        frames.append({"index": index, "time_ms": time_ms, "pulses": pulses})
    if total > MAX_TOTAL_MS:
        raise D6AError(f"{path}: longer than {MAX_TOTAL_MS / 60000:.0f} minutes")
    return {"servo_ids": ids, "frames": frames, "total_ms": total}


def describe(path):
    a = read_action(path)
    lines = [f"{os.path.basename(path)}: {len(a['frames'])} frames, {a['total_ms'] / 1000:.2f} s, "
             f"servo columns {a['servo_ids']}"]
    for sid in a["servo_ids"]:
        vals = [f["pulses"][sid] for f in a["frames"] if f["pulses"][sid] is not None]
        nulls = len(a["frames"]) - len(vals)
        rng = f"{min(vals)}-{max(vals)}" if vals else "no values"
        lines.append(f"  Servo{sid:<2} pulses {rng}{f', {nulls} empty' if nulls else ''}")
    return "\n".join(lines)


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        raise SystemExit(__doc__)
    status = 0
    for p in paths:
        try:
            print(describe(p))
        except (D6AError, sqlite3.DatabaseError) as e:
            print(f"{p}: ERROR {e}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
