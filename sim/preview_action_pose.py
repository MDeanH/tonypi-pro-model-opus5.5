"""Check a .d6a action group and optionally pose (and render) one of its frames on the fixed-base twin.

Usage:
  .venv/bin/python -I sim/preview_action_pose.py <file.d6a> [--frame N] [--calibration unit|reference]
                                                 [--render out.png]

Without --calibration it only validates and lists the file. With one, it converts the frame's pulses to joint angles
(model/servo_map.csv), sets those coordinates and calls mj_forward: no dynamics step, no inference of what the
servos would actually reach. Files inside data/action-groups/ are checked against its MANIFEST.json first.
Rendering creates a new PNG only: it refuses existing files and the data, vendor, build and model folders.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

import mujoco

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/action-groups"
MODEL = ROOT / "build/tonypi_pro_as_is.xml"
PROTECTED = [ROOT / d for d in ("data", "vendor", "build", "model")]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "sim"))
import d6a_info  # noqa: E402
import servo_map  # noqa: E402
from play_action import HAND_IDS, load_targets  # noqa: E402


def verified_if_listed(path: Path):
    """If the file lives in data/action-groups, its bytes must match MANIFEST.json; returns the entry or None."""
    resolved = path.resolve()
    if not resolved.is_relative_to(DATA.resolve()):
        return None
    manifest = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
    rel = resolved.relative_to(DATA.resolve()).as_posix()
    entry = next((e for e in manifest["files"] if e["path"] == rel), None)
    if entry is None:
        raise ValueError(f"{rel} is in data/action-groups but not in its MANIFEST.json")
    data = path.read_bytes()
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{rel} differs from data/action-groups/MANIFEST.json")
    return entry


def validate_render_output(output: Path) -> None:
    resolved = output.resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("render output must be a .png file")
    if any(resolved.is_relative_to(p.resolve()) for p in PROTECTED):
        raise ValueError("render output cannot be placed in data/, vendor/, build/ or model/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output}")


def render(model, data, output: Path) -> None:
    validate_render_output(output)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    # The fixed-base XML keeps MuJoCo's default 640x480 offscreen buffer; enlarge it in memory only.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 540)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    with mujoco.Renderer(model, height=720, width=540) as renderer:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.0, 0.0, 0.185)
        camera.distance, camera.azimuth, camera.elevation = 0.9, 120.0, -14.0
        renderer.update_scene(data, camera=camera, scene_option=opt)
        image = renderer.render()

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xffffffff)
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(image.shape[0]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:  # exclusive create: never overwrites
        stream.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", image.shape[1], image.shape[0],
                                                                         8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def pose_frame(frames, wanted):
    """Joint targets after frames 0..wanted, carrying forward each servo's last value (NULL = keep previous)."""
    if not 0 <= wanted < len(frames):
        raise IndexError(f"frame {wanted} outside the action group ({len(frames)} frames)")
    pose = {}
    for _, targets in frames[:wanted + 1]:
        pose.update({j: v for j, v in targets.items() if v is not None})
    return pose


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", type=Path)
    ap.add_argument("--frame", type=int, default=0, help="0-based frame (row in Index order)")
    ap.add_argument("--calibration", choices=servo_map.CALIBRATIONS)
    ap.add_argument("--render", type=Path, help="save a PNG of the posed frame (requires --calibration)")
    a = ap.parse_args(argv)
    if a.frame < 0 or (a.render and not a.calibration):
        ap.error("--frame must be nonnegative; --render requires --calibration")
    if a.render:
        validate_render_output(a.render)
    entry = verified_if_listed(a.action)
    action = d6a_info.read_action(a.action)
    result = {"status": "file/software evidence; joint mapping from upstream, calibration as chosen, physical fit "
                        "unverified",
              "action": a.action.name, "manifest_checked": entry is not None, "frames": len(action["frames"]),
              "duration_s": action["total_ms"] / 1000.0, "servo_columns": action["servo_ids"],
              "hand_columns_present": [i for i in action["servo_ids"] if i in HAND_IDS],
              "selected_frame": a.frame}
    if a.frame >= len(action["frames"]):
        raise SystemExit(f"frame {a.frame} outside the action group ({len(action['frames'])} frames)")
    if a.calibration:
        try:
            frames, _ = load_targets(a.action, a.calibration)
        except servo_map.CalibrationError as e:
            raise SystemExit(f"refused: {e}")
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = mujoco.MjData(model)
        pose = pose_frame(frames, a.frame)
        outside = []
        for name, value in pose.items():
            lo, hi = model.joint(name).range
            if not lo <= value <= hi:
                outside.append(name)
            data.joint(name).qpos[0] = value
        mujoco.mj_forward(model, data)
        result.update({"calibration": a.calibration, "posed_joints": len(pose),
                       "joints_outside_model_range": outside,
                       "pose_rad": {k: round(v, 6) for k, v in sorted(pose.items())}})
        if a.render:
            render(model, data, a.render)
            result["render"] = str(a.render)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
