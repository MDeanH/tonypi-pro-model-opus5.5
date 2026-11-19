"""Render the README preview: the floor-contact twin at its `home` keyframe, front three-quarter view.

Usage: .venv/bin/python -I tools/render_preview.py [--out media/preview.png] [--model build/tonypi_pro_contact_candidate.xml]

The image is an illustration of the model's zero pose drawn by MuJoCo's renderer (2x supersampled). It is not a
photograph or scan of a TonyPi Pro and shows the vendor's simplified hull meshes, not Hiwonder's real parts.
Sensor sites are hidden. Writes a PNG with only the standard library (no Pillow).
"""
import argparse
import os
import struct
import sys
import zlib

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO, "build", "tonypi_pro_contact_candidate.xml")
DEFAULT_OUT = os.path.join(REPO, "media", "preview.png")
PROTECTED = [os.path.join(REPO, d) for d in ("vendor", "build", "model", "data")]
WIDTH, HEIGHT, SUPERSAMPLE = 540, 720, 2
CAMERA = dict(lookat=(0.0, 0.0, 0.185), distance=0.9, azimuth=120.0, elevation=-14.0)  # robot faces -Y


def write_png(path, img):
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def render(model_path, key="home"):
    m = mujoco.MjModel.from_xml_path(model_path)
    m.vis.quality.offsamples = 0  # no MSAA: its noise made re-renders differ; the 2x supersample antialiases
    d = mujoco.MjData(m)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    if k >= 0:
        mujoco.mj_resetDataKeyframe(m, d, k)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAMERA["lookat"]
    cam.distance, cam.azimuth, cam.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    with mujoco.Renderer(m, HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        big = r.render().astype(np.float64)
    small = big.reshape(HEIGHT, SUPERSAMPLE, WIDTH, SUPERSAMPLE, 3).mean(axis=(1, 3))
    return np.clip(np.rint(small), 0, 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if not out.lower().endswith(".png"):
        raise SystemExit("output must be a .png file")
    if any(os.path.commonpath([out, p]) == p for p in PROTECTED):
        raise SystemExit("refusing to write a render into vendor/, build/, model/ or data/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_png(out, render(os.path.abspath(a.model)))
    print(f"rendered {os.path.relpath(out, REPO).replace(os.sep, '/')} ({WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
