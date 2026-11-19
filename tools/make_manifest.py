"""Write or check SOURCE_MANIFEST.json: byte count and SHA-256 of every file this repository ships.

Usage:
  .venv/bin/python -I tools/make_manifest.py           rewrite SOURCE_MANIFEST.json
  .venv/bin/python -I tools/make_manifest.py --check   verify every listed file, and that no shipped file is unlisted

Shipped = every file under the repository root except .git/, .venv/, caches, generated scratch (see SKIP) and the
manifest itself. Paths use forward slashes so the manifest is identical on Windows and macOS.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "SOURCE_MANIFEST.json")
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "generated", ".pytest_cache"}
SKIP_SUFFIXES = (".pyc", ".mjb", ".tmp")
SKIP_FILES = {"SOURCE_MANIFEST.json", "MUJOCO_LOG.TXT", "preview-local.png", ".DS_Store"}


def shipped():
    out = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if f in SKIP_FILES or f.endswith(SKIP_SUFFIXES) or f.startswith("rebuilt"):
                continue
            out.append(os.path.relpath(os.path.join(dirpath, f), REPO).replace(os.sep, "/"))
    return sorted(out)


def entry(rel):
    with open(os.path.join(REPO, rel), "rb") as f:
        data = f.read()
    return {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        with open(MANIFEST, encoding="utf-8") as f:
            listed = json.load(f)["files"]
        bad = [e["path"] for e in listed if not os.path.exists(os.path.join(REPO, e["path"]))
               or entry(e["path"]) != e]
        unlisted = sorted(set(shipped()) - {e["path"] for e in listed})
        for p in bad:
            print(f"CHANGED OR MISSING: {p}")
        for p in unlisted:
            print(f"NOT IN MANIFEST: {p}")
        print(f"{len(listed) - len(bad)}/{len(listed)} listed files match; {len(unlisted)} unlisted")
        return 1 if bad or unlisted else 0
    files = [entry(p) for p in shipped()]
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"note": "SHA-256 of every shipped file; regenerate with tools/make_manifest.py",
                   "files": files}, f, indent=2)
        f.write("\n")
    print(f"wrote SOURCE_MANIFEST.json: {len(files)} files, {sum(e['bytes'] for e in files)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
