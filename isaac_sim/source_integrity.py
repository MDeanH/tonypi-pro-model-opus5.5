"""Verify every local model input before Isaac import and record USD provenance.

SOURCE_MANIFEST.json pins the fixed-base twin MJCF, the parameter ledger, the vendor model, its STL meshes and
the tools that define them. The floating contact MJCF must also reproduce byte-for-byte from
sim/build_contact_scene.py (and match its manifest entry). The generated URDF is not in the manifest (generated
files never are); it is bound to its inputs by its build record, isaac_sim/generated/tonypi_pro_candidate.build.json:
the URDF bytes, the generator and every input the generator read must still have the recorded hashes. A
byte-for-byte regeneration of the URDF needs MuJoCo (the ledger check compiles the vendor model), so it is a
separate step in the project venv: isaac_sim/build_tonypi_urdf.py --check.

This module uses only the Python standard library and runs without Isaac Sim, including inside Isaac Sim's own
python. A hash match proves file identity and reproducible derivation, not physical accuracy for a TonyPi Pro.

Usage (repository root):
  python isaac_sim/source_integrity.py --source urdf
  python isaac_sim/source_integrity.py --source mjcf [--floating]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "SOURCE_MANIFEST.json"
GENERATED = REPO / "isaac_sim/generated"
BASELINE = REPO / "build/tonypi_pro_as_is.xml"
CONTACT = REPO / "build/tonypi_pro_contact_candidate.xml"
URDF = GENERATED / "tonypi_pro_candidate.urdf"
URDF_RECORD = GENERATED / "tonypi_pro_candidate.build.json"
EXPECTED_JOINTS = 20

# Files that determine the model's numbers and structure. Meshes are added from each source's own references.
COMMON_PINNED = (
    "model/ledger.csv",
    "requirements-sim.txt",
    "tools/inspect_mjcf.py",
    "tools/seed_ledger.py",
    "vendor/tonypi/SHA256SUMS",
    "vendor/tonypi/model/robot.xml",
    "vendor/tonypi/model/scene.xml",  # the vendor floor's friction, used for every scene's ground
)
MJCF_PINNED = COMMON_PINNED + ("build/tonypi_pro_as_is.xml", "model/build_twin.py")
CONTACT_PINNED = MJCF_PINNED + ("sim/build_contact_scene.py",)
RECORD_SCHEMA = 2


class IntegrityError(ValueError):
    """An input changed, is absent, or cannot be accounted for."""


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError as exc:
        raise IntegrityError(f"file is outside the TonyPi Pro repository: {resolved}") from exc


def fingerprint(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise IntegrityError(f"missing file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise IntegrityError(f"cannot read {path}: {exc}") from exc
    return {"path": repo_relative(path), "bytes": size, "sha256": digest.hexdigest()}


def load_manifest() -> tuple[dict[str, dict], dict[str, object]]:
    identity = fingerprint(MANIFEST)
    try:
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))
        entries = document["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IntegrityError(f"invalid SOURCE_MANIFEST.json: {exc}") from exc
    if not isinstance(entries, list):
        raise IntegrityError("SOURCE_MANIFEST.json: files must be a list")
    by_path: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise IntegrityError("SOURCE_MANIFEST.json: file entry must be an object")
        name = entry.get("path")
        if not isinstance(name, str) or not name or name in by_path:
            raise IntegrityError(f"SOURCE_MANIFEST.json: duplicate or invalid path {name!r}")
        expected_size, expected_hash = entry.get("bytes"), entry.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise IntegrityError(f"SOURCE_MANIFEST.json: invalid byte count for {name}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(
            char not in "0123456789abcdef" for char in expected_hash
        ):
            raise IntegrityError(f"SOURCE_MANIFEST.json: invalid SHA-256 for {name}")
        # Only canonical repository-relative paths can be used as identities.
        candidate = (REPO / name).resolve()
        if repo_relative(candidate) != name.replace("\\", "/"):
            raise IntegrityError(f"SOURCE_MANIFEST.json: unsafe path {name!r}")
        by_path[name] = entry
    return by_path, identity


def verify_pinned(path: Path, entries: dict[str, dict]) -> dict[str, object]:
    actual = fingerprint(path)
    name = actual["path"]
    if name not in entries:
        raise IntegrityError(f"input is not pinned by SOURCE_MANIFEST.json: {name}")
    expected = entries[name]
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise IntegrityError(
            f"manifest hash or size mismatch: {name} (review provenance; if the change is intended, "
            "regenerate SOURCE_MANIFEST.json with tools/make_manifest.py)"
        )
    return actual


def resolve_local(reference: str | None, base: Path, label: str) -> Path:
    if not reference or "://" in reference:
        raise IntegrityError(f"{label}: unsupported or missing local file reference {reference!r}")
    path = Path(reference)
    if path.is_absolute():
        raise IntegrityError(f"{label}: absolute file reference {reference!r}; generated sources must be portable")
    resolved = (base / path).resolve()
    repo_relative(resolved)  # Reject files outside the pinned repository.
    return resolved


def mjcf_references(source: Path) -> list[Path]:
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise IntegrityError(f"cannot parse MJCF {source}: {exc}") from exc
    if root.tag != "mujoco":
        raise IntegrityError(f"{source}: expected <mujoco> root")
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    basedir = resolve_local(meshdir, source.parent, "MJCF meshdir") if meshdir else source.parent
    assets = root.find("asset")
    if assets is None:
        raise IntegrityError(f"{source}: missing <asset> section")
    references: list[Path] = []
    for element in assets.iter():
        if "file" in element.attrib:
            # Texture files would resolve against texturedir, not meshdir. Fail closed if one appears.
            if element.tag != "mesh":
                raise IntegrityError(f"{source}: unsupported external asset <{element.tag} file=...>")
            references.append(resolve_local(element.get("file"), basedir, "MJCF mesh"))
    if not references:
        raise IntegrityError(f"{source}: no external mesh assets")
    return references


def urdf_references(source: Path) -> list[Path]:
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise IntegrityError(f"cannot parse URDF {source}: {exc}") from exc
    if root.tag != "robot":
        raise IntegrityError(f"{source}: expected <robot> root")
    references: list[Path] = []
    for element in root.iter():
        if "filename" in element.attrib:
            if element.tag != "mesh":
                raise IntegrityError(f"{source}: unsupported external asset <{element.tag} filename=...>")
            references.append(resolve_local(element.get("filename"), source.parent, "URDF mesh"))
    if not references:
        raise IntegrityError(f"{source}: no external mesh assets")
    return references


def verify_urdf_record(source: Path) -> tuple[dict, dict[str, object]]:
    """The URDF and everything its generator read must still match the build record."""
    record_path = source.with_name(source.stem + ".build.json")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrityError(
            f"missing or unreadable URDF build record {record_path}: {exc}; run isaac_sim/build_tonypi_urdf.py"
        ) from exc
    if not isinstance(record, dict) or record.get("schema_version") != RECORD_SCHEMA:
        raise IntegrityError(f"{record_path}: unsupported build record schema")
    try:
        urdf_entry, generator, inputs = record["urdf"], record["generator"], record["inputs"]
        joints, placeholders = record["joints"], record["velocity_placeholders"]
        home = record["home_root_pose"]
        terms = record["importer_joint_terms"]["joints"]
        contact_classes = record["contact_classes"]
        collision_sources = record["collision_sources"]
    except (KeyError, TypeError) as exc:
        raise IntegrityError(f"{record_path}: missing key {exc}") from exc
    if fingerprint(source) != urdf_entry:
        raise IntegrityError(
            f"{repo_relative(source)} differs from its build record (hand-edited or stale); "
            "run isaac_sim/build_tonypi_urdf.py"
        )
    for entry in [generator, *inputs]:
        path = (REPO / str(entry.get("path", ""))).resolve()
        if fingerprint(path) != entry:
            raise IntegrityError(
                f"{entry.get('path')} changed since {repo_relative(source)} was built; "
                "run isaac_sim/build_tonypi_urdf.py again"
            )
    if len(joints) != EXPECTED_JOINTS or not set(placeholders) <= set(joints):
        raise IntegrityError(f"{record_path}: unexpected joint list or velocity placeholders")
    if set(home) != {"x", "y", "z", "qw", "qx", "qy", "qz"}:
        raise IntegrityError(f"{record_path}: incomplete home_root_pose")
    if set(terms) != set(joints) or not isinstance(contact_classes, dict) or not collision_sources:
        raise IntegrityError(f"{record_path}: importer joint terms, contact classes or collision sources incomplete")
    return record, fingerprint(record_path)


def verify_derivative(source_type: str, source: Path, floating: bool) -> dict[str, object]:
    """Bind a derived source to its manifest-verified parents. The caller also pins the parent files."""
    if source_type == "mjcf" and floating:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        try:
            from sim.build_contact_scene import generate
        except ImportError as exc:
            raise IntegrityError(f"cannot import sim/build_contact_scene.py: {exc}") from exc
        try:
            expected = generate()  # itself refuses unless build/tonypi_pro_as_is.xml matches the manifest
        except (ValueError, OSError) as exc:
            raise IntegrityError(f"contact scene generator refused: {exc}") from exc
        if source.read_bytes() != expected:
            raise IntegrityError(
                f"derived source differs from its generator: {repo_relative(source)}; "
                "run python sim/build_contact_scene.py --check"
            )
        return {"method": "byte-for-byte regeneration by sim/build_contact_scene.py"}
    if source_type == "urdf":
        record, record_identity = verify_urdf_record(source)
        return {
            "method": ("build record hash chain (URDF, generator and every generator input); byte-for-byte "
                       "regeneration is isaac_sim/build_tonypi_urdf.py --check in the project venv"),
            "build_record": record_identity,
            "joints": record["joints"],
            "velocity_placeholders": record["velocity_placeholders"],
            "velocity_placeholder_value": record.get("velocity_placeholder_value"),
            "home_root_pose": record["home_root_pose"],
            "importer_joint_terms": record["importer_joint_terms"],
            "contact_classes": record["contact_classes"],
            "collision_sources": record["collision_sources"],
        }
    return {"method": "source is pinned directly by SOURCE_MANIFEST.json"}


def expected_source(source_type: str, floating: bool) -> Path:
    if source_type == "urdf":
        return URDF
    return CONTACT if floating else BASELINE


def verify_source_bundle(source_type: str, source: Path, floating: bool = False) -> dict[str, object]:
    """Return a JSON-ready record after verifying all import input files.

    ``source_type`` is ``mjcf`` or ``urdf``. The fixed twin is directly manifest-pinned; the floating MJCF must
    also reproduce byte-for-byte from its generator, and the URDF must match its build record. Referenced meshes
    must be pinned.
    """
    if source_type not in ("mjcf", "urdf"):
        raise IntegrityError(f"unknown source type: {source_type}")
    source = source.resolve()
    if source != expected_source(source_type, floating).resolve():
        raise IntegrityError(f"unexpected {source_type} source: {source}")
    entries, manifest_identity = load_manifest()
    names = COMMON_PINNED if source_type == "urdf" else (CONTACT_PINNED if floating else MJCF_PINNED)
    pinned_paths = {REPO / name for name in names}
    mesh_refs = mjcf_references(source) if source_type == "mjcf" else urdf_references(source)
    pinned_paths.update(mesh_refs)
    pinned = [verify_pinned(path, entries) for path in sorted(pinned_paths)]
    derivation = verify_derivative(source_type, source, floating)
    listed = repo_relative(source) in entries
    source_identity = verify_pinned(source, entries) if listed else fingerprint(source)

    local_tools = [REPO / "isaac_sim/import_scene.py", Path(__file__).resolve()]
    if source_type == "urdf":
        local_tools.append(REPO / "isaac_sim/build_tonypi_urdf.py")
    if source_type == "mjcf":
        local_tools.append(REPO / "isaac_sim/mujoco_compat.py")
    if source_type == "mjcf" and floating:
        local_tools.append(REPO / "sim/build_contact_scene.py")
    return {
        "schema_version": 1,
        "source_type": source_type,
        "floating": floating,
        "source_manifest": manifest_identity,
        "source": {**source_identity, "manifest_pinned": listed},
        "derivation": derivation,
        "pinned_inputs": pinned,
        "referenced_mesh_count": len(mesh_refs),
        "local_tools": [fingerprint(path) for path in local_tools],
        "scope": "file identity and reproducible derivation; not physical validation of a TonyPi Pro",
    }


def comparable(bundle: dict[str, object]) -> dict[str, object]:
    """The parts of a verified bundle that identify the model inputs.

    The manifest file's own hash is left out: another tool may regenerate SOURCE_MANIFEST.json (for example to
    pin new files) without changing any input of this import. Every pinned input is still compared by hash."""
    return {key: value for key, value in bundle.items() if key != "source_manifest"}


def recheck(preflight: dict[str, object]) -> dict[str, object]:
    """Verify the sources again and refuse if any pinned input, the source or its derivation changed."""
    source_type = preflight["source_type"]
    floating = preflight["floating"]
    source_path = REPO / preflight["source"]["path"]  # type: ignore[index]
    current = verify_source_bundle(source_type, source_path, floating)  # type: ignore[arg-type]
    before, after = comparable(preflight), comparable(current)
    if before != after:
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        if "pinned_inputs" in changed:
            old = {item["path"]: item for item in before["pinned_inputs"]}  # type: ignore[union-attr,index]
            new = {item["path"]: item for item in after["pinned_inputs"]}  # type: ignore[union-attr,index]
            files = sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))
            changed = [c for c in changed if c != "pinned_inputs"] + [f"pinned input {p}" for p in files]
        raise IntegrityError(f"model inputs changed during the import: {', '.join(changed)}")
    return current


def write_integrity_report(preflight: dict[str, object], asset: Path, scene: Path,
                           extra: dict[str, object] | None = None) -> Path:
    """Recheck sources after import, then save hashes beside the imported USD."""
    current = recheck(preflight)
    asset, scene = asset.resolve(), scene.resolve()
    repo_relative(asset)
    repo_relative(scene)
    if asset.suffix.lower() not in (".usd", ".usda", ".usdc"):
        raise IntegrityError(f"importer output is not a USD file: {asset}")
    report_path = asset.with_name(asset.stem + ".integrity.json")
    assets = []
    for path in sorted(asset.parent.rglob("*")):
        if path.is_file() and not path.name.endswith(".integrity.json") and not path.name.startswith(".tonypi_integrity_"):
            assets.append(fingerprint(path))
    if not any(item["path"] == repo_relative(asset) for item in assets):
        raise IntegrityError(f"imported USD is missing: {asset}")
    report = {**current, "generated_scene": fingerprint(scene), "generated_assets": assets, **(extra or {})}
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=asset.parent, prefix=".tonypi_integrity_",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, report_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("urdf", "mjcf"), default="urdf")
    parser.add_argument("--floating", action="store_true", help="MJCF only: the floating contact candidate")
    args = parser.parse_args()
    if args.source == "urdf" and args.floating:
        parser.error("--floating selects the MJCF contact candidate; the URDF is base-agnostic")
    try:
        result = verify_source_bundle(args.source, expected_source(args.source, args.floating), args.floating)
    except (IntegrityError, OSError, ValueError) as exc:
        print(f"Integrity check failed: {exc}", file=sys.stderr)
        return 1
    summary = {
        "source": result["source"]["path"],  # type: ignore[index]
        "manifest_pinned_files": len(result["pinned_inputs"]),  # type: ignore[arg-type]
        "mesh_references": result["referenced_mesh_count"],
        "derivation": result["derivation"]["method"],  # type: ignore[index]
        "status": "verified",
    }
    if args.source == "urdf":
        summary["velocity_placeholders"] = result["derivation"]["velocity_placeholders"]  # type: ignore[index]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
