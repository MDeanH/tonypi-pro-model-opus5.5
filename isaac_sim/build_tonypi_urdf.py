"""Derive an Isaac Sim import candidate URDF for the TonyPi Pro twin from model/ledger.csv.

Spine 2, as in model/build_twin.py: every number in the URDF comes from an ACTIVE ledger row -- each joint's
frame offset and orientation, axis, range and torque cap, its passive damping and friction, and each link's
mass, centre of mass and full inertia tensor. Nothing is re-read from build/tonypi_pro_as_is.xml, whose numbers
MuJoCo's writer rounds to about 6 significant figures. Structure and geometry come from the sha-pinned vendor
MJCF (vendor/tonypi/model/robot.xml): the link tree, the visual STL meshes (each geom's own mesh attribute, so
the vendor's left/right file swaps are followed) and the collision primitives. URDF has no capsule, so each
vendor capsule becomes a cylinder plus two end spheres, which together are exactly the same solid.

Units: every ledger row the builder reads must carry the unit its URDF field expects (m, rad, -, kg, kg*m^2,
N*m, N*m*s/rad, N*m/rad, rad/s; see UNIT_RULES). A row in any other unit (for example a Hiwonder speed in deg/s
or s/60deg) stops the build instead of being written mis-scaled.

Servo speed comes from the ledger's joint.<j>.limit.velocity rows. A joint whose row has status missing is
written with velocity="0", an explicit UNKNOWN placeholder as in the A2 build_luka_urdf.py, NOT a measured
limit; it is listed in the printed summary and in the build record. Every importer reads velocity="0" as a
0 rad/s limit (a frozen joint), so import the URDF only through isaac_sim/import_scene.py, which blocks the
placeholder. An active row replaces the placeholder on the next build. A joint with neither an active nor a
missing velocity row stops the build.

URDF cannot carry, so this candidate does not contain: joint armature (upstream ESTIMATE, and 66-99.9% of each
joint's inertia at the zero pose), the servo position gains (actuator.*.kp, dampratio, ctrlrange), MuJoCo
contact semantics (frictionless condim-1 self-collision guards, priority-2 sole and pad contacts with friction
0.9, the parent/child contact excludes and the sole-sole pair), the IMU and foot-touch sites, and the sensors.
So the URDF matches the twin in kinematics, mass, centres of mass and link inertia only, not in dynamics. The
build record therefore also carries, per joint, the terms an importer needs to restore the twin's dynamics:
ledger armature, passive damping and friction, kp, dampratio and the servo damping kv that MuJoCo derives from
dampratio (kv = 2 dampratio sqrt(kp / (M^-1)_jj) at the zero pose, M including armature), computed here by
compiling the URDF in MuJoCo with the ledger armature. isaac_sim/import_scene.py authors them in its scene layer.
The root link is torso at the URDF origin; the twin welds it at the ledger's scene.home.root pose (recorded in
the build record), and the importer decides whether the base is fixed or floating.

This is a model candidate, NOT validated against a physical TonyPi Pro. Only isaac_sim/generated/ is written;
the ledger, the vendor files and the twin build are read-only inputs, and all of them must lie inside this
repository. The ledger check reuses tools/seed_ledger.ledger_problems, which compiles the vendor model with
MuJoCo, so run this with the project venv.

Usage (from the repository root; on Windows use .venv\\Scripts\\python.exe in place of .venv/bin/python):
  .venv/bin/python -I isaac_sim/build_tonypi_urdf.py           write the URDF and its build record
  .venv/bin/python -I isaac_sim/build_tonypi_urdf.py --check   regenerate in memory; exit 1 if stale
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO / "vendor/tonypi"
VENDOR_MJCF = VENDOR_DIR / "model/robot.xml"
VENDOR_SUMS = VENDOR_DIR / "SHA256SUMS"
LEDGER = REPO / "model/ledger.csv"
TOOLS = REPO / "tools"
LEDGER_TOOLS = (TOOLS / "seed_ledger.py", TOOLS / "inspect_mjcf.py")
GENERATED_DIR = REPO / "isaac_sim/generated"
DEFAULT_OUTPUT = GENERATED_DIR / "tonypi_pro_candidate.urdf"
ROBOT_NAME = "tonypi_pro_candidate"
VISUAL_MATERIAL = "tonypi_visual"
VELOCITY_PLACEHOLDER = 0.0
INERTIA_KEYS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
ORIGIN_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
HOME_KEYS = ("x", "y", "z", "qw", "qx", "qy", "qz")
COMMENT = re.compile(r"<!--.*?-->", re.S)
# Vendor geom classes and what each becomes. "internal" boxes carry the vendor's per-part masses, which are
# already folded into the ledger's link inertials; they neither collide nor render, so they are not exported.
GEOM_ROLES = {"visual": "visual", "selfcol": "collision", "collision": "collision", "internal": "mass-only"}
COLLISION_KIND = {"selfcol": "self-collision guard (vendor: condim 1, friction 0)",
                  "collision": "contact geometry (vendor: priority 2, friction 0.9)"}
ORIENTATION_ATTRS = ("euler", "axisangle", "xyaxes", "zaxis")
SCHEMA_VERSION = 2
# The unit each ledger parameter family must carry to be written as is (URDF: SI, radians).
UNIT_RULES = tuple((re.compile(pattern), unit) for pattern, unit in (
    (r"link\.[^.]+\.mass", "kg"),
    (r"link\.[^.]+\.com\.[xyz]", "m"),
    (r"link\.[^.]+\.inertia\.i(xx|yy|zz|xy|xz|yz)", "kg*m^2"),
    (r"joint\.[^.]+\.origin\.[xyz]", "m"),
    (r"joint\.[^.]+\.origin\.(roll|pitch|yaw)", "rad"),
    (r"joint\.[^.]+\.axis\.[xyz]", "-"),
    (r"joint\.[^.]+\.limit\.(lower|upper)", "rad"),
    (r"joint\.[^.]+\.limit\.effort", "N*m"),
    (r"joint\.[^.]+\.limit\.velocity", "rad/s"),
    (r"joint\.[^.]+\.dynamics\.damping", "N*m*s/rad"),
    (r"joint\.[^.]+\.dynamics\.frictionloss", "N*m"),
    (r"joint\.[^.]+\.dynamics\.armature", "kg*m^2"),
    (r"actuator\.[^.]+\.kp", "N*m/rad"),
    (r"actuator\.[^.]+\.dampratio", "-"),
    (r"scene\.home\.root\.[xyz]", "m"),
    (r"scene\.home\.root\.q[wxyz]", "-"),
))
SERVO_RULE = ("kv = 2 * dampratio * sqrt(kp / (M^-1)_jj), M = joint-space mass matrix at the zero pose including "
              "the ledger armature (MuJoCo's dampratio rule for position actuators), computed by compiling this "
              "URDF in MuJoCo")


def required_unit(param: str) -> str:
    for pattern, unit in UNIT_RULES:
        if pattern.fullmatch(param):
            return unit
    raise BuildError(f"{param}: no unit rule in build_tonypi_urdf.UNIT_RULES; extend it before using this row")


def check_unit(row: dict[str, str]) -> None:
    expected = required_unit(row["param"])
    if row.get("unit", "") != expected:
        raise BuildError(f"{row['param']}: ledger unit {row.get('unit')!r}, but the URDF needs {expected!r}; "
                         "convert the row (the builder never rescales)")


class BuildError(ValueError):
    """Input is incomplete or inconsistent; no candidate is written."""


def finite(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{label}: expected a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise BuildError(f"{label}: number must be finite, got {value!r}")
    return number


def numbers(text: str | None, count: int, label: str) -> list[float]:
    parts = (text or "").split()
    if len(parts) != count:
        raise BuildError(f"{label}: expected {count} values, got {text!r}")
    return [finite(part, label) for part in parts]


def fmt(number: float) -> str:
    """Shortest text that parses back to exactly the same double (and no negative zero)."""
    return repr(float(number) + 0.0)


def fmt_all(values) -> str:
    return " ".join(fmt(v) for v in values)


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError as exc:
        raise BuildError(f"file is outside the repository: {resolved}") from exc


def fingerprint(path: Path, data: bytes | None = None) -> dict[str, object]:
    if data is None:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise BuildError(f"cannot read {path}: {exc}") from exc
    return {"path": repo_relative(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


# --------------------------------------------------------------------------------------------------- inputs


def vendor_sums() -> dict[str, str]:
    try:
        lines = VENDOR_SUMS.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BuildError(f"cannot read {VENDOR_SUMS}: {exc}") from exc
    sums: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64 or name in sums:
            raise BuildError(f"{VENDOR_SUMS}: malformed line {line!r}")
        sums[name] = digest
    return sums


def pinned_vendor_bytes(path: Path, sums: dict[str, str]) -> bytes:
    """Bytes of a vendor file, refusing unless they match vendor/tonypi/SHA256SUMS."""
    resolved = path.resolve()
    try:
        key = resolved.relative_to(VENDOR_DIR.resolve()).as_posix()
    except ValueError as exc:
        raise BuildError(f"vendor input must live under {VENDOR_DIR}: {resolved}") from exc
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise BuildError(f"cannot read vendor file {resolved}: {exc}") from exc
    if not data:
        raise BuildError(f"vendor file is empty: {resolved}")
    if sums.get(key) != hashlib.sha256(data).hexdigest():
        raise BuildError(f"vendor {key} differs from vendor/tonypi/SHA256SUMS (or is not listed there)")
    return data


@dataclass
class LedgerValues:
    """Active ledger values. Records every parameter read, so the summary can say where each number came from."""

    active: dict[str, dict[str, str]]
    missing: dict[str, dict[str, str]]
    used: dict[str, dict[str, str]] = field(default_factory=dict)

    def number(self, param: str) -> float:
        row = self.active.get(param)
        if row is None:
            raise BuildError(f"missing active ledger value: {param}")
        check_unit(row)
        value = finite(row["value"], param)
        self.used[param] = row
        return value

    def optional(self, param: str) -> float | None:
        return self.number(param) if param in self.active else None

    def peek(self, param: str) -> float | None:
        """An active value that is reported or recorded but not written into the URDF (not counted as used)."""
        row = self.active.get(param)
        if row is None:
            return None
        check_unit(row)
        return finite(row["value"], param)

    def is_missing(self, param: str) -> bool:
        row = self.missing.get(param)
        if row is not None:
            check_unit(row)
        return row is not None


def load_ledger(ledger_bytes: bytes, vendor: Path) -> LedgerValues:
    """Parse the ledger and refuse it if tools/seed_ledger.ledger_problems reports anything."""
    try:
        text = ledger_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildError(f"ledger is not UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    if not rows:
        raise BuildError("ledger has no rows; run tools/seed_ledger.py first")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import seed_ledger  # noqa: E402  (read-only reuse of the repository's spine-2 checks)

    if not set(seed_ledger.LEDGER_COLUMNS).issubset(reader.fieldnames or []):
        raise BuildError(f"ledger columns {reader.fieldnames} lack {seed_ledger.LEDGER_COLUMNS}")
    problems = seed_ledger.ledger_problems(rows, path=str(vendor))
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  ... {len(problems) - 10} more" if len(problems) > 10 else ""
        raise BuildError(f"ledger has {len(problems)} problem(s); fix them first:\n  {shown}{more}")
    return LedgerValues(active={r["param"]: r for r in rows if r["status"] == "active"},
                        missing={r["param"]: r for r in rows if r["status"] == "missing"})


@dataclass
class VendorModel:
    root_body: str
    bodies: list[dict]          # document order: name, parent, joint, geoms (attribute dicts)
    joints: list[str]           # document order
    meshes: dict[str, tuple[str, list[float]]]   # mesh asset name -> (file relative to the model folder, scale)
    defaults: dict[str, dict[str, dict[str, str]]]
    sites: int
    model_dir: Path


def parse_vendor(data: bytes, model_dir: Path) -> VendorModel:
    """Structure of the vendor MJCF. Comments are stripped in memory (upstream comments contain '--')."""
    try:
        root = ET.fromstring(COMMENT.sub("", data.decode("utf-8")))
    except (ET.ParseError, UnicodeDecodeError) as exc:
        raise BuildError(f"cannot parse the vendor MJCF: {exc}") from exc
    if root.tag != "mujoco":
        raise BuildError("vendor MJCF must have a <mujoco> root")
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle") != "radian":
        raise BuildError("vendor MJCF must declare <compiler angle=\"radian\">")
    for attr in ("meshdir", "assetdir"):
        if compiler.get(attr):
            raise BuildError(f"vendor MJCF sets compiler {attr}; extend build_tonypi_urdf to resolve it")
    defaults: dict[str, dict[str, dict[str, str]]] = {}
    top_default = root.find("default")
    for element in (top_default.findall("default") if top_default is not None else []):
        defaults[element.get("class") or ""] = {child.tag: dict(child.attrib) for child in element}
    meshes: dict[str, tuple[str, list[float]]] = {}
    asset = root.find("asset")
    for mesh in (asset.findall("mesh") if asset is not None else []):
        name, reference = mesh.get("name"), mesh.get("file")
        if not name or not reference or name in meshes:
            raise BuildError(f"vendor mesh asset needs a unique name and a file: {mesh.attrib}")
        if "://" in reference or Path(reference).is_absolute():
            raise BuildError(f"vendor mesh {name}: unsupported file reference {reference!r}")
        meshes[name] = (reference, numbers(mesh.get("scale", "1 1 1"), 3, f"mesh {name} scale"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise BuildError("vendor MJCF has no <worldbody>")
    tops = worldbody.findall("body")
    if len(tops) != 1:
        raise BuildError(f"vendor MJCF must have exactly one root body, found {len(tops)}")
    bodies: list[dict] = []
    joints: list[str] = []
    sites = 0

    def walk(element: ET.Element, parent: str) -> None:
        nonlocal sites
        for body in element.findall("body"):
            name = body.get("name")
            if not name or any(b["name"] == name for b in bodies):
                raise BuildError(f"vendor body needs a unique name: {body.attrib}")
            for attr in ("quat",) + ORIENTATION_ATTRS:
                if body.get(attr) is not None:
                    raise BuildError(f"vendor body {name} has {attr}=...; the ledger frames assume none")
            hinge = body.findall("joint")
            free = body.findall("freejoint")
            joint = None
            if parent == "world":
                if hinge or len(free) != 1:
                    raise BuildError(f"vendor root body {name} must have exactly one freejoint and no joint")
            else:
                if free or len(hinge) != 1:
                    raise BuildError(f"vendor body {name} must have exactly one hinge joint")
                j = hinge[0]
                joint = j.get("name")
                if joint != name:
                    raise BuildError(f"vendor joint {joint!r} must be named like its child body {name!r}")
                if j.get("type", "hinge") != "hinge":
                    raise BuildError(f"vendor joint {joint} is not a hinge")
                if any(abs(v) > 0.0 for v in numbers(j.get("pos", "0 0 0"), 3, f"joint {joint} pos")):
                    raise BuildError(f"vendor joint {joint} is offset from its body origin; extend the builder")
                joints.append(joint)
            sites += len(body.findall("site"))
            bodies.append(dict(name=name, parent=parent, joint=joint,
                               geoms=[dict(g.attrib) for g in body.findall("geom")]))
            walk(body, name)

    walk(worldbody, "world")
    return VendorModel(root_body=bodies[0]["name"], bodies=bodies, joints=joints, meshes=meshes,
                       defaults=defaults, sites=sites, model_dir=model_dir)


# ----------------------------------------------------------------------------------------------- geometry


def quat_to_rpy(q: list[float], label: str) -> list[float]:
    """URDF roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)) of a MuJoCo quaternion (w x y z)."""
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-12:
        raise BuildError(f"{label}: zero quaternion")
    w, x, y, z = (v / norm for v in q)
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    if abs(r20) >= 1.0 - 1e-12:
        raise BuildError(f"{label}: orientation at pitch +-90 deg; extend quat_to_rpy for gimbal lock")
    return [math.atan2(r21, r22), math.atan2(-r20, math.hypot(r00, r10)), math.atan2(r10, r00)]


def geom_frame(geom: dict, label: str) -> tuple[list[float], list[float]]:
    for attr in ORIENTATION_ATTRS + ("fromto",):
        if geom.get(attr) is not None:
            raise BuildError(f"{label}: {attr}=... is not supported for this geom")
    xyz = numbers(geom.get("pos", "0 0 0"), 3, f"{label} pos")
    rpy = quat_to_rpy(numbers(geom["quat"], 4, f"{label} quat"), label) if geom.get("quat") else [0.0, 0.0, 0.0]
    return xyz, rpy


def origin_element(parent: ET.Element, xyz, rpy) -> ET.Element:
    return ET.SubElement(parent, "origin", {"xyz": fmt_all(xyz), "rpy": fmt_all(rpy)})


def resolved_type(geom: dict, defaults: dict) -> str:
    cls = geom.get("class", "")
    return geom.get("type") or defaults.get(cls, {}).get("geom", {}).get("type") or "sphere"


def geom_role(geom: dict, defaults: dict, label: str) -> str:
    cls = geom.get("class")
    if cls not in GEOM_ROLES:
        raise BuildError(f"{label}: unknown geom class {cls!r}; known: {sorted(GEOM_ROLES)}")
    base = defaults.get(cls, {}).get("geom", {})
    for attr in ("contype", "conaffinity", "group"):
        if attr in geom:
            raise BuildError(f"{label}: per-geom {attr} override; the builder trusts the class defaults only")
    colliding = base.get("contype", "1") != "0" or base.get("conaffinity", "1") != "0"
    role = GEOM_ROLES[cls]
    if role == "collision" and not colliding:
        raise BuildError(f"{label}: class {cls} no longer collides in the vendor defaults")
    if role != "collision" and colliding:
        raise BuildError(f"{label}: class {cls} collides in the vendor defaults; it cannot be dropped")
    return role


def capsule_parts(fromto: list[float], radius: float, label: str):
    """Cylinder (centre, rpy, length) and the two end-sphere centres of a capsule given by fromto."""
    p1, p2 = fromto[:3], fromto[3:]
    d = [b - a for a, b in zip(p1, p2)]
    length = math.sqrt(sum(v * v for v in d))
    if not length > 0.0 or not radius > 0.0:
        raise BuildError(f"{label}: capsule needs a positive length and radius")
    axis = [v / length for v in d]
    # A cylinder is symmetric under a half turn, so orient its local z with a non-negative leading component.
    for v in (axis[2], axis[1], axis[0]):
        if v != 0.0:
            if v < 0.0:
                axis = [-a for a in axis]
            break
    pitch = math.acos(max(-1.0, min(1.0, axis[2])))
    yaw = math.atan2(axis[1], axis[0]) if (axis[0] or axis[1]) else 0.0
    centre = [(a + b) / 2 for a, b in zip(p1, p2)]
    return centre, [0.0, pitch, yaw], length, p1, p2


def collision_elements(geom: dict, defaults: dict, label: str) -> tuple[list[ET.Element], str]:
    """URDF <collision> elements for one vendor collision geom, and the primitive kind it came from."""
    name = geom.get("name")
    if not name:
        raise BuildError(f"{label}: collision geom needs a name")
    gtype = resolved_type(geom, defaults)
    size = (geom.get("size") or "").split()

    def element(suffix: str, xyz, rpy, shape: str, attrs: dict[str, str]) -> ET.Element:
        col = ET.Element("collision", {"name": name + suffix})
        origin_element(col, xyz, rpy)
        ET.SubElement(ET.SubElement(col, "geometry"), shape, attrs)
        return col

    if gtype == "box":
        half = numbers(" ".join(size), 3, f"{label} size")
        if min(half) <= 0:
            raise BuildError(f"{label}: box half-sizes must be positive")
        xyz, rpy = geom_frame(geom, label)
        return [element("", xyz, rpy, "box", {"size": fmt_all(2 * h for h in half)})], "box"
    if gtype == "sphere":
        (radius,) = numbers(" ".join(size), 1, f"{label} size")
        if radius <= 0:
            raise BuildError(f"{label}: sphere radius must be positive")
        xyz, _ = geom_frame(geom, label)
        return [element("", xyz, [0.0, 0.0, 0.0], "sphere", {"radius": fmt(radius)})], "sphere"
    if gtype == "capsule":
        if geom.get("fromto") is None:
            raise BuildError(f"{label}: capsule without fromto is not supported")
        for attr in ("pos", "quat") + ORIENTATION_ATTRS:
            if geom.get(attr) is not None:
                raise BuildError(f"{label}: capsule with both fromto and {attr} is ambiguous")
        (radius,) = numbers(" ".join(size), 1, f"{label} size")
        centre, rpy, length, p1, p2 = capsule_parts(numbers(geom["fromto"], 6, f"{label} fromto"), radius, label)
        zero = [0.0, 0.0, 0.0]
        return [element("_cylinder", centre, rpy, "cylinder", {"radius": fmt(radius), "length": fmt(length)}),
                element("_end_from", p1, zero, "sphere", {"radius": fmt(radius)}),
                element("_end_to", p2, zero, "sphere", {"radius": fmt(radius)})], "capsule"
    raise BuildError(f"{label}: collision geom type {gtype!r} is not supported (box, sphere, capsule)")


def validate_inertia(mass: float, values: dict[str, float], label: str) -> None:
    if mass <= 0:
        raise BuildError(f"{label}: mass must be positive")
    a, b, c = values["ixx"], values["iyy"], values["izz"]
    d, e, f = values["ixy"], values["ixz"], values["iyz"]
    # Sylvester's criterion for a symmetric 3x3 inertia tensor.
    second_minor = a * b - d * d
    determinant = a * b * c + 2 * d * e * f - a * f * f - b * e * e - c * d * d
    if a <= 0 or second_minor <= 0 or determinant <= 0:
        raise BuildError(f"{label}: inertia tensor is not positive definite")


def contact_class(defaults: dict, cls: str) -> dict[str, object]:
    """Vendor contact semantics of a collision class, for importers that can author physics materials."""
    base = defaults.get(cls, {}).get("geom", {})
    condim = int(base.get("condim", "3"))
    friction = numbers(base.get("friction", "1 0.005 0.0001"), 3, f"vendor class {cls} friction")
    priority = int(base.get("priority", "0"))
    return {"condim": condim, "friction": friction, "priority": priority,
            # condim 1 is frictionless whatever the friction coefficients say.
            "sliding_friction": 0.0 if condim == 1 else friction[0]}


def joint_importer_terms(urdf: bytes, terms: dict[str, dict[str, float | None]]) -> dict[str, dict[str, object]]:
    """Per joint: the ledger's servo and passive terms plus MuJoCo's dampratio-derived servo damping kv.

    The URDF is compiled in MuJoCo without visuals (no mesh is read); the ledger armature is added to each DOF,
    and kv = 2 dampratio sqrt(kp / (M^-1)_jj) at the zero pose, which is how MuJoCo resolves a position
    actuator's dampratio (the twin's actuator_biasprm reproduces this; see test_build_tonypi_urdf.py)."""
    import mujoco
    import numpy as np

    marker = f'<robot name="{ROBOT_NAME}">'
    text = urdf.decode("utf-8")
    if text.count(marker) != 1:
        raise BuildError("generated URDF lacks its robot element")
    extension = '<mujoco><compiler discardvisual="true" fusestatic="false" balanceinertia="false"/></mujoco>'
    try:
        model = mujoco.MjModel.from_xml_string(text.replace(marker, marker + extension, 1))
    except ValueError as exc:
        raise BuildError(f"MuJoCo cannot compile the generated URDF: {exc}") from exc
    if model.nv != len(terms):
        raise BuildError(f"MuJoCo compiled {model.nv} DOFs from the URDF, expected {len(terms)}")
    dofs = {}
    for name, term in terms.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise BuildError(f"MuJoCo lost joint {name} of the generated URDF")
        dofs[name] = int(model.jnt_dofadr[joint])
        model.dof_armature[dofs[name]] = float(term["armature"])  # type: ignore[arg-type]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    inverse = np.linalg.inv(mass)
    out: dict[str, dict[str, object]] = {}
    for name, term in terms.items():
        dof = dofs[name]
        reflected = 1.0 / float(inverse[dof, dof])
        kv = 2.0 * float(term["dampratio"]) * math.sqrt(float(term["kp"]) * reflected)  # type: ignore[arg-type]
        diagonal = float(mass[dof, dof])
        out[name] = {**term, "kv": kv, "reflected_inertia": reflected, "mass_matrix_diagonal": diagonal,
                     "armature_share_zero_pose": float(term["armature"]) / diagonal}  # type: ignore[arg-type]
    return out


# ----------------------------------------------------------------------------------------------- assembly


@dataclass
class Candidate:
    urdf: bytes
    record: bytes
    summary: list[str]
    urdf_path: Path
    record_path: Path


def record_path_for(output: Path) -> Path:
    return output.with_name(output.stem + ".build.json")


def validated_output(output: Path) -> Path:
    """Keep every write inside the dedicated derived-artifact directory (rejects '..' and symlink escapes)."""
    resolved = output.resolve()
    allowed = GENERATED_DIR.resolve()
    if resolved.suffix.lower() != ".urdf" or allowed not in resolved.parents:
        raise BuildError(f"output must be a .urdf below {allowed}")
    return resolved


def generate(ledger: Path = LEDGER, output: Path = DEFAULT_OUTPUT, vendor: Path = VENDOR_MJCF) -> Candidate:
    """Build the URDF and its build record in memory. Writes nothing."""
    output = validated_output(output)
    ledger, vendor = ledger.resolve(), vendor.resolve()
    for label, path in (("ledger", ledger), ("vendor model", vendor)):
        # The build record names every input by its repository path, so refuse outside inputs before any work.
        try:
            path.relative_to(REPO)
        except ValueError as exc:
            raise BuildError(f"the {label} must lie inside the repository {REPO}: {path}") from exc
    if output in (ledger, vendor) or record_path_for(output) in (ledger, vendor):
        raise BuildError("output must not overwrite an input file")
    sums = vendor_sums()
    vendor_bytes = pinned_vendor_bytes(vendor, sums)
    try:
        ledger_bytes = ledger.read_bytes()
    except OSError as exc:
        raise BuildError(f"cannot read ledger {ledger}: {exc}") from exc
    model = parse_vendor(vendor_bytes, vendor.parent)
    led = load_ledger(ledger_bytes, vendor)

    visual_cls = model.defaults.get("visual", {}).get("geom", {})
    rgba = numbers(visual_cls.get("rgba"), 4, "vendor visual class rgba")
    robot = ET.Element("robot", {"name": ROBOT_NAME})
    header = ET.Comment(
        " GENERATED by isaac_sim/build_tonypi_urdf.py from model/ledger.csv (every number) and the sha-pinned "
        "vendor/tonypi/model/robot.xml (pab47/TonyPi: tree, STL meshes, collision primitives). Do not hand-edit. "
        "Model candidate, NOT validated against a physical TonyPi Pro. velocity=\"0\" marks an UNKNOWN servo "
        "speed (ledger status missing), not a physical limit. Not representable in URDF and absent here: joint "
        "armature, servo gains, MuJoCo contact parameters and excludes, sites, sensors. Capsules are written as "
        "a cylinder plus two end spheres. ")
    robot.append(header)
    material = ET.SubElement(robot, "material", {"name": VISUAL_MATERIAL})
    ET.SubElement(material, "color", {"rgba": fmt_all(rgba)})

    mesh_inputs: dict[str, tuple[Path, bytes]] = {}
    counts = {"box": 0, "sphere": 0, "capsule": 0}
    collision_sources: list[dict[str, object]] = []
    skipped_mass_only = 0
    total_mass = 0.0
    link_elements: list[ET.Element] = []
    for body in model.bodies:
        name = body["name"]
        link = ET.Element("link", {"name": name})
        mass = led.number(f"link.{name}.mass")
        com = [led.number(f"link.{name}.com.{k}") for k in "xyz"]
        inertia = {k: led.number(f"link.{name}.inertia.{k}") for k in INERTIA_KEYS}
        validate_inertia(mass, inertia, f"link {name}")
        total_mass += mass
        inertial = ET.SubElement(link, "inertial")
        origin_element(inertial, com, [0.0, 0.0, 0.0])  # the ledger tensor is about the COM in link axes
        ET.SubElement(inertial, "mass", {"value": fmt(mass)})
        ET.SubElement(inertial, "inertia", {k: fmt(inertia[k]) for k in INERTIA_KEYS})
        visuals = 0
        for geom in body["geoms"]:
            label = f"body {name} geom {geom.get('name', '<unnamed>')}"
            role = geom_role(geom, model.defaults, label)
            if role == "mass-only":
                skipped_mass_only += 1
                continue
            if role == "visual":
                if resolved_type(geom, model.defaults) != "mesh" or geom.get("mesh") not in model.meshes:
                    raise BuildError(f"{label}: visual geom must reference a vendor mesh asset")
                reference, scale = model.meshes[geom["mesh"]]
                mesh_path = (model.model_dir / reference).resolve()
                if mesh_path not in mesh_inputs:
                    mesh_inputs[mesh_path] = (mesh_path, pinned_vendor_bytes(mesh_path, sums))
                xyz, rpy = geom_frame(geom, label)
                visual = ET.SubElement(link, "visual", {"name": geom.get("name") or f"{name}_visual"})
                origin_element(visual, xyz, rpy)
                relative = Path(os.path.relpath(mesh_path, output.parent)).as_posix()
                ET.SubElement(ET.SubElement(visual, "geometry"), "mesh",
                              {"filename": relative, "scale": fmt_all(scale)})
                ET.SubElement(visual, "material", {"name": VISUAL_MATERIAL})
                visuals += 1
                continue
            elements, kind = collision_elements(geom, model.defaults, label)
            counts[kind] += 1
            collision_sources.append({"link": name, "vendor_geom": geom["name"], "vendor_class": geom["class"],
                                      "kind": kind, "urdf_collisions": [e.get("name") for e in elements]})
            for element in elements:
                link.append(element)
        if visuals != 1:
            raise BuildError(f"link {name}: expected exactly one visual mesh, found {visuals}")
        link_elements.append(link)
    for link in link_elements:
        robot.append(link)

    placeholders: list[str] = []
    velocity_rows: dict[str, float] = {}
    gaps: list[str] = []
    dropped_armature: dict[str, float] = {}
    joint_terms: dict[str, dict[str, float | None]] = {}
    parent_of = {b["name"]: b["parent"] for b in model.bodies}
    for joint_name in model.joints:
        prefix = f"joint.{joint_name}"
        origin = [led.number(f"{prefix}.origin.{k}") for k in ORIGIN_KEYS]
        axis = [led.number(f"{prefix}.axis.{k}") for k in "xyz"]
        if math.sqrt(sum(v * v for v in axis)) < 1e-12:
            raise BuildError(f"{prefix}: zero joint axis")
        lower, upper = led.number(f"{prefix}.limit.lower"), led.number(f"{prefix}.limit.upper")
        if not lower < upper:
            raise BuildError(f"{prefix}: lower limit must be below upper limit")
        effort = led.number(f"{prefix}.limit.effort")
        if effort <= 0:
            raise BuildError(f"{prefix}.limit.effort must be positive")
        dynamics = {}
        for key in ("damping", "frictionloss"):
            value = led.optional(f"{prefix}.dynamics.{key}")
            if value is None:
                # Same rule as model/build_twin.py: an unknown passive term is 0 there too, and reported.
                gaps.append(f"{prefix}.dynamics.{key}")
                value = 0.0
            if value < 0:
                raise BuildError(f"{prefix}.dynamics.{key} must not be negative")
            dynamics[key] = value
        armature = led.peek(f"{prefix}.dynamics.armature")
        if armature is not None:
            if armature < 0:
                raise BuildError(f"{prefix}.dynamics.armature must not be negative")
            dropped_armature[joint_name] = armature
        velocity_param = f"{prefix}.limit.velocity"
        velocity = led.optional(velocity_param)
        is_placeholder = False
        if velocity is not None:
            if velocity <= 0:
                raise BuildError(f"{velocity_param} must be positive")
            velocity_rows[joint_name] = velocity
        elif led.is_missing(velocity_param):
            # Explicit UNKNOWN placeholder (as in the A2 build_luka_urdf.py), never a measured speed limit.
            velocity = VELOCITY_PLACEHOLDER
            is_placeholder = True
            placeholders.append(joint_name)
        else:
            raise BuildError(f"{velocity_param}: the ledger has neither an active nor a missing row")
        kp, dampratio = led.peek(f"actuator.{joint_name}.kp"), led.peek(f"actuator.{joint_name}.dampratio")
        if kp is None or dampratio is None or armature is None:
            raise BuildError(f"{joint_name}: the ledger needs active actuator kp, dampratio and joint armature rows "
                             "for the importer terms in the build record")
        if kp <= 0 or dampratio < 0:
            raise BuildError(f"{joint_name}: actuator kp must be positive and dampratio not negative")
        joint_terms[joint_name] = {
            "kp": kp, "dampratio": dampratio, "armature": armature, "damping": dynamics["damping"],
            "frictionloss": dynamics["frictionloss"], "effort": effort,
            "velocity": None if is_placeholder else velocity,
        }
        joint = ET.SubElement(robot, "joint", {"name": joint_name, "type": "revolute"})
        if is_placeholder:
            joint.append(ET.Comment(f" velocity=\"0\": placeholder, ledger {velocity_param} is missing "))
        origin_element(joint, origin[:3], origin[3:])
        ET.SubElement(joint, "parent", {"link": parent_of[joint_name]})
        ET.SubElement(joint, "child", {"link": joint_name})
        ET.SubElement(joint, "axis", {"xyz": fmt_all(axis)})
        ET.SubElement(joint, "limit", {"lower": fmt(lower), "upper": fmt(upper),
                                       "effort": fmt(effort), "velocity": fmt(velocity)})
        ET.SubElement(joint, "dynamics", {"damping": fmt(dynamics["damping"]),
                                          "friction": fmt(dynamics["frictionloss"])})

    home: dict[str, float] = {}
    for key in HOME_KEYS:  # recorded for the importer's floating-base spawn; not part of the URDF itself
        value = led.peek(f"scene.home.root.{key}")
        if value is None:
            raise BuildError(f"missing active ledger value: scene.home.root.{key}")
        home[key] = value
    ET.indent(robot, space="  ")
    urdf = ('<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(robot, encoding="unicode") + "\n").encode("utf-8")

    links, joints = len(model.bodies), len(model.joints)
    collisions = counts["box"] + counts["sphere"] + 3 * counts["capsule"]
    by_source: dict[str, int] = {}
    for row in led.used.values():
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    import mujoco  # the ledger check above compiled the vendor model with it

    importer_terms = joint_importer_terms(urdf, joint_terms)
    contact_classes = {cls: contact_class(model.defaults, cls) for cls in COLLISION_KIND}

    inputs = [fingerprint(ledger, ledger_bytes), fingerprint(vendor, vendor_bytes), fingerprint(VENDOR_SUMS)]
    inputs += [fingerprint(path) for path in LEDGER_TOOLS]
    inputs += [fingerprint(path, data) for path, data in sorted(mesh_inputs.values())]
    record = {
        "schema_version": SCHEMA_VERSION,
        "scope": "derivation record for an uncalibrated model candidate; not physical validation of a TonyPi Pro",
        "urdf": fingerprint(output, urdf),
        "generator": fingerprint(Path(__file__).resolve()),
        "inputs": inputs,
        "ledger_check": {"function": "tools/seed_ledger.ledger_problems", "problems": 0,
                         "mujoco_version": mujoco.__version__},
        "counts": {"links": links, "revolute_joints": joints, "visual_meshes": len(mesh_inputs),
                   "collision_elements": collisions, "vendor_boxes": counts["box"],
                   "vendor_spheres": counts["sphere"], "vendor_capsules": counts["capsule"],
                   "vendor_mass_only_boxes_skipped": skipped_mass_only, "vendor_sites_dropped": model.sites,
                   "ledger_values_in_urdf": len(led.used)},
        "ledger_values_by_source": dict(sorted(by_source.items())),
        "total_mass_kg": total_mass,
        "root_link": model.root_body,
        "home_root_pose": home,
        "joints": list(model.joints),
        "velocity_placeholders": placeholders,
        "velocity_placeholder_value": VELOCITY_PLACEHOLDER,
        "velocity_from_ledger": velocity_rows,
        "dynamics_gaps_written_as_zero": gaps,
        "dropped_not_representable": {
            "joint_armature": dropped_armature,
            "servo_gains": "actuator.*.kp, actuator.*.dampratio, actuator.*.ctrlrange",
            "contact": ("MuJoCo condim/friction/priority per class, parent-child <exclude>s and the sole-sole "
                        "<pair> of the vendor model"),
            "sites_and_sensors": "IMU and foot-touch sites, all vendor sensors",
        },
        "collision_sources": collision_sources,
        "collision_kinds": COLLISION_KIND,
        "contact_classes": contact_classes,
        "importer_joint_terms": {
            "scope": ("ledger terms URDF cannot carry, for importers to author (isaac_sim/import_scene.py); "
                      "SI units, per radian"),
            "kv_rule": SERVO_RULE,
            "mujoco_version": mujoco.__version__,
            "joints": importer_terms,
        },
    }
    record_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    summary = [
        f"links {links}, revolute joints {joints}, visual meshes {len(mesh_inputs)}, collision elements "
        f"{collisions} ({counts['box']} boxes, {counts['sphere']} spheres, {counts['capsule']} capsules -> "
        f"{counts['capsule']} cylinders + {2 * counts['capsule']} end spheres); {skipped_mass_only} vendor "
        f"mass-only boxes not exported (their masses are in the ledger inertials)",
        f"total mass {total_mass:.6f} kg; {len(led.used)} ledger values written to the URDF, by source: "
        + ", ".join(f"{k} {v}" for k, v in sorted(by_source.items())),
        (f"velocity placeholders (ledger status missing, written as velocity=\"0\" = UNKNOWN, not a limit): "
         f"{len(placeholders)} of {joints} joints" + (f": {', '.join(placeholders)}" if placeholders else ""))
        + (f"; from the ledger: {len(velocity_rows)}" if velocity_rows else ""),
        f"dynamics gaps written as 0 (as in model/build_twin.py): {len(gaps)}",
        f"not representable in URDF, dropped: armature on {len(dropped_armature)} joints, servo kp/dampratio/"
        f"ctrlrange, MuJoCo contact parameters/excludes/sole-sole pair, {model.sites} sites, sensors",
        "armature share of each joint's zero-pose inertia (so the URDF alone is not the twin's dynamics): "
        f"{min(t['armature_share_zero_pose'] for t in importer_terms.values()):.3f} to "
        f"{max(t['armature_share_zero_pose'] for t in importer_terms.values()):.4f}; armature, passive terms and "
        "servo kp/kv recorded in the build record for the importer",
        f"root link {model.root_body} at the URDF origin; the twin welds it at scene.home.root "
        f"({home['x']}, {home['y']}, {home['z']}) m; the importer decides fixed or floating base",
    ]
    return Candidate(urdf=urdf, record=record_bytes, summary=summary, urdf_path=output,
                     record_path=record_path_for(output))


def atomic_write(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".tonypi_urdf_", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(ledger: Path = LEDGER, output: Path = DEFAULT_OUTPUT, vendor: Path = VENDOR_MJCF) -> Candidate:
    candidate = generate(ledger, output, vendor)
    candidate.urdf_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(candidate.urdf_path, candidate.urdf)
    atomic_write(candidate.record_path, candidate.record)  # written second: a URDF without it fails integrity
    return candidate


def stale_outputs(ledger: Path = LEDGER, output: Path = DEFAULT_OUTPUT, vendor: Path = VENDOR_MJCF) -> list[str]:
    """Generated files that are missing or differ from a fresh in-memory build."""
    candidate = generate(ledger, output, vendor)
    stale = []
    for path, data in ((candidate.urdf_path, candidate.urdf), (candidate.record_path, candidate.record)):
        if not path.is_file() or path.read_bytes() != data:
            stale.append(repo_relative(path))
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="regenerate in memory and compare with the files")
    args = parser.parse_args(argv)
    try:
        if args.check:
            stale = stale_outputs(args.ledger, args.out)
            if stale:
                print("Stale or missing (run isaac_sim/build_tonypi_urdf.py): " + ", ".join(stale), file=sys.stderr)
                return 1
            print(f"URDF and build record match the generator: {repo_relative(args.out)}")
            return 0
        candidate = build(args.ledger, args.out)
    except BuildError as exc:
        print(f"Build stopped: {exc}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(candidate.urdf).hexdigest()
    print(f"Wrote {repo_relative(candidate.urdf_path)} ({len(candidate.urdf)} bytes, sha256 {digest[:16]}...)")
    for line in candidate.summary:
        print(f"  {line}")
    print(f"Build record: {repo_relative(candidate.record_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
