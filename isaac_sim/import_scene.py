"""Import the TonyPi Pro twin into Isaac Sim 6.2 and save a USD scene with provenance hashes.

--source urdf (default) imports the ledger-derived isaac_sim/generated/tonypi_pro_candidate.urdf with NVIDIA's
URDF importer; --source mjcf imports the twin MJCF with the MJCF importer (build/tonypi_pro_as_is.xml, or the
floating contact candidate build/tonypi_pro_contact_candidate.xml with --floating). Every input is verified first
(isaac_sim/source_integrity.py). The scene must contain exactly one articulation root, the twin's 20 revolute
joints by name and exactly one ground collider, or nothing is saved.

Run it with Isaac Sim's bundled python.bat / python.sh. --no-kit skips SimulationApp and calls the same importer
classes directly; that works where the importer runs without Kit (for example the pip isaacsim-asset-isolated
package in an Isaac Lab venv). It runs the same scene checks, but its output has not been compared with a Kit
import, and there is no PhysX run (run_scene.py needs Kit).

Transaction: the importer writes into an empty staging folder (the 6.2 importer never overwrites; it would pick
<name>_1, _2, ...). The importer's layer documentation, which holds the absolute staging path, is cleared, and
any absolute path left in a generated layer stops the import. The scene is built and checked against the
staged asset, the inputs are verified again (every pinned file by hash), and only then do the asset folder,
scene and integrity report replace the previous ones. On any failure the previous files are restored and the
staging folder is deleted.

Scene layer (the imported asset itself is never edited; everything is listed in the scene's customLayerData):
- Joint dynamics. URDF cannot carry armature or servo gains, and the 6.2 importer authors passive damping and
  friction only as newton:* attributes. For every joint the scene authors, from the URDF build record (ledger
  values) or, for --source mjcf, from the MJCF as MuJoCo compiles it: drive stiffness = servo kp and drive
  damping = servo kv (USD angular units, per degree), newton:armature and physxJoint:armature, and, where the
  ledger has a speed, physxJoint:maxJointVelocity (deg/s) next to the importer's newton:velocityLimit. The
  importer's maxForce, newton:damping and newton:friction are checked against the same values. PhysX has no
  passive viscous damping or Coulomb friction-torque attribute in this schema, so under PhysX the passive terms
  are absent (newton:* only).
- Velocity placeholders: joints whose ledger speed is missing carry velocity="0" in the URDF, which the importer
  writes as newton:velocityLimit = 0 (a frozen joint). The scene blocks that authored zero, so no limit applies,
  as in the MuJoCo twin.
- Self-collision is off for PhysX too (physxArticulation:enabledSelfCollisions = false on the articulation root;
  the importer only writes newton:selfCollisionEnabled = 0, and the PhysX schema default is on).
- Contact materials: each robot collider gets the vendor class's sliding friction (self-collision guards 0,
  combine mode min, as their condim 1 makes them frictionless; soles and gripper pads 0.9), and the ground gets
  the vendor floor's friction (vendor/tonypi/model/scene.xml). condim, priority and solref have no PhysX form.
- World geometry imported with the robot (the contact candidate's own floor plane) is deactivated, so the scene's
  ground is the only ground collider.
- Physics step: physxScene:timeStepsPerSecond = 1 / the twin's <option timestep> (250 Hz).

Placement: a floating URDF is spawned at the ledger's scene.home.root pose (recorded in the URDF build record);
the MJCF sources carry their own torso pose. Floating scenes get a floor whose top is z = 0, as in the vendor
scene. Fixed-base scenes get an uncalibrated model-check floor 5 mm below the lowest collider, so the welded robot
never touches it. No standing controller or balance behaviour is assumed.

Neither source is a verified physical match to a TonyPi Pro.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
GENERATED = REPO / "isaac_sim" / "generated"
VENDOR_ROBOT = REPO / "vendor" / "tonypi" / "model" / "robot.xml"
VENDOR_SCENE = REPO / "vendor" / "tonypi" / "model" / "scene.xml"
EXPECTED_JOINTS = 20
FIXED_BASE_GROUND_CLEARANCE_M = 0.005
PLACEHOLDER_ATTRIBUTES = ("newton:velocityLimit", "physxJoint:maxJointVelocity")
DEG = math.pi / 180.0
FLOAT32_REL = 1e-6
MATERIALS = "/World/PhysicsMaterials"
PHYSICS_PURPOSE = "physics"  # UsdShade material purpose read by USD Physics
COMMENT = re.compile(r"<!--.*?-->", re.S)  # vendor comments contain "--", which XML parsers reject
VENV_PYTHON = ".venv\\Scripts\\python.exe" if sys.platform == "win32" else ".venv/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("urdf", "mjcf"), default="urdf",
                        help="model to import (default: the ledger-derived URDF)")
    parser.add_argument("--floating", action="store_true",
                        help="floating base (URDF: importer fix_base=False; MJCF: the contact candidate)")
    parser.add_argument("--headless", action="store_true", help="run Isaac Sim without a viewport")
    parser.add_argument("--no-kit", action="store_true",
                        help="do not start SimulationApp; use the importer classes directly (conversion and checks only)")
    args = parser.parse_args()
    if args.no_kit and args.headless:
        parser.error("--headless has no meaning with --no-kit")
    return args


def source_path(source_type: str, floating: bool) -> Path:
    if source_type == "urdf":
        return GENERATED / "tonypi_pro_candidate.urdf"
    return REPO / "build" / ("tonypi_pro_contact_candidate.xml" if floating else "tonypi_pro_as_is.xml")


def parse_xml(path: Path) -> ET.Element:
    return ET.fromstring(COMMENT.sub("", path.read_text(encoding="utf-8")))


def option_timestep(path: Path) -> float:
    option = parse_xml(path).find("option")
    if option is None or option.get("timestep") is None:
        raise ValueError(f"{path}: no <option timestep>; refusing to guess the physics step")
    step = float(option.get("timestep"))
    rate = 1.0 / step
    if not math.isfinite(step) or step <= 0 or abs(rate - round(rate)) > 1e-9:
        raise ValueError(f"{path}: timestep {step} is not 1/N s for an integer N")
    return step


def vendor_floor_friction() -> float:
    """Sliding friction of the vendor scene's floor (vendor/tonypi/model/scene.xml, manifest-pinned)."""
    floors = [g for g in parse_xml(VENDOR_SCENE).iter("geom") if g.get("name") == "floor"]
    if len(floors) != 1 or floors[0].get("type") != "plane":
        raise ValueError(f"{VENDOR_SCENE}: expected one floor plane")
    if floors[0].get("condim", "3") == "1":
        return 0.0
    friction = float(floors[0].get("friction", "1").split()[0])
    if not math.isfinite(friction) or friction < 0:
        raise ValueError(f"{VENDOR_SCENE}: invalid floor friction")
    return friction


def preflight_mjcf(source: Path, floating: bool) -> list[str]:
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    base = (source.parent / meshdir).resolve() if meshdir else source.parent.resolve()
    meshes = root.findall("./asset/mesh")
    if not meshes:
        raise ValueError(f"{source}: no MJCF <asset><mesh> entries")
    for mesh in meshes:
        reference = mesh.get("file")
        if not reference or "://" in reference:
            raise ValueError(f"{source}: unsupported mesh reference {reference!r}")
        resolved = (base / reference).resolve()
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise FileNotFoundError(f"MJCF mesh {reference!r} is missing or empty: {resolved}")
    hinges = [j.get("name") for j in root.findall("./worldbody//joint") if j.get("type", "hinge") == "hinge"]
    if len(hinges) != EXPECTED_JOINTS or len(set(hinges)) != EXPECTED_JOINTS:
        raise ValueError(f"{source}: expected {EXPECTED_JOINTS} uniquely named hinge joints, found {hinges}")
    freejoints = root.findall("./worldbody//freejoint")
    if len(freejoints) != int(floating):
        raise ValueError(f"{source}: unexpected floating-base joint count {len(freejoints)}")
    print(f"Preflight: {len(hinges)} MJCF hinges and {len(meshes)} local meshes verified")
    return hinges


def mjcf_model_terms(source: Path, joints: list[str]) -> tuple[dict[str, dict], dict[str, float]]:
    """Joint terms and collider frictions of the MJCF source as MuJoCo compiles it (the importer drops the servo).

    mujoco is a dependency of Isaac Sim's MJCF importer (mujoco-usd-converter), so it is present wherever that
    importer runs."""
    try:
        import mujoco
    except ImportError as exc:
        raise ValueError(f"--source mjcf needs the mujoco package of the MJCF importer: {exc}") from exc
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mujoco_compat import load_mjcf

    model, edits = load_mjcf(source)
    for edit in edits:
        print(f"Joint terms: {edit}")
    by_joint: dict[str, int] = {}
    for actuator in range(model.nu):
        if (model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                or model.actuator_gainprm[actuator, 0] != -model.actuator_biasprm[actuator, 1]
                or model.actuator_biasprm[actuator, 0] != 0):
            raise ValueError(f"{source}: actuator {actuator} is not a joint position servo")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[actuator, 0]))
        if name in by_joint:
            raise ValueError(f"{source}: two actuators drive {name}")
        by_joint[name] = actuator
    terms: dict[str, dict] = {}
    for name in joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or name not in by_joint:
            raise ValueError(f"{source}: joint {name} or its servo is missing")
        dof, actuator = int(model.jnt_dofadr[joint]), by_joint[name]
        low, high = (float(v) for v in model.actuator_forcerange[actuator])
        if not model.actuator_forcelimited[actuator] or low != -high:
            raise ValueError(f"{source}: servo {name} needs a symmetric force range")
        terms[name] = {
            "kp": float(model.actuator_gainprm[actuator, 0]), "kv": float(-model.actuator_biasprm[actuator, 2]),
            "armature": float(model.dof_armature[dof]), "damping": float(model.dof_damping[dof]),
            "frictionloss": float(model.dof_frictionloss[dof]), "effort": high,
            "velocity": None,  # MJCF has no joint speed limit; the twin has none
        }
    frictions: dict[str, float] = {}
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0 or (model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if not name or name in frictions:
            raise ValueError(f"{source}: colliding geoms need unique names (geom {geom})")
        frictions[name] = 0.0 if model.geom_condim[geom] == 1 else float(model.geom_friction[geom, 0])
    return terms, frictions


def preflight_urdf(source: Path, derivation: dict) -> list[str]:
    root = ET.parse(source).getroot()
    joints = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
    if len(joints) != EXPECTED_JOINTS or joints != list(derivation["joints"]):
        raise ValueError(f"{source}: expected the build record's {EXPECTED_JOINTS} revolute joints, found {joints}")
    if len(root.findall("joint")) != EXPECTED_JOINTS:
        raise ValueError(f"{source}: unexpected non-revolute joints")
    meshes = root.findall(".//mesh")
    for mesh in meshes:
        reference = mesh.get("filename")
        if not reference or "://" in reference or Path(reference).is_absolute():
            raise ValueError(f"{source}: unsupported mesh reference {reference!r}")
        resolved = (source.parent / reference).resolve()
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise FileNotFoundError(f"URDF mesh {reference!r} is missing or empty: {resolved}")
    placeholders = derivation["velocity_placeholders"]
    print(f"Preflight: {len(joints)} URDF revolute joints and {len(meshes)} local meshes verified; "
          f"velocity placeholders (unknown speed): {placeholders or 'none'}")
    return joints


def urdf_collider_frictions(derivation: dict) -> dict[str, float]:
    """URDF collision element name -> the vendor class's sliding friction (build record)."""
    classes = derivation["contact_classes"]
    out: dict[str, float] = {}
    for source in derivation["collision_sources"]:
        friction = float(classes[source["vendor_class"]]["sliding_friction"])
        for name in source["urdf_collisions"]:
            if name in out:
                raise ValueError(f"build record lists collision {name} twice")
            out[name] = friction
    return out


def import_asset(source: Path, output_dir: Path, source_type: str, floating: bool, use_kit: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)  # always a fresh, empty staging folder
    if use_kit:
        from isaacsim.core.experimental.utils.app import enable_extension

        extension = f"isaacsim.asset.importer.{source_type}"
        if not enable_extension(extension):
            raise RuntimeError(f"Could not enable {extension}")
    if source_type == "mjcf":
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

        config = MJCFImporterConfig(
            mjcf_path=str(source),
            usd_path=str(output_dir),  # The API expects a directory, not a USD filename.
            # Skips only the MJCF <option>/simulation settings; worldbody geometry such as the contact
            # candidate's floor is still imported, and build_scene deactivates it.
            import_scene=False,
            merge_mesh=False,
            collision_from_visuals=False,
            allow_self_collision=False,
            robot_type="Humanoid",
            fix_base=None if floating else True,  # The contact candidate carries its own free joint.
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        output = MJCFImporter(config).import_mjcf()
    else:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        config = URDFImporterConfig(
            urdf_path=str(source),
            usd_path=str(output_dir),
            merge_fixed_joints=False,
            merge_mesh=False,
            collision_from_visuals=False,  # The URDF has explicit primitive colliders from the vendor model.
            allow_self_collision=False,    # URDF cannot carry the vendor's parent/child contact excludes.
            robot_type="Humanoid",
            fix_base=not floating,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        output = URDFImporter(config).import_urdf()

    asset = Path(output).resolve()
    if not asset.is_file() or output_dir.resolve() not in asset.parents:
        raise RuntimeError(f"Importer did not create a USD asset inside {output_dir}: {output!r}")
    return asset


def absolute_path_markers() -> list[bytes]:
    markers = {str(REPO), REPO.as_posix(), str(Path.home()), Path.home().as_posix(), ".asset-transformer-"}
    wsl = REPO.as_posix()
    if len(wsl) > 2 and wsl[1] == ":":  # C:/... seen from WSL as /mnt/c/...
        markers.add(f"/mnt/{wsl[0].lower()}{wsl[2:]}")
    return sorted({m.encode("utf-8") for m in markers if len(m) > 3})


def scrub_layers(folder: Path) -> list[str]:
    """Clear importer layer docs that embed the absolute staging path; refuse any absolute path left behind."""
    from pxr import Sdf

    cleared = []
    layers = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".usd", ".usda", ".usdc"))
    for path in layers:
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise RuntimeError(f"Cannot open generated layer {path}")
        if layer.documentation:
            layer.documentation = ""
            if not layer.Save():
                raise RuntimeError(f"Cannot save {path}")
            cleared.append(path.relative_to(folder).as_posix())
    markers = absolute_path_markers()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        data = path.read_bytes()
        for marker in markers:
            if marker in data:
                raise RuntimeError(f"generated file {path.name} contains a machine-specific path ({marker.decode()!r})")
    return cleared


def crate_content_hashes(folder: Path) -> dict[str, str]:
    """SHA-256 of the text export of each binary (crate) layer.

    The importer's crate files can differ byte-for-byte between identical imports while their content is the
    same; this hash is reproducible."""
    import hashlib

    from pxr import Sdf

    out = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".usd", ".usdc")):
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise RuntimeError(f"Cannot open generated layer {path}")
        out[path.relative_to(folder).as_posix()] = hashlib.sha256(layer.ExportToString().encode("utf-8")).hexdigest()
    return out


def select_physics_variant(prim) -> str:
    """Select a PhysX-compatible Physics payload on the referenced robot prim."""
    variant_sets = prim.GetVariantSets()
    if "Physics" not in variant_sets.GetNames():
        raise RuntimeError(
            "Imported robot has no Physics variant set. The Isaac Sim 6.2 "
            "asset transformer may not have produced a layered physics asset."
        )
    variant_set = variant_sets.GetVariantSet("Physics")
    choices = variant_set.GetVariantNames()
    # A dedicated PhysX overlay is preferable; 'physics' carries the shared USD Physics payload that PhysX also
    # consumes. Never select 'none' (no joints) or 'mujoco' (MuJoCo-only mjc:* schema) here.
    selected = next((name for name in ("physx", "physics") if name in choices), None)
    if selected is None:
        raise RuntimeError(
            "Imported robot has no PhysX-compatible Physics variant "
            f"('physx' or 'physics'); available variants: {choices}"
        )
    if not variant_set.SetVariantSelection(selected) or variant_set.GetVariantSelection() != selected:
        raise RuntimeError(f"Could not select Physics={selected!r} on {prim.GetPath()}")
    return selected


def rigid_body_ancestor(prim):
    from pxr import UsdPhysics

    while prim and prim.IsValid() and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        prim = prim.GetParent()
    return None


def colliders(stage, under=None) -> list:
    from pxr import Usd, UsdPhysics

    out = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if under is not None and not prim.GetPath().HasPrefix(under):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            out.append(prim)
    return out


def derive_ground_placement(stage, robot_path, clearance_m: float) -> dict[str, object]:
    """Place a model-check floor below enabled robot collision geometry, not markers."""
    from pxr import Usd, UsdGeom

    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("Ground clearance must be a finite, nonnegative distance")
    # Imported colliders have guide purpose and can be hidden in the viewport.
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        useExtentsHint=False,
        ignoreVisibility=True,
    )
    minimum_z = math.inf
    collision_count = 0
    for prim in colliders(stage, robot_path):
        if not prim.IsA(UsdGeom.Boundable):
            raise RuntimeError(f"Collision geometry is not boundable: {prim.GetPath()}")
        world_bounds = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        if world_bounds.IsEmpty() or not all(
            math.isfinite(value)
            for point in (world_bounds.GetMin(), world_bounds.GetMax())
            for value in point
        ):
            raise RuntimeError(f"Collision geometry has invalid bounds: {prim.GetPath()}")
        minimum_z = min(minimum_z, float(world_bounds.GetMin()[2]))
        collision_count += 1
    if not collision_count:
        raise RuntimeError("Cannot place model-check ground: robot has no enabled collision geometry")
    return {
        "method": "minimum world-space collision bound minus clearance",
        "collision_count": collision_count,
        "collision_min_z_m": minimum_z,
        "clearance_m": clearance_m,
        "ground_top_z_m": minimum_z - clearance_m,
        "calibrated": False,
    }


def block_velocity_placeholders(stage, robot_path, placeholders: list[str]) -> list[dict[str, object]]:
    """Block authored zero speed limits on placeholder joints in the scene layer (schema fallback applies)."""
    from pxr import UsdPhysics

    blocked: list[dict[str, object]] = []
    wanted = set(placeholders)
    found: set[str] = set()
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(robot_path) or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if prim.GetName() not in wanted:
            continue
        found.add(prim.GetName())
        for name in PLACEHOLDER_ATTRIBUTES:
            attribute = prim.GetAttribute(name)
            if not attribute or not attribute.HasAuthoredValue():
                continue
            value = attribute.Get()
            if value is None or not math.isfinite(float(value)) or float(value) != 0.0:
                raise RuntimeError(
                    f"{prim.GetPath()}.{name} = {value!r}, but the ledger has no speed for this joint; "
                    "expected the importer to carry the URDF placeholder 0"
                )
            attribute.Block()
            if attribute.HasAuthoredValue():
                raise RuntimeError(f"Could not block {prim.GetPath()}.{name}")
            blocked.append({"joint": prim.GetName(), "attribute": name, "importer_value": float(value)})
    if found != wanted:
        raise RuntimeError(f"Placeholder joints missing from the imported asset: {sorted(wanted - found)}")
    return blocked


def close(a: float, b: float) -> bool:
    return abs(a - b) <= FLOAT32_REL * max(abs(a), abs(b), 1e-12)


def float_attribute(prim, name: str, value: float, expected_import: float | None = None) -> dict[str, object]:
    """Author a float attribute in the scene layer; if the importer authored it, it must match expected_import."""
    from pxr import Sdf

    attribute = prim.GetAttribute(name)
    imported = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
    if imported is not None and expected_import is not None and not close(float(imported), expected_import):
        raise RuntimeError(f"{prim.GetPath()}.{name}: importer value {imported} differs from the source's "
                           f"{expected_import}")
    if not attribute:
        attribute = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
    attribute.Set(float(value))
    return {"value": float(value), "importer_value": None if imported is None else float(imported)}


def author_joint_dynamics(stage, robot_path, terms: dict[str, dict]) -> dict[str, dict[str, object]]:
    """Servo gains, armature and speed limits in the scene layer (see the module docstring)."""
    from pxr import UsdPhysics

    authored: dict[str, dict[str, object]] = {}
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(robot_path) or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        if name not in terms:
            raise RuntimeError(f"No dynamics terms for joint {prim.GetPath()}")
        t = terms[name]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        record: dict[str, object] = {}
        record["drive:angular:physics:maxForce"] = float_attribute(
            prim, "drive:angular:physics:maxForce", t["effort"], expected_import=t["effort"])
        record["drive:angular:physics:stiffness"] = float_attribute(
            prim, "drive:angular:physics:stiffness", t["kp"] * DEG)
        record["drive:angular:physics:damping"] = float_attribute(
            prim, "drive:angular:physics:damping", t["kv"] * DEG)
        if drive.GetTypeAttr().Get() not in (None, "force"):
            raise RuntimeError(f"{prim.GetPath()}: drive type {drive.GetTypeAttr().Get()!r}, expected force")
        record["newton:armature"] = float_attribute(prim, "newton:armature", t["armature"],
                                                    expected_import=t["armature"])
        prim.AddAppliedSchema("PhysxJointAPI")
        record["physxJoint:armature"] = float_attribute(prim, "physxJoint:armature", t["armature"])
        record["newton:damping"] = float_attribute(prim, "newton:damping", t["damping"] * DEG,
                                                   expected_import=t["damping"] * DEG)
        record["newton:friction"] = float_attribute(prim, "newton:friction", t["frictionloss"],
                                                    expected_import=t["frictionloss"])
        if t["velocity"] is not None:
            limit = t["velocity"] / DEG
            imported = prim.GetAttribute("newton:velocityLimit")
            if not imported or not imported.HasAuthoredValue() or not close(float(imported.Get()), limit):
                raise RuntimeError(f"{prim.GetPath()}: newton:velocityLimit does not carry the ledger speed {limit} deg/s")
            record["physxJoint:maxJointVelocity"] = float_attribute(prim, "physxJoint:maxJointVelocity", limit)
        authored[name] = record
    if set(authored) != set(terms):
        raise RuntimeError(f"Joints without authored dynamics: {sorted(set(terms) - set(authored))}")
    return authored


def disable_physx_self_collision(root_prim) -> dict[str, object]:
    from pxr import Sdf

    newton = root_prim.GetAttribute("newton:selfCollisionEnabled")
    newton_value = newton.Get() if newton and newton.HasAuthoredValue() else None
    if newton_value not in (False, 0):
        raise RuntimeError(f"{root_prim.GetPath()}: expected the importer's newton:selfCollisionEnabled = 0, "
                           f"found {newton_value!r}")
    root_prim.AddAppliedSchema("PhysxArticulationAPI")
    attribute = root_prim.GetAttribute("physxArticulation:enabledSelfCollisions")
    if not attribute:
        attribute = root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool)
    attribute.Set(False)
    return {"prim": str(root_prim.GetPath()), "newton:selfCollisionEnabled": bool(newton_value),
            "physxArticulation:enabledSelfCollisions": False}


def friction_material(stage, friction: float):
    from pxr import Sdf, UsdPhysics, UsdShade

    path = f"{MATERIALS}/sliding_friction_{format(friction, 'g').replace('.', 'p')}"
    material = UsdShade.Material.Get(stage, path)
    if material:
        return material
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(0.0)
    if friction == 0.0:
        # PhysX averages the two materials' friction by default; frictionless guards must win.
        material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
        material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    return material


def bind_physics_material(prim, material) -> None:
    from pxr import UsdPhysics, UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.strongerThanDescendants, PHYSICS_PURPOSE)
    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(PHYSICS_PURPOSE)
    if not bound or bound.GetPath() != material.GetPath():
        raise RuntimeError(f"Could not bind {material.GetPath()} to {prim.GetPath()}")


def assign_contact_materials(stage, robot_path, frictions: dict[str, float]) -> dict[str, object]:
    found: dict[str, float] = {}
    for prim in colliders(stage, robot_path):
        name = prim.GetName()
        if prim.IsInstanceProxy():
            raise RuntimeError(f"Collider {prim.GetPath()} is inside an instance; cannot bind a material")
        if name not in frictions:
            raise RuntimeError(f"Robot collider {prim.GetPath()} has no vendor contact class")
        if name in found:
            raise RuntimeError(f"Two robot colliders are named {name}")
        bind_physics_material(prim, friction_material(stage, frictions[name]))
        found[name] = frictions[name]
    if set(found) != set(frictions):
        raise RuntimeError(f"Source colliders missing from the imported asset: {sorted(set(frictions) - set(found))}")
    counts: dict[str, int] = {}
    for value in found.values():
        counts[format(value, "g")] = counts.get(format(value, "g"), 0) + 1
    return {"robot_colliders": len(found), "by_sliding_friction": counts}


def build_scene(asset: Path, scene_path: Path, expected_names: list[str], *, terms: dict[str, dict],
                frictions: dict[str, float], floor_friction: float, timestep: float,
                spawn_pose: dict[str, float] | None = None, ground_clearance_m: float | None = None,
                placeholders: list[str] | None = None) -> dict[str, object]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    imported = Usd.Stage.Open(str(asset))
    if imported is None or not imported.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Imported asset has no valid default prim: {asset}")
    del imported

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = scene_path.with_name(f".{scene_path.stem}.{uuid.uuid4().hex}.usda")
    try:
        stage = Usd.Stage.CreateNew(str(temporary))
        if stage is None:
            raise RuntimeError(f"Could not create temporary scene: {temporary}")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(1e9)
        stage.SetTimeCodesPerSecond(60.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        # Keep the transform on a parent prim: the referenced asset's default prim carries its Physics variant
        # set and must retain its selection.
        spawn = UsdGeom.Xform.Define(stage, "/World/TonyPi")
        robot = UsdGeom.Xform.Define(stage, "/World/TonyPi/Robot")
        reference = Path(os.path.relpath(asset, scene_path.parent)).as_posix()
        robot.GetPrim().GetReferences().AddReference(reference)
        if spawn_pose is not None:
            # The URDF root starts at the origin; the twin's floating keyframe puts the torso at this ledger pose.
            spawn.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(spawn_pose["x"], spawn_pose["y"], spawn_pose["z"]))
            quat = [spawn_pose[k] for k in ("qw", "qx", "qy", "qz")]
            norm = math.sqrt(sum(v * v for v in quat))
            if not norm > 0:
                raise RuntimeError("ledger home root quaternion is zero")
            spawn.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(quat[0] / norm, Gf.Vec3d(quat[1] / norm, quat[2] / norm, quat[3] / norm)))

        physics_variant = select_physics_variant(robot.GetPrim())
        if not stage.GetPrimAtPath("/World/TonyPi/Robot").IsValid():
            raise RuntimeError("Robot USD reference did not compose")
        robot_path = robot.GetPath()

        # World geometry the importer brought along (the contact candidate's floor plane): the scene's own ground
        # replaces it, so the floor neither duplicates nor moves with the robot's placement.
        deactivated = []
        for prim in colliders(stage, robot_path):
            if rigid_body_ancestor(prim) is None:
                deactivated.append(str(prim.GetPath()))
        for path in deactivated:
            stage.OverridePrim(path).SetActive(False)

        blocked = block_velocity_placeholders(stage, robot_path, placeholders or [])
        dynamics = author_joint_dynamics(stage, robot_path, terms)
        roots = [
            prim for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if len(roots) != 1:
            raise RuntimeError(f"Expected one articulation root, found {[str(p.GetPath()) for p in roots]}")
        self_collision = disable_physx_self_collision(roots[0])
        materials = assign_contact_materials(stage, robot_path, frictions)

        placement = None
        ground_top_z = 0.0
        if ground_clearance_m is not None:
            placement = derive_ground_placement(stage, robot_path, ground_clearance_m)
            ground_top_z = float(placement["ground_top_z_m"])

        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ground_top_z - 0.05))
        ground.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.1))
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.24)])
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        bind_physics_material(ground.GetPrim(), friction_material(stage, floor_friction))

        physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics.CreateGravityMagnitudeAttr(9.81)
        rate = int(round(1.0 / timestep))
        physics.GetPrim().AddAppliedSchema("PhysxSceneAPI")
        physics.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(rate)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2000.0)
        sun.CreateAngleAttr(0.53)

        grounds = [str(p.GetPath()) for p in colliders(stage) if rigid_body_ancestor(p) is None]
        if grounds != ["/World/Ground"]:
            raise RuntimeError(f"Expected exactly one ground collider (/World/Ground), found {grounds}")

        revolute = [
            prim for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.IsA(UsdPhysics.RevoluteJoint)
        ]
        other_movable = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path)
            and prim.IsA(UsdPhysics.Joint)
            and not prim.IsA(UsdPhysics.RevoluteJoint)
            and not prim.IsA(UsdPhysics.FixedJoint)
        ]
        names = sorted(prim.GetName() for prim in revolute)
        if names != sorted(expected_names) or other_movable:
            raise RuntimeError(
                f"Imported scene failed closed: expected the {len(expected_names)} named revolute joints; "
                f"found revolute={names}, other_movable={other_movable}"
            )

        layer_data: dict[str, object] = {
            "tonypi_velocity_placeholders_blocked": json.dumps(blocked),
            "tonypi_joint_dynamics": json.dumps(dynamics, sort_keys=True),
            "tonypi_self_collision": json.dumps(self_collision, sort_keys=True),
            "tonypi_contact_materials": json.dumps({**materials, "ground_sliding_friction": floor_friction},
                                                   sort_keys=True),
            "tonypi_deactivated_world_colliders": json.dumps(
                [Path(p).relative_to("/World/TonyPi/Robot").as_posix() for p in deactivated]),
            "tonypi_physics_step": json.dumps({"timestep_s": timestep, "physxScene:timeStepsPerSecond": rate}),
        }
        if placement is not None:
            layer_data["tonypi_ground_placement"] = json.dumps(placement, sort_keys=True)
        stage.GetRootLayer().customLayerData = layer_data
        stage.GetRootLayer().comment = (
            "TonyPi Pro twin model candidate (Hiwonder humanoid, pab47/TonyPi geometry + model/ledger.csv). "
            "Geometry, inertials and servo parameters are not validated against a physical TonyPi Pro."
        )
        if placement is not None:
            stage.GetRootLayer().comment += (
                f" Model-check ground is {ground_clearance_m:g} m below the minimum "
                "collision bound; this placement is not a calibrated standing pose."
            )
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"Could not save temporary scene: {temporary}")
        # A scene reference stays relative to this directory after the rename.
        del stage
        os.replace(temporary, scene_path)
        return {
            "root": str(roots[0].GetPath()), "joints": [str(p.GetPath()) for p in revolute],
            "physics_variant": physics_variant, "blocked": blocked, "placement": placement,
            "deactivated": deactivated, "materials": materials, "physics_rate": rate,
            "colliders_total": len(colliders(Usd.Stage.Open(str(scene_path)))),
        }
    finally:
        temporary.unlink(missing_ok=True)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    variant = "floating" if args.floating else "fixed"
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from isaac_sim.source_integrity import IntegrityError, recheck, verify_source_bundle, write_integrity_report

    source = source_path(args.source, args.floating)
    if args.source == "urdf" and not source.is_file():
        raise SystemExit(f"Derived URDF is missing: {source}\n"
                         f"Run {VENV_PYTHON} -I isaac_sim/build_tonypi_urdf.py from the repository root first.")
    try:
        integrity = verify_source_bundle(args.source, source, args.floating)
        derivation = integrity["derivation"]
        floor_friction = vendor_floor_friction()
        if args.source == "mjcf":
            expected_names = preflight_mjcf(source, args.floating)
            terms, frictions = mjcf_model_terms(source, expected_names)
            placeholders: list[str] = []
            spawn_pose = None
            timestep = option_timestep(source)
            terms_from = f"{integrity['source']['path']} compiled by MuJoCo"  # type: ignore[index]
        else:
            expected_names = preflight_urdf(source, derivation)  # type: ignore[arg-type]
            terms = dict(derivation["importer_joint_terms"]["joints"])  # type: ignore[index]
            frictions = urdf_collider_frictions(derivation)  # type: ignore[arg-type]
            placeholders = list(derivation["velocity_placeholders"])  # type: ignore[index]
            spawn_pose = dict(derivation["home_root_pose"]) if args.floating else None  # type: ignore[index]
            timestep = option_timestep(VENDOR_ROBOT)
            terms_from = "the URDF build record (model/ledger.csv)"
    except (IntegrityError, OSError, ValueError, KeyError, ET.ParseError) as exc:
        raise SystemExit(f"Model preflight failed: {exc}") from exc
    ground_clearance_m = None if args.floating else FIXED_BASE_GROUND_CLEARANCE_M
    asset_dir = GENERATED / "usd" / f"tonypi_{args.source}_{variant}"
    scene = GENERATED / f"tonypi_{args.source}_{variant}_scene.usda"
    scene_options = dict(terms=terms, frictions=frictions, floor_friction=floor_friction, timestep=timestep,
                         spawn_pose=spawn_pose, ground_clearance_m=ground_clearance_m, placeholders=placeholders)

    def run() -> None:
        token = uuid.uuid4().hex[:12]
        staging = asset_dir.parent / f".staging-{asset_dir.name}-{token}"
        previous_dir = asset_dir.parent / f".previous-{asset_dir.name}-{token}"
        validation_scene = GENERATED / f".validate-{scene.stem}-{token}.usda"
        previous_scene = GENERATED / f".previous-{scene.stem}-{token}.usda"
        swapped = False
        try:
            staged = import_asset(source, staging, args.source, args.floating, use_kit=not args.no_kit)
            cleared = scrub_layers(staging)
            build_scene(staged, validation_scene, expected_names, **scene_options)  # checks the staged asset
            validation_scene.unlink()
            recheck(integrity)  # every pinned input still has its preflight hash
            if asset_dir.exists():
                os.replace(asset_dir, previous_dir)
            if scene.exists():
                os.replace(scene, previous_scene)
            swapped = True
            os.replace(staging, asset_dir)
            asset = asset_dir / staged.relative_to(staging)
            result = build_scene(asset, scene, expected_names, **scene_options)
            report = write_integrity_report(integrity, asset, scene, extra={
                "scene_layer": {"joint_terms_from": terms_from, "cleared_layer_docs": cleared},
                "crate_layer_content_sha256": crate_content_hashes(asset_dir)})
        except BaseException:
            if swapped:
                remove_tree(asset_dir)
                scene.unlink(missing_ok=True)
                if previous_dir.exists():
                    os.replace(previous_dir, asset_dir)
                if previous_scene.exists():
                    os.replace(previous_scene, scene)
                print("Import failed after the swap; the previous asset and scene were restored.", file=sys.stderr)
            remove_tree(staging)
            validation_scene.unlink(missing_ok=True)
            raise
        remove_tree(previous_dir)
        previous_scene.unlink(missing_ok=True)
        print(f"Imported asset: {asset}")
        print(f"Cleared importer layer docs (absolute staging path): {cleared or 'none'}")
        print(f"Selected Physics variant: {result['physics_variant']}")
        print(f"Validated articulation root: {result['root']}")
        print(f"Validated revolute joints: {len(result['joints'])} (names match the source)")
        print(f"Joint dynamics authored in the scene layer from {terms_from}: drive stiffness/damping = servo "
              f"kp/kv, armature (newton + physx), physxJoint:maxJointVelocity for "
              f"{sum(1 for t in terms.values() if t['velocity'] is not None)} joints")
        print(f"Self-collision: off for Newton and PhysX")
        print(f"Contact materials: {result['materials']}; ground sliding friction {floor_friction:g}")
        print(f"Deactivated world colliders from the source: {result['deactivated'] or 'none'}")
        print(f"Colliders in the scene: {result['colliders_total']} (robot + one ground)")
        print(f"Physics step: physxScene:timeStepsPerSecond = {result['physics_rate']} (timestep {timestep} s)")
        if result["placement"] is not None:
            print(f"Uncalibrated model-check ground placement: {json.dumps(result['placement'], sort_keys=True)}")
        if spawn_pose is not None:
            print(f"Floating spawn at the ledger home pose: {spawn_pose}")
        print(f"Velocity placeholders blocked in the scene layer: {result['blocked'] or 'none'}")
        print(f"Saved scene: {scene}")
        print(f"Integrity report: {report}")

    if args.no_kit:
        try:
            import isaacsim.asset.importer  # noqa: F401
            import pxr  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                f"Model preflight passed, but --no-kit needs the Isaac Sim 6.2 importer and USD in this Python ({exc}); "
                "for example the pip package isaacsim-asset-isolated 6.2 of an Isaac Lab venv."
            ) from exc
        run()
        return
    # Isaac/Omniverse imports above this point would be too early: SimulationApp must be created first in a
    # standalone Python process.
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise SystemExit(
            "Model preflight passed, but Isaac Sim's SimulationApp is unavailable "
            f"({exc}). Run this script with Isaac Sim 6.2's bundled python.bat or python.sh on a "
            "supported RTX host, or pass --no-kit to convert and check the USD without Kit."
        ) from exc

    app = SimulationApp({"headless": args.headless})
    try:
        run()
    except BaseException:
        # Kit's fast shutdown exits the process, so print and preserve failures before closing instead of
        # allowing close() to replace them with exit 0.
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()


if __name__ == "__main__":
    main()
