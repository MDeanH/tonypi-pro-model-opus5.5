"""Import the TonyPi Pro twin into Isaac Sim 6.2 and save a USD scene with provenance hashes.

--source urdf (default) imports the ledger-derived isaac_sim/generated/tonypi_pro_candidate.urdf with NVIDIA's
URDF importer; --source mjcf imports the twin MJCF with the MJCF importer (build/tonypi_pro_as_is.xml, or the
floating contact candidate build/tonypi_pro_contact_candidate.xml with --floating). Every input is verified first
(isaac_sim/source_integrity.py). The scene must contain exactly one articulation root and the twin's 20 revolute
joints, by name, or nothing is saved.

Run it with Isaac Sim's bundled python.bat / python.sh. --no-kit skips SimulationApp and calls the same importer
classes directly; that works where the importer runs without Kit (for example the pip isaacsim-asset-isolated
package in an Isaac Lab venv) and produces the same USD and checks, but no PhysX run (run_scene.py needs Kit).

Placement: a floating URDF is spawned at the ledger's scene.home.root pose (recorded in the URDF build record);
the MJCF sources carry their own torso pose. Floating scenes get a floor whose top is z = 0, as in the vendor
scene. Fixed-base scenes get an uncalibrated model-check floor 5 mm below the lowest collider, so the welded robot
never touches it. No standing controller or balance behaviour is assumed.

Velocity placeholders: joints whose ledger speed is missing carry velocity="0" in the URDF, which the 6.2 importer
writes as newton:velocityLimit = 0 (a frozen joint for a Newton-based simulation). The scene layer blocks that
authored zero for exactly those joints, so the schema fallback (no limit) applies, as in the MuJoCo twin. The
imported asset itself is not edited, and the blocked joints are listed in the scene's customLayerData.

Neither source is a verified physical match to a TonyPi Pro.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
GENERATED = REPO / "isaac_sim" / "generated"
EXPECTED_JOINTS = 20
FIXED_BASE_GROUND_CLEARANCE_M = 0.005
PLACEHOLDER_ATTRIBUTES = ("newton:velocityLimit", "physxJoint:maxJointVelocity")


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


def import_asset(source: Path, output_dir: Path, source_type: str, floating: bool, use_kit: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
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
            import_scene=False,        # The script adds its own floor, light and physics scene.
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
    # consumes. Never select 'none' (no joints) or 'mujoco' here.
    selected = next((name for name in ("physx", "physics") if name in choices), None)
    if selected is None:
        raise RuntimeError(
            "Imported robot has no PhysX-compatible Physics variant "
            f"('physx' or 'physics'); available variants: {choices}"
        )
    if not variant_set.SetVariantSelection(selected) or variant_set.GetVariantSelection() != selected:
        raise RuntimeError(f"Could not select Physics={selected!r} on {prim.GetPath()}")
    return selected


def derive_ground_placement(robot, clearance_m: float) -> dict[str, object]:
    """Place a model-check floor below enabled collision geometry, not markers."""
    from pxr import Usd, UsdGeom, UsdPhysics

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
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
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


def build_scene(asset: Path, scene_path: Path, expected_names: list[str], *,
                spawn_pose: dict[str, float] | None = None, ground_clearance_m: float | None = None,
                placeholders: list[str] | None = None) -> tuple[str, list[str], str, list[dict[str, object]]]:
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    imported = Usd.Stage.Open(str(asset))
    if imported is None or not imported.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Imported asset has no valid default prim: {asset}")

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
        blocked = block_velocity_placeholders(stage, robot_path, placeholders or [])

        placement = None
        ground_top_z = 0.0
        layer_data: dict[str, object] = {"tonypi_velocity_placeholders_blocked": json.dumps(blocked)}
        if ground_clearance_m is not None:
            placement = derive_ground_placement(robot.GetPrim(), ground_clearance_m)
            ground_top_z = float(placement["ground_top_z_m"])
            layer_data["tonypi_ground_placement"] = json.dumps(placement, sort_keys=True)
        stage.GetRootLayer().customLayerData = layer_data

        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ground_top_z - 0.05))
        ground.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.1))
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.24)])
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics.CreateGravityMagnitudeAttr(9.81)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2000.0)
        sun.CreateAngleAttr(0.53)

        stage.GetRootLayer().comment = (
            "TonyPi Pro twin model candidate (Hiwonder humanoid, pab47/TonyPi geometry + model/ledger.csv). "
            "Geometry, inertials and servo parameters are not validated against a physical TonyPi Pro."
        )
        if placement is not None:
            stage.GetRootLayer().comment += (
                f" Model-check ground is {ground_clearance_m:g} m below the minimum "
                "collision bound; this placement is not a calibrated standing pose."
            )
        roots = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
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
        if len(roots) != 1 or names != sorted(expected_names) or other_movable:
            raise RuntimeError(
                "Imported scene failed closed: expected one articulation root and the "
                f"{len(expected_names)} named revolute joints; found roots={roots}, "
                f"revolute={names}, other_movable={other_movable}"
            )
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"Could not save temporary scene: {temporary}")
        # A scene reference stays relative to this directory after the rename.
        del stage
        os.replace(temporary, scene_path)
        if placement is not None:
            print(f"Uncalibrated model-check ground placement: {json.dumps(placement, sort_keys=True)}")
        return roots[0], [str(prim.GetPath()) for prim in revolute], physics_variant, blocked
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    variant = "floating" if args.floating else "fixed"
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from isaac_sim.source_integrity import IntegrityError, verify_source_bundle, write_integrity_report

    source = source_path(args.source, args.floating)
    if args.source == "urdf" and not source.is_file():
        raise SystemExit(f"Derived URDF is missing: {source}\n"
                         "Run .venv/Scripts/python.exe -I isaac_sim/build_tonypi_urdf.py from the repository root first.")
    try:
        integrity = verify_source_bundle(args.source, source, args.floating)
        derivation = integrity["derivation"]
        if args.source == "mjcf":
            expected_names = preflight_mjcf(source, args.floating)
            placeholders: list[str] = []
            spawn_pose = None
        else:
            expected_names = preflight_urdf(source, derivation)  # type: ignore[arg-type]
            placeholders = list(derivation["velocity_placeholders"])  # type: ignore[index]
            spawn_pose = dict(derivation["home_root_pose"]) if args.floating else None  # type: ignore[index]
    except (IntegrityError, OSError, ValueError) as exc:
        raise SystemExit(f"Model preflight failed: {exc}") from exc
    ground_clearance_m = None if args.floating else FIXED_BASE_GROUND_CLEARANCE_M
    asset_dir = GENERATED / "usd" / f"tonypi_{args.source}_{variant}"
    scene = GENERATED / f"tonypi_{args.source}_{variant}_scene.usda"

    def run() -> None:
        asset = import_asset(source, asset_dir, args.source, args.floating, use_kit=not args.no_kit)
        root, joints, physics_variant, blocked = build_scene(
            asset, scene, expected_names, spawn_pose=spawn_pose,
            ground_clearance_m=ground_clearance_m, placeholders=placeholders,
        )
        report = write_integrity_report(integrity, asset, scene)
        print(f"Imported asset: {asset}")
        print(f"Selected Physics variant: {physics_variant}")
        print(f"Validated articulation root: {root}")
        print(f"Validated revolute joints: {len(joints)} (names match the source)")
        if spawn_pose is not None:
            print(f"Floating spawn at the ledger home pose: {spawn_pose}")
        print(f"Velocity placeholders blocked in the scene layer: {blocked or 'none'}")
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
