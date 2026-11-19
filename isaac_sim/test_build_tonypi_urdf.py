"""Checks for the derived TonyPi Pro URDF, using the real ledger, vendor files and twin build.

The offline equivalence check needs no Isaac Sim: it loads the generated URDF in MuJoCo (with a temporary copy
that adds a <mujoco><compiler> block so meshes resolve and no link is fused) and compares it with
build/tonypi_pro_as_is.mjb, the exact binary twin. Compared, in the torso frame: the world pose of every link at
random joint configurations (forward kinematics), link masses, centres of mass, full inertia tensors about the
COM, joint axes, anchors and ranges, passive damping and friction, the placement of every visual mesh vertex and
every collision primitive (each vendor capsule against its cylinder plus two end spheres). Max errors are printed.

Usage (repository root): .venv/Scripts/python.exe -I isaac_sim/test_build_tonypi_urdf.py [-v]
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_tonypi_urdf import (  # noqa: E402
    BuildError, DEFAULT_OUTPUT, GENERATED_DIR, LEDGER, REPO, ROBOT_NAME, VENDOR_MJCF, build, capsule_parts,
    collision_elements, generate, parse_vendor, quat_to_rpy, record_path_for, vendor_sums,
)

MJB = REPO / "build/tonypi_pro_as_is.mjb"
TWIN_XML = REPO / "build/tonypi_pro_as_is.xml"
CONFIGS = 8            # random joint configurations (plus the zero pose)
SEED = 20261119
TOL_POS = 1e-9         # m
TOL_ROT = 1e-9         # rotation-matrix entries
TOL_REL = 1e-9         # relative, for masses and inertia tensors
EXPECTED_LINKS = 21
EXPECTED_JOINTS = 20


def ledger_rows(path: Path = LEDGER) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_ledger(directory: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    path = directory / f"ledger_{len(list(directory.glob('ledger_*.csv')))}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def active(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r["param"]: r for r in rows if r["status"] == "active"}


def quat_mat(quat) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


def load_urdf_in_mujoco(urdf: Path, scratch: Path) -> mujoco.MjModel:
    """MuJoCo parses URDF natively. Its URDF defaults fuse static links (torso would merge into the world),
    discard visuals and strip mesh paths, so a temporary copy turns those off and points meshdir at the
    URDF's own folder, where its relative mesh paths start."""
    text = urdf.read_text(encoding="utf-8")
    marker = f'<robot name="{ROBOT_NAME}">'
    if text.count(marker) != 1:
        raise AssertionError(f"{urdf}: expected one {marker}")
    extension = (f'<mujoco><compiler meshdir="{urdf.parent.resolve().as_posix()}" strippath="false" '
                 'fusestatic="false" discardvisual="false" balanceinertia="false"/></mujoco>')
    copy = scratch / f"{urdf.stem}.mujoco_check.urdf"
    copy.write_text(text.replace(marker, marker + extension, 1), encoding="utf-8")
    try:
        return mujoco.MjModel.from_xml_path(str(copy))
    finally:
        copy.unlink(missing_ok=True)


def names(model: mujoco.MjModel, kind, count: int) -> dict[str, int]:
    return {mujoco.mj_id2name(model, kind, i): i for i in range(count)}


def full_inertia(model: mujoco.MjModel, body: int) -> np.ndarray:
    rotation = quat_mat(model.body_iquat[body])
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def equivalence(urdf: Path, scratch: Path, home: dict[str, float]) -> dict[str, object]:
    """Max errors between the URDF (loaded by MuJoCo) and the binary twin. Raises if structure differs."""
    twin = mujoco.MjModel.from_binary_path(str(MJB))
    cand = load_urdf_in_mujoco(urdf, scratch)
    tb, cb = names(twin, mujoco.mjtObj.mjOBJ_BODY, twin.nbody), names(cand, mujoco.mjtObj.mjOBJ_BODY, cand.nbody)
    tj, cj = names(twin, mujoco.mjtObj.mjOBJ_JOINT, twin.njnt), names(cand, mujoco.mjtObj.mjOBJ_JOINT, cand.njnt)
    if set(tb) != set(cb) or set(tj) != set(cj):
        raise AssertionError(f"body/joint names differ: {sorted(set(tb) ^ set(cb))} {sorted(set(tj) ^ set(cj))}")
    bodies = [n for n in tb if n != "world"]
    for name in bodies:
        tp = mujoco.mj_id2name(twin, mujoco.mjtObj.mjOBJ_BODY, int(twin.body_parentid[tb[name]]))
        cp = mujoco.mj_id2name(cand, mujoco.mjtObj.mjOBJ_BODY, int(cand.body_parentid[cb[name]]))
        if tp != cp:
            raise AssertionError(f"{name}: parent {tp} in the twin, {cp} in the URDF")
    err: dict[str, float] = {k: 0.0 for k in (
        "link_pos_m", "link_rot", "com_local_m", "com_world_m", "mass_rel", "inertia_local_rel", "inertia_world_rel",
        "joint_axis_local", "joint_axis_world", "joint_anchor_m", "joint_pos_local_m", "joint_range_rad",
        "damping", "frictionloss", "mesh_vertex_m", "collision_pos_m", "collision_size_m", "collision_rot",
        "torso_vs_ledger_home_m", "total_mass_kg")}

    def bump(key: str, value: float) -> None:
        err[key] = max(err[key], float(value))

    for name in bodies:
        i, k = tb[name], cb[name]
        bump("mass_rel", abs(twin.body_mass[i] - cand.body_mass[k]) / twin.body_mass[i])
        bump("com_local_m", np.abs(twin.body_ipos[i] - cand.body_ipos[k]).max())
        ti, ci = full_inertia(twin, i), full_inertia(cand, k)
        bump("inertia_local_rel", np.abs(ti - ci).max() / np.abs(ti).max())
    err["total_mass_kg"] = abs(float(twin.body_mass.sum()) - float(cand.body_mass.sum()))
    joints = sorted(tj, key=lambda n: tj[n])
    armature = {}
    for name in joints:
        i, k = tj[name], cj[name]
        if twin.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE or cand.jnt_type[k] != mujoco.mjtJoint.mjJNT_HINGE:
            raise AssertionError(f"{name}: not a hinge in both models")
        bump("joint_range_rad", np.abs(twin.jnt_range[i] - cand.jnt_range[k]).max())
        bump("joint_axis_local", np.abs(twin.jnt_axis[i] - cand.jnt_axis[k]).max())
        bump("joint_pos_local_m", np.abs(twin.jnt_pos[i] - cand.jnt_pos[k]).max())
        ti, ci = twin.jnt_dofadr[i], cand.jnt_dofadr[k]
        bump("damping", abs(twin.dof_damping[ti] - cand.dof_damping[ci]))
        bump("frictionloss", abs(twin.dof_frictionloss[ti] - cand.dof_frictionloss[ci]))
        armature[name] = (float(twin.dof_armature[ti]), float(cand.dof_armature[ci]))
        if not cand.jnt_limited[k]:
            raise AssertionError(f"{name}: URDF joint is not limited in MuJoCo")
    # Effort: MuJoCo's URDF parser maps <limit effort> to the joint's actuator force range when it supports it.
    effort = {"mapped_by_parser": bool(np.any(cand.jnt_actfrclimited)), "max_abs_error": 0.0}
    if effort["mapped_by_parser"]:
        for a in range(twin.nu):
            jname = mujoco.mj_id2name(twin, mujoco.mjtObj.mjOBJ_JOINT, int(twin.actuator_trnid[a, 0]))
            effort["max_abs_error"] = max(effort["max_abs_error"], float(np.abs(
                twin.actuator_forcerange[a] - cand.jnt_actfrcrange[cj[jname]]).max()))

    home_pos = np.array([home["x"], home["y"], home["z"]])
    home_rot = quat_mat(np.array([home[k] for k in ("qw", "qx", "qy", "qz")]) /
                        np.linalg.norm([home[k] for k in ("qw", "qx", "qy", "qz")]))
    twin_meshes = mesh_geoms(twin)
    cand_meshes = mesh_geoms(cand)
    if sorted(twin_meshes) != sorted(cand_meshes) or len(twin_meshes) != EXPECTED_LINKS:
        raise AssertionError(f"visual mesh geoms per link differ: {sorted(twin_meshes)} vs {sorted(cand_meshes)}")
    td, cd = mujoco.MjData(twin), mujoco.MjData(cand)
    rng = np.random.default_rng(SEED)
    collision_counts = None
    for config in range(CONFIGS + 1):
        for name in joints:
            lo, hi = twin.jnt_range[tj[name]]
            q = 0.0 if config == 0 else rng.uniform(lo, hi)
            td.qpos[twin.jnt_qposadr[tj[name]]] = q
            cd.qpos[cand.jnt_qposadr[cj[name]]] = q
        mujoco.mj_kinematics(twin, td)
        mujoco.mj_kinematics(cand, cd)
        p0, r0 = td.xpos[tb["torso"]].copy(), td.xmat[tb["torso"]].reshape(3, 3).copy()
        q0, s0 = cd.xpos[cb["torso"]].copy(), cd.xmat[cb["torso"]].reshape(3, 3).copy()
        bump("torso_vs_ledger_home_m", max(np.abs(p0 - home_pos).max(), np.abs(r0 - home_rot).max()))
        if np.abs(q0).max() > 0 or np.abs(s0 - np.eye(3)).max() > 0:
            raise AssertionError("URDF root link is not at the origin in MuJoCo")

        def rel_p(p):
            return r0.T @ (p - p0)

        def rel_r(r):
            return r0.T @ r

        for name in bodies:
            i, k = tb[name], cb[name]
            bump("link_pos_m", np.abs(rel_p(td.xpos[i]) - cd.xpos[k]).max())
            bump("link_rot", np.abs(rel_r(td.xmat[i].reshape(3, 3)) - cd.xmat[k].reshape(3, 3)).max())
            bump("com_world_m", np.abs(rel_p(td.xipos[i]) - cd.xipos[k]).max())
            ti = rel_r(td.ximat[i].reshape(3, 3))
            tw = ti @ np.diag(twin.body_inertia[i]) @ ti.T
            ci = cd.ximat[k].reshape(3, 3)
            cw = ci @ np.diag(cand.body_inertia[k]) @ ci.T
            bump("inertia_world_rel", np.abs(tw - cw).max() / np.abs(tw).max())
        for name in joints:
            i, k = tj[name], cj[name]
            bump("joint_anchor_m", np.abs(rel_p(td.xanchor[i]) - cd.xanchor[k]).max())
            bump("joint_axis_world", np.abs(r0.T @ td.xaxis[i] - cd.xaxis[k]).max())
        for body, (tg, cg) in ((b, (twin_meshes[b], cand_meshes[b])) for b in twin_meshes):
            tv = mesh_world_vertices(twin, td, tg)
            cv = mesh_world_vertices(cand, cd, cg)
            if tv.shape != cv.shape:
                raise AssertionError(f"{body}: visual mesh vertex count {tv.shape} vs {cv.shape}")
            bump("mesh_vertex_m", np.abs((tv - p0) @ r0 - cv).max())
        collision_counts = compare_collisions(twin, td, cand, cd, rel_p, rel_r, bump)
    return {"errors": err, "armature_twin_vs_urdf": armature, "effort": effort,
            "collision_counts": collision_counts, "configs": CONFIGS + 1, "links": len(bodies), "joints": len(joints)}


def mesh_geoms(model: mujoco.MjModel) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
            if body in out:
                raise AssertionError(f"{body}: more than one mesh geom")
            out[body] = g
    return out


def mesh_world_vertices(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start + count].astype(float)
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def collision_geoms(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    out = []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        out.append(dict(id=g, body=int(model.geom_bodyid[g]), type=int(model.geom_type[g]),
                        size=model.geom_size[g].copy(), pos=data.geom_xpos[g].copy(),
                        rot=data.geom_xmat[g].reshape(3, 3).copy()))
    return out


def compare_collisions(twin, td, cand, cd, rel_p, rel_r, bump) -> dict[str, int]:
    """Match every twin collision primitive with its URDF counterpart(s); none may be left over."""
    gt = mujoco.mjtGeom
    tgeoms = collision_geoms(twin, td)
    remaining = collision_geoms(cand, cd)
    counts = {"twin_primitives": len(tgeoms), "urdf_primitives": len(remaining)}

    def body_name(model, g):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, g["body"])

    def take(body: str, gtype: int, centre: np.ndarray) -> dict:
        best = None
        for g in remaining:
            if g["type"] == gtype and body_name(cand, g) == body:
                d = np.abs(g["pos"] - centre).max()
                if best is None or d < best[0]:
                    best = (d, g)
        if best is None:
            raise AssertionError(f"{body}: no URDF collision geom of type {gtype} left to match")
        remaining.remove(best[1])
        bump("collision_pos_m", best[0])
        return best[1]

    for g in tgeoms:
        body = body_name(twin, g)
        centre, rot = rel_p(g["pos"]), rel_r(g["rot"])
        if g["type"] == gt.mjGEOM_BOX:
            match = take(body, gt.mjGEOM_BOX, centre)
            bump("collision_size_m", np.abs(match["size"] - g["size"]).max())
            bump("collision_rot", np.abs(match["rot"] - rot).max())
        elif g["type"] == gt.mjGEOM_SPHERE:
            match = take(body, gt.mjGEOM_SPHERE, centre)
            bump("collision_size_m", abs(match["size"][0] - g["size"][0]))
        elif g["type"] == gt.mjGEOM_CAPSULE:
            radius, half = g["size"][0], g["size"][1]
            axis = rot[:, 2]
            cyl = take(body, gt.mjGEOM_CYLINDER, centre)
            bump("collision_size_m", max(abs(cyl["size"][0] - radius), abs(cyl["size"][1] - half)))
            bump("collision_rot", np.linalg.norm(np.cross(cyl["rot"][:, 2], axis)))  # parallel axes
            for end in (centre - half * axis, centre + half * axis):
                cap = take(body, gt.mjGEOM_SPHERE, end)
                bump("collision_size_m", abs(cap["size"][0] - radius))
        else:
            raise AssertionError(f"{body}: unexpected twin collision geom type {g['type']}")
    if remaining:
        raise AssertionError(f"{len(remaining)} URDF collision geoms have no twin counterpart")
    return counts


class BuildTonyPiUrdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        cls._tmp = tempfile.TemporaryDirectory(dir=GENERATED_DIR)
        cls.tmp = Path(cls._tmp.name)
        cls.output = cls.tmp / "candidate.urdf"
        cls.candidate = build(output=cls.output)
        cls.root = ET.parse(cls.output).getroot()
        cls.fieldnames, cls.rows = ledger_rows()
        cls.active = active(cls.rows)
        cls.vendor = parse_vendor(VENDOR_MJCF.read_bytes(), VENDOR_MJCF.parent)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # --------------------------------------------------------------------------------------- structure

    def test_tree_has_21_links_and_20_revolute_joints_matching_the_vendor(self) -> None:
        links = [link.get("name") for link in self.root.findall("link")]
        joints = self.root.findall("joint")
        self.assertEqual(self.root.get("name"), ROBOT_NAME)
        self.assertEqual(len(links), EXPECTED_LINKS)
        self.assertEqual(len(joints), EXPECTED_JOINTS)
        self.assertTrue(all(j.get("type") == "revolute" for j in joints))
        self.assertEqual(links, [b["name"] for b in self.vendor.bodies])
        self.assertEqual([j.get("name") for j in joints], self.vendor.joints)
        parents = {b["name"]: b["parent"] for b in self.vendor.bodies}
        for joint in joints:
            self.assertEqual(joint.find("child").get("link"), joint.get("name"))
            self.assertEqual(joint.find("parent").get("link"), parents[joint.get("name")])
        children = {j.find("child").get("link") for j in joints}
        self.assertEqual(set(links) - children, {"torso"})  # one root, the importer decides fixed or floating

    def test_every_mesh_is_relative_exists_is_pinned_and_follows_the_vendor_geom(self) -> None:
        text = self.output.read_text(encoding="utf-8")
        for absolute in {str(REPO), REPO.as_posix()}:
            self.assertNotIn(absolute, text)
        sums = vendor_sums()
        vendor_mesh = {}
        for body in self.vendor.bodies:
            for geom in body["geoms"]:
                if geom.get("class") == "visual":
                    vendor_mesh[body["name"]] = self.vendor.meshes[geom["mesh"]][0]
        for link in self.root.findall("link"):
            meshes = link.findall("visual/geometry/mesh")
            self.assertEqual(len(meshes), 1, link.get("name"))
            reference = meshes[0].get("filename")
            self.assertNotIn("://", reference)
            self.assertFalse(Path(reference).is_absolute() or reference.startswith("/") or ":" in reference)
            resolved = (self.output.parent / reference).resolve()
            self.assertTrue(resolved.is_file() and resolved.stat().st_size > 0, resolved)
            key = resolved.relative_to((REPO / "vendor/tonypi").resolve()).as_posix()
            self.assertEqual(hashlib.sha256(resolved.read_bytes()).hexdigest(), sums[key])
            # r_* links use l_*.stl files (and vice versa) in the vendor model: the geom's mesh attribute rules.
            self.assertEqual(resolved, (VENDOR_MJCF.parent / vendor_mesh[link.get("name")]).resolve())
            self.assertEqual(meshes[0].get("scale"), "0.001 0.001 0.001")
        self.assertTrue(self.root.find("link[@name='r_hip_roll']/visual/geometry/mesh")
                        .get("filename").endswith("/l_hip_roll.stl"))

    def test_collision_primitives_cover_every_vendor_collider(self) -> None:
        shapes = [c.find("geometry")[0].tag for c in self.root.iter("collision")]
        record = self.candidate_record()
        caps = record["counts"]["vendor_capsules"]
        self.assertEqual(caps, 18)
        self.assertEqual(shapes.count("cylinder"), caps)
        self.assertEqual(shapes.count("box"), record["counts"]["vendor_boxes"])
        self.assertEqual(shapes.count("sphere"), record["counts"]["vendor_spheres"] + 2 * caps)
        self.assertEqual(len(shapes), 63)

    def test_every_number_equals_the_active_ledger_value_not_the_rounded_xml(self) -> None:
        def floats(text: str) -> list[float]:
            return [float(v) for v in text.split()]

        led = {p: float(r["value"]) for p, r in self.active.items()}
        for link in self.root.findall("link"):
            name = link.get("name")
            inertial = link.find("inertial")
            self.assertEqual(floats(inertial.find("origin").get("xyz")), [led[f"link.{name}.com.{k}"] for k in "xyz"])
            self.assertEqual(float(inertial.find("mass").get("value")), led[f"link.{name}.mass"])
            for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                self.assertEqual(float(inertial.find("inertia").get(key)), led[f"link.{name}.inertia.{key}"])
        placeholders = []
        for joint in self.root.findall("joint"):
            name, p = joint.get("name"), f"joint.{joint.get('name')}"
            origin = joint.find("origin")
            self.assertEqual(floats(origin.get("xyz")), [led[f"{p}.origin.{k}"] for k in "xyz"])
            self.assertEqual(floats(origin.get("rpy")), [led[f"{p}.origin.{k}"] for k in ("roll", "pitch", "yaw")])
            self.assertEqual(floats(joint.find("axis").get("xyz")), [led[f"{p}.axis.{k}"] for k in "xyz"])
            limit = joint.find("limit")
            for key in ("lower", "upper", "effort"):
                self.assertEqual(float(limit.get(key)), led[f"{p}.limit.{key}"])
            if f"{p}.limit.velocity" in led:
                self.assertEqual(float(limit.get("velocity")), led[f"{p}.limit.velocity"])
            else:
                self.assertEqual(float(limit.get("velocity")), 0.0)
                placeholders.append(name)
            self.assertEqual(float(joint.find("dynamics").get("damping")), led[f"{p}.dynamics.damping"])
            self.assertEqual(float(joint.find("dynamics").get("friction")), led[f"{p}.dynamics.frictionloss"])
        self.assertEqual(placeholders, self.candidate_record()["velocity_placeholders"])
        # The compiled twin XML carries ~6 significant figures; the URDF carries the ledger's full value.
        xml_torso = ET.parse(TWIN_XML).getroot().find(".//body[@name='torso']/inertial").get("pos")
        urdf_torso = self.root.find("link[@name='torso']/inertial/origin").get("xyz")
        self.assertNotEqual(floats(xml_torso), floats(urdf_torso))
        self.assertEqual(floats(urdf_torso), [led[f"link.torso.com.{k}"] for k in "xyz"])

    def test_rebuild_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory(dir=GENERATED_DIR) as directory:
            other = Path(directory) / "candidate.urdf"  # same depth below generated/: same relative mesh paths
            second = build(output=other)
            self.assertEqual(second.urdf, self.output.read_bytes())
        again = build(output=self.output)
        self.assertEqual(again.urdf, self.output.read_bytes())
        self.assertEqual(again.record, record_path_for(self.output).read_bytes())
        self.assertEqual(generate(output=self.output).urdf, again.urdf)

    def test_default_output_is_current(self) -> None:
        if not DEFAULT_OUTPUT.is_file():
            self.skipTest(f"{DEFAULT_OUTPUT} not built yet (run isaac_sim/build_tonypi_urdf.py)")
        fresh = generate(output=DEFAULT_OUTPUT)
        self.assertEqual(DEFAULT_OUTPUT.read_bytes(), fresh.urdf,
                         "stale generated URDF: run isaac_sim/build_tonypi_urdf.py")
        self.assertEqual(record_path_for(DEFAULT_OUTPUT).read_bytes(), fresh.record, "stale build record")

    # ----------------------------------------------------------------------------------- equivalence

    def test_mujoco_equivalence_with_the_binary_twin(self) -> None:
        home = {k: float(self.active[f"scene.home.root.{k}"]["value"]) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")}
        targets = [self.output] + ([DEFAULT_OUTPUT] if DEFAULT_OUTPUT.is_file() else [])
        for urdf in targets:
            with self.subTest(urdf=urdf.relative_to(REPO).as_posix()):
                result = equivalence(urdf, self.tmp, home)
                errors = result["errors"]
                print(f"\nMuJoCo equivalence: {urdf.relative_to(REPO).as_posix()} vs "
                      f"{MJB.relative_to(REPO).as_posix()} ({result['links']} links, {result['joints']} joints, "
                      f"{result['configs']} configurations incl. zero pose, compared in the torso frame)")
                for key, value in errors.items():
                    print(f"  max {key:<24} {value:.3e}")
                counts = result["collision_counts"]
                print(f"  collision primitives matched: twin {counts['twin_primitives']} -> URDF "
                      f"{counts['urdf_primitives']}; effort mapped by MuJoCo's URDF parser: "
                      f"{result['effort']['mapped_by_parser']} (max error {result['effort']['max_abs_error']:.3e})")
                arm = result["armature_twin_vs_urdf"]
                print(f"  armature (not representable in URDF): twin {sorted({v[0] for v in arm.values()})}, "
                      f"URDF-in-MuJoCo {sorted({v[1] for v in arm.values()})}")
                for key in ("link_pos_m", "com_local_m", "com_world_m", "joint_anchor_m", "joint_pos_local_m",
                            "mesh_vertex_m", "collision_pos_m", "collision_size_m", "torso_vs_ledger_home_m"):
                    self.assertLessEqual(errors[key], TOL_POS, key)
                for key in ("link_rot", "joint_axis_local", "joint_axis_world", "collision_rot", "joint_range_rad"):
                    self.assertLessEqual(errors[key], TOL_ROT, key)
                for key in ("mass_rel", "inertia_local_rel", "inertia_world_rel"):
                    self.assertLessEqual(errors[key], TOL_REL, key)
                self.assertLessEqual(errors["total_mass_kg"], 1e-12)
                self.assertLessEqual(errors["damping"], 1e-12)
                self.assertLessEqual(errors["frictionloss"], 1e-12)
                self.assertTrue(all(v[1] == 0.0 for v in arm.values()))
                if result["effort"]["mapped_by_parser"]:
                    self.assertLessEqual(result["effort"]["max_abs_error"], 1e-12)

    def test_equivalence_detects_a_one_micron_frame_change_and_a_mass_change(self) -> None:
        """Negative control: the check is not vacuous. A 1 um joint offset or a 1 ppm mass change is caught."""
        home = {k: float(self.active[f"scene.home.root.{k}"]["value"]) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")}
        for param, delta, key, limit in (("joint.r_knee.origin.z", 1e-6, "link_pos_m", TOL_POS),
                                         ("link.l_elbow.mass", 1e-6 * 0.076, "mass_rel", TOL_REL)):
            with self.subTest(param=param):
                rows = [dict(r) for r in self.rows]
                row = next(r for r in rows if r["param"] == param and r["status"] == "active")
                row["status"] = "deprecated"
                rows.append(dict(row, value=repr(float(row["value"]) + delta), source="estimate", status="active",
                                 evidence="negative control", note="negative control"))
                with tempfile.TemporaryDirectory(dir=GENERATED_DIR) as directory:
                    root = Path(directory)
                    out = root / "perturbed.urdf"
                    build(ledger=write_ledger(root, self.fieldnames, rows), output=out)
                    errors = equivalence(out, root, home)["errors"]
                self.assertGreater(errors[key], 100 * limit, f"{param} change went undetected: {errors[key]}")

    # ------------------------------------------------------------------------------------ refusals

    def test_ledger_problem_stops_before_writing(self) -> None:
        rows = [dict(r) for r in self.rows]
        next(r for r in rows if r["param"] == "link.head_pan.mass" and r["status"] == "active")["value"] = "nan"
        self.assert_refused(rows, "ledger has .* problem")

    def test_missing_active_mass_stops_before_writing(self) -> None:
        rows = [dict(r) for r in self.rows]
        next(r for r in rows if r["param"] == "link.head_pan.mass" and r["status"] == "active")["status"] = "deprecated"
        self.assert_refused(rows, "missing active ledger value: link.head_pan.mass")

    def test_invalid_inertia_is_rejected(self) -> None:
        rows = [dict(r) for r in self.rows]
        row = next(r for r in rows if r["param"] == "link.head_pan.inertia.ixx" and r["status"] == "active")
        row["status"] = "deprecated"
        rows.append(dict(row, value="-1", source="estimate", status="active", evidence="test", note="test"))
        self.assert_refused(rows, "not positive definite")

    def test_velocity_placeholder_and_ledger_value(self) -> None:
        joint = "head_pan"
        param = f"joint.{joint}.limit.velocity"
        base = [dict(r) for r in self.rows if r["param"] != param]
        placeholder = base + [dict(param=param, value="", unit="rad/s", source="", status="missing",
                                   date="2026-11-19", evidence="", note="test")]
        estimate = base + [dict(param=param, value="7.5", unit="rad/s", source="estimate", status="active",
                                date="2026-11-19", evidence="test", note="test")]
        with tempfile.TemporaryDirectory(dir=GENERATED_DIR) as directory:
            root = Path(directory)
            out = root / "candidate.urdf"
            first = build(ledger=write_ledger(root, self.fieldnames, placeholder), output=out)
            limit = ET.parse(out).getroot().find(f"joint[@name='{joint}']/limit")
            self.assertEqual(float(limit.get("velocity")), 0.0)
            self.assertIn(joint, __import__("json").loads(first.record)["velocity_placeholders"])
            self.assertIn(f"placeholder, ledger {param} is missing", out.read_text(encoding="utf-8"))
            second = build(ledger=write_ledger(root, self.fieldnames, estimate), output=out)
            limit = ET.parse(out).getroot().find(f"joint[@name='{joint}']/limit")
            self.assertEqual(float(limit.get("velocity")), 7.5)
            self.assertNotIn(joint, __import__("json").loads(second.record)["velocity_placeholders"])
            self.assert_refused(base, "neither an active nor a missing row")
            zero = base + [dict(param=param, value="0", unit="rad/s", source="estimate", status="active",
                                date="2026-11-19", evidence="test", note="test")]
            self.assert_refused(zero, "must be positive")

    def test_outputs_stay_in_generated_and_never_overwrite_evidence(self) -> None:
        for evidence in (REPO / "SOURCE_MANIFEST.json", LEDGER, VENDOR_MJCF, MJB):
            with self.subTest(evidence=evidence.name):
                original = evidence.read_bytes()
                with self.assertRaisesRegex(BuildError, "output must be a .urdf below"):
                    build(output=evidence)
                self.assertEqual(evidence.read_bytes(), original)
        with self.assertRaisesRegex(BuildError, "output must be a .urdf below"):
            build(output=GENERATED_DIR / ".." / "escape.urdf")

    def test_build_record_binds_the_urdf_for_source_integrity(self) -> None:
        """isaac_sim/source_integrity.py (stdlib only, runs inside Isaac Sim) accepts the record, refuses edits."""
        from source_integrity import IntegrityError, verify_urdf_record

        with tempfile.TemporaryDirectory(dir=GENERATED_DIR) as directory:
            out = Path(directory) / "candidate.urdf"
            build(output=out)
            record, _ = verify_urdf_record(out)
            self.assertEqual(record["joints"], self.vendor.joints)
            data = bytearray(out.read_bytes())
            data[data.index(b'effort="1.667"') + 8] ^= 1  # one changed digit in a torque cap
            out.write_bytes(bytes(data))
            with self.assertRaisesRegex(IntegrityError, "differs from its build record"):
                verify_urdf_record(out)
            record_path_for(out).unlink()
            with self.assertRaisesRegex(IntegrityError, "missing or unreadable URDF build record"):
                verify_urdf_record(out)

    def test_vendor_input_must_be_the_pinned_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=GENERATED_DIR) as directory:
            copy = Path(directory) / "robot.xml"
            copy.write_bytes(VENDOR_MJCF.read_bytes())
            with self.assertRaisesRegex(BuildError, "vendor input must live under"):
                generate(output=Path(directory) / "candidate.urdf", vendor=copy)

    def test_unsupported_vendor_geometry_is_refused(self) -> None:
        defaults = self.vendor.defaults
        cases = [
            ({"name": "g", "class": "selfcol", "type": "cylinder", "size": "0.01 0.02"}, "not supported"),
            ({"name": "g", "class": "selfcol", "type": "capsule", "size": "0.01 0.02"}, "without fromto"),
            ({"name": "g", "class": "selfcol", "type": "box", "size": "0.1 0.1 0.1", "euler": "0 0 1"},
             "euler=... is not supported"),
            ({"name": "g", "class": "selfcol", "type": "box", "size": "0.1 0.1"}, "expected 3 values"),
            ({"name": "g", "class": "selfcol", "type": "capsule", "size": "0.01",
              "fromto": "0 0 0 0 0 0"}, "positive length"),
        ]
        for geom, message in cases:
            with self.subTest(geom=geom):
                with self.assertRaisesRegex(BuildError, message):
                    collision_elements(geom, defaults, "test")

    def test_capsule_and_rotation_helpers(self) -> None:
        rng = np.random.default_rng(SEED)
        for _ in range(50):
            quat = rng.normal(size=4)
            roll, pitch, yaw = quat_to_rpy(list(quat), "test")
            rpy_matrix = (rot("z", yaw) @ rot("y", pitch) @ rot("x", roll))
            self.assertLess(np.abs(rpy_matrix - quat_mat(quat / np.linalg.norm(quat))).max(), 1e-12)
            p1, p2 = rng.normal(size=3), rng.normal(size=3)
            centre, (r, p, y), length, a, b = capsule_parts(list(p1) + list(p2), 0.01, "test")
            z = (rot("z", y) @ rot("y", p) @ rot("x", r))[:, 2]
            direction = (p2 - p1) / np.linalg.norm(p2 - p1)
            self.assertLess(np.linalg.norm(np.cross(z, direction)), 1e-12)
            self.assertAlmostEqual(length, float(np.linalg.norm(p2 - p1)), places=12)
            self.assertLess(np.abs(np.asarray(centre) - (p1 + p2) / 2).max(), 1e-15)

    # ------------------------------------------------------------------------------------- helpers

    def candidate_record(self) -> dict:
        return __import__("json").loads(self.candidate.record)

    def assert_refused(self, rows: list[dict[str, str]], message: str) -> None:
        with tempfile.TemporaryDirectory(dir=GENERATED_DIR) as directory:
            root = Path(directory)
            out = root / "refused.urdf"
            with self.assertRaisesRegex(BuildError, message):
                build(ledger=write_ledger(root, self.fieldnames, rows), output=out)
            self.assertFalse(out.exists())
            self.assertFalse(record_path_for(out).exists())


def rot(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
