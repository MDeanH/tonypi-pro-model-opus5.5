"""Read the vendor TonyPi MJCF: its literal numbers, its same-line comments, and what MuJoCo compiles from it.

Usage: .venv/bin/python -I tools/inspect_mjcf.py [vendor/tonypi/model/robot.xml]

Read-only on the vendor file. The upstream XML has comments that contain "--", which strict XML parsers such as
ElementTree reject, so comments are removed in memory before parsing; the file on disk is never changed.

Two kinds of number come out of this module:
  literal   - copied character-for-character from an attribute in the file (frames, axes, ranges, gains, ...)
  compiled  - computed by MuJoCo from the file (link mass, centre of mass and inertia: the vendor gives no
              <inertial> elements, so MuJoCo infers them from the explicit per-geom masses and primitive shapes)
"""
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, "vendor", "tonypi", "model", "robot.xml")
COMMENT = re.compile(r"<!--.*?-->", re.S)
TRAILING = re.compile(r'<(\w+)\b[^>]*\bname="([^"]+)"[^>]*/?>\s*<!--(.*?)-->')
INERTIA_KEYS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")


def strip_comments(text):
    return COMMENT.sub("", text)


def trailing_comments(text):
    """{(tag, name): comment} for comments that sit on the same line as a named element (upstream qualifiers such
    as 'servo 5', 'orientation MEASURED', 'orientation ESTIMATE')."""
    out = {}
    for line in text.splitlines():
        m = TRAILING.search(line)
        if m:
            out[(m.group(1), m.group(2))] = " ".join(m.group(3).split())
    return out


class Vendor:
    """Literal structure of the vendor MJCF, in file order."""

    def __init__(self, path=VENDOR):
        self.path = os.path.abspath(path)
        with open(self.path, encoding="utf-8") as f:
            self.text = f.read()
        self.comments = trailing_comments(self.text)
        root = ET.fromstring(strip_comments(self.text))
        self.model_name = root.get("model")
        self.compiler = dict(root.find("compiler").attrib)
        self.option = dict(root.find("option").attrib)
        self.defaults = {}
        for d in root.find("default").findall("default"):
            cls = d.get("class")
            self.defaults[cls] = {child.tag: dict(child.attrib) for child in d}
        self.meshes = {m.get("name"): m.get("file") for m in root.find("asset").findall("mesh")}
        self.bodies, self.joints = [], []

        def walk(el, parent):
            for b in el.findall("body"):
                body = dict(name=b.get("name"), parent=parent, pos=b.get("pos", "0 0 0").split(),
                            quat=b.get("quat"), euler=b.get("euler"),
                            meshes=[g.get("mesh") for g in b.findall("geom") if g.get("type") == "mesh"],
                            geoms=[dict(g.attrib) for g in b.findall("geom")], joint=None, free=False)
                if b.find("freejoint") is not None:
                    body["free"] = True
                for j in b.findall("joint"):
                    cls = j.get("class")
                    joint = dict(name=j.get("name"), body=body["name"], cls=cls, axis=j.get("axis").split(),
                                 range=j.get("range").split())
                    body["joint"] = joint["name"]
                    self.joints.append(joint)
                self.bodies.append(body)
                walk(b, body["name"])

        walk(root.find("worldbody"), "world")
        self.actuators = []
        for a in root.find("actuator"):
            self.actuators.append(dict(tag=a.tag, name=a.get("name"), joint=a.get("joint"), cls=a.get("class"),
                                       kp=a.get("kp"), ctrlrange=a.get("ctrlrange").split(),
                                       comment=self.comments.get((a.tag, a.get("name")), "")))
        self.keys = {k.get("name"): dict(qpos=k.get("qpos", "").split(), ctrl=k.get("ctrl", "").split())
                     for k in root.find("keyframe").findall("key")}
        self.sensors = [(s.tag, s.get("name")) for s in root.find("sensor")]

    def body(self, name):
        return next(b for b in self.bodies if b["name"] == name)

    def joint(self, name):
        return next(j for j in self.joints if j["name"] == name)

    def joint_default(self, cls, attr):
        return self.defaults[cls]["joint"][attr]

    def position_default(self, cls, attr):
        return self.defaults[cls]["position"][attr]

    def kp(self, actuator):
        """Literal kp string: the actuator's own override, else its class default."""
        return actuator["kp"] if actuator["kp"] is not None else self.position_default(actuator["cls"], "kp")


def compiled_model(path=VENDOR):
    import mujoco
    return mujoco.MjModel.from_xml_path(os.path.abspath(path))


def compiled_inertials(path=VENDOR):
    """{body: (mass, com[3], full inertia [ixx, iyy, izz, ixy, ixz, iyz] about the COM in body axes)}."""
    import mujoco
    import numpy as np
    m = compiled_model(path)
    out = {}
    for i in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_iquat[i])
        R = R.reshape(3, 3)
        full = R @ np.diag(m.body_inertia[i]) @ R.T
        out[name] = (float(m.body_mass[i]), [float(x) for x in m.body_ipos[i]],
                     [float(full[0, 0]), float(full[1, 1]), float(full[2, 2]), float(full[0, 1]),
                      float(full[0, 2]), float(full[1, 2])])
    return out


def zero_pose(path=VENDOR, key="home"):
    """World pose at the vendor `home` keyframe (all joints 0): {body: (R 3x3, origin xyz)} plus the model/data."""
    import mujoco
    import numpy as np
    m = compiled_model(path)
    d = mujoco.MjData(m)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    mujoco.mj_resetDataKeyframe(m, d, k)
    mujoco.mj_forward(m, d)
    poses = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): (d.xmat[i].reshape(3, 3).copy(), d.xpos[i].copy())
             for i in range(1, m.nbody)}
    return poses, m, d


def mesh_extents(m, d, group=2):
    """World-space bounds of every visual mesh geom (group 2 in the vendor file): (lo[3], hi[3])."""
    import mujoco
    import numpy as np
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != group:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo, hi


def geom_box_world(m, d, name):
    """Centre and half-sizes of a box geom in world axes (axis-aligned at the zero pose for the vendor soles)."""
    import mujoco
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    return d.geom_xpos[g].copy(), m.geom_size[g].copy(), d.geom_xmat[g].reshape(3, 3).copy()


def summary(path=VENDOR):
    v = Vendor(path)
    lines = [f"{os.path.relpath(path, REPO)}: model {v.model_name!r}, {len(v.bodies)} bodies, {len(v.joints)} hinge "
             f"joints, {len(v.actuators)} actuators, {len(v.sensors)} sensors, keyframes {list(v.keys)}"]
    for j in v.joints:
        a = next(x for x in v.actuators if x["joint"] == j["name"])
        lines.append(f"  {j['name']:<17} class {j['cls']:<10} axis {' '.join(j['axis']):<7} "
                     f"range [{j['range'][0]:>7}, {j['range'][1]:>7}] kp {v.kp(a):>3}  {a['comment']}")
    inert = compiled_inertials(path)
    total = sum(x[0] for x in inert.values())
    lines.append(f"compiled total mass {total:.6f} kg over {len(inert)} links")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(sys.argv[1] if len(sys.argv) > 1 else VENDOR))
