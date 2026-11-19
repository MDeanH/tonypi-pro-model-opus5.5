"""Compile a twin MJCF with whichever MuJoCo the current Python has.

The twin XML is written by the project venv's MuJoCo (3.17), whose writer adds input="pos" to position-input
actuator defaults. Older MuJoCo releases (the WSL Isaac Lab venv has 3.12, a dependency of Isaac Sim's MJCF
importer) reject that attribute. input="pos" is the default actuator input, so removing exactly that value is
semantically neutral; any other input value is refused. The edit happens in memory only, and only after the
unmodified file has failed to compile; mesh paths are resolved against the file's own meshdir.
"""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

COMMENT = re.compile(r"<!--.*?-->", re.S)


def load_mjcf(path: Path):
    """Return (mujoco.MjModel, list of in-memory compatibility edits)."""
    import mujoco

    try:
        return mujoco.MjModel.from_xml_path(str(path)), []
    except ValueError as exc:
        if "input is only available" not in str(exc):
            raise
        first_error = str(exc).splitlines()[0]
    root = ET.fromstring(COMMENT.sub("", path.read_text(encoding="utf-8")))
    removed = 0
    for element in root.iter():
        value = element.attrib.get("input")
        if value is None:
            continue
        if value != "pos":
            raise ValueError(f"{path}: actuator input={value!r} cannot be dropped for MuJoCo {mujoco.__version__}")
        del element.attrib["input"]
        removed += 1
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attribute in ("meshdir", "assetdir", "texturedir"):
        if compiler.get(attribute):
            compiler.set(attribute, (path.parent / compiler.get(attribute)).resolve().as_posix())
    if not compiler.get("meshdir") and not compiler.get("assetdir"):
        compiler.set("meshdir", path.parent.resolve().as_posix())
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, [f"MuJoCo {mujoco.__version__} rejected the file ({first_error}); removed input=\"pos\" "
                   f"(the default) from {removed} element(s) in memory"]
