from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _mesh_filenames(root: ET.Element) -> list[str]:
    """Return every ``filename`` referenced by a ``<mesh>`` element, in document order."""
    filenames = []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn is not None:
            filenames.append(fn)
    return filenames


def _strip_uri_scheme(ref: str) -> str:
    """Strip a leading ``file://`` scheme if present (kept verbatim otherwise)."""
    if ref.startswith("file://"):
        return ref[len("file://") :]
    return ref


def _classify_mesh_ref(ref: str, urdf_dir: Path) -> str:
    """Classify a single mesh reference.

    Returns one of: ``"package"`` (package:// URI), ``"ok"`` (resolves on disk),
    or ``"broken"`` (absolute/relative path that does not resolve).
    """
    if ref.startswith("package://"):
        return "package"
    candidate = _strip_uri_scheme(ref)
    path = Path(candidate)
    if not path.is_absolute():
        path = urdf_dir / candidate
    try:
        exists = path.exists()
    except OSError:
        exists = False
    return "ok" if exists else "broken"


def copy_robot_to_models(name: str, data_path: PathLike, dataset_name: str = "omnigibson-robot-assets") -> Path:
    """Copy an imported robot from ``objects/robot/<name>/`` into ``models/<name>/`` (idempotent) so
    ``REGISTERED_ROBOTS`` discovers it. The ``<name>.yaml`` definition is placed there separately.
    Returns the destination ``models/<name>/`` directory.
    """
    src = Path(data_path) / dataset_name / "objects" / "robot" / name
    dst = Path(data_path) / dataset_name / "models" / name
    assert src.exists(), f"imported robot not found at {src} (run import_custom_robot.py first)"
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    return dst


def link_has_dedicated_collision(link_element: ET.Element) -> bool:
    """True if the link ships a usable collision proxy: a ``<collision>`` that is a primitive, or a
    mesh whose file differs from the link's visual meshes. Such collisions are preserved on import;
    otherwise (missing, or the same mesh as the visual) one must be generated.
    """
    visual_mesh_files = {
        mesh.get("filename") for vis in link_element.findall("visual") for mesh in vis.findall("geometry/mesh")
    }
    for col in link_element.findall("collision"):
        geom = col.find("geometry/*")
        if geom is None:
            continue
        if geom.tag != "mesh" or geom.get("filename") not in visual_mesh_files:
            return True
    return False


def override_joint_limits(tree: ET.ElementTree, limits: dict) -> int:
    """Set ``<limit>`` attributes on named joints (in place), repairing zeroed/degenerate limits that
    make a joint non-drivable. ``limits`` maps ``joint_name -> {attr: value}``. Returns the count updated.
    """
    root = tree.getroot()
    updated = 0
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name in limits:
            limit_el = joint.find("limit")
            if limit_el is None:
                limit_el = ET.SubElement(joint, "limit")
            for attr, value in limits[name].items():
                limit_el.set(attr, str(value))
            updated += 1
    return updated


def rewrite_mesh_paths(tree: ET.ElementTree, replacements) -> int:
    """Apply ordered ``(old, new)`` substring replacements to every ``<mesh filename>`` (in place),
    e.g. to turn ``package://`` URIs or broken relative paths into resolvable absolute paths.
    Returns the count of references changed.
    """
    root = tree.getroot()
    updated = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn is None:
            continue
        new = fn
        for old, repl in replacements:
            new = new.replace(old, repl)
        if new != fn:
            mesh.set("filename", new)
            updated += 1
    return updated


def strip_mimic_joints(tree: ET.ElementTree) -> int:
    """Remove every ``<mimic>`` element (in place); returns the count removed. The importer writes no
    DriveAPI on a mimic joint, leaving that finger non-drivable -- drive both fingers at runtime instead.
    """
    root = tree.getroot()
    removed = 0
    for joint in root.findall("joint"):
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
            removed += 1
    return removed


def stage_meshes(tree: ET.ElementTree, dest_dir: PathLike, invalid_chars: str = " ()", replacement: str = "_") -> int:
    """Copy every referenced mesh into ``dest_dir`` under a USD-safe basename (space/paren -> ``_``),
    rewriting the references (in place); returns the number of files copied. Avoids the importer's
    ill-formed-SdfPath failure on mesh filenames with spaces (it names geometry prims after the
    basename). Run after `rewrite_mesh_paths` so refs resolve. Identical sources are copied
    once; distinct sources colliding on a sanitized name get a numeric suffix.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    root = tree.getroot()

    def _clean_base(name):
        for ch in invalid_chars:
            name = name.replace(ch, replacement)
        return name

    src_to_dest: dict = {}  # absolute source -> staged absolute path
    used_names: dict = {}  # staged basename -> source (collision detection)
    copied = 0
    for mesh in root.iter("mesh"):
        ref = mesh.get("filename")
        if ref is None:
            continue
        src = Path(_strip_uri_scheme(ref))
        if str(src) in src_to_dest:
            mesh.set("filename", src_to_dest[str(src)])
            continue
        base = _clean_base(src.name)
        # Disambiguate distinct sources that sanitize to the same basename.
        if base in used_names and used_names[base] != str(src):
            stem, suffix = Path(base).stem, Path(base).suffix
            i = 1
            while f"{stem}_{i}{suffix}" in used_names:
                i += 1
            base = f"{stem}_{i}{suffix}"
        target = dest / base
        if src.exists():
            shutil.copy2(src, target)
            copied += 1
        used_names[base] = str(src)
        src_to_dest[str(src)] = str(target)
        mesh.set("filename", str(target))
    return copied


def sanitize_names(tree: ET.ElementTree, invalid_chars: str = " ()", replacement: str = "_") -> dict:
    """Replace ``invalid_chars`` (space/paren) with ``replacement`` in every link/joint name and all
    parent/child/mimic references (in place). The importer otherwise renames these silently, breaking
    downstream name matching. Returns the ``{old: new}`` map so callers can update their own references.
    """
    root = tree.getroot()

    def _clean(name):
        if name is None:
            return None
        out = name
        for ch in invalid_chars:
            out = out.replace(ch, replacement)
        return out

    renames: dict = {}
    for link in root.findall("link"):
        old = link.get("name")
        new = _clean(old)
        if new != old:
            link.set("name", new)
            renames[old] = new
    for joint in root.findall("joint"):
        old = joint.get("name")
        new = _clean(old)
        if new != old:
            joint.set("name", new)
            renames[old] = new
        for tag in ("parent", "child"):
            el = joint.find(tag)
            if el is not None and el.get("link") is not None:
                el.set("link", _clean(el.get("link")))
        mimic = joint.find("mimic")
        if mimic is not None and mimic.get("joint") is not None:
            mimic.set("joint", _clean(mimic.get("joint")))
    return renames


def set_joint_dynamics(tree: ET.ElementTree, joint_names, **attrs) -> int:
    """Set ``<dynamics>`` attributes (e.g. ``friction=0``) on named joints (in place); returns the count updated."""
    root = tree.getroot()
    names = set(joint_names)
    updated = 0
    for joint in root.findall("joint"):
        if joint.get("name") in names:
            dyn = joint.find("dynamics")
            if dyn is None:
                dyn = ET.SubElement(joint, "dynamics")
            for attr, value in attrs.items():
                dyn.set(attr, str(value))
            updated += 1
    return updated


# Repair for Stretch sg3 gripper fingers shipped with zeroed limits (matches the OG stretch reference).
STRETCH_FINGER_LIMITS = {
    "joint_gripper_finger_left": {"effort": 100, "lower": -0.6, "upper": 0.6, "velocity": 1.0},
    "joint_gripper_finger_right": {"effort": 100, "lower": -0.6, "upper": 0.6, "velocity": 1.0},
}

# Stretch wheel joints whose axle friction is zeroed to match the OG reference.
STRETCH_WHEEL_JOINTS = ["joint_left_wheel", "joint_right_wheel"]


def urdf_audit(urdf_path: PathLike) -> dict:
    """Return a deterministic, GPU-free fact sheet for a URDF (XML-aware, so multi-line tags count
    correctly). Non-resolving mesh paths are reported as ``broken`` (never raised); ``package://`` URIs
    counted separately. Keys: ``n_links``, ``n_joints``, ``joint_type_counts``, ``mimic_joints``,
    ``mesh_refs``, ``mesh_formats``, ``package_uris``, ``nonascii_paths``, ``broken_relative_paths``,
    ``links_without_inertial``, ``root_link``.
    """
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = list(root.findall("link"))
    joints = list(root.findall("joint"))

    joint_type_counts: dict[str, int] = {}
    mimic_joints = 0
    child_link_names: set[str] = set()
    for joint in joints:
        jtype = joint.get("type", "unknown")
        joint_type_counts[jtype] = joint_type_counts.get(jtype, 0) + 1
        if joint.find("mimic") is not None:
            mimic_joints += 1
        child_el = joint.find("child")
        if child_el is not None and child_el.get("link") is not None:
            child_link_names.add(child_el.get("link"))

    link_names = [link.get("name") for link in links if link.get("name") is not None]
    links_without_inertial = sum(1 for link in links if link.find("inertial") is None)

    # The root link is the one never referenced as a joint's child.
    root_candidates = [name for name in link_names if name not in child_link_names]
    root_link: Optional[str] = root_candidates[0] if len(root_candidates) == 1 else None

    mesh_refs = _mesh_filenames(root)
    mesh_formats: dict[str, int] = {}
    package_uris = 0
    nonascii_paths = 0
    broken_relative_paths = 0
    for ref in mesh_refs:
        ext = Path(_strip_uri_scheme(ref).split("?")[0]).suffix.lower()
        if ext:
            mesh_formats[ext] = mesh_formats.get(ext, 0) + 1
        if not _is_ascii(ref):
            nonascii_paths += 1
        kind = _classify_mesh_ref(ref, urdf_dir)
        if kind == "package":
            package_uris += 1
        elif kind == "broken":
            broken_relative_paths += 1

    return {
        "n_links": len(links),
        "n_joints": len(joints),
        "joint_type_counts": dict(sorted(joint_type_counts.items())),
        "mimic_joints": mimic_joints,
        "mesh_refs": len(mesh_refs),
        "mesh_formats": dict(sorted(mesh_formats.items())),
        "package_uris": package_uris,
        "nonascii_paths": nonascii_paths,
        "broken_relative_paths": broken_relative_paths,
        "links_without_inertial": links_without_inertial,
        "root_link": root_link,
    }
