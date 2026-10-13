"""Find the joints that keep objects from being scaled anisotropically, and decide their fixes.

Scenes scale objects with MuJoCo frame scales. A non-uniform frame scale needs
every hinge and slide joint of the object to be aligned with its base link's
axes (see export_objects_mujoco.anisotropic_scaling_problems). This script
finds every object that some scene or task instance scales non-uniformly and,
for each unaligned joint of those objects, decides:

* `snap`: the axis is within --snap_tolerance_deg of a base-link axis, so it is
  rotated onto that axis;
* `fix`: the axis is further off, which in practice means a cosmetic part
  mounted at an angle (a toilet's flush lever, a washer's dials on a slanted
  panel, a lamp's head), so the joint is made fixed at its zero position.

It writes metadata/mujoco_joint_fixes.json, which export_objects_mujoco.py
applies and which documents the changes for the dataset's authors.

Scales come from the task-instance archive (settled scenes and task
templates) and from the base scenes (`_best`; `_with_clutter` isn't exported): the scene
URDFs in scenes.zip when it's
available, or else the `scale` attributes of a previous MuJoCo scene export
given with --scene_mjcf_dir.

    python -m b1k_pipeline.find_mujoco_joint_fixes [--scene_mjcf_dir og_dataset_mujoco]
"""

import argparse
import glob
import io
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils
from b1k_pipeline import mujoco_common

PARALLELS = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "parallels"
DEFAULT_INSTANCES_ZIP = b1k_pipeline.utils.PIPELINE_ROOT / "2026-challenge-task-instances.zip"
OUTPUT = b1k_pipeline.utils.PIPELINE_ROOT / "metadata" / "mujoco_joint_fixes.json"
FORMAT = "b1k_mujoco_joint_fixes/1"
DEFAULT_SNAP_TOLERANCE_DEG = 10.0
INSTANCE_STATE_PATTERN = re.compile(r"^scenes/(?P<scene>[^/]+)/json/(?P<file>[^/]+(_template|_stable)\.json)$")
AXIS_NAMES = ("x", "y", "z")


def is_anisotropic(scale):
    return not (mujoco_common.is_unit_scale(scale) or mujoco_common.is_uniform_scale(scale))


def scales_from_instances(instances_zip):
    """(category, model) -> [use] from settled scenes and task templates."""
    uses = {}
    with zipfile.ZipFile(instances_zip) as zf:
        for member in sorted(zf.namelist()):
            match = INSTANCE_STATE_PATTERN.match(member)
            if not match:
                continue
            state = json.loads(zf.read(member))
            for name, info in state["objects_info"]["init_info"].items():
                args = info["args"]
                if "category" not in args or "model" not in args or not args.get("scale"):
                    continue
                if is_anisotropic(args["scale"]):
                    uses.setdefault((args["category"], args["model"]), []).append(
                        {
                            "source": f"{os.path.basename(str(instances_zip))}:{member}",
                            "object": name,
                            "scale": [round(float(s), 6) for s in args["scale"]],
                        }
                    )
    return uses


def scales_from_scene_urdfs(scenes_zip, objects_zf):
    """(category, model) -> [use] from the `_best` scene URDFs (bbox ratios, as DatasetObject)."""
    uses, native = {}, {}
    with zipfile.ZipFile(scenes_zip) as zf:
        for member in sorted(zf.namelist()):
            if not re.match(r"^scenes/[^/]+/urdf/[^/]+_best\.urdf$", member):
                continue
            root = ET.fromstring(zf.read(member))
            for link in root.findall("link"):
                if link.attrib.get("name") == "world":
                    continue
                key = (link.attrib["category"], link.attrib["model"])
                if key not in native:
                    md = json.loads(objects_zf.read(f"objects/{key[0]}/{key[1]}/misc/metadata.json"))
                    native[key] = np.array(md["bbox_size"], dtype=np.float64)
                bbox = np.fromstring(link.attrib["bounding_box"], sep=" ")
                scale = np.ones(3)
                valid = native[key] > 1e-4
                scale[valid] = bbox[valid] / native[key][valid]
                if is_anisotropic(scale):
                    uses.setdefault(key, []).append(
                        {"source": member, "object": link.attrib["name"], "scale": np.round(scale, 6).tolist()}
                    )
    return uses


def scales_from_scene_mjcf(dataset_dir):
    """(category, model) -> [use] from a previous MuJoCo scene export's attach scales."""
    uses = {}
    for path in sorted(glob.glob(os.path.join(dataset_dir, "scenes", "*", "mjcf", "*_best.xml"))):
        root = ET.parse(path).getroot()
        for parent in root.iter():
            attach = parent.find("attach")
            if attach is None or "scale" not in attach.attrib:
                continue
            category, model = attach.attrib["model"].rsplit("-", 1)
            scale = [float(s) for s in attach.attrib["scale"].split()]
            if is_anisotropic(scale):
                uses.setdefault((category, model), []).append(
                    {
                        "source": os.path.relpath(path, dataset_dir),
                        "object": parent.attrib.get("name"),
                        "scale": [round(s, 6) for s in scale],
                    }
                )
    return uses


def _rpy_matrix(origin):
    if origin is None or "rpy" not in origin.attrib:
        return np.eye(3)
    return R.from_euler("xyz", np.fromstring(origin.attrib["rpy"], sep=" ")).as_matrix()


def joint_alignment(urdf_root):
    """Movable joints with their axes in the base link's frame.

    Returns:
        [(joint element, child link, axis in the joint frame, axis in the base
        link frame)], in URDF order.
    """
    joints = urdf_root.findall("joint")
    by_child = {j.find("child").attrib["link"]: j for j in joints}
    rotation_cache = {}

    def link_rotation(link):
        if link not in by_child:
            return np.eye(3)
        if link not in rotation_cache:
            joint = by_child[link]
            parent = joint.find("parent").attrib["link"]
            rotation_cache[link] = link_rotation(parent) @ _rpy_matrix(joint.find("origin"))
        return rotation_cache[link]

    result = []
    for joint in joints:
        if joint.attrib["type"] not in ("revolute", "continuous", "prismatic"):
            continue
        child = joint.find("child").attrib["link"]
        axis_element = joint.find("axis")
        axis = np.fromstring(axis_element.attrib["xyz"], sep=" ") if axis_element is not None else np.array([1.0, 0, 0])
        axis = axis / np.linalg.norm(axis)
        result.append((joint, child, axis, link_rotation(child) @ axis))
    return result


def decide_fixes(urdf_root, metadata, snap_tolerance_deg):
    """Fixes for one object's unaligned joints: name -> description."""
    openable = {pair[1] for pair in metadata.get("openable_joint_ids") or []}
    fixes = {}
    for joint, child, axis, base_axis in joint_alignment(urdf_root):
        k = int(np.argmax(np.abs(base_axis)))
        off_axis_deg = float(np.degrees(np.arccos(min(1.0, abs(base_axis[k])))))
        if off_axis_deg <= mujoco_common.ALIGNMENT_TOLERANCE_DEG:
            continue
        limit = joint.find("limit")
        action = "snap" if off_axis_deg <= snap_tolerance_deg else "fix"
        snapped_base = np.zeros(3)
        snapped_base[k] = np.sign(base_axis[k])
        fixes[joint.attrib["name"]] = {
            "action": action,
            "type": joint.attrib["type"],
            "child_link": child,
            "openable": joint.attrib["name"] in openable,
            "off_axis_deg": round(off_axis_deg, 2),
            "nearest_base_axis": ("+" if snapped_base[k] > 0 else "-") + AXIS_NAMES[k],
            "axis": np.round(axis, 9).tolist(),
            "axis_in_base_frame": np.round(base_axis, 9).tolist(),
            "limit": [float(limit.attrib.get("lower", 0)), float(limit.attrib.get("upper", 0))]
            if limit is not None
            else None,
        }
    return fixes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instances_zip", default=str(DEFAULT_INSTANCES_ZIP))
    parser.add_argument(
        "--scene_mjcf_dir",
        help="A previous MuJoCo dataset export whose scene MJCFs carry attach scales; used when scenes.zip is missing.",
    )
    parser.add_argument("--snap_tolerance_deg", type=float, default=DEFAULT_SNAP_TOLERANCE_DEG)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    objects_zf = zipfile.ZipFile(PARALLELS / "objects.zip")
    sources, uses = [], {}

    def merge(new_uses, source):
        sources.append(source)
        for key, entries in new_uses.items():
            uses.setdefault(key, []).extend(entries)

    merge(scales_from_instances(args.instances_zip), os.path.basename(args.instances_zip))
    scenes_zip = PARALLELS / "scenes.zip"
    if scenes_zip.exists():
        merge(scales_from_scene_urdfs(scenes_zip, objects_zf), "scenes.zip (scene URDFs)")
    elif args.scene_mjcf_dir:
        merge(scales_from_scene_mjcf(args.scene_mjcf_dir), "previous MuJoCo scene export (attach scales)")
    else:
        raise SystemExit("scenes.zip is missing; pass --scene_mjcf_dir")

    objects = {}
    for (category, model), entries in sorted(uses.items()):
        urdf_member = f"objects/{category}/{model}/urdf/{model}.urdf"
        if urdf_member not in objects_zf.namelist():
            continue
        urdf_root = ET.fromstring(objects_zf.read(urdf_member))
        metadata = json.loads(objects_zf.read(f"objects/{category}/{model}/misc/metadata.json"))
        fixes = decide_fixes(urdf_root, metadata, args.snap_tolerance_deg)
        if not fixes:
            continue
        objects[f"{category}/{model}"] = {
            "joints": fixes,
            "num_anisotropic_uses": len(entries),
            "anisotropic_uses": entries,
        }

    summary = {
        "objects": len(objects),
        "snapped_joints": sum(1 for o in objects.values() for j in o["joints"].values() if j["action"] == "snap"),
        "fixed_joints": sum(1 for o in objects.values() for j in o["joints"].values() if j["action"] == "fix"),
        "openable_joints_affected": sorted(
            f"{key}:{name}" for key, o in objects.items() for name, j in o["joints"].items() if j["openable"]
        ),
    }
    out = {
        "format": FORMAT,
        "dataset_version": mujoco_common.DATASET_VERSION,
        "description": (
            "Joints of objects that some scene or task instance scales non-uniformly, whose axes aren't "
            "aligned with the object's base-link axes. A non-uniform scale can't stretch such a joint "
            "exactly, so the MuJoCo export snaps joints within snap_tolerance_deg of a base-link axis "
            "onto it ('snap', axis replaced by nearest_base_axis) and makes the others fixed at their "
            "zero position ('fix'). Axes are unit vectors: 'axis' in the joint frame as in the URDF, "
            "'axis_in_base_frame' in the base link's frame. Limits are in the URDF's units."
        ),
        "alignment_tolerance_deg": mujoco_common.ALIGNMENT_TOLERANCE_DEG,
        "snap_tolerance_deg": args.snap_tolerance_deg,
        "scale_sources": sources,
        "summary": summary,
        "objects": objects,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
