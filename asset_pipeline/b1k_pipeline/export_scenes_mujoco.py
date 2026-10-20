"""Export every scene as MuJoCo base-scene parts plus composed scenes.

A scene's objects are split into parts by OmniGibson's loading rule (see
mujoco_scene_utils.part_for_object):

* `shell`: walls, ceilings, fences, roofs, background and doors, which load
  whatever the rooms, plus the global cameras;
* one part per distinct room set of the other objects (in practice one per
  room, plus a few objects that span rooms), with that room's cameras.

Each part is a scene MJCF of its own: catalog objects are attached from their
archives (export_objects_mujoco.py) and structure is inlined. Parts compose with
attach prefix `{part}__`, so an object N in part P has root body `P__N_root`
and contents `P__N_*` in any scene built from them. The composed
`{scene}_best.xml` attaches every part, after a skybox; task scenes
(export_tasks_mujoco.py) attach a subset.

Poses, scales and fixedness come from the scene's settled `{scene}_stable.json`
when the task-instance archive has one (the state task templates were built
from), and from the scene's `_best` URDF otherwise. Cameras come from the URDF.
The `_with_clutter` variant isn't exported.

Output layout, in scenes_mujoco.zip:

    skybox.png
    scenes/{scene}/mjcf/{scene}_best.xml
    scenes/{scene}/mjcf/{scene}_parts.json
    scenes/{scene}/mjcf/parts/{part}/model.xml
    scenes/{scene}/mjcf/parts/{part}/structure/{category}_{model}/...
"""

import argparse
import io
import json
import os
import shutil
import traceback
import xml.etree.ElementTree as ET
import zipfile
from concurrent import futures
from xml.dom import minidom

import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils
from b1k_pipeline import mujoco_common
from b1k_pipeline import mujoco_scene_utils

PARALLELS = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "parallels"
DEFAULT_INSTANCES_ZIP = b1k_pipeline.utils.PIPELINE_ROOT / "2026-challenge-task-instances.zip"

# Bright midday sky skybox, shipped at the root of the archive and shared by every
# scene. Derived from the CC0 Poly Haven HDRI "Kloofendal 48d Partly Cloudy (Pure
# Sky)" (https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky), resampled
# into a cube map laid out as a 3x4 grid. The faces are authored in the Y-up frame
# MuJoCo expects for skyboxes (it draws them rotated 90 degrees about +X).
SKYBOX_SOURCE = b1k_pipeline.utils.PIPELINE_ROOT / "b1k_pipeline" / "assets" / "skybox.png"
SKYBOX_FILENAME = "skybox.png"
SKYBOX_GRIDSIZE = "3 4"
SKYBOX_GRIDLAYOUT = ".U..LFRB.D.."

SCENE_MANIFEST_FORMAT = "b1k_mujoco_scene/1"

# Tolerance when cross-checking stable.json against the scene URDF.
SCALE_CHECK_TOLERANCE = 1e-3


def _fmt(values):
    return mujoco_common.fmt(values)


def scene_dir(scene):
    return f"scenes/{scene}/mjcf"


def part_dir(scene, part):
    return f"{scene_dir(scene)}/parts/{part}"


def compiler_xml(root):
    g = mujoco_common.COLLISION_GROUP
    return ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true", "inertiagrouprange": f"{g} {g}"})


def add_skybox(asset_xml, from_dir):
    ET.SubElement(
        asset_xml,
        "texture",
        {
            "name": "skybox",
            "type": "skybox",
            "file": mujoco_scene_utils.relative_path(from_dir, SKYBOX_FILENAME),
            "gridsize": SKYBOX_GRIDSIZE,
            "gridlayout": SKYBOX_GRIDLAYOUT,
        },
    )


def _rooms_list(rooms):
    if rooms is None:
        return []
    if isinstance(rooms, str):
        return [r for r in rooms.replace(",", " ").split() if r]
    return list(rooms)


def load_stable_objects(instances_zip, scene):
    """Objects of a scene's settled state, or None if the archive has none.

    Returns:
        {name: record}, where a record has category, model, pos, quat (wxyz,
        base link), scale, fixed, visual_only and rooms.
    """
    if instances_zip is None or not os.path.exists(instances_zip):
        return None
    member = f"scenes/{scene}/json/{scene}_stable.json"
    with zipfile.ZipFile(instances_zip) as zf:
        if member not in zf.namelist():
            return None
        state = json.loads(zf.read(member))
    registry = state["state"]["registry"]["object_registry"]
    objects = {}
    for name, info in state["objects_info"]["init_info"].items():
        if "Robot" in info["class_name"] or info["class_name"] in ("R1Pro",):
            continue
        args = info["args"]
        root_link = registry[name]["root_link"]
        objects[name] = {
            "category": args["category"],
            "model": args["model"],
            "pos": np.asarray(root_link["pos"], dtype=np.float64),
            "quat": mujoco_common.xyzw_to_wxyz(root_link["ori"]),
            "scale": np.asarray(args.get("scale") or [1, 1, 1], dtype=np.float64),
            "fixed": bool(args.get("fixed_base")),
            "visual_only": bool(args.get("visual_only")),
            "rooms": _rooms_list(args.get("in_rooms")),
            "joint_pos": list(registry[name].get("joint_pos") or []),
        }
    return objects


def load_urdf_scene(scenes_fs, objects_fs, scene, suffix, metadata_cache):
    """Objects and cameras of a scene URDF.

    Returns:
        ({name: record}, [camera]), records as in load_stable_objects, with
        the base link pose and scale derived from the URDF's bounding boxes
        the way DatasetObject does.
    """
    with scenes_fs.open(f"scenes/{scene}/urdf/{scene}_{suffix}.urdf", "r") as f:
        scene_urdf = ET.parse(io.StringIO(f.read())).getroot()
    joints_by_child = {joint.find("child").attrib["link"]: joint for joint in scene_urdf.findall("joint")}

    objects = {}
    for link in scene_urdf.findall("link"):
        if link.attrib.get("name") == "world":
            continue
        category = link.attrib["category"]
        model = link.attrib["model"]
        urdf_name = link.attrib["name"]
        bbox_size = np.fromstring(link.attrib["bounding_box"], sep=" ")
        joint = joints_by_child[urdf_name]
        origin = joint.find("origin")
        bbox_center = np.fromstring(origin.attrib["xyz"], sep=" ")
        bbox_rot = R.from_euler("xyz", np.fromstring(origin.attrib["rpy"], sep=" "))

        if (category, model) not in metadata_cache:
            path = f"objects/{category}/{model}/misc/metadata.json"
            assert objects_fs.exists(path), f"No metadata found for object {category}/{model}"
            with objects_fs.open(path, "r") as f:
                metadata_cache[(category, model)] = json.load(f)
        metadata = metadata_cache[(category, model)]
        native_bbox = np.array(metadata["bbox_size"], dtype=np.float64)
        base_link_offset = np.array(metadata["base_link_offset"], dtype=np.float64)
        # The scene specifies the desired bounding box; the scale is relative to
        # the object's native bbox, as in DatasetObject.
        scale = np.ones(3)
        valid = native_bbox > 1e-4
        scale[valid] = bbox_size[valid] / native_bbox[valid]
        # The URDF places the bbox center; MJCF places the base link
        # (DatasetObject.set_bbox_center_position_orientation).
        base_link_pos = bbox_center - bbox_rot.apply(scale * base_link_offset)
        objects[mujoco_common.object_name(urdf_name)] = {
            "category": category,
            "model": model,
            "pos": base_link_pos,
            "quat": mujoco_common.xyzw_to_wxyz(bbox_rot.as_quat()),
            "scale": scale,
            "fixed": joint.attrib["type"] == "fixed",
            "visual_only": False,
            "rooms": _rooms_list(link.attrib.get("rooms", "")),
        }

    cameras = []
    for camera in scene_urdf.findall("camera"):
        cameras.append(
            {
                "name": f"camera_{camera.attrib['name']}",
                "pos": np.fromstring(camera.attrib["xyz"], sep=" "),
                "quat": mujoco_common.xyzw_to_wxyz(np.fromstring(camera.attrib["quat"], sep=" ")),
                "rooms": _rooms_list(camera.attrib.get("rooms", "")),
            }
        )
    return objects, cameras


class PartBuilder:
    """One base-scene part being assembled."""

    def __init__(self, scene, part, texture_max_size, object_metadata):
        self.scene = scene
        self.part = part
        self.dir = part_dir(scene, part)
        self.root = ET.Element("mujoco", {"model": f"{scene}__{part}"})
        compiler_xml(self.root)
        self.asset_xml = ET.SubElement(self.root, "asset")
        self.worldbody_xml = ET.SubElement(self.root, "worldbody")
        self.custom_xml = ET.SubElement(self.root, "custom")
        self.inliner = mujoco_scene_utils.StructureInliner(
            part, self.asset_xml, self.custom_xml, texture_max_size=texture_max_size
        )
        self.object_metadata = object_metadata
        self.declared_models = set()
        self.objects = []
        self.inlined = []
        self.cameras = []

    def add_catalog_object(self, name, record):
        mujoco_common.check_scalable(
            name, self.object_metadata(record["category"], record["model"]), record["scale"]
        )
        asset = mujoco_common.model_asset_name(record["category"], record["model"])
        if asset not in self.declared_models:
            archive = mujoco_common.object_archive_path(record["category"], record["model"])
            ET.SubElement(
                self.asset_xml,
                "model",
                {"name": asset, "file": mujoco_scene_utils.relative_path(self.dir, archive)},
            )
            self.declared_models.add(asset)
        mujoco_scene_utils.add_object(
            self.worldbody_xml, name, asset, record["pos"], record["quat"], record["fixed"], record["scale"]
        )
        self.objects.append(name)

    def add_inlined_object(self, name, record, archive):
        if not record["fixed"]:
            raise ValueError(f"Inlined object {name} must be fixed")
        root = self.inliner.add(
            self.worldbody_xml,
            name,
            record["category"],
            record["model"],
            archive,
            record["pos"],
            record["quat"],
            record["scale"],
        )
        if record["visual_only"]:
            for geom in root.iter("geom"):
                geom.attrib["contype"] = "0"
                geom.attrib["conaffinity"] = "0"
        self.inlined.append(name)

    def add_rooms_text(self, name, rooms):
        if rooms:
            ET.SubElement(self.custom_xml, "text", {"name": f"rooms::{name}", "data": ",".join(rooms)})

    def add_camera(self, camera):
        ET.SubElement(
            self.worldbody_xml,
            "camera",
            {"name": camera["name"], "pos": _fmt(camera["pos"]), "quat": _fmt(camera["quat"])},
        )
        self.add_rooms_text(camera["name"], camera["rooms"])
        self.cameras.append(camera["name"])

    def write(self, out_root):
        if self.inliner.excludes:
            contact_xml = ET.Element("contact")
            for body1, body2 in self.inliner.excludes:
                ET.SubElement(contact_xml, "exclude", {"body1": body1, "body2": body2})
            # MJCF sections may come in any order; keep custom last.
            self.root.insert(list(self.root).index(self.custom_xml), contact_xml)
        if len(self.custom_xml) == 0:
            self.root.remove(self.custom_xml)
        out = os.path.join(out_root, self.dir)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, mujoco_common.ARCHIVE_ROOT_FILE), "w") as f:
            f.write(minidom.parseString(ET.tostring(self.root)).toprettyxml(indent="  "))
        for rel, data in self.inliner.files.items():
            path = os.path.join(out, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)


def write_composed_scene(out_root, scene, model_name, parts):
    """A scene that attaches the given parts, with the skybox."""
    root = ET.Element("mujoco", {"model": model_name})
    compiler_xml(root)
    asset_xml = ET.SubElement(root, "asset")
    add_skybox(asset_xml, scene_dir(scene))
    worldbody_xml = ET.SubElement(root, "worldbody")
    for part in parts:
        ET.SubElement(
            asset_xml,
            "model",
            {
                "name": f"part-{part}",
                "file": mujoco_scene_utils.relative_path(
                    scene_dir(scene), f"{part_dir(scene, part)}/{mujoco_common.ARCHIVE_ROOT_FILE}"
                ),
            },
        )
        ET.SubElement(
            worldbody_xml,
            "attach",
            {"model": f"part-{part}", "prefix": mujoco_common.part_attach_prefix(part)},
        )
    out = os.path.join(out_root, scene_dir(scene), f"{model_name}.xml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="  "))


def _object_mjz(objects_mujoco_zf, category, model):
    member = mujoco_common.object_archive_path(category, model)
    return io.BytesIO(objects_mujoco_zf.read(member))


def _inlined(record):
    return record["category"] in mujoco_common.STRUCTURE_CATEGORIES or record["visual_only"]


def process_scene(scene, out_root, instances_zip, texture_max_size, objects_mujoco_zip):
    """Export one scene's parts, composed scenes and manifest.

    Returns:
        The scene's issues (warnings), as a list of strings.
    """
    issues = []
    metadata_cache = {}
    scenes_zip = PARALLELS / "scenes.zip"
    with b1k_pipeline.utils.ParallelZipFS("objects.zip") as objects_fs, zipfile.ZipFile(
        objects_mujoco_zip
    ) as objects_mujoco_zf:
        urdf = {}
        cameras = []
        if scenes_zip.exists():
            with b1k_pipeline.utils.ParallelZipFS("scenes.zip") as scenes_fs:
                urdf["best"], cameras = load_urdf_scene(scenes_fs, objects_fs, scene, "best", metadata_cache)
        else:
            issues.append("scenes.zip is missing: no cameras")

        best = load_stable_objects(instances_zip, scene)
        pose_source = "stable.json"
        if best is None:
            if "best" not in urdf:
                raise ValueError(f"{scene}: neither a stable.json nor a scene URDF")
            best = urdf["best"]
            pose_source = "urdf"
        elif "best" in urdf:
            # Cross-check the settled state against the URDF. The settled state
            # is what task templates were built from, so it wins.
            only_urdf = sorted(set(urdf["best"]) - set(best))
            only_stable = sorted(set(best) - set(urdf["best"]))
            if only_urdf:
                issues.append(f"objects in the URDF but not stable.json, dropped: {only_urdf}")
            if only_stable:
                issues.append(f"objects in stable.json but not the URDF: {only_stable}")
            for name, record in best.items():
                if name not in urdf["best"]:
                    continue
                ref = urdf["best"][name]
                if (record["category"], record["model"]) != (ref["category"], ref["model"]):
                    raise ValueError(f"{scene}: {name} has a different model in stable.json")
                if np.max(np.abs(record["scale"] - ref["scale"])) > SCALE_CHECK_TOLERANCE:
                    issues.append(f"{name}: stable.json scale {record['scale']} vs URDF {ref['scale']}")

        # Parts start with every joint at 0; settled joint positions can only be
        # restored by keyframes (task scenes do).
        for name, record in sorted(best.items()):
            joint_pos = record.get("joint_pos") or []
            if joint_pos and max(abs(x) for x in joint_pos) > 1e-3:
                issues.append(f"{name}: settled joint positions {joint_pos} are 0 at qpos0")

        parts = {}
        metadata_by_model = {}

        def object_metadata(category, model):
            """b1k_metadata of an object archive."""
            if (category, model) not in metadata_by_model:
                with zipfile.ZipFile(_object_mjz(objects_mujoco_zf, category, model)) as zf:
                    root = ET.fromstring(zf.read(mujoco_common.ARCHIVE_ROOT_FILE))
                text = root.find(f"custom/text[@name='{mujoco_common.METADATA_TEXT}']")
                metadata_by_model[(category, model)] = json.loads(text.attrib["data"])
            return metadata_by_model[(category, model)]

        def part_builder(part):
            if part not in parts:
                parts[part] = PartBuilder(scene, part, texture_max_size, object_metadata)
            return parts[part]

        manifest_objects = {}
        for name in sorted(best):
            record = best[name]
            part = mujoco_scene_utils.part_for_object(record["category"], record["rooms"])
            builder = part_builder(part)
            if _inlined(record):
                builder.add_inlined_object(
                    name, record, _object_mjz(objects_mujoco_zf, record["category"], record["model"])
                )
            else:
                builder.add_catalog_object(name, record)
            builder.add_rooms_text(name, record["rooms"])
            manifest_objects[name] = {
                "category": record["category"],
                "model": record["model"],
                "scale": record["scale"].tolist(),
                "fixed": record["fixed"],
                "visual_only": record["visual_only"],
                "rooms": record["rooms"],
                "part": part,
                "inlined": _inlined(record),
                "archive": None
                if _inlined(record)
                else mujoco_common.object_archive_path(record["category"], record["model"]),
            }

        for camera in cameras:
            part = (
                mujoco_scene_utils.part_for_object("camera", camera["rooms"])
                if camera["rooms"]
                else mujoco_scene_utils.SHELL_PART
            )
            part_builder(part).add_camera(camera)

        for builder in parts.values():
            builder.write(out_root)

        write_composed_scene(out_root, scene, f"{scene}_best", sorted(parts))

        manifest = {
            "format": SCENE_MANIFEST_FORMAT,
            "dataset_version": mujoco_common.DATASET_VERSION,
            "scene": scene,
            "pose_source": pose_source,
            "scenes": {"best": f"{scene}_best.xml"},
            "parts": {
                part: {
                    "file": f"parts/{part}/{mujoco_common.ARCHIVE_ROOT_FILE}",
                    "rooms": mujoco_scene_utils.part_rooms(part),
                    "objects": builder.objects,
                    "inlined": builder.inlined,
                    "cameras": builder.cameras,
                }
                for part, builder in sorted(parts.items())
            },
            "objects": manifest_objects,
            "issues": issues,
        }
        with open(os.path.join(out_root, scene_dir(scene), f"{scene}_parts.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", nargs="*", help="Only export these scenes (default: final_scenes).")
    parser.add_argument(
        "--instances_zip",
        default=str(DEFAULT_INSTANCES_ZIP),
        help="Task-instance archive providing scenes/{scene}/json/{scene}_stable.json.",
    )
    parser.add_argument(
        "--structure_texture_max_size",
        type=int,
        default=1024,
        help="Cap on the side of inlined structure textures (0: no cap).",
    )
    parser.add_argument(
        "--objects_mujoco_zip",
        default=str(PARALLELS / "objects_mujoco.zip"),
        help="Object archives to inline structure from (for debugging with a partial export).",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    targets = [t.split("/")[-1] for t in b1k_pipeline.utils.get_targets("final_scenes")]
    if args.scenes:
        targets = [t for t in targets if t in set(args.scenes)]
    out_name = "scenes_mujoco_debug.zip" if args.scenes else "scenes_mujoco.zip"

    with b1k_pipeline.utils.ParallelZipFS(out_name, write=True) as archive_fs:
        out_root = archive_fs.getsyspath("/")
        shutil.copyfile(SKYBOX_SOURCE, os.path.join(out_root, SKYBOX_FILENAME))
        errors, warnings = {}, {}
        with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            target_futures = {
                executor.submit(
                    process_scene,
                    scene,
                    out_root,
                    args.instances_zip,
                    args.structure_texture_max_size,
                    args.objects_mujoco_zip,
                ): scene
                for scene in targets
            }
            for future in tqdm.tqdm(futures.as_completed(target_futures), total=len(target_futures)):
                scene = target_futures[future]
                try:
                    issues = future.result()
                    if issues:
                        warnings[scene] = issues
                except Exception:
                    errors[scene] = traceback.format_exc()
                    traceback.print_exc()

        print("Finished processing")

    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    out_json = "export_scenes_mujoco_debug.json" if args.scenes else "export_scenes_mujoco.json"
    with pipeline_fs.pipeline_output().open(out_json, "w") as f:
        json.dump({"success": not errors, "errors": errors, "warnings": warnings}, f, indent=1)


if __name__ == "__main__":
    main()
