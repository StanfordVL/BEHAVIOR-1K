import io
import json
import os
import traceback
import xml.etree.ElementTree as ET
from concurrent import futures
from xml.dom import minidom

import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils

# Relative path from scenes/{scene_name}/mjcf/ to an object's MJZ, mirroring the
# original dataset layout with the usd subdirectory replaced by mjz.
OBJECT_MJZ_PATH_TEMPLATE = "../../../objects/{category}/{model}/mjz/{model}.mjz"


def _fmt(values):
    return " ".join(f"{v:.10g}" for v in np.asarray(values, dtype=np.float64).flatten())


def _load_object_metadata(objects_fs, category, model):
    metadata_path = f"objects/{category}/{model}/misc/metadata.json"
    assert objects_fs.exists(metadata_path), f"No metadata found for object {category}/{model}"
    with objects_fs.open(metadata_path, "r") as f:
        return json.load(f)


def process_target(target, scenes_dir):
    scene_name = os.path.split(target)[-1]

    with b1k_pipeline.utils.ParallelZipFS("scenes.zip") as scenes_fs, b1k_pipeline.utils.ParallelZipFS(
        "objects.zip"
    ) as objects_fs:
        scene_urdf_dir = scenes_fs.opendir(f"scenes/{scene_name}/urdf")

        # Cache native bbox / base link offset per object model.
        metadata_cache = {}

        for suffix in ["best", "with_clutter"]:
            urdf_name = f"{scene_name}_{suffix}.urdf"
            with scene_urdf_dir.open(urdf_name, "r") as f:
                scene_urdf = ET.parse(io.StringIO(f.read())).getroot()

            # Index the scene joints (fixedness + world pose of each object's bbox center).
            joints_by_child = {}
            for joint in scene_urdf.findall("joint"):
                child = joint.find("child").attrib["link"]
                joints_by_child[child] = joint

            root = ET.Element("mujoco", {"model": f"{scene_name}_{suffix}"})
            root.append(
                ET.Comment(
                    " Scene MJCF referencing per-object MJZ archives (see export_objects_mujoco.py). "
                    "Each attach carries the per-instance scale via the scale attribute, following "
                    "the proposed MuJoCo attach-scaling feature (see mujoco-design.md). Until that "
                    "feature is available, loaders must unpack each MJZ and apply the scale (also "
                    "stored in the custom section) by transforming the child spec (mesh scales, "
                    "geom/body/joint/site positions and sizes) before attaching. Fixed (non-loose) "
                    "objects are attached statically (no freejoint); loose objects are attached "
                    "under a freejointed wrapper body. "
                )
            )
            asset_xml = ET.SubElement(root, "asset")
            worldbody_xml = ET.SubElement(root, "worldbody")
            custom_xml = ET.SubElement(root, "custom")

            declared_models = set()
            for link in scene_urdf.findall("link"):
                if link.attrib.get("name") == "world":
                    continue
                obj_category = link.attrib["category"]
                obj_model = link.attrib["model"]
                obj_name = link.attrib["name"]
                obj_rooms = link.attrib.get("rooms", "")
                bbox_size = np.fromstring(link.attrib["bounding_box"], sep=" ")

                joint = joints_by_child[obj_name]
                is_fixed = joint.attrib["type"] == "fixed"
                origin = joint.find("origin")
                bbox_center = np.fromstring(origin.attrib["xyz"], sep=" ")
                bbox_rot = R.from_euler("xyz", np.fromstring(origin.attrib["rpy"], sep=" "))

                # Scaling: the scene specifies the desired bounding box; the scale
                # is relative to the object's native bbox, as in DatasetObject.
                if (obj_category, obj_model) not in metadata_cache:
                    metadata_cache[(obj_category, obj_model)] = _load_object_metadata(
                        objects_fs, obj_category, obj_model
                    )
                obj_metadata = metadata_cache[(obj_category, obj_model)]
                native_bbox = np.array(obj_metadata["bbox_size"], dtype=np.float64)
                base_link_offset = np.array(obj_metadata["base_link_offset"], dtype=np.float64)
                scale = np.ones(3)
                valid = native_bbox > 1e-4
                scale[valid] = bbox_size[valid] / native_bbox[valid]

                # The scene URDF places the object's bbox center; MJCF frames place
                # the base link, so apply the scaled base link offset (mirrors
                # DatasetObject.set_bbox_center_position_orientation).
                base_link_pos = bbox_center - bbox_rot.apply(scale * base_link_offset)
                quat_xyzw = bbox_rot.as_quat()
                quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

                # Declare the object model asset once per (category, model).
                model_asset_name = f"{obj_category}-{obj_model}"
                if model_asset_name not in declared_models:
                    ET.SubElement(
                        asset_xml,
                        "model",
                        {
                            "name": model_asset_name,
                            "file": OBJECT_MJZ_PATH_TEMPLATE.format(category=obj_category, model=obj_model),
                        },
                    )
                    declared_models.add(model_asset_name)

                # Instantiate the object at its base link pose. The object MJZs
                # have no freejoint: fixed objects are attached statically under
                # a frame (exactly rigid, unlike a soft weld constraint), while
                # loose objects get a freejointed wrapper body.
                pose_attrs = {"name": obj_name, "pos": _fmt(base_link_pos), "quat": _fmt(quat_wxyz)}
                if is_fixed:
                    parent_xml = ET.SubElement(worldbody_xml, "frame", pose_attrs)
                else:
                    parent_xml = ET.SubElement(worldbody_xml, "body", pose_attrs)
                    ET.SubElement(parent_xml, "freejoint")
                # The scale attribute follows the proposed MuJoCo attach-scaling
                # feature (see mujoco-design.md). Current MuJoCo releases ignore
                # unknown attach attributes, so until the feature ships, loaders
                # must apply the scale themselves (also mirrored in the custom
                # section below).
                ET.SubElement(
                    parent_xml,
                    "attach",
                    {
                        "model": model_asset_name,
                        "body": "base_link",
                        "prefix": f"{obj_name}_",
                        "scale": _fmt(scale),
                    },
                )

                # Per-instance data that MJCF cannot express natively.
                ET.SubElement(
                    custom_xml,
                    "numeric",
                    {"name": f"scale::{obj_name}", "size": "3", "data": _fmt(scale)},
                )
                if obj_rooms:
                    ET.SubElement(custom_xml, "text", {"name": f"rooms::{obj_name}", "data": obj_rooms})

            # Emit the scene cameras (stored as custom <camera> elements in the
            # scene URDF by export_scenes_global.py) as native MuJoCo cameras.
            for camera in scene_urdf.findall("camera"):
                cam_name = camera.attrib["name"]
                cam_pos = np.fromstring(camera.attrib["xyz"], sep=" ")
                cam_quat_xyzw = np.fromstring(camera.attrib["quat"], sep=" ")
                cam_quat_wxyz = [
                    cam_quat_xyzw[3],
                    cam_quat_xyzw[0],
                    cam_quat_xyzw[1],
                    cam_quat_xyzw[2],
                ]
                mj_cam_name = f"camera_{cam_name}"
                ET.SubElement(
                    worldbody_xml,
                    "camera",
                    {
                        "name": mj_cam_name,
                        "pos": _fmt(cam_pos),
                        "quat": _fmt(cam_quat_wxyz),
                    },
                )
                # Room assignment (empty means the camera is global).
                cam_rooms = camera.attrib.get("rooms", "")
                if cam_rooms:
                    ET.SubElement(
                        custom_xml,
                        "text",
                        {"name": f"rooms::{mj_cam_name}", "data": cam_rooms},
                    )

            # Drop empty sections for cleanliness.
            for section in [custom_xml]:
                if len(section) == 0:
                    root.remove(section)

            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            out_dir = os.path.join(scenes_dir, scene_name, "mjcf")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{scene_name}_{suffix}.xml"), "w") as f:
                f.write(xmlstr)


def main():
    with b1k_pipeline.utils.ParallelZipFS("scenes_mujoco.zip", write=True) as archive_fs:
        scenes_dir = archive_fs.makedir("scenes").getsyspath("/")
        errors = {}
        target_futures = {}

        with futures.ProcessPoolExecutor(max_workers=16) as target_executor:
            targets = b1k_pipeline.utils.get_targets("final_scenes")
            for target in tqdm.tqdm(targets):
                target_futures[
                    target_executor.submit(process_target, target, scenes_dir)
                ] = target

            with tqdm.tqdm(total=len(target_futures)) as pbar:
                for future in futures.as_completed(target_futures.keys()):
                    try:
                        future.result()
                    except:
                        name = target_futures[future]
                        errors[name] = traceback.format_exc()

                    pbar.update(1)

        print("Finished processing")

    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    with pipeline_fs.pipeline_output().open("export_scenes_mujoco.json", "w") as f:
        json.dump({"success": not errors, "errors": errors}, f)


if __name__ == "__main__":
    main()
