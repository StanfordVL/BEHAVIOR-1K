import re
import sys
import traceback

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import collections
import glob
import json
import os
import random
import shutil
import string
from typing import Dict, List

import numpy as np
import pymxs
import tqdm
import trimesh.transformations
import yaml
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils
from b1k_pipeline.max.new_sanity_check import SanityCheck
from b1k_pipeline.urdfpy import URDF, Joint, Link

rt = pymxs.runtime

CONFIRM_EACH = True
INTERACTIVE_MODE = True
JOINT_TYPES = {"prismatic": "P", "revolute": "R", "fixed": "F", "continuous": "R"}
IN_DATASET_ROOT = r"C:\Users\Cem\research\iGibson-dev\igibson\data\ig_dataset"
OUTPUT_ROOT = r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects"
TRANSLATION_PATH = os.path.join(IN_DATASET_ROOT, "metadata", "model_rename.yaml")
with open(TRANSLATION_PATH, "r") as f:
    TRANSLATION_DICT = yaml.load(f, Loader=yaml.SafeLoader)
RENDER_PRESET_FILENAME = str(
    (b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute()
)
BLACKLIST = {
    ("door", "rrsovh"),
    ("car", "takwdb"),
}


class AutomationError(ValueError):
    pass


def get_maps_from_mat(mat):
    results = set()
    if mat:
        for i in range(mat.numsubs):
            submat = mat[i]
            if submat:
                if rt.classOf(submat) == rt.SubAnim:
                    if rt.isKindOf(submat.object, rt.textureMap):
                        results.add(submat.object)
                results.update(get_maps_from_mat(submat))
    return results


def get_all_maps():
    materials = {x.material for x in rt.objects}
    return {map for mat in materials for map in get_maps_from_mat(mat)}


def load_objs_from_urdf(fn, pose=None):
    pass


def process_urdf(old_category_name, old_model_name):
    model_dir = os.path.join(
        IN_DATASET_ROOT, "objects", old_category_name, old_model_name
    )
    intervention_request_msgs = []

    # Convert to new model name.
    new_category_name, new_model_name = TRANSLATION_DICT[
        old_category_name + "/" + old_model_name
    ].split("/")

    # Load the URDF file into urdfpy
    urdf_filename = old_model_name + ".urdf"
    urdf_path = os.path.join(model_dir, "urdf", urdf_filename)
    robot = URDF.load(urdf_path)

    # Load the metadata file too
    metadata_path = os.path.join(model_dir, "misc", "metadata.json")
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    # Check the list of links that end at openable joints.
    openable_joint_ids = metadata.get("openable_joint_ids", [])
    openable_links = {
        robot.link_map[robot.joint_map[openable_joint_data[1]].child]
        for openable_joint_data in openable_joint_ids
    }

    # Do FK with everything at the lower position
    joint_cfg = {
        joint.name: joint.limit.lower
        for joint in robot.joints
        if joint.joint_type in ("prismatic", "revolute")
    }
    lfk: Dict[Link, np.ndarray] = robot.link_fk(cfg=joint_cfg)

    # Get the base link base mesh transform, revert it
    base_link = robot.base_link

    if len(base_link.visuals) >= 1:
        assert np.allclose(
            base_link.visuals[0].origin[:3, :3], np.eye(3)
        ), "Rotational offsets are not allowed."
        real_base_link_transform = np.linalg.inv(base_link.visuals[0].origin)
    else:
        intervention_request_msgs.append(
            "Base link has no visual mesh. Check if the visuals are reasonable."
        )
        real_base_link_transform = np.eye(4)

    # Start tracking objects per link
    lower_objects_by_link = {}

    def process_link(link, link_pose, is_upper_side):
        found_identity_transform = False
        if len(link.visuals) == 0:
            intervention_request_msgs.append(
                f"Encountered link with no visual meshes: {link.name}"
            )
            return None, found_identity_transform

        # Actually start the processing.
        first_mesh = None
        for visual in link.visuals:
            assert (
                visual.geometry.mesh is not None
            ), "Encountered link with primitive visuals."
            filename = os.path.abspath(
                os.path.join(model_dir, visual.geometry.mesh.filename)
            )
            joint_type = None

            # Compute the pose
            pose = real_base_link_transform @ link_pose @ visual.origin
            if visual.geometry.mesh.scale is not None:
                S = np.eye(4, dtype=np.float64)
                S[:3, :3] = np.diag(visual.geometry.mesh.scale)
                pose = pose @ S

            # Count this for check that at least one thing has trivial pose
            if np.allclose(pose, trimesh.transformations.identity_matrix()):
                found_identity_transform = True

            # Convert the pose to a 3ds Max transform
            pose_formatted = pose[:3].T.tolist()
            mat = rt.Matrix3(
                rt.Point3(*pose_formatted[0]),
                rt.Point3(*pose_formatted[1]),
                rt.Point3(*pose_formatted[2]),
                rt.Point3(*pose_formatted[3]),
            )

            # Compute the name
            if link == robot.base_link:
                obj_name = f"{new_category_name}-{new_model_name}-0-base_link"
            else:
                # Get the name based on the joint type/side.
                link_name = re.sub(r"[^a-z0-9_]", "", link.name.lower())
                joints: List[Joint] = robot.joints
                (j,) = [j for j in joints if j.child == link.name]

                # Save this for use later.
                joint_type = j.joint_type
                # if joint_type == "continuous":
                #     intervention_request_msgs.append("Continuous joint found. Check bounds.")

                if joint_type == "floating":
                    # Floating joints should get converted to plain objects.
                    obj_name = f"{new_category_name}_{link_name}-TODOfixme-0"
                    intervention_request_msgs.append(
                        f"{obj_name} needs a reasonable category / ID."
                    )
                else:
                    parent_name = re.sub(r"[^a-z0-9_]", "", j.parent.lower())
                    if parent_name == robot.base_link.name:
                        parent_name = "base_link"
                    joint_type_str = JOINT_TYPES[joint_type]
                    side = "upper" if is_upper_side else "lower"
                    obj_name = f"{new_category_name}-{new_model_name}-0-{link_name}-{parent_name}-{joint_type_str}-{side}"

                    # Add openable tag to relevant joints
                    if link in openable_links:
                        obj_name = obj_name + "-Topenable"

            # Check if we are processing the 2nd instance (upper) of a link. If so, just copy an instance and we can return now since we just need
            # to process the first visual mesh in that case.
            if link in lower_objects_by_link:
                lower_obj = lower_objects_by_link[link]
                instance_objs = []
                success, instance_objs = rt.maxOps.cloneNodes(
                    lower_obj,
                    cloneType=rt.name("instance"),
                    newNodes=pymxs.byref(instance_objs),
                )
                assert success, "Could not clone object for upper end."
                (obj,) = instance_objs
                obj.transform = mat
                obj.name = obj_name
                return obj, found_identity_transform

            # Otherwise, import the object into 3ds Max
            rt.importFile(filename, pymxs.runtime.Name("noPrompt"))

            # Are there multiple objects? Combine them.
            obj_candidates = list(rt.selection)
            obj = obj_candidates[0]
            for extra_obj in obj_candidates[1:]:
                intervention_request_msgs.append(
                    f"Found extra object during import when importing {obj_name}. Take a look."
                )
                rt.polyop.attach(obj, extra_obj)
            obj.transform = mat
            obj.name = obj_name

            # If this is not the first mesh, we just attach it to the first mesh which is already fully processed.
            if first_mesh is not None:
                rt.polyop.attach(first_mesh, obj)
                continue

            # Add meta-links.
            if link == robot.base_link:
                meta_links = metadata.get("links", {})
                for metalink_name, metalink_info in meta_links.items():
                    printable_metalink_name = re.sub(
                        r"[^a-z0-9]", "", metalink_name.lower()
                    )

                    metalink_type = metalink_info["geometry"]

                    # Get its position w.r.t. the URDF base link, transform it
                    metalink_orig_pos = np.array(metalink_info["xyz"])
                    metalink_pos = trimesh.transformations.transform_points(
                        metalink_orig_pos[None, :], real_base_link_transform
                    ).flatten()

                    if metalink_type is None:
                        # For point metalinks, we use a PointHelper.
                        point = rt.Point()
                        point.cross = False
                        point.name = f"{new_category_name}-{new_model_name}-0-base_link-M{printable_metalink_name}"
                        point.position = rt.Point3(*metalink_pos.tolist())
                        point.rotation = obj.rotation
                        point.parent = obj
                    elif metalink_type == "box":
                        # For box metalinks, we use a Box.
                        box = rt.Box()
                        box.name = f"{new_category_name}-{new_model_name}-0-base_link-M{printable_metalink_name}"
                        box.rotation = obj.rotation * rt.Quat(
                            *R.from_euler("xyz", metalink_info["rpy"])
                            .as_quat()
                            .tolist()
                        )
                        box.position = rt.Point3(*metalink_pos.tolist())
                        box.parent = obj
                        box.width, box.length, box.height = metalink_info["size"]
                        box.objectoffsetpos = rt.Point3(0, 0, box.height / 2.0)
                        intervention_request_msgs.append(
                            f"Found box metalink {box.name}. Take a look."
                        )
                    else:
                        # We don't know how to handle anything else really.
                        raise ValueError(f"Unknown metalink type {metalink_type}")

            # If this is a floating link object let's toss it under the parent.
            if joint_type == "floating":
                parent_obj = lower_objects_by_link[robot.base_link]
                obj.parent = parent_obj

            # Keep track of the first mesh we process for attachment of later meshes.
            first_mesh = obj

        return first_mesh, found_identity_transform

    # First, get the lower range of everything based on the default lfk.
    found_identity_transform = False
    for link, link_pose in lfk.items():
        obj, any_identity_transform = process_link(link, link_pose, is_upper_side=False)
        lower_objects_by_link[link] = obj
        found_identity_transform = found_identity_transform or any_identity_transform
    assert found_identity_transform, "No object has identity transform."

    # Then, for each joint, add the upper limit.
    joints: List[Joint] = robot.joints
    for joint in joints:
        if joint.joint_type not in {"prismatic", "revolute", "continuous"}:
            continue

        # Get some joint info.
        joint_name = joint.name
        if joint.joint_type == "continuous":
            # Set a fixed 180 degree limit on continuous joints. Typically buttons.
            joint_upper = np.pi - 0.01
        else:
            joint_upper = joint.limit.upper
        child_link_name = joint.child
        link = robot.link_map[child_link_name]

        # Apply FK with only this joint set to upper position.
        joint_lfk = robot.link_fk(cfg={joint_name: joint_upper})
        link_pose = joint_lfk[link]

        # Process the child link with the joint's upper position.
        process_link(link, link_pose, is_upper_side=True)

    return intervention_request_msgs, lower_objects_by_link[base_link]


def process_object_dir(model_dir):
    # Process the model name
    old_category_name = os.path.basename(os.path.dirname(model_dir))
    old_model_name = os.path.basename(model_dir)

    # Convert to new model name.
    new_category_name, new_model_name = TRANSLATION_DICT[
        old_category_name + "/" + old_model_name
    ].split("/")

    if (new_category_name, new_model_name) in BLACKLIST:
        return

    # Get output path.
    obj_output_dir = os.path.join(
        OUTPUT_ROOT, f"legacy_{new_category_name}-{new_model_name}"
    )

    # If the path already exists, assume the file has already been processed.
    if os.path.exists(obj_output_dir):
        print(f"{obj_output_dir} already exists!")
        return

    # Reset the scene
    rt.resetMaxFile(rt.name("noPrompt"))

    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    # Start processing.
    try:
        intervention_request_msgs, base_obj = process_urdf(
            old_category_name, old_model_name
        )

        # Add the parts
        metadata_path = os.path.join(model_dir, "misc", "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        if "object_parts" in metadata:
            part_type_instance_idx = collections.Counter()
            for part in metadata["object_parts"]:
                # Load the metadata
                part_cat = part["category"]
                part_model = part["model"]
                expected_part_name = f"{part_cat}-{part_model}-0"
                new_part_cat, new_part_model = TRANSLATION_DICT[
                    part_cat + "/" + part_model
                ].split("/")
                part_pos = np.array(part["pos"])
                part_orn = R.from_quat(part["orn"]).as_matrix()
                part_pose = np.zeros((3, 4))
                part_pose[:, :3] = part_orn
                part_pose[:, 3] = part_pos
                pose_formatted = part_pose.T.tolist()
                pose_mat = rt.Matrix3(
                    rt.Point3(*pose_formatted[0]),
                    rt.Point3(*pose_formatted[1]),
                    rt.Point3(*pose_formatted[2]),
                    rt.Point3(*pose_formatted[3]),
                )

                # Keep track of the objects we already had
                existing_objs = set(rt.objects)

                # Load the part
                intervention_request_msgs.extend(process_urdf(part_cat, part_model)[0])

                # Find the new objects
                new_objs = set(rt.objects) - existing_objs

                for new_obj in new_objs:
                    # Update the name, transform and parent
                    part_idx = part_type_instance_idx[expected_part_name]
                    part_type_instance_idx[expected_part_name] += 1
                    new_obj.name = (
                        f"B-{new_part_cat}-{new_part_model}-{part_idx}-Tsubpart"
                    )
                    new_obj.transform = new_obj.transform * pose_mat
                    new_obj.parent = base_obj

                if len(new_objs) != 1:
                    part_names = ",".join([x.name for x in new_objs])
                    intervention_request_msgs.append(
                        f"Found new objects != 1 after importing part {part_cat}-{part_model}: {part_names}"
                    )

        # Fix any bad materials, set the reflection color of the vray materials to be 0
        rt.select(rt.objects)
        rt.convertToVRay(True)
        for obj in rt.objects:
            if obj.material:
                obj.material.reflection = rt.Color(0, 0, 0)

        # Update the image paths.
        texture_dir = os.path.join(obj_output_dir, "textures")
        os.makedirs(texture_dir, exist_ok=True)
        for map in get_all_maps():
            source_path = map.filename
            assert os.path.exists(
                source_path
            ), f"Could not find texture map {source_path}"
            source_name, source_ext = os.path.splitext(os.path.basename(source_path))

            # Pick a semi-random target path
            while True:
                target_name = (
                    source_name
                    + "-"
                    + "".join(random.choices(string.ascii_letters, k=6))
                )
                target_path = os.path.join(texture_dir, target_name + source_ext)
                if not os.path.exists(target_path):
                    break

            # Copy the file to its new spot
            shutil.copy2(source_path, target_path)

            # Update the bitmap to have the right reference.
            print(target_path)
            map.filename = target_path

        # Stop execution if the sanity check has failed.
        sc = SanityCheck().run()
        if sc["ERROR"]:
            intervention_request_msgs.append(
                "Sanitycheck issues:\n" + "\n".join(sc["ERROR"])
            )

        # Finally, save the file.
        max_path = os.path.join(obj_output_dir, "processed.max")
        assert rt.saveMaxFile(max_path), "Could not save the file at {max_path}."

        print("Processed ", obj_output_dir)

        if CONFIRM_EACH:
            intervention_request_msgs.append(
                "Confirm this file - requested due to CONFIRM_EACH=True"
            )

        # If we are not silencing automation errors, raise now so that the user can examine the file.
        if len(intervention_request_msgs) > 0:
            if INTERACTIVE_MODE:
                raise AutomationError("\n".join(intervention_request_msgs))
            else:
                raise ValueError(
                    "Failed due to automation errors happening in headless mode. Retry in interactive."
                )
    except AutomationError:
        # No cleanup needed for automation errors.
        raise
    except:
        # Otherwise, remove the incomplete file.
        shutil.rmtree(obj_output_dir, ignore_errors=True)
        raise


def import_legacy_objs():
    # Grab all the files from the dataset.
    obj_dirs = sorted(
        p
        for p in glob.glob(os.path.join(IN_DATASET_ROOT, "objects/*/*"))
        if os.path.isdir(p)
    )

    whitelist = {"pot_plant-jatssq", "washer-omeuop"}

    # Narrow down to objects that don't already exist.
    split_names = [
        (os.path.basename(os.path.dirname(model_dir)), os.path.basename(model_dir))
        for model_dir in obj_dirs
    ]
    new_split_names = [
        TRANSLATION_DICT[old_category_name + "/" + old_model_name].split("/")
        for old_category_name, old_model_name in split_names
    ]
    # is_whitelisted = [f"{cat}-{model}" in whitelist for cat, model in new_split_names]
    is_whitelisted = [True for cat, model in new_split_names]
    obj_output_dirs = [
        os.path.join(OUTPUT_ROOT, f"legacy_{new_category_name}-{new_model_name}")
        for new_category_name, new_model_name in new_split_names
    ]

    remaining_objs = [
        obj_dir
        for obj_dir, obj_output_dir, obj_whitelisted in zip(
            obj_dirs, obj_output_dirs, is_whitelisted
        )
        if obj_whitelisted and not os.path.exists(obj_output_dir)
    ]

    failures = {}
    for obj_dir in tqdm.tqdm(remaining_objs):
        try:
            process_object_dir(obj_dir)
        except AutomationError as e:
            print(
                f"Stopping after object {obj_dir} due to non-automatable fixes needed. Please correct these, save, and re-run script to continue."
            )
            print(e)
            break
        except Exception as e:
            failures[obj_dir] = str(e)  # traceback.format_exc()
    else:
        # This means we finished iterating without stopping at an object due to a Sanitycheck Error.
        print("Finished importing files.")
        print(f"Failures: {len(failures)} / {len(remaining_objs)}")
        sorted_failure_fns = sorted(failures.keys())
        for i, failure_fn in enumerate(sorted_failure_fns):
            print(f"{i + 1} / {len(failures)} - {failure_fn}: {failures[failure_fn]}")


def import_legacy_objs_button():
    try:
        import_legacy_objs()
    except Exception as e:
        # Print message
        rt.messageBox(str(e))


if __name__ == "__main__":
    import_legacy_objs_button()
