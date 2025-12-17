import re
import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import glob
import json
import os
from typing import Dict, List

import numpy as np
import pymxs
import tqdm
import yaml
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils
from b1k_pipeline.max.new_sanity_check import SanityCheck
from b1k_pipeline.urdfpy import URDF, Joint, Link

rt = pymxs.runtime

CONFIRM_EACH = False
INTERACTIVE_MODE = True
IN_DATASET_ROOT = r"C:\Users\Cem\research\iGibson-dev\igibson\data\ig_dataset"
OUTPUT_ROOT = r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects"
RECORD_PATH = os.path.join(
    r"D:\BEHAVIOR-1K\asset_pipeline\metadata\patched_rotations.json"
)
TRANSLATION_PATH = os.path.join(IN_DATASET_ROOT, "metadata", "model_rename.yaml")
with open(TRANSLATION_PATH, "r") as f:
    TRANSLATION_DICT = yaml.load(f, Loader=yaml.SafeLoader)


class AutomationError(ValueError):
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

    def process_link(link, link_pose, is_upper_side):
        visual = link.visuals[0]
        assert (
            visual.geometry.mesh is not None
        ), "Encountered link with primitive visuals."

        # Compute the pose
        pose = real_base_link_transform @ link_pose @ visual.origin
        if visual.geometry.mesh.scale is not None:
            S = np.eye(4, dtype=np.float64)
            S[:3, :3] = np.diag(visual.geometry.mesh.scale)
            pose = pose @ S

        # Convert the pose to a 3ds Max transform
        pose_formatted = pose[:3].T.tolist()
        mat = rt.Matrix3(
            rt.Point3(*pose_formatted[0]),
            rt.Point3(*pose_formatted[1]),
            rt.Point3(*pose_formatted[2]),
            rt.Point3(*pose_formatted[3]),
        )

        # Get the name based on the joint type/side.
        link_name = re.sub(r"[^a-z0-9_]", "", link.name.lower())
        joints: List[Joint] = robot.joints
        (j,) = [j for j in joints if j.child == link.name]

        # Save this for use later.
        parent_name = re.sub(r"[^a-z0-9_]", "", j.parent.lower())
        if parent_name == robot.base_link.name:
            parent_name = "base_link"
        obj_name = (
            f"{new_category_name}-{new_model_name}-0-{link_name}-{parent_name}-R-upper"
        )

        (obj,) = [
            x
            for x in rt.objects
            if b1k_pipeline.utils.parse_name(x.name).group("mesh_basename") == obj_name
        ]
        obj.transform = mat

    # Then, for each joint, add the upper limit.
    joints: List[Joint] = robot.joints
    for joint in joints:
        if joint.joint_type != "revolute":
            continue

        # Exclusively target joints that are more than 179 degrees
        joint_range = np.abs(np.rad2deg(joint.limit.upper - joint.limit.lower))
        if joint_range <= 179:
            continue
        elif joint_range > 200:
            intervention_request_msgs.append(
                f"Large joint range of {joint_range}. Take a look."
            )

        # Get some joint info.
        joint_name = joint.name
        assert joint.limit.lower < joint.limit.upper, "Lower above upper"
        joint_upper = joint.limit.lower + np.deg2rad(179)
        child_link_name = joint.child
        link = robot.link_map[child_link_name]

        # Apply FK with only this joint set to upper position.
        joint_lfk = robot.link_fk(cfg={joint_name: joint_upper})
        link_pose = joint_lfk[link]

        # Process the child link with the joint's upper position.
        process_link(link, link_pose, is_upper_side=True)

    return intervention_request_msgs


def process_object_dir(model_dir):
    # Process the model name
    old_category_name = os.path.basename(os.path.dirname(model_dir))
    old_model_name = os.path.basename(model_dir)

    # Convert to new model name.
    new_category_name, new_model_name = TRANSLATION_DICT[
        old_category_name + "/" + old_model_name
    ].split("/")

    # Get output path.
    obj_output_dir = os.path.join(
        OUTPUT_ROOT, f"legacy_{new_category_name}-{new_model_name}"
    )

    if not os.path.exists(obj_output_dir):
        print(f"{obj_output_dir} doesn't exist!")
        return

    # Reset the scene
    filename = os.path.join(obj_output_dir, "processed.max")
    assert rt.loadMaxFile(str(filename), useFileUnits=False)

    # Start processing.
    try:
        intervention_request_msgs = process_urdf(old_category_name, old_model_name)

        # Stop execution if the sanity check has failed.
        sc = SanityCheck().run()
        if sc["ERROR"]:
            intervention_request_msgs.append(
                "Sanitycheck issues:\n" + "\n".join(sc["ERROR"])
            )

        # Finally, save the file.
        max_path = os.path.join(obj_output_dir, "processed.max")
        assert rt.saveMaxFile(max_path), "Could not save the file at {max_path}."

        # Update the done path record
        done_paths = []
        if os.path.exists(RECORD_PATH):
            with open(RECORD_PATH, "r") as f:
                done_paths = json.load(f)
        done_paths.append(os.path.normpath(obj_output_dir))
        with open(RECORD_PATH, "w") as f:
            done_paths = json.dump(done_paths, f)

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


def fix_legacy_obj_rots():
    # Grab all the files from the dataset.
    obj_dirs = sorted(
        p
        for p in glob.glob(os.path.join(IN_DATASET_ROOT, "objects/*/*"))
        if os.path.isdir(p)
    )

    whitelist = {
        "bottom_cabinet-immwzb",
        "bottom_cabinet-hrdeys",
        "bottom_cabinet-xiurwn",
        "bottom_cabinet-plccav",
        "bottom_cabinet_no_top-vdedzt",
        "briefcase-rxvbea",
        "briefcase-ohelib",
        "briefcase-fdpjtj",
        "carton-causya",
        "dryer-wivnic",
        "floor_lamp-pfuqec",
        "fridge-hivvdf",
        "jar-vzwhbg",
        "laptop-nvulcs",
        "package-msfzpz",
        "stove-qbjiva",
        "table_lamp-tggobp",
        "table_lamp-zpqejt",
        "trash_can-pdmzhv",
        "trash_can-nuoypc",
        "trash_can-mdojox",
        "trash_can-ifzxzj",
        "window-mjssrd",
    }

    # Narrow down to objects that don't already exist.
    split_names = [
        (os.path.basename(os.path.dirname(model_dir)), os.path.basename(model_dir))
        for model_dir in obj_dirs
    ]
    new_split_names = [
        TRANSLATION_DICT[old_category_name + "/" + old_model_name].split("/")
        for old_category_name, old_model_name in split_names
    ]
    is_whitelisted = [f"{cat}-{model}" in whitelist for cat, model in new_split_names]
    obj_output_dirs = [
        os.path.join(OUTPUT_ROOT, f"legacy_{new_category_name}-{new_model_name}")
        for new_category_name, new_model_name in new_split_names
    ]

    done_paths = []
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH, "r") as f:
            done_paths = json.load(f)

    remaining_objs = [
        obj_dir
        for obj_dir, obj_output_dir, obj_whitelisted in zip(
            obj_dirs, obj_output_dirs, is_whitelisted
        )
        if obj_whitelisted and not os.path.normpath(obj_output_dir) in done_paths
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
        print("Finished patching files.")
        print(f"Failures: {len(failures)} / {len(remaining_objs)}")
        sorted_failure_fns = sorted(failures.keys())
        for i, failure_fn in enumerate(sorted_failure_fns):
            print(f"{i + 1} / {len(failures)} - {failure_fn}: {failures[failure_fn]}")


if __name__ == "__main__":
    fix_legacy_obj_rots()
