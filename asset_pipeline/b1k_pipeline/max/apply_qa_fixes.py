from collections import defaultdict
import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pathlib
import json

import pymxs
from scipy.spatial.transform import Rotation as R
import numpy as np

import b1k_pipeline.utils

rt = pymxs.runtime
local_coordsys = pymxs.runtime.Name("local")
parent_coordsys = pymxs.runtime.Name("parent")


def rotate_pivot(tgt_obj, delta_rot_local):
    # Unparent all of the children
    children = list(tgt_obj.children)
    for child in children:
        child.parent = None

    delta_quat = rt.quat(*delta_rot_local.as_quat().tolist())
    coordsys = getattr(pymxs.runtime, "%coordsys_context")
    prev_coordsys = coordsys(local_coordsys, None)
    tgt_obj.rotation *= delta_quat
    tgt_obj.objectOffsetRot *= delta_quat
    tgt_obj.objectOffsetPos *= delta_quat
    coordsys(prev_coordsys, None)

    # Reparent the children
    for child in children:
        child.parent = tgt_obj


def scale_pivot(tgt_obj, delta_scale):
    # Update the object offset of the original
    print(tgt_obj.name, tgt_obj.position, delta_scale)
    coordsys = getattr(pymxs.runtime, "%coordsys_context")
    prev_coordsys = coordsys(parent_coordsys, None)
    tgt_obj.position *= delta_scale
    tgt_obj.objectOffsetPos *= delta_scale
    if rt.classOf(tgt_obj) != rt.Point:
        tgt_obj.objectOffsetScale *= delta_scale
    coordsys(prev_coordsys, None)

    # Recurse through the children
    for c in tgt_obj.children:
        scale_pivot(c, delta_scale)


def apply_qa_fixes_in_open_file(apply_scale=True, apply_orientation=True):
    # Load the fixes
    fixes = defaultdict(dict)
    with open(
        r"D:\BEHAVIOR-1K\asset_pipeline\metadata\orientation_and_scale_edits.json"
    ) as f:
        data = json.load(f)
        for model, orientation in data["orientations"].items():
            if not np.allclose(orientation, [0, 0, 0, 1], atol=1e-3):
                fixes[model]["orientation"] = orientation
        for model, scale in data["scales"].items():
            if not np.allclose(scale, [1, 1, 1], atol=1e-3):
                assert np.allclose(
                    scale, scale[0], atol=1e-3
                ), "Non-uniform scale not supported."
                fixes[model]["scale"] = scale[0]

    # Prior to starting, let's get the minimum y value and add a bit of margin to it. This is
    # where we'll put our rescaled objects.
    bboxes = []
    for node in rt.objects:
        bbox_min, bbox_max = rt.NodeGetBoundingBox(node, rt.Matrix3(1))
        bboxes.extend([np.array(bbox_min), np.array(bbox_max)])
    bboxes = np.array(bboxes)
    max_y = np.max(bboxes[:, 1])

    # Go through all objects and apply rotation fix if they are the base link and not a metalink
    objects_by_model_and_id = defaultdict(list)
    base_links_by_model_and_id = {}

    # Make a pass for orientations
    for obj in rt.objects:
        pn = b1k_pipeline.utils.parse_name(obj.name)
        if not pn:
            continue

        # If there are no fixes for this model, we can continue
        model_id = pn.group("model_id")
        if model_id not in fixes:
            continue
        model_fixes = fixes[model_id]

        # If this is a meta link, continue
        if pn.group("meta_type"):
            continue

        print("Applying fixes to", obj.name)

        # Apply orientation fixes (the pivot needs to be rotated by the inverse of the fix)
        # Note that child links also get their pivots rotated here. Is that good? idk.
        # But we definitely don't want to rotate lights.
        if (
            apply_orientation
            and "orientation" in model_fixes
            and rt.ClassOf(obj) == rt.Editable_Poly
        ):
            rotate_pivot(obj, R.from_quat(model_fixes["orientation"]).inv())

        # Record the object for possible scale fixes. We're doing this only for root-level
        # objects. We will reparent these objects for the process of applying scale fixes.
        if "scale" in model_fixes and not obj.parent:
            is_base_link = pn.group("link_name") in (None, "", "base_link")
            key = (model_id, pn.group("instance_id"))
            if is_base_link:
                base_links_by_model_and_id[key] = obj
            else:
                objects_by_model_and_id[key].append(obj)

    # Make a pass for scales, only for object files
    if apply_scale and "objects" in pathlib.Path(rt.maxFilePath).parts:
        # Assert that all children have bases
        assert set(base_links_by_model_and_id.keys()).issuperset(
            set(objects_by_model_and_id.keys())
        ), "Not all objects have base links."

        # Go through the base links
        current_x = 0
        align_y = max_y + 1000.0
        for key, base_link in base_links_by_model_and_id.items():
            # If there are links, temporarily parent them under the base link
            children = (
                objects_by_model_and_id[key] if key in objects_by_model_and_id else []
            )
            for child in children:
                child.parent = base_link

            # Apply scale fixes for the base link
            scale_fix = fixes[key[0]]["scale"]
            print("Applying scale fix to object", key, scale_fix)
            scale_pivot(base_link, scale_fix)

            # If the scale is greater than 1, let's move the object to be safe
            if scale_fix > 1:
                # Get the current guy's bbox
                bbox_min, bbox_max = rt.NodeGetBoundingBox(base_link, rt.Matrix3(1))
                bbox_min = np.array(bbox_min)
                bbox_max = np.array(bbox_max)
                bbox_extent = bbox_max - bbox_min
                bbox_center = (bbox_max + bbox_min) / 2
                base_link_wrt_bbox = np.array(base_link.position) - bbox_center

                # Lay out the bbox
                bbox_center_goal_position = np.array(
                    [
                        current_x + bbox_extent[0] / 2,
                        align_y + bbox_extent[1] / 2,
                        bbox_extent[2] / 2,
                    ]
                )
                base_link_goal_position = bbox_center_goal_position + base_link_wrt_bbox
                base_link.position = rt.Point3(*base_link_goal_position.tolist())

                # Move the current x
                current_x += bbox_extent[0] + 1000.0

            # Unparent all of the children
            for child in children:
                child.parent = None


if __name__ == "__main__":
    apply_qa_fixes_in_open_file()
