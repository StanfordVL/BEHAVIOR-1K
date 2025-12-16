from collections import defaultdict
import json
import sys

import tqdm

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import mat2arr, parse_name, PIPELINE_ROOT

import numpy as np
from scipy.spatial.transform import Rotation as R

SET_VIA_TRANSFORM = False
_EPS = np.finfo(float).eps * 4.0

inventory_path = PIPELINE_ROOT / "artifacts" / "pipeline" / "object_inventory.json"
with open(inventory_path, "r") as f:
    providers = {k.split("-")[-1]: v for k, v in json.load(f)["providers"].items()}


def quat2arr(q):
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)


def transform2mat(transform):
    # Transpose the rotation/scale part to go from right hand to left hand
    transform_copy = transform.copy()
    # transform_copy[:3, :3] = transform_copy[:3, :3].T

    # Convert a 4x4 numpy transform in our right-hand multiplication system to a 4x3 matrix in 3ds Max's left-hand multiplication system
    return rt.Matrix3(
        *[
            rt.Point3(*row)
            for row in transform_copy[:3, :].T.astype(np.float32).tolist()
        ]
    )


def mat2transform(mat):
    transform = np.hstack([mat2arr(mat, dtype=np.float64), [[0], [0], [0], [1]]]).T

    # Transpose the rotation/scale part to go from left hand to right hand
    # transform[:3, :3] = transform[:3, :3].T

    return transform


def get_vert_sets_including_children(node, only_canonical_parts_of=None):
    if only_canonical_parts_of is not None:
        parsed_name = parse_name(node.name)
        if not parsed_name:
            return []

        if parsed_name.group("model_id") != only_canonical_parts_of:
            return []

        if parsed_name.group("meta_type") or parsed_name.group("joint_side") == "upper":
            return []
    vert_sets = []
    if rt.classOf(node) == rt.Editable_Poly:
        vert_sets.append(
            np.array(
                [
                    rt.polyop.getVert(node, i + 1)
                    for i in range(rt.polyop.GetNumVerts(node))
                ],
                dtype=np.float64,
            )
        )
    for child in node.children:
        vert_sets.extend(
            get_vert_sets_including_children(
                child, only_canonical_parts_of=only_canonical_parts_of
            )
        )
    return vert_sets


def bounding_box_from_verts(verts):
    return np.min(verts, axis=0), np.max(verts, axis=0)


def node_bounding_box_incl_children(node, transform=None, only_canonical=None):
    if only_canonical:
        parsed_name = parse_name(node.name)
        if not parsed_name:
            return None
        only_canonical = parsed_name.group("model_id")
    verts_world = np.concatenate(
        get_vert_sets_including_children(node, only_canonical_parts_of=only_canonical),
        axis=0,
    )
    if transform is None or np.allclose(transform, np.eye(4)):
        verts = verts_world
    else:
        inv_transform = np.linalg.inv(transform)
        verts = np.concatenate(
            [verts_world, np.ones((verts_world.shape[0], 1))], axis=1
        )
        verts = (inv_transform @ verts.T).T[:, :3]
    return bounding_box_from_verts(verts)


def get_rotation_from_transform(matrix):
    M = np.array(matrix, dtype=np.float64).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3,))
    shear = [0.0, 0.0, 0.0]

    if any(abs(M[:3, 3]) > _EPS):
        M[:, 3] = 0.0, 0.0, 0.0, 1.0

    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = np.linalg.norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = np.linalg.norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = np.linalg.norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        np.negative(scale, scale)
        np.negative(row, row)

    return row.T, scale


def rotation_only_transform(mat):
    transform = mat2transform(mat)
    rotation, _ = get_rotation_from_transform(
        transform
    )  # R.from_quat(quat2arr(mat.rotation)).as_matrix()
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result


def apply_transform(points, transform):
    with_ones = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return np.dot(transform, with_ones.T).T[:, :3]


def import_bad_model_originals(model_id):
    # Get the provider file
    provider = providers[model_id]
    filename = str(PIPELINE_ROOT / "cad" / provider / "processed.max")

    # Get the right object names from the file
    f_objects = rt.getMAXFileObjectNames(filename, quiet=True)
    visual_objects = {}
    for obj in f_objects:
        m = parse_name(obj)
        if not m:
            continue
        if (
            m.group("model_id") != model_id
            or m.group("instance_id") != "0"
            or m.group("joint_side") == "upper"
        ):
            continue

        # We have no way of checking if the object is an editable poly so we just apply some heuristics
        if m.group("meta_type") and m.group("meta_type") != "collision":
            continue
        if m.group("light_id"):
            continue

        # If we get here this is a lower link of an instance zero
        assert not m.group("bad"), "Bad objects should not be in the inventory"

        link_name = m.group("link_name") if m.group("link_name") else "base_link"
        joint_type = m.group("joint_type")
        if "Mcollision" in obj:
            continue

        visual_objects[(link_name, m.group("parent_link_name"), joint_type)] = obj

    objects_to_import = sorted(set(visual_objects.values()))

    success, imported_meshes = rt.mergeMaxFile(
        filename,
        objects_to_import,
        rt.Name("select"),
        rt.Name("autoRenameDups"),
        rt.Name("useSceneMtlDups"),
        rt.Name("neverReparent"),
        rt.Name("noRedraw"),
        quiet=True,
        mergedNodes=pymxs.byref(None),
    )
    assert success, f"Failed to import {model_id}."
    imported_objs_by_name = {obj.name: obj for obj in imported_meshes}
    assert set(objects_to_import) == set(imported_objs_by_name.keys()), (
        "Not all objects were imported. Missing: "
        + str(set(objects_to_import) - set(imported_objs_by_name.keys()))
    )

    zero_layer = rt.LayerManager.getLayer(0)
    zero_layer.current = True
    for mesh in imported_meshes:
        zero_layer.addNode(mesh)

    return {k: imported_objs_by_name[v] for k, v in visual_objects.items()}


def replace_object_instances(obj, respect_aspect_ratio=False):
    # Parse the name and assert it's the base link of a bad object and instance 0
    parsed_name = parse_name(obj.name)
    assert parsed_name, f"Object {obj.name} has no parsed name"
    assert parsed_name.group("bad"), f"Object {obj.name} is not a bad object"
    assert (
        parsed_name.group("instance_id") == "0"
    ), f"Object {obj.name} is not instance 0"
    assert not (
        parsed_name.group("link_name") and parsed_name.group("link_name") != "base_link"
    ), f"Object {obj.name} is not the base link"

    # Get the model ID
    loose = parsed_name.group("loose")
    if not loose:
        loose = ""
    category = parsed_name.group("category")
    model_id = parsed_name.group("model_id")
    tags = parsed_name.group("tag")
    if (
        not tags
    ):  # This is for retaining any part tag on the base link. Other tags dont care.
        tags = ""

    # First, find all the instances of this object and validate that their pivots are the same
    all_same_model_id = [
        x
        for x in rt.objects
        if parse_name(x.name) and parse_name(x.name).group("model_id") == model_id
    ]
    all_base_links = [
        x
        for x in all_same_model_id
        if parse_name(x.name).group("link_name") == parsed_name.group("link_name")
    ]
    same_model_id_parsed_names = [parse_name(x.name) for x in all_same_model_id]
    assert all(
        same_model_id_parsed_names
    ), "All same model ID objects should have a parsed name"

    # Check that once grouped by instance ID, they all have the same set of links
    links_by_instance_id = defaultdict(dict)
    for inst, inst_parsed_name in zip(all_same_model_id, same_model_id_parsed_names):
        link_name = inst_parsed_name.group("link_name")
        if not link_name:
            link_name = "base_link"
        links_by_instance_id[inst_parsed_name.group("instance_id")][link_name] = inst
    all_links = set(frozenset(x.keys()) for x in links_by_instance_id.values())
    # assert (
    #     len(all_links) == 1
    # ), f"All instances of {obj.name} do not share the same set of links: {links_by_instance_id}"

    # Check that they are all marked as BAD
    assert all(
        x.group("bad") for x in same_model_id_parsed_names
    ), "All instances of the same model ID should be marked as bad."

    # Check that they do NOT have any children
    assert not any(
        len(x.children) > 0 for x in all_same_model_id
    ), "Instances of the same model ID should not have children: {all_same_model_id}"

    # Parent all links under their bases for efficient computation of the bounding box
    all_parented = set()
    for instance, links in links_by_instance_id.items():
        imported_base_link = links.pop("base_link")
        for link in links.values():
            link.parent = imported_base_link
        all_parented.add(imported_base_link)
    assert all_parented == set(all_base_links), "Not all instances were parented."

    # Record the transforms of all of the instances as well as their local and world BBs
    instance_layers = [x.layer for x in all_base_links]
    instance_parents = [inst.parent for inst in all_base_links]
    instance_transforms = [inst.transform for inst in all_base_links]
    instance_rotations = [inst.rotation for inst in all_base_links]
    instance_lowbbs = [
        node_bounding_box_incl_children(inst, rotation_only_transform(inst.transform))
        for inst in all_base_links
    ]
    instance_world_bbs = [
        node_bounding_box_incl_children(inst) for inst in all_base_links
    ]

    # Delete all the objects
    for same_model_id in all_same_model_id:
        rt.delete(same_model_id)

    # Import the original mesh for just the zero instance
    imported = import_bad_model_originals(model_id)
    print("Imported originals for", model_id)

    # Parent all the imported meshes under the base link
    imported_base_link = imported[("base_link", None, None)]
    for (link_name, parent_link, joint_type), link in imported.items():
        if link_name == "base_link":
            continue
        link.parent = imported_base_link

    # Set the rotation of the imported base link to be the identity rotation
    imported_base_link.transform = rt.Matrix3(1)

    # Flatten the copyables into a list
    copyables = [("base_link", None, None, imported_base_link)] + [
        (*k, v) for k, v in imported.items() if k[0] != "base_link"
    ]
    copyable_meshes = [x[-1] for x in copyables]

    comparison_data = {}

    # Now for each of the original bozos make a copy of the whole thing, scale it up, rotate it, and shift it into place
    for instance_id, (
        instance_parent,
        instance_transform,
        instance_rotation,
        instance_lowbb,
        instance_world_bb,
        instance_layer,
    ) in enumerate(
        zip(
            instance_parents,
            instance_transforms,
            instance_rotations,
            instance_lowbbs,
            instance_world_bbs,
            instance_layers,
        )
    ):
        # First copy the object.
        success, child_copy = rt.maxOps.cloneNodes(
            copyable_meshes,
            cloneType=rt.name("instance"),
            newNodes=pymxs.byref(None),
        )
        assert success, f"Could not clone {imported_base_link.name}"
        print("Cloned", model_id, instance_id)

        # Move everything to the correct layer
        for mesh in child_copy:
            instance_layer.addNode(mesh)

        # Find the base link in the child copies. It's the one whose parent is not one of the child copies
        (base_copy,) = [x for x in child_copy if x.parent not in child_copy]
        assert (
            base_copy == child_copy[0]
        ), "The base link should be the first element in the list of child"

        # Get the points
        base_copy_points = np.concatenate(
            get_vert_sets_including_children(base_copy), axis=0
        )

        # Scale the imported mesh to match the instance
        # The target bounding box is the instance's local bounding box
        instance_lowbb_size = instance_lowbb[1] - instance_lowbb[0]

        # The current world bounding box is equal to the local bounding box, since we start
        # with an identity transform (e.g. the points are the same in world
        # and local frame)
        base_copy_lowbb = bounding_box_from_verts(base_copy_points)
        base_copy_lowbb_size = base_copy_lowbb[1] - base_copy_lowbb[0]

        # The target scale is the ratio of the two sizes
        if respect_aspect_ratio:
            relative_scale_from_now = np.min(
                instance_lowbb_size / base_copy_lowbb_size
            ) * np.ones(3)
        else:
            relative_scale_from_now = instance_lowbb_size / base_copy_lowbb_size

        # This scale needs to be applied BEFORE the rotation since it's in local frame
        scale_transform = np.diag(relative_scale_from_now.tolist() + [1])

        # Combine the scale with the rotation
        rotation_transform = rotation_only_transform(instance_transform)
        combined_transform = rotation_transform @ scale_transform

        # Assert the validity of the rotation and scale
        print("Rotation transform", rotation_transform)
        print("Scale transform", scale_transform)
        print(
            "Combined transform components",
            get_rotation_from_transform(combined_transform),
        )
        target_orn = R.from_quat(quat2arr(instance_transform.rotation))
        current_orn = R.from_matrix(get_rotation_from_transform(combined_transform)[0])
        delta_orn = target_orn * current_orn
        delta_mag = delta_orn.magnitude()
        print("Delta orientation", delta_orn.as_rotvec(), "magnitude", delta_mag)

        # Position it to match bottom face center (this allows for the object to not be floating
        # if respect_aspect_ratio is True)
        base_copy_world_bb = bounding_box_from_verts(
            apply_transform(base_copy_points, combined_transform)
        )
        base_copy_world_bb_bottom_center = (
            base_copy_world_bb[0] + base_copy_world_bb[1]
        ) / 2
        base_copy_world_bb_bottom_center[2] = base_copy_world_bb[0][2]
        instance_world_bb_bottom_center = (
            instance_world_bb[0] + instance_world_bb[1]
        ) / 2
        instance_world_bb_bottom_center[2] = instance_world_bb[0][2]
        move_center_by = (
            instance_world_bb_bottom_center - base_copy_world_bb_bottom_center
        )
        combined_transform[:3, 3] = move_center_by

        # Convert and apply the transform to the 3ds Max object
        if SET_VIA_TRANSFORM:
            base_copy.transform = transform2mat(combined_transform)
        else:
            quat = R.from_matrix(rotation_transform[:3, :3]).as_quat()
            base_copy.scale = rt.Point3(*relative_scale_from_now.tolist())
            base_copy.rotation = rt.Quat(*quat.tolist())
            base_copy.position = rt.Point3(*move_center_by.tolist())

        max_orn = R.from_quat(quat2arr(base_copy.rotation))
        numpy_rot, numpy_scale = get_rotation_from_transform(combined_transform)
        numpy_orn = R.from_matrix(numpy_rot)
        delta_orn = max_orn.inv() * numpy_orn
        delta_mag = delta_orn.magnitude()
        print("Max orientation differs from local orientation by", delta_mag)
        print(
            "Max scale",
            base_copy.transform.scale,
            "vs Numpy scale",
            numpy_scale,
            "vs Relative scale",
            relative_scale_from_now,
        )

        # Assert the new world bb is the same as the original world bb and the numpy bb
        # Compute the transformed bb via numpy
        computed_world_bb = bounding_box_from_verts(
            apply_transform(base_copy_points, combined_transform)
        )
        computed_size = computed_world_bb[1] - computed_world_bb[0]
        computed_center = (computed_world_bb[0] + computed_world_bb[1]) / 2
        computed_min, computed_max = computed_world_bb

        # Decompose the original bb
        orig_min, orig_max = instance_world_bb
        orig_center = (orig_min + orig_max) / 2
        orig_size = orig_max - orig_min

        # Decompose the new 3ds Max bb
        base_copy_world_bb = node_bounding_box_incl_children(base_copy)
        new_min, new_max = base_copy_world_bb
        new_center = (new_min + new_max) / 2
        new_size = new_max - new_min

        print("Bounding box info")
        print(
            "  Original min",
            orig_min,
            "vs New min",
            new_min,
            "vs Computed min",
            computed_world_bb[0],
        )
        print(
            "  Original max",
            orig_max,
            "vs New max",
            new_max,
            "vs Computed max",
            computed_world_bb[1],
        )
        print(
            "  Original size",
            orig_size,
            "vs New size",
            new_size,
            "vs Computed size",
            computed_size,
        )
        print(
            "  Original center",
            orig_center,
            "vs New center",
            new_center,
            "vs Computed center",
            computed_center,
        )

        # Note the much higher tolerance due to rotation errors :(
        if not respect_aspect_ratio:
            assert np.allclose(
                orig_center, computed_center, atol=100
            ), "Computed center is not the same as the original center"
            assert np.allclose(
                orig_size, computed_size, atol=100
            ), "Computed size is not the same as the original size"

            assert np.allclose(
                orig_center, new_center, atol=100
            ), "New center is not the same as the original center"
            assert np.allclose(
                orig_size,
                new_size,
                atol=100,
            ), "New size is not the same as the original size"

        # Record the comparison data
        comparison_data[instance_id] = {
            "original_transform": mat2transform(instance_transform).tolist(),
            "calculated_transform": combined_transform.tolist(),
            "current_transform": mat2transform(base_copy.transform).tolist(),
            "original_world_bb": [x.tolist() for x in instance_world_bb],
            "computed_world_bb": [x.tolist() for x in computed_world_bb],
            "current_world_bb": [x.tolist() for x in base_copy_world_bb],
            "original_lowbb": [x.tolist() for x in instance_lowbb],
            "current_lowbb": [x.tolist() for x in base_copy_lowbb],
            "respect_aspect_ratio": respect_aspect_ratio,
        }

        # Name each of the children correctly, and parent them to the original owner's parents
        for (link_name, parent_link, joint_type, _), link in zip(copyables, child_copy):
            link.parent = instance_parent
            if link_name == "base_link":
                link.name = (
                    f"B-{loose}{category}-{model_id}-{instance_id}-{link_name}{tags}"
                )
            else:
                link.name = f"B-{loose}{category}-{model_id}-{instance_id}-{link_name}-{parent_link}-{joint_type}-lower"
        print("Replaced", model_id, instance_id)

    # Finally, after all is done, remove all of the copyables
    rt.delete(copyable_meshes)

    return comparison_data


def replace_all_bad_legacy_objects_in_open_file():
    bad_objects_to_remove = {}
    for obj in rt.objects:
        # Check that it's an editable poly
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        # Check that the parsed name contains the B- tag
        parsed_name = parse_name(obj.name)
        if not parsed_name or not parsed_name.group("bad"):
            continue

        # Check that the link is the base link
        if (
            parsed_name.group("link_name")
            and parsed_name.group("link_name") != "base_link"
        ):
            continue

        # Check that the instance ID is 0
        if parsed_name.group("instance_id") != "0":
            continue

        # Check that its not a meta link
        if parsed_name.group("meta_type"):
            continue

        # Check that the provider is legacy_
        model_id = parsed_name.group("model_id")
        provider = providers[model_id]
        if "legacy_" not in provider:
            continue

        # Check that we havent picked an object for this model ID yet.
        assert (
            model_id not in bad_objects_to_remove
        ), f"Found multiple base objects for {model_id}. First: {bad_objects_to_remove[model_id].name}, Second: {obj.name}"

        # Add to the list of bad objects to remove
        bad_objects_to_remove[model_id] = obj

    print(
        "Found",
        len(bad_objects_to_remove),
        "bad objects to remove:",
        ", ".join(bad_objects_to_remove.keys()),
    )

    # Go through all the bad objects to remove
    comparison_data = {}
    for model_id, example_obj in tqdm.tqdm(bad_objects_to_remove.items()):
        comparison_data[model_id] = replace_object_instances(example_obj)

    print("\nDon't forget to reset scale!")

    return comparison_data


def main():
    replace_object_instances(rt.selection[0], respect_aspect_ratio=False)


if __name__ == "__main__":
    main()
