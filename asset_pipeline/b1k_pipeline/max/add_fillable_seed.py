import pymxs

rt = pymxs.runtime

import numpy as np
import networkx as nx

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name


def add_fillable_seed():
    # Assert that the currently selected object is a seed point
    assert len(rt.selection) == 1, "Expected exactly one object to be selected"
    (target_obj,) = rt.selection

    # Some checks about the target object
    parsed_name = parse_name(target_obj.name)
    assert parsed_name, "Failed to parse the name of the target object"
    assert not parsed_name.group("bad"), "Expected the target object to not be bad"
    assert (
        parsed_name.group("instance_id") == "0"
    ), "Expected the target object to be the 0 instance"
    assert not parsed_name.group(
        "meta_type"
    ), "Expected the target object to not have a meta type"
    assert (
        parsed_name.group("joint_side") != "upper"
    ), "Expected the target object to not be an upper joint"
    target_link_name = parsed_name.group("link_name") or "base_link"

    # Browse the entire scene to find the articulation tree of the target object
    articulation_tree = nx.DiGraph()
    link_objs = {}
    for obj in rt.objects:
        candidate_parsed_name = parse_name(obj.name)
        if not candidate_parsed_name:
            continue
        if candidate_parsed_name.group("model_id") != parsed_name.group("model_id"):
            continue
        if candidate_parsed_name.group("instance_id") != parsed_name.group(
            "instance_id"
        ):
            continue
        if candidate_parsed_name.group("bad"):
            continue
        if candidate_parsed_name.group("meta_type"):
            continue
        if candidate_parsed_name.group("joint_side") == "upper":
            continue
        if candidate_parsed_name.group("light_id"):
            continue
        link_name = candidate_parsed_name.group("link_name") or "base_link"

        # Assert that it has a collision mesh
        collision_mesh_objects = []
        for child in obj.children:
            if rt.classOf(child) == rt.Editable_Poly and "Mcollision" in child.name:
                collision_mesh_objects.append(child)
        assert (
            len(collision_mesh_objects) == 1
        ), f"Expected {obj.name} to have exactly one collision mesh"
        collision_mesh_object = collision_mesh_objects[0]

        # Add it into the graph
        articulation_tree.add_node(link_name)
        link_objs[link_name] = collision_mesh_object
        if link_name != "base_link":
            parent_link_name = candidate_parsed_name.group("parent_link_name")
            articulation_tree.add_edge(parent_link_name, link_name)

    # Get all the descendants of the target link. These will be used when computing the fillable volume
    links_to_include = set(nx.descendants(articulation_tree, target_link_name)) | {
        target_link_name
    }
    collision_mesh_objects_to_include = [
        link_objs[link_name] for link_name in links_to_include
    ]

    # Get the bounding box
    vert_sets = np.concatenate(
        [
            np.array(
                [
                    rt.polyop.getVert(node, i + 1)
                    for i in range(rt.polyop.GetNumVerts(node))
                ],
                dtype=np.float64,
            )
            for node in collision_mesh_objects_to_include
        ],
        axis=0,
    )
    bb_min = np.min(vert_sets, axis=0)
    bb_max = np.max(vert_sets, axis=0)
    bb_center = (bb_min + bb_max) / 2

    # Create the point
    seed = rt.Point()
    seed.name = "fillable_seed"
    seed.rotation = target_obj.rotation
    seed.position = rt.Point3(*bb_center.tolist())
    seed.parent = target_obj


if __name__ == "__main__":
    add_fillable_seed()
