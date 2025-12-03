import csv
import json
import os
import re

import networkx as nx
import numpy as np
import tqdm
import trimesh
import fs.path
from fs.zipfs import ZipFS
from scipy.spatial.transform import Rotation as R

from b1k_pipeline.utils import parse_name, load_mesh, load_points, PipelineFS

SCALE_FACTOR = 0.001
SCALE_MATRIX = trimesh.transformations.scale_matrix(0.001)

PARTICLE_APPLIER_CONE_LENGTH = 0.3

CONVEX_MESH_TYPES = {"collision", "fillable", "openfillable"}
NON_COLLISION_CONVEX_MESH_TYPES = CONVEX_MESH_TYPES - {"collision"}

# Load the rename file
RENAMES = {}
with PipelineFS().open("metadata/object_renames.csv") as f:
    for row in csv.DictReader(f):
        key = (row["Original category (auto)"], row["ID (auto)"])
        RENAMES[key] = row["New Category"]

# Load the deletion file
DELETION_QUEUE = set()
with PipelineFS().open("metadata/deletion_queue.csv", "r") as f:
    for row in csv.DictReader(f):
        DELETION_QUEUE.add(row["Object"].strip().split("-")[1])


def maybe_rename_category(cat, model):
    if (cat, model) in RENAMES:
        return RENAMES[(cat, model)]
    return cat


def build_mesh_tree(
    target,
    load_upper=True,
    load_bad=True,
    load_nonzero=True,
    load_meshes=True,
    filter_nodes=None,
    show_progress=False,
):
    G = nx.DiGraph()

    pipeline_fs = PipelineFS()
    target_output_fs = pipeline_fs.target_output(target)

    # Open the mesh filesystems
    mesh_fs = ZipFS(target_output_fs.open("meshes.zip", "rb"))

    # Load the object list for the file
    with target_output_fs.open("object_list.json", "r") as f:
        object_list = json.load(f)

    # Get the mesh list and bboxes
    mesh_list = object_list["meshes"]
    object_bounding_boxes = object_list["bounding_boxes"]

    # Invert the bbox rotations and update their scales
    # TODO: Remove .inv() after fixing the object_list wrong rotation.
    for model_id, instances in object_bounding_boxes.items():
        for instance_id, bbox in instances.items():
            bbox["rotation"] = R.from_quat(bbox["rotation"]).inv().as_quat().tolist()
            bbox["extent"] = (np.array(bbox["extent"]) * SCALE_FACTOR).tolist()
            bbox["position"] = (np.array(bbox["position"]) * SCALE_FACTOR).tolist()

    pbar = tqdm.tqdm(mesh_list) if show_progress else mesh_list
    for mesh_name in pbar:
        if show_progress:
            pbar.set_description(mesh_name)
        match = parse_name(mesh_name)
        is_broken = match.group("bad")
        is_randomization_fixed = match.group("randomization_disabled")
        is_loose = match.group("loose")
        obj_model = match.group("model_id")
        obj_cat = maybe_rename_category(match.group("category"), obj_model)
        obj_inst_id = match.group("instance_id")
        link_name = match.group("link_name")
        parent_link_name = match.group("parent_link_name")
        joint_type = match.group("joint_type")
        joint_side = match.group("joint_side")
        tags_str = match.group("tag")

        if obj_model in DELETION_QUEUE:
            continue

        if joint_side == "upper" and not load_upper:
            continue

        if is_broken and not load_bad:
            continue

        if int(obj_inst_id) != 0 and not load_nonzero:
            continue

        link_name = "base_link" if link_name is None else link_name

        tags = []
        if tags_str:
            tags = sorted([x[1:] for x in tags_str.split("-") if x])

        node_key = (obj_cat, obj_model, obj_inst_id, link_name)

        if filter_nodes is not None and node_key not in filter_nodes:
            continue

        if node_key not in G.nodes:
            G.add_node(node_key)
        G.nodes[node_key]["is_broken"] = is_broken
        G.nodes[node_key]["is_loose"] = is_loose
        G.nodes[node_key]["is_randomization_fixed"] = is_randomization_fixed
        G.nodes[node_key]["tags"] = tags

        should_load_data = not is_broken and int(obj_inst_id) == 0

        if should_load_data:
            # Get the path for the mesh
            mesh_dir = mesh_fs.opendir(mesh_name)
            mesh_fn = f"{mesh_name}.obj"
            with mesh_dir.open(f"{mesh_name}.json", "r") as metadata_file:
                metadata = json.load(metadata_file)

            # Rename parts
            renamed_parts = []
            for part_name in metadata["parts"]:
                part_name_parsed = parse_name(part_name)
                part_cat = part_name_parsed.group("category")
                part_model = part_name_parsed.group("model_id")
                part_name_renamed = part_name.replace(
                    part_cat, maybe_rename_category(part_cat, part_model)
                )
                renamed_parts.append(part_name_renamed)
            metadata["parts"] = renamed_parts

            # Grab orientation from metadata
            canonical_orientation = metadata["orientation"]
            del metadata["orientation"]

            # Grab meta links from metadata and delete original to avoid confusion
            meta_links = metadata["meta_links"]
            del metadata["meta_links"]

            # Apply the scaling factor.
            for meta_type, meta_link_id_to_subid in meta_links.items():
                for meta_link_subid_to_link in meta_link_id_to_subid.values():
                    for meta_link in meta_link_subid_to_link:
                        meta_link["position"] = (
                            np.array(meta_link["position"]) * SCALE_FACTOR
                        )

                        # TODO: Remove this after it's fixed in export_meshes
                        # Fix inverted meta link orientations
                        meta_link["orientation"] = (
                            R.from_quat(meta_link["orientation"])
                            .inv()
                            .as_quat()
                            .tolist()
                        )

                        if "length" in meta_link:
                            meta_link["length"] *= SCALE_FACTOR
                        if "width" in meta_link:
                            meta_link["width"] *= SCALE_FACTOR
                        if "size" in meta_link:
                            meta_link["size"] = (
                                np.asarray(meta_link["size"]) * SCALE_FACTOR
                            ).tolist()

                            # Fix any negative sizes.
                            z_negative = meta_link["size"][2] < 0
                            meta_link["size"] = np.abs(meta_link["size"]).tolist()
                            if z_negative and meta_link["type"] in (
                                "box",
                                "cylinder",
                                "cone",
                            ):
                                # These objects are not symmetrical around the Z axis & need to be rotated
                                new_orientation = R.from_quat(
                                    meta_link["orientation"]
                                ) * R.from_euler("x", np.pi)
                                meta_link["orientation"] = new_orientation.as_quat()

                        # TODO: Remove this once it is moved to a better place
                        # Apply the meta link scaling rules here
                        if meta_type == "particleapplier":
                            coefficient = (
                                PARTICLE_APPLIER_CONE_LENGTH / meta_link["size"][2]
                            )

                            if coefficient > 1:
                                meta_link["size"] = (
                                    np.asarray(meta_link["size"]) * coefficient
                                ).tolist()

            # Add the data for the position onto the node.
            if joint_side == "upper":
                assert (
                    "upper_points" not in G.nodes[node_key]
                ), f"Found two upper meshes for {node_key}"
                if load_meshes:
                    upper_points = load_points(mesh_dir, mesh_fn)
                    G.nodes[node_key]["upper_points"] = (
                        trimesh.transformations.transform_points(
                            upper_points, SCALE_MATRIX
                        )
                    )
            else:
                G.nodes[node_key]["metadata"] = metadata
                G.nodes[node_key]["meta_links"] = meta_links
                G.nodes[node_key]["canonical_orientation"] = canonical_orientation

                if load_meshes:
                    assert (
                        "lower_mesh" not in G.nodes[node_key]
                    ), f"Found two lower meshes for {node_key}"
                    lower_mesh = load_mesh(
                        mesh_dir, mesh_fn, process=False, force="mesh"
                    )
                    lower_mesh.apply_transform(SCALE_MATRIX)
                    G.nodes[node_key]["lower_mesh"] = lower_mesh

                    lower_points = load_points(mesh_dir, mesh_fn)
                    G.nodes[node_key]["lower_points"] = (
                        trimesh.transformations.transform_points(
                            lower_points, SCALE_MATRIX
                        )
                    )

                    # Load the texture map paths and convert them to absolute paths
                    G.nodes[node_key]["material_maps"] = {}
                    bakery_fs = pipeline_fs.target(target).opendir("bakery")
                    for channel, path_rel_to_bakery in metadata[
                        "material_maps"
                    ].items():
                        # TODO: Remove this once it is corrected.
                        if channel == "Transparency Color Map":
                            channel = "Transparency Map"
                        G.nodes[node_key]["material_maps"][channel] = (
                            bakery_fs.getsyspath(path_rel_to_bakery)
                        )

                    # Load convexmesh meta links
                    for cm_type in CONVEX_MESH_TYPES:
                        # Check if a collision mesh exist in the same directory
                        pattern_str = (
                            r"^.*-M" + cm_type + r"(?:_[A-Za-z0-9]+)?-(\d+).obj$"
                        )
                        selection_matching_pattern = re.compile(pattern_str)
                        cm_filenames = [
                            x
                            for x in mesh_dir.listdir("/")
                            if selection_matching_pattern.fullmatch(x)
                        ]

                        if cm_filenames:
                            # Match the files
                            selection_matches = [
                                (selection_matching_pattern.fullmatch(x), x)
                                for x in cm_filenames
                            ]
                            indexed_matches = {
                                int(match.group(1)): fn
                                for match, fn in selection_matches
                                if match
                            }
                            expected_keys = set(range(len(indexed_matches)))
                            found_keys = set(indexed_matches.keys())
                            assert (
                                expected_keys == found_keys
                            ), f"Missing {cm_type} meshes for {node_key}: {expected_keys - found_keys}"
                            ordered_cm_filenames = [
                                indexed_matches[i] for i in range(len(indexed_matches))
                            ]

                            convex_meshes = []
                            for convex_mesh_filename in ordered_cm_filenames:
                                convex_mesh = load_mesh(
                                    mesh_dir,
                                    convex_mesh_filename,
                                    force="mesh",
                                    process=False,
                                    skip_materials=True,
                                )
                                convex_mesh.apply_transform(SCALE_MATRIX)
                                convex_meshes.append(convex_mesh)
                            G.nodes[node_key][f"{cm_type}_mesh"] = convex_meshes

        # Add the edge in from the parent
        if link_name != "base_link":
            assert (
                parent_link_name
            ), f"Non-base_link {node_key} should have a parent link name."
            parent_key = (obj_cat, obj_model, obj_inst_id, parent_link_name)
            G.add_edge(parent_key, node_key, joint_type=joint_type)
            if "is_loose" not in G.nodes[parent_key]:
                G.nodes[parent_key]["is_loose"] = None

    # Quick validation.
    for node, data in G.nodes(data=True):
        should_load_data = not data["is_broken"] and int(node[2]) == 0
        needs_upper = False
        if node[-1] != "base_link":
            assert len(G.in_edges(node)) == 1, node
            ((_, _, d),) = G.in_edges(node, data=True)
            joint_type = d["joint_type"]
            needs_upper = load_upper and not data["is_broken"] and joint_type != "F"
        assert (
            not should_load_data
            or not load_meshes
            or not needs_upper
            or "upper_points" in data
        ), f"{node} does not have upper mesh."
        assert (
            not should_load_data or not load_meshes or "lower_mesh" in data
        ), f"{node} does not have lower mesh."
        assert (
            not should_load_data or not load_meshes or "lower_points" in data
        ), f"{node} does not have lower mesh."
        assert (
            not should_load_data or "metadata" in data
        ), f"{node} does not have metadata."

        if "upper_points" in data:
            lower_vertices = len(data["lower_points"])
            upper_vertices = len(data["upper_points"])
            assert (
                lower_vertices == upper_vertices
            ), f"{node} lower mesh has {lower_vertices} while upper mesh has {upper_vertices}. They should be equal."

        if node[3] == "base_link":
            assert (
                G.in_degree(node) == 0
            ), f"Base_link node {node} should have in degree 0."
        else:
            assert (
                G.in_degree(node) != 0
            ), f"Non-base_link node {node} should not have in degree 0."

    # Find roots to look them up in bounding boxes.
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    # Assert that the roots keys are a subset of the bounding boxes keys.
    roots_to_model_and_instance = sorted([(node[1], node[2]) for node in roots])
    roots_to_model_and_instance_set = set(roots_to_model_and_instance)
    assert len(roots_to_model_and_instance) == len(
        roots_to_model_and_instance_set
    ), f"Duplicate root nodes found in {roots_to_model_and_instance}"
    bbox_keys = set(
        [
            (model_id, instance_id)
            for model_id, instances in object_bounding_boxes.items()
            for instance_id in instances
        ]
    )
    assert roots_to_model_and_instance_set.issubset(
        bbox_keys
    ), f"Root nodes do not match the bounding boxes. Roots: {roots_to_model_and_instance}, BBoxes: {bbox_keys}"

    for root in roots:
        # First find the object bounding box.
        G.nodes[root]["object_bounding_box"] = object_bounding_boxes[root[1]][root[2]]

        # Check that the bounding box orientation is the same as the canonical orientation when one is available.
        bbox_orientation = G.nodes[root]["object_bounding_box"]["rotation"]
        bbox_orientation = R.from_quat(bbox_orientation)
        if "canonical_orientation" in G.nodes[root]:
            canonical_orientation = G.nodes[root]["canonical_orientation"]
            canonical_orientation = R.from_quat(canonical_orientation)
            delta_orientation = bbox_orientation.inv() * canonical_orientation
            assert (
                delta_orientation.magnitude() < 1e-5
            ), f"Root node {root} has a bounding box with orientation {bbox_orientation.as_quat()} that does not match the canonical orientation {canonical_orientation.as_quat()}."

        # Create combined mesh for each root node and add some data.
        if load_meshes:
            nodes = list(nx.dfs_preorder_nodes(G, root))
            meshes = [
                G.nodes[node]["lower_mesh"]
                for node in nodes
                if "lower_mesh" in G.nodes[node]
            ]
            combined_mesh = trimesh.util.concatenate(meshes)
            G.nodes[root]["combined_mesh"] = combined_mesh

            for cm_type in CONVEX_MESH_TYPES:
                if all(f"{cm_type}_mesh" in G.nodes[node] for node in nodes):
                    convex_meshes = [
                        cm for node in nodes for cm in G.nodes[node][f"{cm_type}_mesh"]
                    ]
                    combined_convex_mesh = trimesh.util.concatenate(convex_meshes)
                    G.nodes[root][f"combined_{cm_type}_mesh"] = combined_convex_mesh

    return G
