import pathlib
import re
import traceback
import numpy as np
import json
from fs.zipfs import ZipFS

import pymxs

rt = pymxs.runtime

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import PipelineFS, get_targets, parse_name, load_mesh


def import_collision_mesh(obj, collision_selections, collision_mesh_fs):
    print("Importing collision mesh for", obj.name)

    if rt.classOf(obj) != rt.Editable_Poly:
        return

    parsed_name = parse_name(obj.name)
    if not parsed_name:
        return

    category = parsed_name.group("category")
    model_id = parsed_name.group("model_id")
    instance_id = parsed_name.group("instance_id")
    link_name = parsed_name.group("link_name")
    link_name = link_name if link_name else "base_link"
    node_key = (category, model_id, instance_id, link_name)

    # Does it already have a collision mesh? If so, move on.
    for child in obj.children:
        parsed_child_name = parse_name(child.name)
        if not parsed_child_name:
            continue

        # Skip parts etc.
        if parsed_child_name.group("mesh_basename") != parsed_name.group(
            "mesh_basename"
        ):
            continue

        if parsed_child_name.group("meta_type") == "collision":
            print("Collision mesh already exists for", obj.name, ", skipping.")
            return

    # Try to load a collision mesh selection
    collision_key = (model_id, link_name)
    if collision_key not in collision_selections:
        print("No collision selection found for", obj.name)
        return

    collision_selection = collision_selections[collision_key]
    print("Collision selection for", obj.name, "is", collision_selection)

    if not collision_mesh_fs.exists(obj.name):
        print("No collision mesh found for", obj.name)
        return

    collision_fs = collision_mesh_fs.opendir(obj.name)
    collision_filenames = collision_fs.listdir("/")
    selection_matching_pattern = re.compile(collision_selection + r"-(\d+).obj$")

    # Match the files
    if not collision_filenames:
        print("No collision meshes found for", obj.name)
        return

    selection_matches = [
        (selection_matching_pattern.fullmatch(x), x) for x in collision_filenames
    ]
    indexed_matches = {
        int(match.group(1)): fn for match, fn in selection_matches if match
    }
    expected_keys = set(range(len(indexed_matches)))
    found_keys = set(indexed_matches.keys())
    assert (
        expected_keys == found_keys
    ), f"Missing collision meshes for {node_key}: {expected_keys - found_keys}"
    ordered_collision_filenames = [
        indexed_matches[i] for i in range(len(indexed_matches))
    ]

    collision_meshes = []
    for collision_filename in ordered_collision_filenames:
        collision_mesh = load_mesh(
            collision_fs, collision_filename, force="mesh", skip_materials=True
        )
        if not collision_mesh.is_volume:
            collision_mesh = load_mesh(
                collision_fs,
                collision_filename,
                force="mesh",
                process=False,
                skip_materials=True,
            )
        collision_meshes.append(collision_mesh)

    # Get a flattened list of vertices and faces
    all_vertices = []
    all_faces = []
    for split in collision_meshes:
        vertices = [rt.Point3(*v.tolist()) for v in split.vertices]
        # Offsetting here by the past vertex count
        faces = [[v + len(all_vertices) + 1 for v in f.tolist()] for f in split.faces]
        all_vertices.extend(vertices)
        all_faces.extend(faces)

    # Create a new node for the collision mesh
    collision_obj = rt.Editable_Mesh()
    rt.ConvertToPoly(collision_obj)
    collision_obj.name = f"{parsed_name.group('mesh_basename')}-Mcollision"
    collision_obj.rotation = obj.rotation
    collision_obj.position = obj.position

    # Add the vertices
    for v in all_vertices:
        rt.polyop.createVert(collision_obj, v)

    # Add the faces
    for f in all_faces:
        rt.polyop.createPolygon(collision_obj, f)

    # Optionally set its wire color
    collision_obj.wirecolor = rt.yellow

    # Update the mesh to reflect changes
    rt.update(collision_obj)

    # Parent the mesh
    collision_obj.parent = obj

    # Check that the new element count is the same as the split count
    elems = {
        tuple(rt.polyop.GetElementsUsingFace(collision_obj, i + 1))
        for i in range(rt.polyop.GetNumFaces(collision_obj))
    }
    assert len(elems) == len(
        collision_meshes
    ), f"{obj.name} has different number of faces in collision mesh than in splits"
    elems = np.array(list(elems))
    assert not np.any(
        np.sum(elems, axis=0) > 1
    ), f"{obj.name} has same face appear in multiple elements"

    # Hide the mesh
    collision_obj.isHidden = True


def process_target(pipeline_fs, target):
    print("Processing", target)
    filename = str(pipeline_fs.target(target).getsyspath("processed.max"))
    assert rt.loadMaxFile(filename, useFileUnits=False, quiet=True)

    with pipeline_fs.target_output(target) as target_output_fs:
        # If there are no collision selections, move on
        if not target_output_fs.exists("collision_selection.json"):
            print("No collision selections found for", target)
            return

        # Load the collision selections
        with target_output_fs.open("collision_selection.json", "r") as f:
            mesh_to_collision = json.load(f)
            match_to_collision = {
                parse_name(k): v for k, v in mesh_to_collision.items()
            }
            collision_selections = {
                (
                    k.group("model_id"),
                    k.group("link_name") if k.group("link_name") else "base_link",
                ): v
                for k, v in match_to_collision.items()
                if k is not None
            }

        # Open the collision meshes ZIP
        with (
            target_output_fs.open("collision_meshes.zip", "rb") as f,
            ZipFS(f) as collision_mesh_fs,
        ):
            # Iterate over the objects in the scene
            for obj in rt.objects:
                import_collision_mesh(obj, collision_selections, collision_mesh_fs)

    rt.saveMaxFile(filename)


def main():
    with PipelineFS() as pipeline_fs:
        # current_target_dir = pathlib.Path(rt.maxFilePath)
        # current_target_rel = current_target_dir.relative_to(PIPELINE_ROOT)
        # cad, target_type, target_name = current_target_rel.parts
        # assert cad == "cad", f"Current file is not in the cad directory: {current_target_rel}"
        # assert target_type in ("scenes", "objects"), f"Current file is not in the scenes or objects directory: {current_target_rel}"
        # current_target = target_type + "/" + target_name

        for target in get_targets("combined"):
            process_target(pipeline_fs, target)


if __name__ == "__main__":
    main()
