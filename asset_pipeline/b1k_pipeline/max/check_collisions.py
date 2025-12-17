from collections import defaultdict
import json
import pathlib
import sys
import traceback

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import mat2arr, parse_name, PIPELINE_ROOT

import networkx as nx
import numpy as np
import tqdm
import trimesh

MINIMUM_COLLISION_DEPTH_METERS = 0.01  # 1cm of collision
MINIMUM_COLLISION_DEPTH_METERS_SENSITIVE = 0  # any collision

inventory_path = PIPELINE_ROOT / "artifacts" / "pipeline" / "object_inventory.json"
with open(inventory_path, "r") as f:
    providers = {k.split("-")[-1]: v for k, v in json.load(f)["providers"].items()}


def get_collision_meshes_relative_to_base(obj, base_transform):
    # Assert that collision meshes do not share instances in the scene
    assert not [
        x for x in rt.objects if x.baseObject == obj.baseObject and x != obj
    ], f"{obj.name} should not have instances."

    world_to_base = rt.inverse(base_transform)

    # Get vertices and faces into numpy arrays for conversion
    verts = np.array(
        [
            rt.polyop.getVert(obj, i + 1) * world_to_base
            for i in range(rt.polyop.GetNumVerts(obj))
        ]
    )
    faces = (
        np.array(
            rt.polyop.getFacesVerts(
                obj, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
            )
        )
        - 1
    )
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"

    # Split the faces into elements
    elems = {
        tuple(rt.polyop.GetElementsUsingFace(obj, i + 1))
        for i in range(rt.polyop.GetNumFaces(obj))
    }
    assert len(elems) <= 40, f"{obj.name} should not have more than 32 elements."
    elems = np.array(list(elems))
    assert not np.any(
        np.sum(elems, axis=0) > 1
    ), f"{obj.name} has same face appear in multiple elements"

    # Iterate through the elements
    meshes = []
    for i, elem in enumerate(elems):
        # Load the mesh into trimesh and assert convexity
        relevant_faces = faces[elem]
        m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
        m.remove_unreferenced_vertices()
        assert m.is_volume, f"{obj.name} element {i} is not a volume"
        # assert m.is_convex, f"{obj.name} element {i} is not convex"
        assert (
            len(m.split()) == 1
        ), f"{obj.name} element {i} has elements trimesh still finds splittable"

        # Add the mesh to the list
        meshes.append(m)

    return meshes


def import_bad_model_originals(model_id):
    # Get the provider file
    provider = providers[model_id]
    filename = str(PIPELINE_ROOT / "cad" / provider / "processed.max")

    # Get the right object names from the file
    f_objects = rt.getMAXFileObjectNames(filename, quiet=True)
    visual_objects = {}
    collision_objects = {}
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
        if "Mcollision" in obj:
            collision_objects[link_name] = obj
        else:
            visual_objects[link_name] = obj

    assert (
        set(visual_objects.keys()) == set(collision_objects.keys())
    ), f"Visual and collision objects should match in source for {model_id}. Currently: {visual_objects.keys()} vs {collision_objects.keys()}"

    objects_to_import = sorted(
        set(visual_objects.values()) | set(collision_objects.values())
    )

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

    # Make sure the objects all have the right parents
    for link_name in visual_objects.keys():
        visual_obj = imported_objs_by_name[visual_objects[link_name]]
        collision_obj = imported_objs_by_name[collision_objects[link_name]]
        assert visual_obj.parent == None, "Visual objects should not have parents"
        assert (
            collision_obj.parent == visual_obj
        ), "Collision objects should be children of visual objects"

    return imported_meshes


def prepare_scene(use_clutter=False):
    # Get all the collision meshes in the scene that belong to lower, instance-zero meshes
    scene = trimesh.Scene()

    # Get a list of all exportable objects
    all_objects = list(rt.objects)

    # Pre-import meshes for all bad objects
    bad_object_model_ids = set()
    for obj in all_objects:
        parsed_name = parse_name(obj.name)
        if not parsed_name or not parsed_name.group("bad"):
            continue
        bad_object_model_ids.add(parsed_name.group("model_id"))
    imported_bad_models = []
    for model_id in tqdm.tqdm(bad_object_model_ids, desc="Importing bad models"):
        imported_bad_models.extend(import_bad_model_originals(model_id))

    # Get the base link frame for each model
    base_link_frames = {}
    categories_by_model_id = {}
    for obj in tqdm.tqdm(all_objects, desc="Getting base link frames"):
        parsed_name = parse_name(obj.name)
        if not parsed_name:
            continue

        if parsed_name.group("model_id") not in categories_by_model_id:
            categories_by_model_id[parsed_name.group("model_id")] = parsed_name.group(
                "category"
            )
        assert (
            parsed_name.group("category")
            == categories_by_model_id[parsed_name.group("model_id")]
        ), f"{parsed_name.group('model_id')} has different categories: {parsed_name.group('category')} vs {categories_by_model_id[parsed_name.group('model_id')]}."

        if int(parsed_name.group("instance_id")) != 0:
            continue

        if parsed_name.group("bad"):
            continue

        if (
            parsed_name.group("link_name")
            and parsed_name.group("link_name") != "base_link"
        ):
            continue

        if parsed_name.group("meta_type"):
            continue

        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        model_id = parsed_name.group("model_id")
        assert model_id not in base_link_frames, f"{model_id} already exists"
        base_link_frames[model_id] = obj.transform

    # Process the collision meshes into a base-relative dictionary
    meshes = defaultdict(dict)  # meshes[model_id][link_name] = [trimesh.Trimesh]
    collision_objs = [x for x in all_objects if "Mcollision" in x.name]
    for obj in tqdm.tqdm(collision_objs, desc="Processing collision meshes"):
        if "Mcollision" not in obj.name:
            continue

        # Get some information about the parent
        parent = obj.parent
        if parent is None:
            continue

        parsed_name = parse_name(parent.name)
        if not parsed_name:
            continue

        if parsed_name.group("bad") or int(parsed_name.group("instance_id")) != 0:
            continue

        # Record the meshes
        link_name = (
            parsed_name.group("link_name")
            if parsed_name.group("link_name")
            else "base_link"
        )
        model_id = parsed_name.group("model_id")
        assert (
            link_name not in meshes[model_id]
        ), f"{link_name} already exists in {model_id}"
        meshes[model_id][link_name] = get_collision_meshes_relative_to_base(
            obj, base_link_frames[model_id]
        )

    # Remove any bad models that were imported
    for obj in imported_bad_models:
        rt.delete(obj)

    # Then go through every other object in the scene and add it onto the physics scene
    for obj in tqdm.tqdm(all_objects, desc="Building physics scene"):
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        parsed_name = parse_name(obj.name)
        if not parsed_name:
            continue

        if "Mcollision" in obj.name:
            continue

        # For now, skip clutter
        loose_key = parsed_name.group("loose")
        if loose_key and "C" in loose_key and not use_clutter:
            continue

        # Only use the base link to get everything else's position
        if (
            parsed_name.group("link_name")
            and parsed_name.group("link_name") != "base_link"
        ):
            continue

        is_loose = "true" if loose_key else "false"
        model_id = parsed_name.group("model_id")
        category = categories_by_model_id[model_id]
        instance_id = parsed_name.group("instance_id")
        obj_transform = np.hstack([mat2arr(obj.transform), [[0], [0], [0], [1]]]).T

        # Actually add the meshes
        for link_name, link_meshes in meshes[model_id].items():
            for mesh_idx, link_mesh in enumerate(link_meshes):
                node_key = f"{category}-{model_id}-{instance_id}-{link_name}-{mesh_idx}-{is_loose}"
                scene.add_geometry(
                    link_mesh, node_name=node_key, transform=obj_transform
                )

    return scene


def check_collisions(scene):
    cm, _ = trimesh.collision.scene_to_collision(scene)
    _, collision_data = cm.in_collision_internal(return_data=True)

    # Then go through every object in the scene and find out which links can be reached from the base_link
    # using only fixed links
    graphs_by_model = defaultdict(nx.Graph)
    for obj in tqdm.tqdm(list(rt.objects), desc="Building link graphs"):
        parsed_name = parse_name(obj.name)
        if not parsed_name:
            continue

        if int(parsed_name.group("instance_id")) != 0:
            continue

        model_id = parsed_name.group("model_id")
        this_link = (
            parsed_name.group("link_name")
            if parsed_name.group("link_name")
            else "base_link"
        )
        graphs_by_model[model_id].add_node(this_link)

        parent_link = parsed_name.group("parent_link_name")
        joint_type = parsed_name.group("joint_type")
        if parent_link and joint_type == "F":
            graphs_by_model[model_id].add_edge(parent_link, this_link)

    links_fixed_to_base = {
        model_id: set(nx.dfs_tree(graph, "base_link").nodes)
        for model_id, graph in graphs_by_model.items()
    }

    pairs_collision = {}
    for collision in tqdm.tqdm(collision_data, desc="Filtering collisions"):
        left, right = sorted(
            tuple(collision.names)
        )  # Sorting here gives deterministic category ordering
        left_category, left_model, left_instance, left_link, left_body, left_loose = (
            left.split("-")
        )
        left_loose = left_loose == "true"
        (
            right_category,
            right_model,
            right_instance,
            right_link,
            right_body,
            right_loose,
        ) = right.split("-")
        right_loose = right_loose == "true"

        # Exclude self-collisions
        if left_model == right_model and left_instance == right_instance:
            continue

        # Exclude collisions between fixed links of two fixed objects
        left_fixed_link = left_link in links_fixed_to_base[left_model]
        right_fixed_link = right_link in links_fixed_to_base[right_model]
        if not left_loose and not right_loose:
            if left_fixed_link and right_fixed_link:
                continue

        # Exclude collisions between walls and sliding doors
        if left_category == "sliding_door" and right_category == "walls":
            continue

        # Decide what the depth limit is. We are especially sensitive to collisions between moving
        # links of fixed objects and other fixed objects, so we set a lower threshold for those.
        depth_threshold = MINIMUM_COLLISION_DEPTH_METERS
        is_left_moving_part_of_fixed_obj = not left_loose and not left_fixed_link
        is_right_moving_part_of_fixed_obj = not right_loose and not right_fixed_link
        if is_left_moving_part_of_fixed_obj or is_right_moving_part_of_fixed_obj:
            depth_threshold = MINIMUM_COLLISION_DEPTH_METERS_SENSITIVE

        # Print the collision
        depth = float(collision.depth)
        if depth < depth_threshold * 1000:
            continue

        pair = tuple(
            sorted(
                [
                    (left_category, left_model, left_instance),
                    (right_category, right_model, right_instance),
                ]
            )
        )
        if pair not in pairs_collision:
            pairs_collision[pair] = depth
        else:
            pairs_collision[pair] = float(np.maximum(pairs_collision[pair], depth))

    return [(left, right, depth) for (left, right), depth in pairs_collision.items()]


def main():
    opts = rt.maxops.mxsCmdLineArgs
    use_clutter = opts[rt.name("clutter")] == "true"

    scene = None
    error = None
    collisions = []
    try:
        scene = prepare_scene(use_clutter=use_clutter)
        scene.export(r"D:\physics.ply")
        collisions = check_collisions(scene)
        for left, right, depth in collisions:
            print(f"Collision between {left} and {right}: {depth} mm")
    except Exception:
        raise
    #     error = traceback.format_exc()
    #     traceback.print_exc()

    output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / (
        "check_collisions.json"
        if not use_clutter
        else "check_collisions_with_clutter.json"
    )
    results = {
        "success": not error and len(collisions) == 0,
        "collisions": collisions,
        "error": error,
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    return scene


if __name__ == "__main__":
    scene = main()
