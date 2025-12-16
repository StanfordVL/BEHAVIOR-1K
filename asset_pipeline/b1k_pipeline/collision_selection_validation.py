import collections
from concurrent import futures
import traceback
import numpy as np
import tqdm
import json

from b1k_pipeline import mesh_tree
from b1k_pipeline.utils import PipelineFS, get_targets, parse_name

NUKE_SELECTIONS = False


def get_aabb_corners(mesh):
    aabb = mesh.bounding_box
    center = aabb.transform[:3, 3]
    half_extent = aabb.extents / 2
    return np.array([center - half_extent, center + half_extent])


def compare_aabbs(lower_mesh, collision_meshes):
    # Get the AABB of the lower mesh
    lower_aabb = get_aabb_corners(lower_mesh)
    lower_min = np.min(lower_aabb, axis=0)
    lower_max = np.max(lower_aabb, axis=0)

    # Get the AABB of the collision meshes
    collision_aabbs = np.concatenate(
        [get_aabb_corners(mesh) for mesh in collision_meshes], axis=0
    )
    collision_min = np.min(collision_aabbs, axis=0)
    collision_max = np.max(collision_aabbs, axis=0)

    # Check if the corners are close enough
    min_diff = np.linalg.norm(lower_min - collision_min)
    max_diff = np.linalg.norm(lower_max - collision_max)
    if min_diff > 2:
        return f"Lower mesh AABB min is too far ({min_diff}m) away from collision mesh AABB min."
    elif max_diff > 2:
        return f"Lower mesh AABB max is too far ({max_diff}m) away from collision mesh AABB max."

    return None


def process_target(target):
    try:
        with PipelineFS() as pipeline_fs:
            errors = collections.defaultdict(list)

            with pipeline_fs.target_output(target).open(
                "collision_selection.json", "r"
            ) as f:
                collision_selection = json.load(f)

            selected = {}
            for obj in collision_selection:
                m = parse_name(obj)
                model = m.group("model_id")
                link = m.group("link_name")
                if not link:
                    link = "base_link"
                if (model, link) in selected:
                    errors[obj].append(
                        "Duplicate collision selection. Seen before as "
                        + selected[(model, link)]
                        + "."
                    )
                selected[(model, link)] = obj

            # Get the mesh tree
            G = mesh_tree.build_mesh_tree(
                target, load_upper=False, load_bad=False, load_nonzero=False
            )

            # Check that each object matches at least one of the options
            for node in G.nodes:
                # If the object has a collision mesh we are probably good.
                if "collision_mesh" in G.nodes[node]:
                    # Validate the mesh
                    splits = G.nodes[node]["collision_mesh"]
                    if len(splits) == 0:
                        errors[node].append(
                            "Collision mesh was found but contains no meshes."
                        )
                    elif len(splits) > 32:
                        errors[node].append(
                            "Collision mesh was found but contains too many meshes: {len(splits)}."
                        )
                    elif any(not split.is_watertight for split in splits):
                        errors[node].append(
                            "Collision mesh was found but contains non-watertight meshes."
                        )
                    elif any(
                        not split.is_volume or split.volume <= 0 for split in splits
                    ):
                        errors[node].append(
                            "Collision mesh was found but contains zero volume meshes."
                        )
                    elif any(
                        np.any(split.bounding_box.extents == 0) for split in splits
                    ):
                        errors[node].append(
                            "Collision mesh was found but contains zero bbox dimension meshes."
                        )
                    else:
                        aabb_error = compare_aabbs(
                            G.nodes[node]["lower_mesh"], G.nodes[node]["collision_mesh"]
                        )
                        if aabb_error and node[0] not in [
                            "floors",
                            "ceilings",
                            "walls",
                        ]:
                            errors[node].append(aabb_error)

                    # Identify manual collision mesh in error
                    if "manual_collision_filename" in G.nodes[node] and node in errors:
                        errors[node] = ["Manual " + x for x in errors[node]]
                    # If we reach here, no errors!
                elif "manual_collision_filename" in G.nodes[node]:
                    # The object has a manual collision mesh but it was not loaded
                    errors[node].append(
                        "Manual collision mesh was found but could not be loaded."
                    )
                elif "collision_selection" not in G.nodes[node]:
                    # No collision selection was made
                    errors[node].append("No collision selection was made.")
                elif (
                    "collision_options" not in G.nodes[node]
                    or not G.nodes[node]["collision_options"]
                ):
                    errors[node].append("No collision options were available.")
                elif (
                    G.nodes[node]["collision_selection"]
                    not in G.nodes[node]["collision_options"]
                ):
                    errors[node].append(
                        "Collision selection was not in the available options."
                    )
                else:
                    errors[node].append("Something unexpected happened.")

            return errors
    except Exception as e:
        return {"Exception": str(e) + "\n" + traceback.format_exc()}


def main():
    errors = {}
    target_futures = {}
    targets = get_targets("combined")

    with futures.ProcessPoolExecutor(max_workers=16) as executor:
        for target in tqdm.tqdm(targets):
            target_futures[executor.submit(process_target, target)] = target

        with tqdm.tqdm(total=len(target_futures)) as pbar:
            for future in futures.as_completed(target_futures.keys()):
                target = target_futures[future]
                target_errors = future.result()
                if target_errors:
                    errors[target] = {k: v for k, v in target_errors.items()}

                pbar.update(1)

    if errors:
        print("Errors found:")
        for target in errors:
            print("\n", target)
            for node, msg in errors[target].items():
                print("  ", node, msg)

    with PipelineFS() as pipeline_fs:
        if NUKE_SELECTIONS:
            for target in errors:
                # Load the collision selections
                with pipeline_fs.target_output(target).open(
                    "collision_selection.json", "r"
                ) as f:
                    target_selections = json.load(f)

                # Get model-link pairs that have errors
                error_keys = {
                    (model, link)
                    for (_, model, _, link), error_list in errors[target].items()
                    if not all("Manual" in error for error in error_list)
                    # if error in (
                    #     "Collision selection was not in the available options.",
                    # )
                }

                # Build a new collision selection without the errors
                new_selections = {}
                for key, selection in target_selections.items():
                    # Parse the key to get model ID and link name
                    parsed_name = parse_name(key)
                    model_id = parsed_name.group("model_id")
                    link_name = parsed_name.group("link_name")
                    if not link_name:
                        link_name = "base_link"
                    if (model_id, link_name) not in error_keys:
                        new_selections[key] = selection
                    else:
                        print("Removing", key)

                # Save the collision selection
                with pipeline_fs.target_output(target).open(
                    "collision_selection.json", "w"
                ) as f:
                    json.dump(new_selections, f, indent=4)

        with pipeline_fs.pipeline_output().open(
            "collision_selection_validation.json", "w"
        ) as f:
            printable_errors = {
                k: {str(k2): v2 for k2, v2 in v.items()} for k, v in errors.items()
            }
            json.dump({"success": not errors, "errors": printable_errors}, f, indent=4)


if __name__ == "__main__":
    main()
