from collections import defaultdict
import csv
import os
import pathlib

import numpy as np
from PIL import Image
import tqdm

PIPELINE_ROOT = pathlib.Path(__file__).parents[2]

RESOLUTION = 0.01
Z_START = 2.0  # Just above the typical robot height
Z_END = -0.1  # Just below the floor
HALF_Z = (Z_START + Z_END) / 2.0
HALF_HEIGHT = (Z_START - Z_END) / 2.0

WALL_CATEGORIES = ["walls", "rail_fence"]
FLOOR_CATEGORIES = ["floors", "driveway", "lawn"]
DOOR_CATEGORIES = ["door", "sliding_door", "garage_door", "gate"]
IGNORE_CATEGORIES = ["carpet"]
NEEDED_STRUCTURE_CATEGORIES = FLOOR_CATEGORIES + WALL_CATEGORIES

# Segmentation maps will be generated with the data from the below map's overlap query
GENERATE_SEG_MAPS_DURING_FNAME = "floor_trav_no_obj_0.png"

# Segmentation maps will be saved with the below filenames even though they don't have their
# own passes.
SEMSEG_MAP_FNAME = "floor_semseg_0.png"
INSSEG_MAP_FNAME = "floor_insseg_0.png"


MAP_GENERATION_PASSES = [
    # Each outer level list is a single pass of the overlap query. Multiple maps can be generated
    # in a single pass if they do not involve modifying the scene.
    # (filename, load, not_load)
    [
        ("floor_trav_no_obj_0.png", NEEDED_STRUCTURE_CATEGORIES, None),
        ("floor_trav_0.png", None, IGNORE_CATEGORIES),
        ("floor_trav_no_door_0.png", None, DOOR_CATEGORIES + IGNORE_CATEGORIES),
    ],
    # This one must be last since it modifies the scene and leaves it as such.
    [
        ("floor_trav_open_door_0.png", None, IGNORE_CATEGORIES),
    ],
]


def map_to_world(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in map reference frame into world (simulator) reference frame

    Args:
        xy (2-array or (N, 2)-array): 2D location(s) in map reference frame (in image pixel space)

    Returns:
        2-array or (N, 2)-array: 2D location(s) in world reference frame (in metric space)
    """
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip((xy - trav_map_size / 2.0) * trav_map_resolution, axis=axis)


def generate_maps_for_current_scene(save_path):
    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.object_states as object_states

    # Create the output directory
    os.makedirs(save_path, exist_ok=True)

    # Get the room type to room id mapping
    with open(PIPELINE_ROOT / "metadata/allowed_room_types.csv") as f:
        sem_to_id = {
            row["Room Name"].strip(): i + 1 for i, row in enumerate(csv.DictReader(f))
        }

    # Compute the map dimensions by finding the AABB of all objects and calculating max distance from origin.
    floor_objs = {
        floor
        for floor_cat in FLOOR_CATEGORIES
        for floor in og.sim.scenes[0].object_registry("category", floor_cat, [])
    }
    roomless_floor_objs = [
        (floor, len(floor.in_rooms)) for floor in floor_objs if len(floor.in_rooms) != 1
    ]
    assert not roomless_floor_objs, f"Found {len(roomless_floor_objs)} floor objects without exactly one room: {roomless_floor_objs}"
    aabb_corners = np.concatenate([floor.aabb for floor in floor_objs], axis=0)
    combined_low = np.min(list(aabb_corners), axis=0)
    combined_high = np.max(list(aabb_corners), axis=0)
    combined_aabb = np.array([combined_low, combined_high])
    aabb_dist_from_zero = np.abs(combined_aabb)
    dist_from_center = np.max(aabb_dist_from_zero)
    map_size_in_meters = dist_from_center * 2
    map_size_in_pixels = map_size_in_meters / RESOLUTION
    map_size_in_pixels = (
        int(np.ceil(map_size_in_pixels / 2) * 2) + 2
    )  # Round to nearest multiple of 2

    # Get the bounds of the part of the map that we will actually cast rays for (e.g. the occupied section)
    world_to_map_float = lambda xy: np.flip(
        (np.array(xy) / RESOLUTION + map_size_in_pixels / 2.0)
    )

    row_min, col_min = np.floor(world_to_map_float(combined_aabb[0][:2])).astype(int)
    row_max, col_max = np.ceil(world_to_map_float(combined_aabb[1][:2])).astype(int)

    # Assert that all the dimensions are within the map
    assert (
        row_min >= 0 and row_max < map_size_in_pixels
    ), f"Map row bounds: {row_min}, {row_max} vs {map_size_in_pixels}"
    assert (
        col_min >= 0 and col_max < map_size_in_pixels
    ), f"Map column bounds: {col_min}, {col_max} vs {map_size_in_pixels}"

    row_extent = row_max - row_min + 1
    col_extent = col_max - col_min + 1
    total_cells = row_extent * col_extent

    for pass_idx, map_pass in enumerate(MAP_GENERATION_PASSES):
        # Get a list of all of the room instances in the scene
        all_insts = {
            room
            for floor in og.sim.scenes[0].objects
            for room in (floor.in_rooms if floor.in_rooms else [])
        }
        sorted_all_insts = sorted(all_insts)

        # Map those rooms into a contiguous range of integers starting from 1
        inst_to_id = {inst: i + 1 for i, inst in enumerate(sorted_all_insts)}

        # Move the doors to the open position if necessary
        if map_pass[0][0] == "floor_trav_open_door_0.png":
            for door_cat in DOOR_CATEGORIES:
                for door in og.sim.scenes[0].object_registry("category", door_cat, []):
                    if object_states.Open not in door.states:
                        continue
                    door.states[object_states.Open].set_value(True, fully=True)

            og.sim.step()

        allowed_hit_paths_by_fname = {}
        for fname, load_categories, not_load_categories in map_pass:
            # Using the load/not load params, build the set of allowed hits
            allowed_hit_paths_for_fname = {
                link.prim_path: obj
                for obj in og.sim.scenes[0].objects
                for link in obj.links.values()
                if not load_categories or obj.category in load_categories
            }
            if not_load_categories:
                for obj in og.sim.scenes[0].objects:
                    for link in obj.links.values():
                        if obj.category in not_load_categories:
                            allowed_hit_paths_for_fname.pop(link.prim_path, None)

            # Add the allowed hit paths to the dictionary
            allowed_hit_paths_by_fname[fname] = allowed_hit_paths_for_fname

        # Prepare the arrays for the maps
        map_fnames = {fname for fname, _, _ in map_pass}
        if GENERATE_SEG_MAPS_DURING_FNAME in map_fnames:
            map_fnames.add(SEMSEG_MAP_FNAME)
            map_fnames.add(INSSEG_MAP_FNAME)
        map_arrays = {
            fname: np.zeros((map_size_in_pixels, map_size_in_pixels), dtype=np.uint8)
            for fname in map_fnames
        }

        # Do the actual ray casting (actually an overlap query). We make a single pass for each
        # map pass, relying on the callback to filter out the hits we don't want for each map file.
        with tqdm.tqdm(
            total=total_cells, desc=f"Overlap grid for pass {pass_idx}"
        ) as pbar:
            for row in range(row_min, row_max + 1):
                for col in range(col_min, col_max + 1):
                    world_pos = map_to_world(
                        np.array([row, col]), RESOLUTION, map_size_in_pixels
                    )

                    hit_objs_by_fname = {fname: set() for fname in map_fnames}

                    def _check_hit(hit):
                        for (
                            fname,
                            allowed_hit_paths_for_fname,
                        ) in allowed_hit_paths_by_fname.items():
                            if hit.rigid_body in allowed_hit_paths_for_fname:
                                hit_objs_by_fname[fname].add(
                                    allowed_hit_paths_for_fname[hit.rigid_body]
                                )

                        return True

                    # Run the actual overlap query
                    og.sim.psqi.overlap_box(
                        halfExtent=np.array(
                            [RESOLUTION / 2, RESOLUTION / 2, HALF_HEIGHT]
                        ),
                        pos=np.array([world_pos[0], world_pos[1], HALF_Z]),
                        rot=np.array([0, 0, 0, 1.0]),
                        reportFn=_check_hit,
                    )

                    # Use the results from the hit_objs_by_fname to fill in the map arrays
                    for fname, load_categories, not_load_categories in map_pass:
                        # Get the hit object set for this map
                        hit_objs = hit_objs_by_fname[fname]

                        # Check whether or not we only hit a floor
                        only_hit_floor = int(hit_objs.issubset(floor_objs))

                        # Assign the reshaped array to the scannable map
                        map_arrays[fname][row, col] = only_hit_floor * 255

                        # At the same time as the no-obj trav map, we generate the segmentation maps.
                        if fname == GENERATE_SEG_MAPS_DURING_FNAME:
                            # Color the instance segmentation map using the hit object's color
                            first_hit_floor = (
                                next(
                                    iter(
                                        sorted(
                                            hit_objs & floor_objs, key=lambda x: x.name
                                        )
                                    ),
                                    None,
                                )
                                if hit_objs
                                else None
                            )
                            hit_room_inst_name = (
                                first_hit_floor.in_rooms[0]
                                if first_hit_floor and first_hit_floor.in_rooms
                                else None
                            )
                            insseg_val = (
                                inst_to_id[hit_room_inst_name]
                                if hit_room_inst_name
                                else 0
                            )
                            map_arrays[INSSEG_MAP_FNAME][row, col] = insseg_val

                            # Now the same for the semseg map
                            hit_room_type = (
                                hit_room_inst_name.rsplit("_", 1)[0]
                                if hit_room_inst_name
                                else None
                            )
                            semseg_val = (
                                sem_to_id[hit_room_type] if hit_room_type else 0
                            )
                            map_arrays[SEMSEG_MAP_FNAME][row, col] = semseg_val

                    # Update the progress bar
                    pbar.update(1)

        # Save the maps
        for fname, map_array in map_arrays.items():
            # Save the map as a PNG
            full_fname = os.path.join(save_path, fname)
            Image.fromarray(map_array).save(full_fname)
