"""
Script to regenerate segmentation maps for SPOC scenes using floor polygons.
This generates semantic and instance segmentation maps without using OmniGibson.
"""

import argparse
import csv
import hashlib
import pathlib
import traceback
import shutil
import os
from tqdm import tqdm
import json

import numpy as np
from PIL import Image
import shapely
from shapely.geometry import Point, Polygon

# Map generation parameters (matching make_maps.py)
RESOLUTION = 0.01  # 1cm per pixel
PIPELINE_ROOT = pathlib.Path(__file__).parents[1] / "asset_pipeline"

# Output filenames
SEMSEG_MAP_FNAME = "floor_semseg_0.png"
INSSEG_MAP_FNAME = "floor_insseg_0.png"


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


def world_to_map(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in world (simulator) reference frame into map reference frame

    Args:
        xy (2-array or (N, 2)-array): 2D location(s) in world reference frame (in metric space)

    Returns:
        2-array or (N, 2)-array: 2D location(s) in map reference frame (in image pixel space)
    """
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip(np.array(xy) / trav_map_resolution + trav_map_size / 2.0, axis=axis)


def extract_floor_polygons(scene_data):
    """
    Extract 2D floor polygons from a SPOC scene.

    Args:
        scene_data: Parsed JSON scene data

    Returns:
        list: List of (shapely_polygon, room_id, room_type) tuples
    """
    floor_polygons = []

    for i, room in enumerate(scene_data.get("rooms", [])):
        room_id = room.get("id", f"room_{i}")
        # Extract room type from room_id (e.g., "kitchen_0" -> "kitchen")
        room_type = room_id.rsplit("_", 1)[0] if "_" in room_id else room_id

        if "floorPolygon" in room:
            # SPOC uses x, z for horizontal plane (y is up)
            # Convert to our coordinate system: negate x to match OmniGibson convention
            points = np.array([[-pt["x"], pt["z"]] for pt in room["floorPolygon"]])

            if len(points) >= 3:
                try:
                    polygon = Polygon(points)
                    if polygon.is_valid:
                        floor_polygons.append((polygon, room_id, room_type))
                    else:
                        # Try to fix invalid polygon
                        polygon = polygon.buffer(0)
                        if polygon.is_valid and not polygon.is_empty:
                            floor_polygons.append((polygon, room_id, room_type))
                        else:
                            print(f"Invalid floor polygon for room {room_id}")
                except Exception as e:
                    print(f"Failed to create floor polygon for room {room_id}: {e}")

    return floor_polygons


def generate_segmentation_maps(scene_data, save_path, sem_to_id):
    """
    Generate semantic and instance segmentation maps for a SPOC scene.

    Args:
        scene_data: Parsed JSON scene data
        save_path: Directory to save the maps
        sem_to_id: Dictionary mapping room types to semantic IDs
    """
    os.makedirs(save_path, exist_ok=True)

    # Extract floor polygons
    floor_polygons = extract_floor_polygons(scene_data)

    if not floor_polygons:
        raise ValueError("No valid floor polygons found in scene")

    # Create instance ID mapping (contiguous IDs starting from 1)
    sorted_room_ids = sorted(set(room_id for _, room_id, _ in floor_polygons))
    inst_to_id = {room_id: i + 1 for i, room_id in enumerate(sorted_room_ids)}

    # Calculate combined AABB of all floor polygons
    all_bounds = np.array([poly.bounds for poly, _, _ in floor_polygons])
    # bounds format: (minx, miny, maxx, maxy)
    combined_min = np.min(all_bounds[:, :2], axis=0)
    combined_max = np.max(all_bounds[:, 2:], axis=0)

    # Calculate map size in pixels based on max distance from origin
    aabb_dist_from_zero = np.max(np.abs(np.array([combined_min, combined_max])))
    map_size_in_meters = aabb_dist_from_zero * 2
    map_size_in_pixels = map_size_in_meters / RESOLUTION
    map_size_in_pixels = int(np.ceil(map_size_in_pixels / 2) * 2) + 2  # Round to nearest even + 2

    # Get the bounds of the part of the map that we will actually process
    world_to_map_float = lambda xy: np.flip(
        (np.array(xy) / RESOLUTION + map_size_in_pixels / 2.0)
    )

    row_min, col_min = np.floor(world_to_map_float(combined_min)).astype(int)
    row_max, col_max = np.ceil(world_to_map_float(combined_max)).astype(int)

    # Clamp to map bounds
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(map_size_in_pixels - 1, row_max)
    col_max = min(map_size_in_pixels - 1, col_max)

    # Prepare the segmentation map arrays
    semseg_map = np.zeros((map_size_in_pixels, map_size_in_pixels), dtype=np.uint8)
    insseg_map = np.zeros((map_size_in_pixels, map_size_in_pixels), dtype=np.uint8)

    # Prepare shapely polygons with their IDs for faster lookup
    prepared_polygons = [
        (shapely.prepared.prep(poly), room_id, room_type)
        for poly, room_id, room_type in floor_polygons
    ]

    # Generate maps by checking each pixel
    row_extent = row_max - row_min + 1
    col_extent = col_max - col_min + 1
    total_cells = row_extent * col_extent

    with tqdm(total=total_cells, desc="Generating segmentation maps") as pbar:
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                # Convert pixel to world coordinates
                world_pos = map_to_world(
                    np.array([row, col]), RESOLUTION, map_size_in_pixels
                )

                # Check which polygon contains this point
                point = Point(world_pos[0], world_pos[1])

                for prep_poly, room_id, room_type in prepared_polygons:
                    if prep_poly.contains(point):
                        # Instance segmentation: room instance ID
                        insseg_map[row, col] = inst_to_id[room_id]

                        # Semantic segmentation: room type ID
                        if room_type in sem_to_id:
                            semseg_map[row, col] = sem_to_id[room_type]
                        break

                pbar.update(1)

    # Save the maps as PNGs
    Image.fromarray(semseg_map).save(os.path.join(save_path, SEMSEG_MAP_FNAME))
    Image.fromarray(insseg_map).save(os.path.join(save_path, INSSEG_MAP_FNAME))

    return map_size_in_pixels


def get_scene_id(split_path, index):
    """Generate a unique scene ID from split path and index."""
    split_name = pathlib.Path(split_path).stem
    return f"{split_name}_{index}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int, help="Task ID (0-indexed)")
    parser.add_argument("total_tasks", type=int, help="Total number of tasks")
    parser.add_argument("--success-prefix", default="", help="Prefix for success files (e.g., scriptname_jobid)")
    parser.add_argument("--spoc-houses-root", default="/checkpoint/clear/cgokmen/procthor/houses/houses_2023_07_28")
    parser.add_argument("--dataset-name", default="spoc", help="Dataset name (defaults to 'spoc')")
    args = parser.parse_args()

    # Setup paths
    output_root = pathlib.Path("/checkpoint/clear/cgokmen/test-structure")
    output_root.mkdir(exist_ok=True)

    errors_dir = output_root / "errors"
    errors_dir.mkdir(exist_ok=True)

    spoc_houses_root = pathlib.Path(args.spoc_houses_root)

    # Load room type to semantic ID mapping
    with open(PIPELINE_ROOT / "metadata/allowed_room_types.csv") as f:
        sem_to_id = {
            row["Room Name"].strip(): i + 1 for i, row in enumerate(csv.DictReader(f))
        }

    # Collect all scenes
    scenes = []
    for split_file in sorted(spoc_houses_root.glob("*.jsonl")):
        with open(split_file, "r") as f:
            for i, line in enumerate(f):
                scenes.append((str(split_file), i))

    # Sort with fixed salt for deterministic ordering across runs
    scenes.sort(key=lambda x: hashlib.md5((f"{x[0]}_{x[1]}potato").encode()).hexdigest())

    rank = args.task_id
    world_size = args.total_tasks

    # Get scenes for this task
    task_scenes = scenes[rank::world_size]

    for split_path, scene_idx in tqdm(task_scenes, desc="Processing scenes"):
        scene_id = get_scene_id(split_path, scene_idx)

        # Check if this scene is already done
        scene_dir = output_root / scene_id
        scene_success_file = scene_dir / "success"
        if scene_success_file.exists():
            continue

        if scene_dir.exists():
            shutil.rmtree(scene_dir)
        scene_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load the scene data
            with open(split_path, "r") as f:
                for j, line in enumerate(f):
                    if j == scene_idx:
                        scene_data = json.loads(line)
                        break
                else:
                    raise ValueError(f"Scene {scene_idx} not found in {split_path}")

            # Generate segmentation maps
            generate_segmentation_maps(scene_data, scene_dir, sem_to_id)
            scene_success_file.touch()

        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            with open(errors_dir / scene_id, "w") as f:
                f.write(traceback.format_exc())


if __name__ == "__main__":
    main()
