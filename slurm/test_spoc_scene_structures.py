"""
Script to pre-generate floor, wall, and ceiling objects for SPOC scenes.
These are procedurally generated from the scene JSON and need to be converted
to DatasetObjects so they can be saved in scene JSONs.
"""

import argparse
import csv
import hashlib
import pathlib
import traceback
import shutil
import tempfile
import os
from tqdm import tqdm
import json

import numpy as np
import trimesh
import shapely


def polygon_to_trimesh(points, convex_hull=False, extrude_thickness=None, extrude_outward=False):
    """
    Create a trimesh object from a list of 3D points defining a planar polygon.
    Optionally extrude the polygon to give it thickness.

    Args:
        points: numpy array of 3D points
        convex_hull: Whether to use convex hull of the polygon
        extrude_thickness: If provided, extrude the polygon to this thickness (in meters)
        extrude_outward: If True, keep the original (inner) surface in place and extrude
                        away from the normal direction (for floors/ceilings). If False,
                        center the extrusion around the original plane (for walls).

    Returns:
        trimesh.Trimesh: The triangulated mesh (or extruded mesh if thickness specified)
    """
    points = np.copy(points)
    points[:, 0] *= -1  # Invert x-coordinates to match coordinate system

    # Create a 3D path and then convert to mesh
    lines = list(range(len(points))) + [0]  # Close the polygon
    path = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(lines)], vertices=points, process=False)

    # Convert path to 2D, triangulate, then back to 3D
    planar, to_3D = path.to_2D()

    # Get the convex hull if requested
    if convex_hull:
        points_2d = planar.vertices
        multipoint = shapely.MultiPoint(points_2d)
        polygon = multipoint.convex_hull
        points_convex = np.array(polygon.exterior.coords[:-1])
        lines_convex = list(range(len(points_convex))) + [0]
        planar = trimesh.path.Path2D(
            entities=[trimesh.path.entities.Line(lines_convex)],
            vertices=points_convex,
            process=False,
        )

    if extrude_thickness is not None and extrude_thickness > 0:
        # Extrude the 2D polygon to give it thickness
        # Get the shapely polygon from the path
        if planar.polygons_full:
            shapely_polygon = planar.polygons_full[0]
        else:
            # Fallback: create polygon from vertices
            shapely_polygon = shapely.Polygon(planar.vertices)

        # Extrude along Z in 2D space
        mesh = trimesh.creation.extrude_polygon(shapely_polygon, extrude_thickness)

        # Adjust extrusion position based on mode
        if extrude_outward:
            # Keep inner surface in place, extrude away from normal direction
            # (floors extrude down, ceilings extrude up)
            mesh.vertices[:, 2] -= extrude_thickness
        else:
            # Center the extrusion around the original plane (for walls)
            mesh.vertices[:, 2] -= extrude_thickness / 2.0

        # Transform back to 3D world coordinates
        mesh.apply_transform(to_3D)
        return mesh
    else:
        verts_2d, faces = planar.triangulate()
        points_3d = np.hstack((verts_2d, np.zeros((verts_2d.shape[0], 1))))
        mesh = trimesh.Trimesh(vertices=points_3d, faces=faces, process=False)
        mesh.apply_transform(to_3D)
        return mesh


def extract_scene_structures(scene_data, scene_dir):
    """
    Extract floor and wall meshes from a SPOC scene.

    Args:
        scene_data: Parsed JSON scene data

    Returns:
        dict: Dictionary mapping structure names to (trimesh, room_id, material_name) tuples
    """
    structures = {}

    # Process floors from rooms
    for i, room in enumerate(scene_data.get("rooms", [])):
        room_id = room.get("id", f"room_{i}")
        if "floorPolygon" in room:
            points = np.array([[pt["x"], pt["y"], pt["z"]] for pt in room["floorPolygon"]])

            if len(points) >= 3:
                try:
                    # Extrude floors to 30cm thickness, keeping inner surface in place
                    # and extruding downward (outward from room interior)
                    mesh = polygon_to_trimesh(points, convex_hull=False, extrude_thickness=0.30, extrude_outward=True)
                    structures[f"floor_{i}"] = (mesh, room_id)
                except Exception as e:
                    print(f"Failed to create floor mesh {i}: {e}")

    # Process walls
    for i, wall in enumerate(scene_data.get("walls", [])):
        # Try to get room info from wall if available
        room_id = wall.get("roomId", None)

        if "polygon" in wall:
            points = np.array([[pt["x"], pt["y"], pt["z"]] for pt in wall["polygon"]])
            if len(points) >= 3:
                try:
                    # Extrude walls to 1cm thickness centered around the planar position
                    mesh = polygon_to_trimesh(points, convex_hull=True, extrude_thickness=0.01)
                    structures[f"wall_{i}"] = (mesh, room_id)
                except Exception as e:
                    print(f"Failed to create wall mesh {i}: {e}")

    # Process ceilings if present
    for i, room in enumerate(scene_data.get("rooms", [])):
        room_id = room.get("id", f"room_{i}")

        if "ceilingPolygon" in room:
            points = np.array([[pt["x"], pt["y"], pt["z"]] for pt in room["ceilingPolygon"]])
            if len(points) >= 3:
                try:
                    # Extrude ceilings to 30cm thickness, keeping inner surface in place
                    # and extruding upward (outward from room interior)
                    mesh = polygon_to_trimesh(points, convex_hull=False, extrude_thickness=0.30, extrude_outward=True)
                    structures[f"ceiling_{i}"] = (mesh, room_id)
                except Exception as e:
                    print(f"Failed to create ceiling mesh {i}: {e}")

    # Save all the structures as GLBs
    for struct_name, (mesh, _) in structures.items():
        mesh.export(scene_dir / f"{struct_name}.glb")


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

    # Collect all scenes
    scenes = []
    for split_file in sorted(spoc_houses_root.glob("*.jsonl")):
        with open(split_file, "r") as f:
            if  "train" not in split_file.name:
                continue
            for i, line in enumerate(f):
                if i != 505:
                    continue
                scenes.append((str(split_file), i))

    # Sort with fixed salt for deterministic ordering across runs
    scenes.sort(key=lambda x: hashlib.md5((f"{x[0]}_{x[1]}potato").encode()).hexdigest())

    rank = args.task_id
    world_size = args.total_tasks

    # Get scenes for this task
    task_scenes = scenes[rank::world_size]

    for split_path, scene_idx in tqdm(task_scenes):
        scene_id = get_scene_id(split_path, scene_idx)

        # Check if all structures for this scene are already done
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

            # Extract all structure meshes
            extract_scene_structures(scene_data, scene_dir)
            scene_success_file.touch()

        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            with open(errors_dir / scene_id, "w") as f:
                f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
