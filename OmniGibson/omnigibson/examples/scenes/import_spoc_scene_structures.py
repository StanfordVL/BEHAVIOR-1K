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

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_conversion_utils import (
    import_og_asset_from_urdf,
    generate_urdf_for_mesh,
)

gm.HEADLESS = True

RESTART_EVERY = 8

# Material mapping paths
MDL_MATERIAL_ROOT = "/checkpoint/clear/cgokmen/og-materials/Materials/2023_2_1"
AI2_MDL_MAPPING_FN = "/home/cgokmen/projects/BEHAVIOR-1K/slurm/ai2_nvidia_material_mapping.csv"
MDL_PATHS_FN = "/home/cgokmen/projects/BEHAVIOR-1K/slurm/material_paths.csv"

# Default materials for structures
DEFAULT_FLOOR_MATERIAL = "Parquet_Floor"
DEFAULT_WALL_MATERIAL = "Plaster"
DEFAULT_CEILING_MATERIAL = "Plaster"


def convert_csv_to_dict(filepath):
    """
    Converts a 2-column CSV file into a key:value dictionary.
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        return {row[0]: row[1] for row in reader if len(row) >= 2}


# Load material mappings
AI2_MDL_MAPPING = convert_csv_to_dict(AI2_MDL_MAPPING_FN)
MDL_PATHS = convert_csv_to_dict(MDL_PATHS_FN)


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


def get_material_info(material_name):
    """
    Get MDL material path info for a given material name.

    Args:
        material_name: The AI2-THOR material name

    Returns:
        tuple: (mdl_material_name, mdl_path) or (None, None) if not found
    """
    if not material_name or material_name not in AI2_MDL_MAPPING:
        return None, None

    mdl_material_name = AI2_MDL_MAPPING.get(material_name)
    if not mdl_material_name or mdl_material_name not in MDL_PATHS:
        return None, None

    mdl_path = os.path.join(MDL_MATERIAL_ROOT, MDL_PATHS[mdl_material_name], f"{mdl_material_name}.mdl")
    if not os.path.exists(mdl_path):
        return mdl_material_name, None

    return mdl_material_name, mdl_path


def extract_scene_structures(scene_data):
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
            # Extract material name
            material_name = DEFAULT_FLOOR_MATERIAL
            if "floorMaterial" in room and "name" in room["floorMaterial"]:
                material_name = room["floorMaterial"]["name"]

            if len(points) >= 3:
                try:
                    # Extrude floors to 30cm thickness, keeping inner surface in place
                    # and extruding downward (outward from room interior)
                    mesh = polygon_to_trimesh(points, convex_hull=False, extrude_thickness=0.30, extrude_outward=True)
                    structures[f"floor_{i}"] = (mesh, room_id, material_name)
                except Exception as e:
                    print(f"Failed to create floor mesh {i}: {e}")

    # Process walls
    for i, wall in enumerate(scene_data.get("walls", [])):
        # Try to get room info from wall if available
        room_id = wall.get("roomId", None)
        # Extract material name
        material_name = DEFAULT_WALL_MATERIAL
        if "material" in wall and "name" in wall["material"]:
            material_name = wall["material"]["name"]

        if "polygon" in wall:
            points = np.array([[pt["x"], pt["y"], pt["z"]] for pt in wall["polygon"]])
            if len(points) >= 3:
                try:
                    # Extrude walls to 1cm thickness centered around the planar position
                    mesh = polygon_to_trimesh(points, convex_hull=True, extrude_thickness=0.01)
                    structures[f"wall_{i}"] = (mesh, room_id, material_name)
                except Exception as e:
                    print(f"Failed to create wall mesh {i}: {e}")

    # Process ceilings if present
    for i, room in enumerate(scene_data.get("rooms", [])):
        room_id = room.get("id", f"room_{i}")
        # Extract material name (ceilings typically use wall material or a default)
        material_name = DEFAULT_CEILING_MATERIAL
        if "ceilingMaterial" in room and "name" in room["ceilingMaterial"]:
            material_name = room["ceilingMaterial"]["name"]

        if "ceilingPolygon" in room:
            points = np.array([[pt["x"], pt["y"], pt["z"]] for pt in room["ceilingPolygon"]])
            if len(points) >= 3:
                try:
                    # Extrude ceilings to 30cm thickness, keeping inner surface in place
                    # and extruding upward (outward from room interior)
                    mesh = polygon_to_trimesh(points, convex_hull=False, extrude_thickness=0.30, extrude_outward=True)
                    structures[f"ceiling_{i}"] = (mesh, room_id, material_name)
                except Exception as e:
                    print(f"Failed to create ceiling mesh {i}: {e}")

    return structures


def import_structure_object(
    mesh: trimesh.Trimesh,
    category: str,
    model: str,
    dataset_name: str,
    material_name: str = None,
):
    """
    Imports a structure mesh into an OmniGibson-compatible USD format.

    Args:
        mesh: The trimesh object to import
        category: Object category (floors, walls, ceilings)
        model: Model name
        dataset_name: Name of the dataset (e.g., "spoc")
        material_name: Optional AI2-THOR material name to bind to the USD
    """
    dataset_root = pathlib.Path(gm.DATA_PATH) / dataset_name
    model_root = dataset_root / "objects" / category / model
    success_file = model_root / "import.success"

    # Get MDL material info if material name is provided
    mdl_material_name, mdl_path = None, None
    if material_name:
        mdl_material_name, mdl_path = get_material_info(material_name)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export the mesh as OBJ
            asset_path = os.path.join(temp_dir, f"spoc{model}.obj")
            mesh.export(asset_path, file_type="obj")

            # Generate URDF
            urdf_path = generate_urdf_for_mesh(
                asset_path,
                temp_dir,
                category,
                model,
                collision_method="convex",  # Simple convex hull for structures
                hull_count=1,
                scale=1.0,
                check_scale=False,
                rescale=False,
                overwrite=True,
            )
            assert urdf_path is not None, f"Failed to generate URDF for {asset_path}"

            # Convert to USD with optional MDL material binding
            import_og_asset_from_urdf(
                category=category,
                model=model,
                urdf_path=str(urdf_path),
                collision_method=None,
                dataset_name=dataset_name,
                hull_count=1,
                overwrite=False,
                use_usda=False,
                mdl_material_path=mdl_path,
                mdl_material_name=mdl_material_name,
            )

            success_file.touch()
    finally:
        if og.sim:
            og.clear()


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
    parser.add_argument("--restart-every", type=int, default=RESTART_EVERY)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes to process (for testing)")
    args = parser.parse_args()

    # Setup paths
    dataset_name = args.dataset_name
    dataset_root = pathlib.Path(gm.DATA_PATH) / dataset_name
    dataset_root.mkdir(exist_ok=True)

    errors_dir = dataset_root / "errors"
    errors_dir.mkdir(exist_ok=True)
    jobs_dir = dataset_root / "jobs"
    jobs_dir.mkdir(exist_ok=True)

    spoc_houses_root = pathlib.Path(args.spoc_houses_root)

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
    if args.limit:
        task_scenes = task_scenes[: args.limit]

    completed_count = 0
    for split_path, scene_idx in tqdm(task_scenes):
        scene_id = get_scene_id(split_path, scene_idx)

        # Check if all structures for this scene are already done
        scene_success_file = dataset_root / "objects" / "spoc_structures" / f"{scene_id}.success"
        if scene_success_file.exists():
            continue

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
            structures = extract_scene_structures(scene_data)

            if not structures:
                print(f"No structures found in scene {scene_id}")
                continue

            # Import each structure
            all_success = True
            for struct_name, (mesh, room_id, material_name) in structures.items():
                # Determine category based on structure type
                if struct_name.startswith("floor"):
                    category = "floors"
                elif struct_name.startswith("wall"):
                    category = "walls"
                elif struct_name.startswith("ceiling"):
                    category = "ceilings"
                else:
                    category = "structures"

                # Model name includes scene ID and structure name
                model = f"spoc_{scene_id}_{struct_name}".replace("-", "_")

                model_root = dataset_root / "objects" / category / model
                struct_success_file = model_root / "import.success"

                if model_root.exists():
                    if struct_success_file.exists():
                        continue
                    shutil.rmtree(model_root)

                try:
                    import_structure_object(
                        mesh=mesh,
                        category=category,
                        model=model,
                        dataset_name=dataset_name,
                        material_name=material_name,
                    )
                except Exception as e:
                    print(f"Error importing {model}: {e}")
                    with open(errors_dir / f"{scene_id}_{struct_name}", "w") as f:
                        f.write(traceback.format_exc())
                    all_success = False

            # Mark scene as complete if all structures imported
            if all_success:
                scene_success_file.parent.mkdir(parents=True, exist_ok=True)
                scene_success_file.touch()

            completed_count += 1

        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            with open(errors_dir / scene_id, "w") as f:
                f.write(traceback.format_exc())

        if args.restart_every and completed_count >= args.restart_every:
            return

    # If we reach here, we're done. Record the rank success with namespace.
    success_filename = f"{args.success_prefix}_{rank}.success" if args.success_prefix else f"{rank}.success"
    (jobs_dir / success_filename).touch()

    og.shutdown()


if __name__ == "__main__":
    main()
