"""
Script to pre-generate floor, wall, and ceiling objects for vid2room scenes.
These are procedurally generated from the scene boundary polygons and need to be converted
to DatasetObjects so they can be saved in scene JSONs.

Based on logic from mesh_placement.ipynb notebook.
"""

import argparse
import hashlib
import pathlib
import traceback
import shutil
import tempfile
import os
import random
from tqdm import tqdm
import json

import numpy as np
import trimesh
from shapely.geometry import Polygon

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.asset_conversion_utils import (
    import_og_asset_from_urdf,
    generate_urdf_for_mesh,
)

gm.HEADLESS = True

RESTART_EVERY = 16

# Default interesting scenes list
DEFAULT_INTERESTING_SCENES_JSON = "/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/interesting_scenes.json"

# Default wall color (light beige)
DEFAULT_WALL_COLOR_RGB = [210, 206, 181]

# Default ceiling color (off-white)
DEFAULT_CEILING_COLOR_RGB = [245, 245, 240]

# Mapping from VLM floor_material types to MDL material paths
# Format: {vlm_material_type: [(mdl_name, relative_path), ...]}
FLOOR_MATERIAL_MDL_MAPPING = {
    "dark_wood": [
        ("Walnut_Planks", "Wood/Walnut_Planks"),
        ("Mahogany_Planks", "Wood/Mahogany_Planks"),
        ("Cherry_Planks", "Wood/Cherry_Planks"),
    ],
    "light_wood": [
        ("Oak_Planks", "Wood/Oak_Planks"),
        ("Ash_Planks", "Wood/Ash_Planks"),
        ("Birch_Planks", "Wood/Birch_Planks"),
        ("Bamboo_Planks", "Wood/Bamboo_Planks"),
        ("Parquet_Floor", "Wood/Parquet_Floor"),
    ],
    "light_tile": [
        ("Ceramic_Tile_12", "Stone/Ceramic_Tile_12"),
        ("Ceramic_Tile_18", "Stone/Ceramic_Tile_18"),
        ("Porcelain_Tile_4", "Stone/Porcelain_Tile_4"),
        ("Porcelain_Tile_6", "Stone/Porcelain_Tile_6"),
        ("Marble_Tile_12", "Stone/Marble_Tile_12"),
        ("Granite_Light", "Stone/Granite_Light"),
    ],
    "dark_tile": [
        ("Granite_Dark", "Stone/Granite_Dark"),
        ("Slate", "Stone/Slate"),
        ("Marble_Tile_18", "Stone/Marble_Tile_18"),
        ("Terracotta", "Stone/Terracotta"),
    ],
    "concrete": [
        ("Concrete_Polished", "Masonry/Concrete_Polished"),
        ("Concrete_Smooth", "Masonry/Concrete_Smooth"),
        ("Concrete_Rough", "Masonry/Concrete_Rough"),
    ],
    "carpet": [
        ("Carpet_Beige", "Carpet/Carpet_Beige"),
        ("Carpet_Gray", "Carpet/Carpet_Gray"),
        ("Carpet_Cream", "Carpet/Carpet_Cream"),
        ("Carpet_Charcoal", "Carpet/Carpet_Charcoal"),
        ("Carpet_Berber_Gray", "Carpet/Carpet_Berber_Gray"),
    ],
}

# Default structure parameters
DEFAULT_WALL_THICKNESS = 0.1
DEFAULT_ADDITIONAL_BUFFER = 0.1
DEFAULT_FLOOR_THICKNESS = 0.3


def get_floor_mdl_material(floor_material_type):
    """
    Get a random MDL material path for a given VLM floor material type.

    Args:
        floor_material_type: One of "dark_wood", "light_wood", "light_tile",
                            "dark_tile", "concrete", "carpet"

    Returns:
        tuple: (mdl_material_name, mdl_path) or (None, None) if not found
    """
    if floor_material_type not in FLOOR_MATERIAL_MDL_MAPPING:
        # Default to light_wood if unknown
        floor_material_type = "light_wood"

    materials = FLOOR_MATERIAL_MDL_MAPPING[floor_material_type]
    mdl_name, relative_path = random.choice(materials)
    mdl_path = relative_path + ".mdl"

    # assert os.path.exists(mdl_path), f"MDL material path does not exist: {mdl_path}"
    return mdl_name, mdl_path


def create_omnipbr_material_with_color(prim, color_rgb, material_name="wall_material"):
    """
    Create an OmniPBR material with a specific diffuse color and bind it to a prim.

    Args:
        prim: The USD prim to bind the material to
        color_rgb: List of [R, G, B] values (0-255)
        material_name: Name for the material

    Returns:
        bool: True if successful
    """
    # Normalize color to 0-1 range
    color_normalized = [c / 255.0 for c in color_rgb]

    # Create the OmniPBR material
    mtl_created_list = []
    lazy.omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=mtl_created_list,
    )

    if not mtl_created_list:
        print(f"Failed to create OmniPBR material")
        return False

    pbr_mat = lazy.isaacsim.core.utils.prims.get_prim_at_path(mtl_created_list[0])

    # Set the diffuse color constant
    lazy.omni.usd.create_material_input(
        pbr_mat, "diffuse_color_constant", lazy.pxr.Gf.Vec3f(*color_normalized), lazy.pxr.Sdf.ValueTypeNames.Color3f
    )

    # Bind the material to the prim
    shader = lazy.pxr.UsdShade.Material(pbr_mat)

    # Find all mesh prims to bind to
    bound_count = 0
    for child in prim.GetChildren():
        visuals_prim_path = f"{child.GetPrimPath().pathString}/visuals"
        visuals_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(visuals_prim_path)

        if not visuals_prim or not visuals_prim.IsValid():
            continue

        if visuals_prim.GetTypeName() == "Mesh":
            lazy.pxr.UsdShade.MaterialBindingAPI(visuals_prim).Bind(
                shader, lazy.pxr.UsdShade.Tokens.strongerThanDescendants
            )
            bound_count += 1
        else:
            for mesh_child in visuals_prim.GetChildren():
                if mesh_child.GetTypeName() == "Mesh":
                    lazy.pxr.UsdShade.MaterialBindingAPI(mesh_child).Bind(
                        shader, lazy.pxr.UsdShade.Tokens.strongerThanDescendants
                    )
                    bound_count += 1

    if bound_count > 0:
        prim.GetStage().Save()
        return True

    return False


def loop_cum_lengths(loop_xy):
    """Calculate cumulative lengths along a polygon loop."""
    n = len(loop_xy)
    cum = np.zeros(n + 1)
    for i in range(n):
        cum[i + 1] = cum[i] + np.linalg.norm(loop_xy[(i + 1) % n] - loop_xy[i])
    return cum, cum[-1]


def point_at_u(loop_xy, u):
    """
    Get the 2D point and direction at a normalized position u along the polygon loop.

    Args:
        loop_xy: Nx2 array of polygon vertices
        u: Normalized position along the polygon (0-1)

    Returns:
        tuple: (center_2d, direction_2d) - the point and tangent direction at u
    """
    u = (u + 0.5) % 1  # Offset to match notebook convention
    cum, total = loop_cum_lengths(loop_xy)
    target = (u % 1.0) * total
    acc = 0.0
    for i in range(len(loop_xy)):
        a = loop_xy[i]
        b = loop_xy[(i + 1) % len(loop_xy)]
        seglen = np.linalg.norm(b - a)
        if acc + seglen >= target:
            t = (target - acc) / (seglen + 1e-12)
            return a + t * (b - a), b - a
        acc += seglen

    raise ValueError("Could not find the point")


def point_to_box_params(center_2d, y_dir, width, height, wall_thickness, sill=0.0):
    """
    Convert a 2D point and direction to a 3D box transform and extents for openings.

    Args:
        center_2d: 2D center point on the wall
        y_dir: 2D direction along the wall
        width: Width of the opening
        height: Height of the opening
        wall_thickness: Thickness of the wall (for depth of opening box)
        sill: Height of the sill (for windows)

    Returns:
        tuple: (transform_4x4, extents_3d)
    """
    y_dir = y_dir / np.linalg.norm(y_dir)
    y_dir = np.array([y_dir[0], y_dir[1], 0])
    z_dir = np.array([0, 0, 1])
    x_dir = np.cross(y_dir, z_dir)
    rotmat = np.array([x_dir, y_dir, z_dir]).T

    ctrpos = np.array([center_2d[0], center_2d[1], sill + height / 2])
    ctrpos += x_dir * (wall_thickness / 2)

    transform = np.eye(4)
    transform[:3, :3] = rotmat
    transform[:3, 3] = ctrpos

    return transform, np.array([wall_thickness, width, height])


def generate_openings(boundary_points, door_openings, window_openings, wall_thickness):
    """
    Generate door and window opening boxes from opening specifications.

    Args:
        boundary_points: Nx3 array of boundary vertices
        door_openings: List of door opening dicts with 'u', 'width', 'height'
        window_openings: List of window opening dicts with 'u', 'width', 'height', 'sill'
        wall_thickness: Thickness of walls

    Returns:
        tuple: (doors, windows) - lists of (transform, extents) tuples
    """
    doors = []
    if door_openings:
        for d in door_openings:
            center_2d, y_dir = point_at_u(boundary_points[:, :2], d["u"])
            doors.append(point_to_box_params(center_2d, y_dir, d["width"], d["height"], wall_thickness))

    windows = []
    if window_openings:
        for w in window_openings:
            center_2d, y_dir = point_at_u(boundary_points[:, :2], w["u"])
            windows.append(
                point_to_box_params(center_2d, y_dir, w["width"], w["height"], wall_thickness, w.get("sill", 0.0))
            )

    return doors, windows


def extract_vid2room_scene_structures(
    scene_data,
    wall_thickness=DEFAULT_WALL_THICKNESS,
    additional_buffer=DEFAULT_ADDITIONAL_BUFFER,
    floor_thickness=DEFAULT_FLOOR_THICKNESS,
    include_ceiling=True,
):
    """
    Extract floor, wall, and ceiling meshes from a vid2room scene.

    Args:
        scene_data: Parsed JSON scene data containing 'boundary3d', 'door_openings',
                    'window_openings', 'ceiling_height', and optionally VLM analysis data
                    like 'wall_color_rgb' and 'floor_material'
        wall_thickness: Thickness of walls in meters
        additional_buffer: Additional buffer for polygon expansion
        floor_thickness: Thickness of the floor in meters
        include_ceiling: Whether to generate a ceiling

    Returns:
        dict: Dictionary mapping structure names to (trimesh, material_info) tuples
              where material_info is a dict with keys like:
              - 'type': 'wall_color' | 'floor_mdl' | 'ceiling_mdl'
              - 'wall_color_rgb': [R, G, B] for walls
              - 'floor_material': VLM floor material type for floors
    """
    structures = {}

    # Extract boundary vertices from scene data
    boundary3d = scene_data.get("boundary3d", [])
    if not boundary3d:
        print("No boundary3d found in scene data")
        return structures

    # Get ceiling height from scene data (may come from vlm_analysis.json)
    ceiling_height = scene_data.get("ceiling_height", 2.7)  # Default to 2.7m if not specified

    # VLM analysis data for materials (required; load_scene_data enforces presence)
    wall_color_rgb = scene_data["wall_color_rgb"]
    floor_material = scene_data["floor_material"]

    # Use only the first half of boundary points (floor vertices, not ceiling)
    vertices = np.array([x[:2] for x in boundary3d])
    vertices = vertices[: len(vertices) // 2]

    if len(vertices) < 3:
        print(f"Not enough vertices to form a polygon: {len(vertices)}")
        return structures

    # Create inner and outer polygons for wall generation
    inner_polygon = Polygon(vertices).buffer(additional_buffer, join_style="mitre")
    outer_polygon = inner_polygon.buffer(wall_thickness, join_style="mitre")

    if not inner_polygon.is_valid or not outer_polygon.is_valid:
        print("Invalid polygon geometry")
        return structures

    # Get door and window openings
    door_openings = scene_data.get("door_openings", [])
    window_openings = scene_data.get("window_openings", [])
    doors, windows = [], []
    if door_openings is not None or window_openings is not None:
        doors, windows = generate_openings(
            boundary_points=np.hstack([vertices, np.zeros((len(vertices), 1))]),
            door_openings=door_openings or [],
            window_openings=window_openings or [],
            wall_thickness=wall_thickness,
        )

    # Generate individual wall segments
    inner_coords = np.array(inner_polygon.exterior.coords)
    outer_coords = np.array(outer_polygon.exterior.coords)

    wall_meshes = []
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        # Get corresponding points on inner and outer polygons
        inner_i = inner_coords[i]
        inner_j = inner_coords[j]
        outer_i = outer_coords[i]
        outer_j = outer_coords[j]

        wall_polygon = Polygon([inner_i, inner_j, outer_j, outer_i])
        if wall_polygon.is_valid and wall_polygon.area > 0:
            try:
                wall_mesh = trimesh.creation.extrude_polygon(wall_polygon, height=ceiling_height)
                wall_meshes.append(wall_mesh)
            except Exception as e:
                print(f"Failed to create wall mesh {i}: {e}")

    # Cut openings from walls using boolean operations
    all_openings = doors + windows
    if all_openings:
        for i in range(len(wall_meshes)):
            for opening_transform, opening_extent in all_openings:
                opening_box = trimesh.creation.box(extents=opening_extent + 0.01, transform=opening_transform)
                try:
                    wall_meshes[i] = trimesh.boolean.difference([wall_meshes[i], opening_box], engine="manifold")
                except Exception as e:
                    print(f"Failed to cut opening from wall {i}: {e}")

    # Add wall meshes to structures with wall color info
    wall_material_info = {
        "type": "wall_color",
        "wall_color_rgb": wall_color_rgb,
    }
    for i, wall_mesh in enumerate(wall_meshes):
        if wall_mesh is not None and hasattr(wall_mesh, "vertices") and len(wall_mesh.vertices) > 0:
            structures[f"wall_{i}"] = (wall_mesh, wall_material_info)

    # Generate floor mesh with floor material info
    floor_material_info = {
        "type": "floor_mdl",
        "floor_material": floor_material,
    }
    try:
        floor_mesh = trimesh.creation.extrude_polygon(outer_polygon, height=floor_thickness)
        # Move floor down so top surface is at z=0
        floor_mesh.vertices -= np.array([0, 0, floor_thickness])

        # Add UV coordinates based on world XY position (for proper texture tiling)
        # Each vertex's UV = its XY world coordinates
        uv_coords = floor_mesh.vertices[:, :2].copy()  # Use X, Y as U, V
        floor_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords)

        structures["floor_0"] = (floor_mesh, floor_material_info)
    except Exception as e:
        print(f"Failed to create floor mesh: {e}")

    # Generate ceiling mesh (optional) with ceiling color info (off-white OmniPBR)
    ceiling_material_info = {
        "type": "ceiling_color",
        "ceiling_color_rgb": DEFAULT_CEILING_COLOR_RGB,
    }
    if include_ceiling:
        try:
            ceiling_mesh = trimesh.creation.extrude_polygon(outer_polygon, height=floor_thickness)
            # Move ceiling to ceiling height
            ceiling_mesh.vertices[:, 2] += ceiling_height
            structures["ceiling_0"] = (ceiling_mesh, ceiling_material_info)
        except Exception as e:
            print(f"Failed to create ceiling mesh: {e}")

    return structures


def import_structure_object(
    mesh: trimesh.Trimesh,
    category: str,
    model: str,
    dataset_name: str,
    material_info: dict,
):
    """
    Imports a structure mesh into an OmniGibson-compatible USD format.

    Args:
        mesh: The trimesh object to import
        category: Object category (floors, walls, ceilings)
        model: Model name
        dataset_name: Name of the dataset (e.g., "vid2room")
        material_info: Material information dict (required). Keys:
            - 'type': 'wall_color' | 'floor_mdl' | 'ceiling_color'
            - 'wall_color_rgb': [R, G, B] for wall_color type (required for walls)
            - 'floor_material': VLM floor material type for floor_mdl type (required for floors)
    """
    dataset_root = pathlib.Path(gm.DATA_PATH) / dataset_name
    model_root = dataset_root / "objects" / category / model
    success_file = model_root / "import.success"

    material_type = material_info["type"]
    mdl_material_name, mdl_path = None, None
    wall_color_rgb = None
    ceiling_color_rgb = None

    if material_type == "floor_mdl":
        floor_material = material_info["floor_material"]
        mdl_material_name, mdl_path = get_floor_mdl_material(floor_material)
        if not mdl_material_name or not mdl_path:
            raise ValueError(f"Failed to resolve MDL material for floor type: {floor_material}")
        print(f"  Floor material: {floor_material} -> {mdl_material_name}")

    elif material_type == "ceiling_color":
        ceiling_color_rgb = material_info["ceiling_color_rgb"]
        print(f"  Ceiling color: RGB{ceiling_color_rgb}")

    elif material_type == "wall_color":
        wall_color_rgb = material_info["wall_color_rgb"]
        print(f"  Wall color: RGB{wall_color_rgb}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export the mesh as OBJ
            asset_path = os.path.join(temp_dir, f"v2r{model}.obj")
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

            # Convert to USD with optional MDL material binding (for floors/ceilings)
            urdf_path_out, usd_path, prim = import_og_asset_from_urdf(
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

            # Walls MUST have wall color applied
            if material_type == "wall_color":
                assert create_omnipbr_material_with_color(prim, wall_color_rgb, f"{model}_wall_material"), f"Failed to apply wall color to {model}"

            # Ceiling color
            if material_type == "ceiling_color" and prim is not None:
                assert create_omnipbr_material_with_color(prim, ceiling_color_rgb, f"{model}_ceiling_material"), f"Failed to apply ceiling color to {model}"

            success_file.touch()
    finally:
        if og.sim:
            og.clear()


def get_scene_id(scene_path):
    """
    Generate a unique scene ID from scene path.

    For vid2room scenes, the path structure is typically:
    .../vid_XXXXX/rooms/room_type_N

    We want to create a unique ID like: vid_XXXXX_room_type_N
    """
    scene_path = pathlib.Path(scene_path)
    room_name = scene_path.name  # e.g., "living_room_0"
    # Go up to find the video ID (parent of "rooms" directory)
    video_id = scene_path.parent.parent.name  # e.g., "vid_1vdXN7X4Af4"
    assert video_id.startswith("vid_"), f"Video ID {video_id} does not start with 'vid_'"
    return f"{video_id}_{room_name}"


def load_scene_data(scene_dir):
    """
    Load scene data from a vid2room scene directory.

    Args:
        scene_dir: Path to the scene directory containing room_parameters.json
                   and vlm_analysis.json (required for floor_material and wall_color_rgb)

    Returns:
        dict: Combined scene data with boundary, openings, ceiling height,
              floor_material, and wall_color_rgb

    Raises:
        FileNotFoundError: If room_parameters.json or vlm_analysis.json is missing
        ValueError: If floor_material or wall_color_rgb is missing from vlm_analysis.json
    """
    scene_dir = pathlib.Path(scene_dir)

    # Load room parameters (boundary, openings)
    floorplan_path = scene_dir / "floorplan2" / "room_parameters.json"
    if not floorplan_path.exists():
        floorplan_path = scene_dir / "room_parameters.json"

    if not floorplan_path.exists():
        raise FileNotFoundError(f"No room_parameters.json found in {scene_dir}")

    scene_data = json.loads(floorplan_path.read_text())

    # VLM analysis is REQUIRED for floor material and wall color
    vlm_analysis_path = scene_dir / "vlm_analysis.json"
    if not vlm_analysis_path.exists():
        raise FileNotFoundError(f"vlm_analysis.json required but not found in {scene_dir}")

    vlm_data = json.loads(vlm_analysis_path.read_text())
    scene_data["ceiling_height"] = vlm_data.get("ceiling_height", 2.7)

    if "floor_material" not in vlm_data:
        raise ValueError(f"floor_material required but missing in {vlm_analysis_path}")
    scene_data["floor_material"] = vlm_data["floor_material"]

    if "wall_color_rgb" not in vlm_data:
        raise ValueError(f"wall_color_rgb required but missing in {vlm_analysis_path}")
    scene_data["wall_color_rgb"] = vlm_data["wall_color_rgb"]

    return scene_data


def main():
    parser = argparse.ArgumentParser(description="Import vid2room scene structures as OmniGibson USD assets")
    parser.add_argument("task_id", type=int, help="Task ID (0-indexed)")
    parser.add_argument("total_tasks", type=int, help="Total number of tasks")
    parser.add_argument("--success-file", type=str, default=None, help="Success file")
    parser.add_argument("--dataset-name", default="vid2room", help="Dataset name (defaults to 'vid2room')")
    parser.add_argument("--restart-every", type=int, default=RESTART_EVERY)
    parser.add_argument("--wall-thickness", type=float, default=DEFAULT_WALL_THICKNESS, help="Wall thickness in meters")
    parser.add_argument(
        "--floor-thickness", type=float, default=DEFAULT_FLOOR_THICKNESS, help="Floor thickness in meters"
    )
    parser.add_argument(
        "--scene-list",
        type=str,
        default=DEFAULT_INTERESTING_SCENES_JSON,
        help="Path to JSON file with list of scene directories to process",
    )
    args = parser.parse_args()

    # Setup paths
    dataset_name = args.dataset_name
    dataset_root = pathlib.Path(gm.DATA_PATH) / dataset_name
    dataset_root.mkdir(exist_ok=True)

    errors_dir = dataset_root / "errors"
    errors_dir.mkdir(exist_ok=True)

    rank = args.task_id
    world_size = args.total_tasks

    # Collect scenes from the interesting scenes JSON (same logic as vid2scene_step10_floorplan.py)
    print("Finding rooms...")
    with open(args.scene_list, "r") as f:
        room_dirs = json.load(f)

    # Convert to pathlib paths
    room_dirs = [pathlib.Path(k) for k in room_dirs]
    # room_dirs = [pathlib.Path("/checkpoint/clear/cgokmen/vid2room/RealEstate10K/vid_1vdXN7X4Af4/rooms/living_room_0")]

    # Exclude bathrooms
    room_dirs = [x for x in room_dirs if "bathroom" not in str(x)]

    # Filter to only rooms where floorplan.success exists (floorplan generation is complete)
    room_dirs = [x for x in room_dirs if (x / "floorplan2.success").exists()]

    print(f"Found {len(room_dirs)} rooms with completed floorplans")

    # Distribute scenes across tasks using hash-based assignment (same as floorplan script)
    task_scenes = [
        x for x in room_dirs if int(hashlib.md5((str(x) + "cucumber").encode()).hexdigest(), 16) % world_size == rank
    ]

    print(f"Task {rank}/{world_size}: Processing {len(task_scenes)} rooms")

    completed_count = 0
    for scene_path in tqdm(task_scenes):
        scene_id = get_scene_id(scene_path)

        # Check if all structures for this scene are already done
        scene_success_file = dataset_root / "objects" / "vid2room_structures" / f"{scene_id}.success"
        if scene_success_file.exists():
            continue

        try:
            # Load the scene data
            scene_data = load_scene_data(scene_path)

            # Extract all structure meshes
            structures = extract_vid2room_scene_structures(
                scene_data,
                wall_thickness=args.wall_thickness,
                floor_thickness=args.floor_thickness,
                include_ceiling=True,
            )

            if not structures:
                print(f"No structures found in scene {scene_id}")
                continue

            # Import each structure
            all_success = True
            for struct_name, (mesh, material_info) in structures.items():
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
                model = f"vid2room_{scene_id}_{struct_name}".replace("-", "_")

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
                        material_info=material_info,
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
    success_file = pathlib.Path(args.success_file)
    success_file.parent.mkdir(parents=True, exist_ok=True)
    success_file.touch()

    og.shutdown()


if __name__ == "__main__":
    main()
