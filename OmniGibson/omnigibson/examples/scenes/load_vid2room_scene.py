import pathlib
import json
import traceback

from omnigibson.scenes.scene_base import Scene
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
import torch as th
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import trimesh
import numpy as np
from tqdm.auto import tqdm
from shapely.geometry import Polygon, Point

import omnigibson as og
from omnigibson.macros import gm


PYTORCH_TO_OPENCV = R.from_euler("z", [180], degrees=True)
PYTORCH_TO_OPENCV_4 = np.eye(4)
PYTORCH_TO_OPENCV_4[:3, :3] = PYTORCH_TO_OPENCV.as_matrix()
Z_UP_TO_Y_UP = R.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
Z_UP_TO_Y_UP_4 = np.eye(4)
Z_UP_TO_Y_UP_4[:3, :3] = Z_UP_TO_Y_UP.as_matrix()
COLLISION_THRESHOLD = 0.30


def snap_rotation(rot, threshold_degrees=15):
    # 1. Create rotation object
    matrix = rot.as_matrix() # 3x3 matrix: columns are Local X, Y, Z
    
    # Extract basis vectors (columns)
    # X=0, Y=1, Z=2
    axes = [matrix[:, 0], matrix[:, 1], matrix[:, 2]]
    
    best_axis_idx = -1
    closest_dot = 0
    sign = 1
    
    # 2. Find which local axis is closest to World Up (0,0,1)
    # We check dot product with (0,0,1), which is just the z-component of the vector
    for i, axis in enumerate(axes):
        z_component = axis[2] 
        if abs(z_component) > abs(closest_dot):
            closest_dot = z_component
            best_axis_idx = i
            # Is it pointing Up (+1) or Down (-1)?
            sign = 1 if z_component > 0 else -1
            
    # 3. Check Threshold
    # Dot product of 1.0 = 0 degrees. 
    # We need to convert degrees to dot product threshold.
    # cos(10 degrees) ~= 0.9848
    threshold_dot = np.cos(np.deg2rad(threshold_degrees))
    
    if abs(closest_dot) < threshold_dot:
        return rot # Not close enough to snap
        
    # 4. Construct Snapped Basis
    # The 'vertical' axis is forced to be exactly World Z
    new_vertical = np.array([0.0, 0.0, float(sign)])
    
    # We need a 'horizontal' axis to preserve the Yaw.
    # We pick a different axis (e.g., if Z is vertical, pick X)
    # If X (idx 0) is vertical, pick Y (idx 1).
    horizontal_idx = (best_axis_idx + 1) % 3
    raw_horizontal = axes[horizontal_idx].copy()
    
    # Flatten horizontal axis to XY plane and normalize
    raw_horizontal[2] = 0 
    new_horizontal = raw_horizontal / np.linalg.norm(raw_horizontal)
    
    # Compute the third axis using Cross Product
    # Order depends on which slot we are filling to maintain Right-Hand Rule
    # We have two known vectors, we need to arrange them into a matrix
    
    new_matrix = np.zeros((3,3))
    
    # Place the vertical axis
    new_matrix[:, best_axis_idx] = new_vertical
    
    # Place the horizontal axis
    new_matrix[:, horizontal_idx] = new_horizontal
    
    # Calculate the remaining axis via cross product
    # To determine cross order (A x B vs B x A), recall: X x Y = Z.
    # It is safer to re-cross depending on indices, 
    # but a simple trick is to fill the matrix and use SVD or QR to orthonormalize, 
    # OR just cross manually:
    
    # Simplified Cross Logic:
    # If we snapped Local Z (2) and used Local X (0): Local Y (1) = Z cross X
    if best_axis_idx == 2: # Z is vertical
        # Y = Z cross X
        new_matrix[:, 1] = np.cross(new_vertical, new_horizontal)
    elif best_axis_idx == 0: # X is vertical
        # Z = X cross Y (Horizontal was Y)
        new_matrix[:, 2] = np.cross(new_vertical, new_horizontal)
    elif best_axis_idx == 1: # Y is vertical
        # X = Y cross Z (Horizontal was Z)
        new_matrix[:, 0] = np.cross(new_vertical, new_horizontal)

    # Convert back to rotation representation
    return R.from_matrix(new_matrix)


def load_collision_meshes_from_npz(npz_path: pathlib.Path) -> list:
    """
    Load pre-computed collision meshes from an NPZ file.

    Args:
        npz_path: Path to the NPZ file containing collision mesh data

    Returns:
        List of trimesh.Trimesh collision meshes
    """
    data = np.load(npz_path)
    collision_meshes = []
    i = 0
    while f"vertices_{i}" in data:
        vertices = data[f"vertices_{i}"]
        faces = data[f"faces_{i}"]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        collision_meshes.append(mesh)
        i += 1
    return collision_meshes


def parse_object_meshes(scene_dir: pathlib.Path) -> dict:
    meshes_root = scene_dir / "obj_meshes_v9_pointmap_decimated"
    collision_root = scene_dir / "obj_meshes_v9_pointmap_collision"
    poses_root = scene_dir / "obj_meshes_v9_pointmap"

    cnp = np.load(scene_dir / "sparse_pi3x/0/cameras_and_points.npz")
    filenames_in_cnp = list(cnp["filenames"])
    pointmaps = cnp["local_points"]
    camera_poses = cnp["camera_poses"]
    meshes = {}
    pose_files = list(poses_root.glob("*.json"))
    for pose_json in tqdm(pose_files):
        # Load the visual mesh
        mesh_path = meshes_root / (pose_json.stem + ".glb")
        if not mesh_path.exists():
            continue
        mesh = trimesh.load(mesh_path, force="mesh")

        # Load the collision meshes
        collision_path = collision_root / (pose_json.stem + ".npz")
        if not collision_path.exists():
            continue
        collision_meshes = load_collision_meshes_from_npz(collision_path)
        
        # Load pose information
        with open(pose_json, "r") as f:
            frame_data = json.load(f)

        # Convert to the correct frames
        for frame_name, frame_info in frame_data.items():
            index = filenames_in_cnp.index(frame_name)
            frame_info["pointmap"] = pointmaps[index].reshape(-1, 3)
            frame_info["camera_pose"] = camera_poses[index]

            frame_info["scale"] = np.array(frame_info["post_postprocess"]["scale"]).reshape(-1)
            frame_info["rotation"] = frame_info["post_postprocess"]["rotation"]
            frame_info["translation"] = frame_info["post_postprocess"]["translation"]

            scale = np.array(frame_info["scale"]).reshape(-1).tolist()
            scale = np.array(scale + [1]) 
            scale_transform = np.diag(scale)
            rotation_transform = np.eye(4)
            rotation_transform[:3, :3] = R.from_quat(np.array(frame_info["rotation"]).reshape(-1), scalar_first=True).as_matrix().T
            translation_transform = np.eye(4)
            translation_transform[:3, 3] = np.array(frame_info["translation"])
            transform = PYTORCH_TO_OPENCV_4 @ translation_transform @ rotation_transform @ scale_transform @ Z_UP_TO_Y_UP_4
            
            frame_info["obj_in_cam"] = transform
            frame_info["obj_in_world"] = frame_info["camera_pose"] @ transform
            
            
            frame_info["world_position"] = frame_info["obj_in_world"][:3, 3]
            frame_info["world_rotation"] = frame_info["obj_in_world"][:3, :3] / frame_info["scale"]

        # Package the data
        data = {
            "frames": frame_data,
            "mesh": mesh,
            "mesh_path": mesh_path,
            "collision_meshes": collision_meshes,
        }
        meshes[mesh_path.stem] = data


    # Load the walls
    floorplan_path = scene_dir / "floorplan" / "room_parameters.json"
    scene_data = json.loads(floorplan_path.read_text())

    vlm_analysis_data = json.loads((scene_dir / "vlm_analysis.json").read_text())
    ceiling_height = vlm_analysis_data["ceiling_height"]

    # 1. Define your vertices (a list of [x, y] tuples)
    vertices = [x[:2] for x in scene_data["boundary3d"]]
    vertices = vertices[:len(vertices)//2]  # Use only the first half, the second are just the ceiling
    vertices = np.array(vertices)

    wall_thickness = 0.1
    additional_buffer = 0.1

    # 2. Create a Shapely polygon
    inner_polygon = Polygon(vertices).buffer(additional_buffer, join_style="mitre")

    collision_manager = trimesh.collision.CollisionManager()
    output_data = {}

    for mesh_name, data in meshes.items():
        if mesh_name.rsplit("-", 2)[0] in ("curtain", "pillow"):
            continue

        # Get the median position
        avg_position = np.median([frame["world_position"] for frame in data["frames"].values()], axis=0)

        # If the average position is not inside the polygon, skip
        if not inner_polygon.contains(Point(avg_position[0], avg_position[1])):
            continue
        if avg_position[2] < 0 or avg_position[2] > ceiling_height:
            continue

        # Pick the median scale
        scales = [np.array(frame["scale"]).reshape(-1) for frame in data["frames"].values()]
        frames_list = list(data["frames"].values())
        scale_norms = [np.linalg.norm(s) for s in scales]
        median_scale_idx = np.argsort(scale_norms)[len(scale_norms) // 2]
        avg_scale = scales[median_scale_idx]

        # Pick the rotation corresponding to the median scale
        avg_rotation = R.from_matrix(frames_list[median_scale_idx]["world_rotation"])
        snapped_rotation = snap_rotation(avg_rotation)

        # Compose into a transform for easy use below
        avg_tf = np.eye(4)
        avg_tf[:3, :3] = snapped_rotation.as_matrix() @ np.diag(avg_scale)
        avg_tf[:3, 3] = avg_position

        # Transform the mesh into the world space
        mesh = data["mesh"].copy()
        mesh.apply_transform(avg_tf)

        # Check if the AABB of the mesh is clipping into the floor. If it is, move it up by the clipping amount
        aabb_min, aabb_max = mesh.bounds
        if aabb_min[2] < 0:
            move_tf = np.eye(4)
            move_tf[2, 3] = -aabb_min[2]
            mesh.apply_transform(move_tf)
            avg_position[2] += aabb_min[2]
            avg_tf = move_tf @ avg_tf

        for i, cmesh in enumerate(data["collision_meshes"]):
            cmesh.apply_transform(avg_tf)
            collision_manager.add_object(f"{mesh_name}-{i}", cmesh)

        output_data[mesh_name] = {
            "scale": avg_scale.tolist(),
            "rotation": snapped_rotation.as_quat().tolist(),
            "position": avg_position.tolist(),
        }

    # Repeatedly remove collision worst offenders until we are clear of collisions above a threshold
    while True:
        in_collision, name_pairs, datas = collision_manager.in_collision_internal(return_names=True, return_data=True)
        if not in_collision:
            break
        max_depths = {}
        for d in datas:
            d_objs = {x.rsplit("-", 1)[0] for x in d.names}
            if len(d_objs) == 1:
                continue
            pair = tuple(sorted(d_objs))
            if max_depths.get(pair, 0) < d.depth:
                max_depths[pair] = d.depth

        if len(max_depths) == 0:
            break
        max_depth_pair = max(max_depths, key=max_depths.get)
        if max_depths[max_depth_pair] < COLLISION_THRESHOLD:
            break
        obj_a, obj_b = sorted(max_depth_pair)

        # Figure out what object to remove
        obj_a_size = sum(cmesh.volume for cmesh in meshes[obj_a]["collision_meshes"])
        obj_b_size = sum(cmesh.volume for cmesh in meshes[obj_b]["collision_meshes"])
        if obj_a_size < obj_b_size:
            obj_to_remove = obj_a
        else:
            obj_to_remove = obj_b

        # Do the actual removal
        for i in range(len(meshes[obj_to_remove]["collision_meshes"])):
            collision_manager.remove_object(f"{obj_to_remove}-{i}")
        del output_data[obj_to_remove]

    # Return the output data dict
    return output_data


def load_object(room_dir, mesh_name, scale):
    """Load a regular object (furniture, etc.) from the dataset."""
    scene_id = get_scene_id(room_dir)
    in_rooms = [room_dir.name]
    # Match naming from import_vid2room_objects.py
    base_category = mesh_name.rsplit("-", 2)[0]
    category = "".join(c if c.isalnum() or c == "_" else "_" for c in base_category.lower())
    model = f"{scene_id}_{mesh_name}".replace("-", "_")
    model = "".join(c if c.isalnum() or c == "_" else "" for c in model.lower())
    fixed_base = True

    i = len(og.sim.scenes[0].objects)
    obj = DatasetObject(
        name=f"{category}_{i}",
        category=category,
        model=model,
        fixed_base=fixed_base,
        dataset_name="vid2room",
        scale=scale,
        in_rooms=in_rooms,
    )

    og.sim.scenes[0].add_object(obj)

    return obj


def load_structure_objects(room_dir):
    """
    Load all pre-generated structure objects (floor, wall, ceiling) as a DatasetObject.

    The material is already baked into the USD during the import process via
    import_vid2room_scene_structures.py.

    Args:
        scene_id: The Vid2Room scene identifier (e.g., "train_0")

    Returns:
        List of DatasetObject: The loaded structure objects
    """
    # Map structure type to category
    scene_id = get_scene_id(room_dir)
    in_rooms = [room_dir.name]
    dataset_root = pathlib.Path(gm.DATA_PATH) / "vid2room"
    objects = []
    for category in ("floors", "walls", "ceilings"):
        category_root = dataset_root / "objects" / category
        model_prefix = f"vid2room_{scene_id}_"
        models = []
        if category_root.exists():
            for model_dir in category_root.iterdir():
                if not model_dir.is_dir():
                    continue
                if not model_dir.name.startswith(model_prefix):
                    continue
                if not (model_dir / "import.success").exists():
                    continue
                models.append(model_dir.name)
        for model in models:
            i = len(og.sim.scenes[0].objects)
            obj = DatasetObject(
                name=f"{category}_{i}",
                category=category,
                model=model,
                fixed_base=True,
                dataset_name="vid2room",
                in_rooms=in_rooms,
            )

            og.sim.scenes[0].add_object(obj)
            objects.append(obj)

    return objects


def get_scene_id(room_dir):
    """
    Generate a unique scene ID from scene path.
    
    For vid2room scenes, the path structure is typically:
    .../vid_XXXXX/rooms/room_type_N
    
    We want to create a unique ID like: vid_XXXXX_room_type_N
    """
    room_name = room_dir.name  # e.g., "living_room_0"
    # Go up to find the video ID (parent of "rooms" directory)
    video_id = room_dir.parent.parent.name  # e.g., "vid_1vdXN7X4Af4"
    assert video_id.startswith("vid_"), f"Video ID {video_id} does not start with 'vid_'"
    return f"{video_id}_{room_name}"


def load_vid2room_scene(room_dir):
    """
    Process and load a Vid2Room scene.

    Args:
        room_dir: Path to the room root directory
    """
    object_data = parse_object_meshes(room_dir)

    ogscene = Scene(use_floor_plane=True, floor_plane_visible=False)
    og.sim.import_scene(ogscene)

    load_structure_objects(room_dir)

    for mesh_name, data in object_data.items():
        obj = load_object(room_dir, mesh_name, data["scale"])
        obj.set_position_orientation(position=th.as_tensor(data["position"]), orientation=th.as_tensor(data["rotation"]))

    og.sim.play()

    for _ in range(10):
        og.sim.step()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_vid2room_scene.py <room_root>")
        sys.exit(1)

    room_root = sys.argv[1]

    if og.sim:
        og.clear()
    else:
        og.launch()

    load_vid2room_scene(room_root)

    while True:
        og.sim.render()
