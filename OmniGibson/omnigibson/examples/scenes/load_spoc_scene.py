import pathlib
import json
import traceback

from omnigibson.scenes.scene_base import Scene
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
import torch as th
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

import omnigibson as og
from omnigibson.macros import gm

AI2_OBJECTS = json.loads((pathlib.Path(gm.DATA_PATH) / "ai2thor" / "object_name_mapping.json").read_text())
SPOC_OBJECTS = json.loads((pathlib.Path(gm.DATA_PATH) / "spoc" / "object_name_mapping.json").read_text())

ROTATE_EVERYTHING_BY = th.as_tensor(R.from_euler("x", 90, degrees=True).as_quat())

# Load SPOC object annotations for scale information
SPOC_ANNOTATIONS_PATH = "/checkpoint/clear/cgokmen/procthor/assets/2023_07_28/annotations.json"
with open(SPOC_ANNOTATIONS_PATH) as f:
    ANNOTATIONS = json.load(f)


def load_object(mesh_name, fixed_base, in_rooms=None):
    """Load a regular object (furniture, etc.) from the dataset."""
    i = len(og.sim.scenes[0].objects)
    if mesh_name in AI2_OBJECTS:
        category, model = AI2_OBJECTS[mesh_name]
        obj = DatasetObject(
            name=f"{category}_{i}",
            category=category,
            model=model,
            fixed_base=fixed_base,
            dataset_name="ai2thor",
            in_rooms=in_rooms,
        )
    elif mesh_name in SPOC_OBJECTS:
        category, model = SPOC_OBJECTS[mesh_name]
        scale = th.ones(3) / ANNOTATIONS[mesh_name]["scale"]
        obj = DatasetObject(
            name=f"{category}_{i}",
            category=category,
            model=model,
            fixed_base=fixed_base,
            dataset_name="spoc",
            scale=scale,
            in_rooms=in_rooms,
        )
    else:
        raise ValueError(f"Unknown mesh name: {mesh_name}")

    og.sim.scenes[0].add_object(obj)

    return obj


def load_structure_object(scene_id, struct_type, struct_idx, in_rooms=None):
    """
    Load a pre-generated structure object (floor, wall, ceiling) as a DatasetObject.

    The material is already baked into the USD during the import process via
    import_spoc_scene_structures.py.

    Args:
        scene_id: The SPOC scene identifier (e.g., "train_0")
        struct_type: Type of structure ("floor", "wall", or "ceiling")
        struct_idx: Index of this structure within the scene
        in_rooms: Room assignment for this structure

    Returns:
        DatasetObject: The loaded structure object
    """
    # Map structure type to category
    category_map = {
        "floor": "floors",
        "wall": "walls",
        "ceiling": "ceilings",
    }
    category = category_map.get(struct_type, "structures")

    # Model name matches what import_spoc_scene_structures.py generates
    model = f"spoc_{scene_id}_{struct_type}_{struct_idx}".replace("-", "_")

    i = len(og.sim.scenes[0].objects)
    obj = DatasetObject(
        name=f"{struct_type}_{i}",
        category=category,
        model=model,
        fixed_base=True,
        dataset_name="spoc",
        in_rooms=in_rooms,
    )

    og.sim.scenes[0].add_object(obj)
    obj.set_position_orientation(orientation=ROTATE_EVERYTHING_BY)

    return obj


def unity_euler_to_rh_quaternion(unity_euler_degrees):
    """
    Converts Euler angles from Unity (left-handed, ZXY order) to a right-handed quaternion.

    Args:
        unity_euler_degrees (list or tuple): A list of three Euler angles [x, y, z] in degrees from Unity.

    Returns:
        numpy.ndarray: The converted quaternion in [x, y, z, w] format.
    """
    # 1. Get Euler angles from Unity
    # Unity's eulerAngles property is (pitch, yaw, roll) -> (x, y, z)
    euler_x, euler_y, euler_z = unity_euler_degrees

    # 2. Adjust for the left-handed to right-handed coordinate system conversion
    # Inverting the Z-axis flips the sign of rotations around X and Y.
    rh_euler_x = -euler_x
    rh_euler_y = -euler_y
    rh_euler_z = euler_z  # Roll around Z remains the same

    # 3. Create a Rotation object in scipy
    # Unity's rotation order is Z-X-Y. Scipy's from_euler method needs this
    # order specified as a string 'zxy'.
    # The angles must be provided in the same order: [z, x, y].
    rotation = R.from_euler("zxy", [rh_euler_z, rh_euler_x, rh_euler_y], degrees=True)

    # 4. Get the quaternion
    # Scipy returns quaternions in [x, y, z, w] format.
    quaternion = rotation.as_quat()

    return quaternion


def process_objects(objects, room_id=None):
    """Process and load objects from the scene JSON."""
    for objinfo in objects:
        try:
            obj_name = objinfo["id"]
            model = objinfo["assetId"]
            in_rooms = [room_id] if room_id else None
            obj = load_object(model, objinfo["kinematic"], in_rooms=in_rooms)
            position = th.tensor([-objinfo["position"]["x"], objinfo["position"]["y"], objinfo["position"]["z"]])
            orn = th.as_tensor(
                unity_euler_to_rh_quaternion(
                    [objinfo["rotation"]["x"], objinfo["rotation"]["y"], objinfo["rotation"]["z"]]
                ),
                dtype=th.float32,
            )

            rotated_pos, rotated_orn = T.pose_transform(th.zeros(3), ROTATE_EVERYTHING_BY, position, orn)

            # rotate the object such that we know the scale inside the bbox
            obj.set_bbox_center_position_orientation(rotated_pos, rotated_orn)
        except Exception as e:
            print(f"Could not load {obj_name} ({model})", traceback.format_exc())

        if "children" in objinfo:
            process_objects(objinfo["children"], room_id=room_id)


def process_scene(scene, scene_id):
    """
    Process and load a SPOC scene.

    Args:
        scene: Parsed scene JSON data
        scene_id: Unique scene identifier (e.g., "train_0")
    """
    ogscene = Scene(use_floor_plane=True, floor_plane_visible=False)
    og.sim.import_scene(ogscene)

    # Build room ID mapping
    room_ids = {}
    for i, room in enumerate(scene.get("rooms", [])):
        room_id = room.get("id")
        assert room_id, "Invalid room ID"
        room_ids[i] = room_id.replace("|", "_")

    print("Processing floors...")
    for i, room in enumerate(scene.get("rooms", [])):
        if "floorPolygon" not in room:
            continue
        room_id = room_ids.get(i)
        try:
            load_structure_object(scene_id, "floor", i, in_rooms=[room_id] if room_id else None)
        except Exception as e:
            print(f"Could not load floor {i}: {traceback.format_exc()}")

    print("Processing walls...")
    for i, wall in enumerate(scene.get("walls", [])):
        if "polygon" not in wall:
            continue
        # Try to get room info from wall if available
        room_id = wall.get("roomId", None)
        try:
            load_structure_object(scene_id, "wall", i, in_rooms=[room_id] if room_id else None)
        except Exception as e:
            print(f"Could not load wall {i}: {traceback.format_exc()}")

    print("Processing ceilings...")
    for i, room in enumerate(scene.get("rooms", [])):
        if "ceilingPolygon" not in room:
            continue
        room_id = room_ids.get(i)
        try:
            load_structure_object(scene_id, "ceiling", i, in_rooms=[room_id] if room_id else None)
        except Exception as e:
            print(f"Could not load ceiling {i}: {traceback.format_exc()}")

    print("Processing objects...")
    process_objects(scene["objects"])

    og.sim.play()

    # Wait for stability
    for _ in range(5000):
        if all(
            [
                th.all(obj.get_linear_velocity() < 1e-4).item()
                for obj in og.sim.scenes[0].objects
                for link in obj.links.values()
                if isinstance(link, RigidDynamicPrim)
            ]
        ):
            break
        og.sim.step()
    else:
        print("Warning: scene did not stabilize")


def get_scene_id(split_path, index):
    """Generate a unique scene ID from split path and index."""
    split_name = pathlib.Path(split_path).stem
    return f"{split_name}_{index}"


def load_spoc_scene(scene_name):
    """
    Load a SPOC scene by name.

    Args:
        scene_name: Scene identifier in format "<split_path>_<index>"
                    e.g., "/path/to/train.jsonl_42"
    """
    split, i = scene_name.rsplit("_", 1)
    i = int(i)
    scene_id = get_scene_id(split, i)

    print(f"Loading SPOC scene: {scene_id}")
    with open(split, "r") as f:
        for j, line in enumerate(f):
            if j < i:
                continue
            print("Found. Loading")
            process_scene(json.loads(line), scene_id)
            break
        else:
            raise ValueError(f"Scene {scene_name} not found.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_spoc_scene.py <scene_name>")
        print("  scene_name: Path to JSONL file with index, e.g.:")
        print("    /checkpoint/clear/cgokmen/procthor/houses/houses_2023_07_28/val.jsonl_3")
        sys.exit(1)

    scene_name = sys.argv[1]

    if og.sim:
        og.clear()
    else:
        og.launch()

    load_spoc_scene(scene_name)

    while True:
        og.sim.render()
