import csv
import gzip
import pathlib
import json
import traceback
import os

import numpy as np
from omnigibson.scenes.scene_base import Scene
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.physx_utils import bind_material
import torch as th
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import create_primitive_mesh, scene_relative_prim_path_to_absolute
import omnigibson.lazy as lazy
import trimesh
import shapely
from scipy.spatial.transform import Rotation as R

import omnigibson as og
from omnigibson.macros import gm

AI2_OBJECTS = json.loads((pathlib.Path(gm.DATA_PATH) / "ai2thor" / "object_name_mapping.json").read_text())
SPOC_OBJECTS = json.loads((pathlib.Path(gm.DATA_PATH) / "spoc" / "object_name_mapping.json").read_text())
MDL_MATERIAL_ROOT = "/fsx-siro/cgokmen/og-materials/Materials/2023_2_1"


def convert_csv_to_dict(filepath):
    """
    Converts a 2-column CSV file into a key:value dictionary.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        dict: A dictionary created from the CSV data.
    """
    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        # Use a dictionary comprehension for a clean and efficient conversion
        return {row[0]: row[1] for row in reader}


AI2_MDL_MAPPING_FN = "/home/cgokmen/projects/BEHAVIOR-1K/slurm/ai2_nvidia_material_mapping.csv"
AI2_MDL_MAPPING = convert_csv_to_dict(AI2_MDL_MAPPING_FN)
MDL_PATHS_FN = "/home/cgokmen/projects/BEHAVIOR-1K/slurm/material_paths.csv"
MDL_PATHS = convert_csv_to_dict(MDL_PATHS_FN)
ROTATE_EVERYTHING_BY = th.as_tensor(R.from_euler("x", 90, degrees=True).as_quat())
with open("/fsx-siro/cgokmen/procthor/assets/2023_07_28/annotations.json") as f:
    ANNOTATIONS = json.load(f)


def load_object(mesh_name, fixed_base):
    i = len(og.sim.scenes[0].objects)
    if mesh_name in AI2_OBJECTS:
        category, model = AI2_OBJECTS[mesh_name]
        obj = DatasetObject(
            name=f"{category}_{i}",
            category=category,
            model=model,
            fixed_base=fixed_base,
            dataset_type="ai2thor",
            # scale=th.tensor([-1., 1., 1.])
        )
    elif mesh_name in SPOC_OBJECTS:
        category, model = SPOC_OBJECTS[mesh_name]
        scale = th.ones(3) / ANNOTATIONS[mesh_name]["scale"]
        obj = DatasetObject(
            name=f"{category}_{i}",
            category=category,
            model=model,
            fixed_base=fixed_base,
            dataset_type="spoc",
            scale=scale,
        )
    else:
        raise ValueError(f"Unknown mesh name: {mesh_name}")

    og.sim.scenes[0].add_object(obj)

    return obj


def polygon_to_mesh(name, points, material_name, convex_hull=False):
    """
    Create a trimesh object from a list of 3D points defining a planar polygon.

    Args:
        points: List or array of 3D points [(x1,y1,z1), (x2,y2,z2), ...]
                Points should be ordered around the polygon boundary.

    Returns:
        trimesh.Trimesh: The triangulated mesh
    """
    points = np.copy(points)
    points[:, 0] *= -1  # Invert x-coordinates
    # Create a 3D path and then convert to mesh.
    lines = list(range(len(points))) + [0]  # Close the polygon by connecting the last point to the first
    path = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(lines)], vertices=points, process=False)

    # Convert path to 2D, triangulate, then back to 3D
    planar, to_3D = path.to_2D()

    # Get the convex hull
    if convex_hull:
        points_2d = planar.vertices
        multipoint = shapely.MultiPoint(points_2d)
        polygon = multipoint.convex_hull
        points_convex = np.array(polygon.exterior.coords[:-1])
        lines_convex = list(range(len(points_convex))) + [0]  # Close the polygon
        planar = trimesh.path.Path2D(
            entities=[trimesh.path.entities.Line(lines_convex)],
            vertices=points_convex,
            process=False,
        )
    verts_2d, faces = planar.triangulate()
    points_3d = np.hstack((verts_2d, np.zeros((verts_2d.shape[0], 1))))  # Add z=0 for 3D mesh
    mesh = trimesh.Trimesh(vertices=points_3d, faces=faces, process=False)
    mesh.apply_transform(to_3D)

    rel_prim_path = f"/{name}"
    abs_prim_path = scene_relative_prim_path_to_absolute(og.sim.scenes[0], rel_prim_path)
    create_primitive_mesh(abs_prim_path, "Plane")

    xp = XFormPrim(rel_prim_path, name=name)
    xp.load(og.sim.scenes[0])
    xp.set_position_orientation(orientation=ROTATE_EVERYTHING_BY)
    prim = xp.prim

    # Convert that trimesh mesh to a mesh prim
    # Update the mesh prim to store the new information. First update the non-configuration-
    # dependent fields
    face_vertex_counts = th.tensor([len(face) for face in mesh.faces], dtype=int).cpu().numpy()
    prim.GetAttribute("points").Set(lazy.pxr.Vt.Vec3fArray.FromNumpy(mesh.vertices))
    prim.GetAttribute("faceVertexCounts").Set(face_vertex_counts)
    prim.GetAttribute("faceVertexIndices").Set(mesh.faces.flatten())
    prim.GetAttribute("normals").Set(lazy.pxr.Vt.Vec3fArray.FromNumpy(mesh.vertex_normals))
    prim.GetAttribute("primvars:st").Set(lazy.pxr.Vt.Vec2fArray.FromNumpy(verts_2d[:, :2][mesh.faces.flatten()]))

    # Create and bind the material
    assert material_name in AI2_MDL_MAPPING, f"Unknown material name: {material_name}"
    mdl_material_name = AI2_MDL_MAPPING[material_name]
    assert mdl_material_name, f"Unknown MDL material name for: {material_name}"
    mdl_path = os.path.join(MDL_MATERIAL_ROOT, MDL_PATHS[mdl_material_name], f"{mdl_material_name}.mdl")
    assert os.path.exists(mdl_path), f"MDL path does not exist: {mdl_path}"
    lazy.omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name=mdl_path,
        mtl_name=mdl_material_name,
        bind_selected_prims=[prim.GetPath()],
    )

    return prim


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


def process_objects(objects):
    for objinfo in objects:
        try:
            obj_name = objinfo["id"]
            model = objinfo["assetId"]
            obj = load_object(model, objinfo["kinematic"])
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
            process_objects(objinfo["children"])  # global_transform)


def process_scene(scene):
    ogscene = Scene(use_floor_plane=True, floor_plane_visible=False)
    og.sim.import_scene(ogscene)

    print("Processing rooms...")
    for i, room in enumerate(scene["rooms"]):
        polygon_to_mesh(
            f"room_{i}_floor",
            np.array([[pt["x"], pt["y"], pt["z"]] for pt in room["floorPolygon"]]),
            room["floorMaterial"]["name"]
            if "floorMaterial" in room and "name" in room["floorMaterial"]
            else "Parquet_Floor",
            convex_hull=False,
        )

    print("Processing walls...")
    for i, wall in enumerate(scene["walls"]):
        polygon_to_mesh(
            f"wall_{i}",
            np.array([[pt["x"], pt["y"], pt["z"]] for pt in wall["polygon"]]),
            wall["material"]["name"] if "material" in wall and "name" in wall["material"] else "Plaster",
            convex_hull=True,
        )

    print("Processing objects...")
    process_objects(scene["objects"])

    og.sim.play()
    for _ in range(30):
        og.sim.step()


def load_spoc_scene(scene_name):
    split, i = scene_name.rsplit("_", 1)
    i = int(i)
    raw_houses = []

    print("Seeking")
    with open(split, "r") as f:
        for j, line in enumerate(f):
            if j < i:
                continue
            print("Found. Loading")
            process_scene(json.loads(line))
            break
        else:
            raise ValueError(f"Scene {scene_name} not found.")


if __name__ == "__main__":
    if og.sim:
        og.clear()
    else:
        og.launch()

    load_spoc_scene("val_3")

    while True:
        og.sim.render()
