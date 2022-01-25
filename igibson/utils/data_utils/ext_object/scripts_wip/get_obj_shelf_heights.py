"""
Credit: Andrey Kurenkov 
"""

import argparse
import os

b
import trimesh
import yaml

from igibson.objects.articulated_object import ArticulatedObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator

"""
Analyzes a model for heights of surfaces within it.
Assumes model is a cabinet or otherwise has sets of shelves within it, so just need to
find the discrete set of heights the shelves are at.
"""

if __name__ == "__main__":
    simulator = Simulator(image_width=640, image_height=640)
    scene = EmptyScene()
    simulator.import_scene(scene)
    parser = argparse.ArgumentParser(description="Finds heights of shelves in a container object.")
    parser.add_argument("object_file", type=str)
    args = parser.parse_args()

    out_dict = {}
    obj = ArticulatedObject(filename=args.object_file, scale=1.0)
    simulator.import_object(obj)
    body_ids = obj.get_body_ids()
    assert len(body_ids) == 1, "Only single-body objects are supported."
    body_id = body_ids[0]
    aabb = pb.getAABB(body_id)
    size = [aabb[1][0] - aabb[0][0], aabb[1][1] - aabb[0][1], aabb[1][2] - aabb[0][2]]

    urdf_dir = os.path.dirname(args.object_file)
    with open(args.object_file, "r") as f:
        past_base_link = False
        past_collision = False
        for line in f:
            if "base_link" in line:
                past_base_link = True
            if past_base_link and "collision" in line:
                past_collision = True
            if past_base_link and past_collision and "mesh filename" in line:
                obj_path = line.split("=")[-1][1:-4]
    obj_path = os.path.join(urdf_dir, obj_path)

    mesh = trimesh.load(obj_path)
    normals = mesh.face_normals
    centers = mesh.triangles_center
    height_counts = {}
    max_height = None
    min_height = None
    for i in range(len(normals)):
        center_height = centers[i][2]
        if max_height is None or center_height > max_height:
            max_height = center_height
        if min_height is None or center_height < min_height:
            min_height = center_height
        normal = normals[i]
        non_z = abs(normal[0]) + abs(normal[1])
        if non_z < 0.01 and normal[2] > 0.99:
            if center_height not in height_counts:
                height_counts[center_height] = 0
            height_counts[center_height] += 1

    heights = []
    for height in height_counts:
        if abs(height - min_height) > 0.05 and abs(height - max_height) > 0.05 and height_counts[height] > 1:
            heights.append(float(height - min_height))

    obj.set_position([0, 0, size[2] / 2])
    aabb = pb.getAABB(body_id)
    heights = sorted(heights)
    shape = pb.createVisualShape(
        pb.GEOM_BOX, rgbaColor=[1, 1, 1, 1], halfExtents=[0.3, 0.3, 0.01], visualFramePosition=[0.0, 0.0, 0.0]
    )
    body_id = pb.createMultiBody(baseVisualShapeIndex=shape, baseCollisionShapeIndex=-1)
    filtered_heights = []
    for height in heights:
        _, old_orn = pb.getBasePositionAndOrientation(body_id)
        pb.resetBasePositionAndOrientation(body_id, [0.0, 0.0, height], old_orn)
        inp = input("Is this a valid shelf height? (y for yes, else no)")
        if inp == "y":
            filtered_heights.append(height)
    out_dict["aabb"] = aabb
    out_dict["size"] = size
    out_dict["shelf_heights"] = filtered_heights

    print("Summary:")
    print(out_dict)

    path = os.path.dirname(args.object_file)
    output_path = os.path.join(path, "info.yaml")

    with open(output_path, "w") as f:
        documents = yaml.dump(out_dict, f)
