"""
Credit: Andrey Kurenkov
"""

import argparse
import json
import math
import os
import signal
import xml.etree.ElementTree as ET

import numpy as np

import trimesh
from pyquaternion import Quaternion

"""
Analyzes a model for possible ways to place it flat on a surface.
Use by running without --ask_probs or --save_json to see rotations, and then with to save them with probabilities.
"""


def viz_transform(body_id, mesh, transform):
    quat = Quaternion(matrix=transform)
    r = quat.real
    v = quat.vector
    p.resetBasePositionAndOrientation(
        body_id, [0, 0, mesh.extents[2] / 2.0], [quat.vector[0], quat.vector[1], quat.vector[2], quat.real]
    )
    return r, v


def compute_poses(mesh, threshold):
    poses = trimesh.poses.compute_stable_poses(mesh, n_samples=5, threshold=threshold)
    return poses


class TimeoutError(Exception):
    pass


def handler(signum, frame):
    raise TimeoutError()


def main(args):
    p.connect(p.GUI)
    cat_ids = []
    if args.object_cat is not None:
        cat_dir = "data/ig_dataset/objects/%s" % (args.object_cat)
        for obj_id in os.listdir(cat_dir):
            cat_ids.append((args.object_cat, obj_id))
    elif args.cat_file is not None:
        with open(args.cat_file, "r") as f:
            for line in f:
                cat = line.strip()
                cat_dir = "data/ig_dataset/objects/%s" % (cat)
                for obj_id in os.listdir(cat_dir):
                    cat_ids.append((cat, obj_id))
    else:
        with open(args.cat_id_file, "r") as f:
            for line in f:
                cat_ids.append(line.strip().split("/"))
    for object_cat, object_id in cat_ids:
        print("Processing %s %s" % (object_cat, object_id))
        metadata_file = "data/ig_dataset/objects/%s/%s/misc/metadata.json" % (object_cat, object_id)
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        if args.skip_processed and "orientations" in metadata and len(metadata["orientations"]) > 0:
            continue

        urdf_file = "data/ig_dataset/objects/%s/%s/%s.urdf" % (object_cat, object_id, object_id)
        tree = ET.parse(urdf_file)
        joints = tree.findall("joint")
        links = tree.findall("link")
        # Find the base link
        children_links = [joint.find("child").attrib["link"] for joint in joints]
        base_link = [link for link in links if link.attrib["name"] not in children_links][0]
        object_file = base_link.find("collision/geometry/mesh").attrib["filename"]
        # Use the collision mesh of the base link
        object_file = "data/ig_dataset/objects/%s/%s/%s" % (object_cat, object_id, object_file)

        mesh = trimesh.load(object_file, force="mesh")

        poses = [np.eye(4)]
        # set the timeout handler
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        try:
            new_poses = compute_poses(mesh, args.threshold)
            for pose in new_poses[0]:
                poses.append(pose)
        except TimeoutError as exc:
            pass
        finally:
            signal.alarm(0)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=object_file)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=object_file)
        body_id = p.createMultiBody(baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId)

        info_dict = {}
        aabb = p.getAABB(body_id)
        print("Showing all stable placement rotations:")
        dicts = []

        for i in range(len(poses)):
            print("Num poses:", len(poses))
            rotation_dict = {}
            transform = poses[i]
            r, v = viz_transform(body_id, mesh, transform)
            prob = 0
            if args.save_json:
                inp = input("Enter probability of rotation, or +x/-x or/ +y/-y or +z/-z to rotate:")
                if inp == "":
                    continue
                while inp[0] == "+" or inp[0] == "-":
                    rot_num = float(inp.split()[1])
                    if inp[0] == "-":
                        rot_num *= -1
                    if inp[1] == "z":
                        rot = np.array(
                            [
                                [math.cos(math.pi * rot_num), -math.sin(math.pi * rot_num), 0.0, 0.0],
                                [math.sin(math.pi * rot_num), math.cos(math.pi * rot_num), 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    elif inp[1] == "y":
                        rot = np.array(
                            [
                                [math.cos(math.pi * rot_num), 0.0, math.sin(math.pi * rot_num), 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [-math.sin(math.pi * rot_num), 0.0, math.cos(math.pi * rot_num), 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    elif inp[1] == "x":
                        rot = np.array(
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, math.cos(math.pi * rot_num), -math.sin(math.pi * rot_num), 0.0],
                                [0.0, math.sin(math.pi * rot_num), math.cos(math.pi * rot_num), 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    transform = np.matmul(rot, transform)
                    r, v = viz_transform(body_id, mesh, transform)
                    inp = input("Enter probability of rotation, or +/- to rotate about Z:")
                prob = float(inp)
                variation = float(input("Enter variation about Z (0-1):"))
                rotation_dict["prob"] = prob
                rotation_dict["variation"] = variation
            else:
                skip = input("Rotation %d: (press enter to continue)" % (i + 1))
            rotation_dict["rotation"] = [float(v[0]), float(v[1]), float(v[2]), float(r)]
            aabb = p.getAABB(body_id)
            size = [aabb[1][0] - aabb[0][0], aabb[1][1] - aabb[0][1], aabb[1][2] - aabb[0][2]]
            print("Bounding box size=%s" % str(size))
            rotation_dict["size"] = size
            if prob > 0:
                dicts.append(rotation_dict)

        print("Summary:")
        for d in dicts:
            print(d)

        if args.save_json:
            metadata["orientations"] = dicts

            with open(metadata_file, "w") as f:
                documents = json.dump(metadata, f)

        p.removeBody(body_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze objects that can be placed in a container for their plausible rotations."
    )
    parser.add_argument("--object_cat", type=str, default=None, help="A category to set rotation for")
    parser.add_argument(
        "--cat_id_file",
        type=str,
        default=None,
        help="A text file containing category and id of each object to set rotation for, one per line",
    )
    parser.add_argument(
        "--cat_file",
        type=str,
        default=None,
        help="A text file containing category and id of each object to set rotation for, one per line",
    )
    parser.add_argument(
        "--save_json", action="store_true", help="Whether to ask for and save orientations and probabilities to json."
    )
    parser.add_argument("--threshold", type=float, default=0.03, help="Threshold for including orientations or not.")
    parser.add_argument("--skip_processed", action="store_true")
    args = parser.parse_args()
    if args.object_cat is None and args.cat_id_file is None and args.cat_file is None:
        raise ValueError("Either object_cat or cat_id_file or cat_file must be set")
    main(args)
