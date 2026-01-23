"""
Script to validate a scene for physics
"""

import json
import sys
import os
import math

import torch as th

from omnigibson.macros import gm

gm.HEADLESS = True
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

import omnigibson as og
import omnigibson.utils.transform_utils as T

MAX_POS_DELTA = 0.05  # 5cm
MAX_ORN_DELTA = math.radians(5)  # 5 degrees
MAX_LINEAR_VEL = 0.01  # 0.01 m/s
MAX_ANGULAR_VEL = math.radians(1)  # 1 degree/s


if __name__ == "__main__":
    dataset_root = sys.argv[1]
    scene = sys.argv[2]
    out_path = os.path.join(sys.argv[3], f"{scene}.json")
    gm.DATASET_PATH = str(dataset_root)

    # Load the sim and do stuff
    # If the scene type is interactive, also check if we want to quick load or full load the scene
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene,
        },
    }

    # Load the environment
    env = og.Environment(configs=cfg)

    # Get all the RigidPrims from the scene
    objs = env.scene.objects
    links = {
        obj.name + "-" + link_name: link
        for obj in objs
        for link_name, link in obj.links.items()
    }

    # Store their poses
    initial_poses = {
        link_name: link.get_position_orientation() for link_name, link in links.items()
    }

    # Run the simulation
    print("Stepping simulation.")
    for _ in range(300):
        env.step([])
    print("Done stepping simulation.")

    # Compare the poses
    mismatches = []
    for link_name, link in links.items():
        old_pos, old_orn = initial_poses[link_name]
        new_pos, new_orn = link.get_position_orientation()
        lin_vel = link.get_linear_velocity()
        ang_vel = link.get_angular_velocity()

        delta_pos = th.linalg.norm(new_pos - old_pos).item()
        if delta_pos > MAX_POS_DELTA:
            mismatches.append(
                f"{link_name} position changed by {delta_pos} meters from {old_pos} to {new_pos}."
            )
        delta_orn_mag = T.get_orientation_diff_in_radian(old_orn, new_orn)
        if delta_orn_mag > MAX_ORN_DELTA:
            mismatches.append(
                f"{link_name} orientation changed by {delta_orn_mag} rads from {old_orn} to {new_orn}."
            )
        if th.any(th.abs(lin_vel) > MAX_LINEAR_VEL):
            mismatches.append(f"{link_name} linear velocity is {lin_vel}.")
        if th.any(th.abs(ang_vel) > MAX_ANGULAR_VEL):
            mismatches.append(f"{link_name} angular velocity is {ang_vel}.")

    # Save the results
    with open(out_path, "w") as f:
        json.dump(mismatches, f)

    og.shutdown()
