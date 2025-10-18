import argparse
import json
import math
import os
import random
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    PlanningContext,
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision
from memory_profiler import profile


def pause_step(time):
    for _ in range(int(time * 100)):
        og.sim.step()


def get_random_joint_position(robot):
    joint_positions = []
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    joints = np.array([joint for joint in robot.joints.values()])
    arm_joints = joints[joint_control_idx]
    for i, joint in enumerate(arm_joints):
        val = random.uniform(joint.lower_limit, joint.upper_limit)
        joint_positions.append(val)
    return joint_positions, joint_control_idx


def append_to_json(file_path, data):
    # Check if the file exists
    if not os.path.exists(file_path):
        # Create the file and initialize it with an empty list
        with open(file_path, 'w') as file:
            json.dump([], file)

    # Open the file in read/write mode ('r+')
    with open(file_path, "r+") as file:
        file.seek(0, 2)  # Move to the end of the file
        file.seek(file.tell() - 1, 0)  # Go back 1 character from the end to overwrite the ']'
        if file.tell() > 1:  # If the file is not empty or contains more than just []
            file.write(",")  # Write a comma to separate the last item
        stringify_data = json.dumps(data)
        append_format = stringify_data[1:-1]
        file.write(append_format)  # Convert the dictionary to JSON and write it
        file.write("]")  # Close the JSON array

@profile
def main(iterations, file_path):
    MAX_JOINT_RANDOMIZATION_ATTEMPTS = 50

    cfg = {
        "env": {
            "action_frequency": 10,
            "physics_frequency": 60,
            "flatten_obs_space": True,
            "flatten_action_space": True,
            "external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {"image_width": 224, "image_height": 224},
                    "local_position": [-0.5, -2.0, 1.0],
                    "local_orientation": [0.707, 0.0, 0.0, 0.707],
                }
            ],
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["proprio"],
                "proprio_obs": ["joint_qpos", "joint_qvel", "eef_0_pos", "eef_0_quat", "grasp_0"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "controller_config": {
                    "base": {
                        "name": "DifferentialDriveController",
                    },
                    # "arm_0": {
                    #     "name": "InverseKinematicsController",
                    #     "motor_type": "velocity",
                    #     "command_input_limits": (np.array([-0.2, -0.2, -0.2, -np.pi, -np.pi, -np.pi]),
                    #     np.array([0.2, 0.2, 0.2, np.pi, np.pi, np.pi])),
                    #     "command_output_limits": None,
                    #     "mode": "pose_absolute_ori",
                    #     "kv": 3.0
                    # },
                    "arm_0": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "use_delta_commands": False,
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "mode": "binary",
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "use_delta_commands": False,
                    },
                },
            }
        ],
        "objects": [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
            },
        ],
    }

    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.HEADLESS = True

    env = og.Environment(configs=cfg)
    # primitive_controller = env.task._primitive_controller
    primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    robot = env.robots[0]
    obj = env.scene.object_registry("name", "cologne")
    place_objects = []
    objects_name = [
        ("shelf_njwsoa_1", object_states.Inside),
        ("shelf_owvfik_0", object_states.Inside),
        ("sofa_mnfbbh_0", object_states.OnTop),
        ("ottoman_ycfbsd_0", object_states.OnTop),
        ("shelf_njwsoa_0", object_states.Inside),
        ("coffee_table_fqluyq_0", object_states.OnTop),
        ("straight_chair_eospnr_0", object_states.OnTop),
        ("countertop_tpuwys_0", object_states.OnTop),
        ("bed_zrumze_0", object_states.OnTop),
        ("sink_zexzrc_0", object_states.OnTop),
        ("oven_wuinhm_0", object_states.OnTop),
        ("straight_chair_amgwaw_0", object_states.OnTop),
        ("breakfast_table_skczfi_0", object_states.OnTop),
        ("bottom_cabinet_jhymlr_0", object_states.OnTop)
    ]
    for name, placement in objects_name:
        o = env.scene.object_registry("name", name)
        place_objects.append((o, placement))
    # for category in place_categories:
    #     o = env.scene.object_registry("category", category)
    #     place_objects += list(o)

    # Randomize the robots joint positions
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    dim = len(joint_control_idx)
    # For Tiago
    if "combined" in robot.robot_arm_descriptor_yamls:
        joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        initial_joint_pos = np.array(robot.get_joint_positions()[joint_combined_idx])
        control_idx_in_joint_pos = np.where(np.in1d(joint_combined_idx, joint_control_idx))[0]
    # For Fetch
    else:
        initial_joint_pos = np.array(robot.get_joint_positions()[joint_control_idx])
        control_idx_in_joint_pos = np.arange(dim)

    progress_bar = tqdm(total=len(place_objects) * iterations, desc="Randomizing poses")
    saved_poses = []
    for place_obj, placement in place_objects:
        progress = 0
        while progress < iterations:
            selected_obj_pose = None
            selected_joint_pos = None
            selected_base_pose = None
            try:
                # Randomize object positions
                selected_obj_pose = primitive_controller._sample_pose_with_object_and_predicate(
                    placement, obj, place_obj
                )
                obj.set_position_orientation(*selected_obj_pose)
                og.sim.step()

                with PlanningContext(
                    env, primitive_controller.robot, primitive_controller.robot_copy, "original"
                ) as context:
                    for _ in range(MAX_JOINT_RANDOMIZATION_ATTEMPTS):
                        joint_pos, joint_control_idx = get_random_joint_position(robot)
                        initial_joint_pos[control_idx_in_joint_pos] = joint_pos
                        if not set_arm_and_detect_collision(context, initial_joint_pos):
                            selected_joint_pos = joint_pos
                            # robot.set_joint_positions(joint_pos, joint_control_idx)
                            # og.sim.step()
                            break

                # Randomize the robot's 2d pose
                grasp_poses = get_grasp_poses_for_object_sticky(obj)
                grasp_pose, _ = random.choice(grasp_poses)
                sampled_pose_2d = primitive_controller._sample_pose_near_object(obj, pose_on_obj=grasp_pose)
                # sampled_pose_2d = [-0.433881, -0.210183, -2.96118]
                robot_pose = primitive_controller._get_robot_pose_from_2d_pose(sampled_pose_2d)
                # robot.set_position_orientation(*robot_pose)
                selected_base_pose = robot_pose

                pose = {
                    "joint_pos": selected_joint_pos,
                    "base_pos": selected_base_pose[0].tolist(),
                    "base_ori": selected_base_pose[1].tolist(),
                    "obj_pos": selected_obj_pose[0].tolist(),
                    "obj_ori": selected_obj_pose[1].tolist(),
                    "place_obj_name": place_obj.name,
                }
                saved_poses.append(pose)
                progress += 1
                progress_bar.update(1)
                if progress % 40 == 0:
                    append_to_json(file_path, saved_poses)
                    saved_poses = []

            except Exception as e:
                # print(e)
                # print(place_obj.name)
                # print("--------------------")
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run worker")
    parser.add_argument("iterations")
    parser.add_argument("file_path")

    args = parser.parse_args()
    main(int(args.iterations), args.file_path)

    # seg semantic - 224 x 224
    # seg instance - 224 x 224
    # depth - 224 x 224
    # rgb - 224 x 224 x 4
