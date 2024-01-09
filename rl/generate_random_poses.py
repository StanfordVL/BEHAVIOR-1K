import argparse
from datetime import datetime
import math
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import yaml
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet, PlanningContext
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision

import random
from tqdm import tqdm
import json

def get_random_joint_position(robot):
    joint_positions = []
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    joints = np.array([joint for joint in robot.joints.values()])
    arm_joints = joints[joint_control_idx]
    for i, joint in enumerate(arm_joints):
        val = random.uniform(joint.lower_limit, joint.upper_limit)
        joint_positions.append(val)
    return joint_positions, joint_control_idx

def main(iterations):
    DIST_COEFF = 0.1
    GRASP_REWARD = 0.3
    MAX_JOINT_RANDOMIZATION_ATTEMPTS = 50

    cfg = {
        "env": {
            "action_timestep": 1 / 10.,
            "physics_timestep": 1 / 60.,
            "flatten_obs_space": True,
            "flatten_action_space": True,
            "external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {
                        "image_width": 224,
                        "image_height": 224
                    },
                    "local_position": [-0.5, -2.0, 1.0],
                    "local_orientation": [0.707, 0.0, 0.0, 0.707]
                }
            ],   
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "coffee_table"],
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb", "depth_linear", "seg_instance", "seg_semantic", "proprio"],
                "proprio_obs": ["robot_pos", "robot_2d_ori", "joint_qpos", "joint_qvel", "eef_0_pos", "eef_0_quat", "grasp_0"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "sensor_config": {
                    "VisionSensor": {
                        "modalities": ["rgb", "depth_linear", "seg_instance", "seg_semantic"],
                        "sensor_kwargs": {
                            "image_width": 224,
                            "image_height": 224
                        }
                    }
                },
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
                        "use_delta_commands": False
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "mode": "binary"
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "use_delta_commands": False
                    }
                }
            }
        ],
        "task": {
            "type": "GraspTask",
            "obj_name": "cologne",
            "termination_config": {
                "max_steps": 400,
            },
            "reward_config": {
                "r_dist_coeff": DIST_COEFF,
                "r_grasp": GRASP_REWARD
            }
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
            },
        ]
    }

    gm.USE_GPU_DYNAMICS = True

    env = og.Environment(configs=cfg)
    primitive_controller = env.task._primitive_controller

    robot = env.robots[0]
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


    saved_poses = []
    for i in tqdm(range(iterations)):
        selected_joint_pos = None
        selected_base_pose = None
        try:
            with PlanningContext(primitive_controller.robot, primitive_controller.robot_copy, "original") as context:
                for _ in range(MAX_JOINT_RANDOMIZATION_ATTEMPTS):
                    joint_pos, joint_control_idx = get_random_joint_position(robot)
                    initial_joint_pos[control_idx_in_joint_pos] = joint_pos
                    if not set_arm_and_detect_collision(context, initial_joint_pos):
                        selected_joint_pos = joint_pos
                        # robot.set_joint_positions(joint_pos, joint_control_idx)
                        # og.sim.step()
                        break

            # Randomize the robot's 2d pose
            obj = env.scene.object_registry("name", "cologne")
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
                "base_ori": selected_base_pose[1].tolist()
            }
            saved_poses.append(pose)

        except Exception as e:
            print("Error in iteration: ", i)
            print(e)
            print('--------------------')

    with open('reset_poses.json', 'w') as f:
        json.dump(saved_poses, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run worker")
    parser.add_argument("iterations")
    
    args = parser.parse_args()
    main(int(args.iterations))

    # seg semantic - 224 x 224
    # seg instance - 224 x 224
    # depth - 224 x 224
    # rgb - 224 x 224 x 4