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


def main():

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

    gm.USE_GPU_DYNAMICS = False
    env = og.Environment(configs=cfg)

    robot = env.robots[0]
    obj = env.scene.object_registry("name", "cologne")
    place_obj = env.scene.object_registry("name", "shelf_njwsoa_1")

    primitive_controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    while True:
        selected_obj_pose = primitive_controller._sample_pose_with_object_and_predicate(
            object_states.Inside, obj, place_obj
        )
        obj.set_position_orientation(*selected_obj_pose)
        for i in range(3000):
            og.sim.step()


    # Open the file and load the data
    # with open("reset_poses_varied.json", "r") as file:
    #     reset_poses = json.load(file)


if __name__ == "__main__":
    main()
