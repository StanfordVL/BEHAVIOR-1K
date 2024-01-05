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
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.sensors.scan_sensor import ScanSensor
from omnigibson.sensors.vision_sensor import VisionSensor
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
import h5py

from PIL import Image
from tqdm import tqdm


def step_sim(time):
    for _ in range(int(time*100)):
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action[0])

def main(iterations):
    DIST_COEFF = 0.1
    GRASP_REWARD = 0.3

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
        
    # Testing primitives with env
    #############################
    # controller = env.task._primitive_controller
    # obj = env.scene.object_registry("name", "cologne")
    # for i in tqdm(range(int(iterations))):
    #     try:
    #         obs = env.reset()
    #         timestep = 0
    #         for action in controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj):
    #             obs, reward, done, truncated, info = env.step(action)
    #             print(reward)
    #             truncated = True if timestep >= 400 else truncated
    #             timestep += 1
    #             if done or timestep >= 400:
    #                 for action in controller._execute_release():
    #                     action = action[0]
    #                     env.step(action)
    #                 break
    #     except Exception as e:
    #         print("Error in iteration: ", i)
    #         print(e)
    #         print('--------------------')

    # Testing random actions with env
    #############################
    for i in tqdm(range(int(iterations))):
        try:
            done = False
            env.reset()
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print("Error in iteration: ", i)
            print(e)
            print('--------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run worker")
    parser.add_argument("iterations")
    
    args = parser.parse_args()
    main(args.iterations)

    # seg semantic - 224 x 224
    # seg instance - 224 x 224
    # depth - 224 x 224
    # rgb - 224 x 224 x 4