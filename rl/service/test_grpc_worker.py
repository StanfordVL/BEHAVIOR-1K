import argparse
from datetime import datetime
import math
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import omnigibson as og
from omnigibson.envs.rl_env import RLEnv
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.sensors.scan_sensor import ScanSensor
from omnigibson.sensors.vision_sensor import VisionSensor
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
import h5py

import grpc
import omnigibson_pb2
import omnigibson_pb2_grpc
import io
from PIL import Image
from tqdm import tqdm, trange


def step_sim(time):
    for _ in range(int(time*100)):
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action[0])

def reset_env(env, initial_poses):
    objs = ["cologne", "coffee_table_fqluyq_0"]
    for o in objs:
        env.scene.object_registry("name", o).set_position_orientation(*initial_poses[o])
    og.sim.step()
    return env.reset()


def main():
    DIST_COEFF = 0.1
    GRASP_REWARD = 0.3
    h5py.get_config().track_order = True

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "coffee_table"],
        },
        "robots": [
            {
                "type": "Tiago",
                "obs_modalities": ["rgb", "depth_linear", "seg_instance", "seg_semantic", "proprio"],
                "proprio_obs": ["joint_qpos", "joint_qvel", "eef_left_pos", "eef_left_quat", "grasp_left"],
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
                        "name": "JointController",
                        "motor_type": "velocity"
                    },
                    "arm_left": {
                        "name": "InverseKinematicsController",
                        "motor_type": "velocity",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "mode": "pose_absolute_ori", 
                        "kv": 3.0
                    },
                    # "arm_left": {
                    #     "name": "JointController",
                    #     "motor_type": "position",
                    #     "command_input_limits": None,
                    #     "command_output_limits": None, 
                    #     "use_delta_commands": False
                    # },
                    "arm_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None, 
                        "use_delta_commands": False
                    },
                    "gripper_left": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        # "use_single_command": True
                    },
                    "gripper_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        # "use_single_command": True
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
                "max_steps": 100000,
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

    reset_positions =  {
        'coffee_table_fqluyq_0': ([-0.4767243 , -1.219805  ,  0.25702515], [-3.69874935e-04, -9.39229270e-04,  7.08872199e-01,  7.05336273e-01]),
        'cologne': ([-0.30000001, -0.80000001,  0.44277492],
                    [0.        , 0.        , 0.        , 1.000000]),
        'robot0': ([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0])
    }

    env_config = {
        "cfg": cfg,
        "reset_positions": reset_positions,
        "action_space_controllers": ["base", "camera", "arm_left", "gripper_left"]
    }
    env = RLEnv(env_config)
    controller = env.env._primitive_controller

    obj = env.scene.object_registry("name", "cologne")

    action = np.zeros(22)

    print("Will try to greet world ...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = omnigibson_pb2_grpc.PolicyStub(channel)
        for _ in trange(1000):
            obs, reward, done, truncated, info = env.step(action)
            # Load the RGB obs into PILLOW
            img = Image.fromarray(obs["robot0"]["robot0:eyes_Camera_sensor_rgb"][..., :3])
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            response = stub.Step(omnigibson_pb2.StepRequest(image=img_byte_arr))
            action = np.array([x for x in response.command])
            if done:
                break


if __name__ == "__main__":
    main()

    # seg semantic - 224 x 224
    # seg instance - 224 x 224
    # depth - 224 x 224
    # rgb - 224 x 224 x 4