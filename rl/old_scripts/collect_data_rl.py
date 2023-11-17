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

from PIL import Image
from tqdm import tqdm


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

class Recorder():
    def __init__(self, folder):
        self.folderpath = f'./rollouts/{folder}'
        self.state_keys = ["robot0:eyes_Camera_sensor_rgb", "robot0:eyes_Camera_sensor_depth_linear", "robot0:eyes_Camera_sensor_seg_instance", "robot0:eyes_Camera_sensor_seg_semantic"]
        self.reset()

    def add(self, timestep, obs, action, reward, done, truncated, proprio):
        for k in self.state_keys:
            self.states[k].append(obs['robot0'][k].copy())
        self.proprios.append(proprio.copy())
        self.truncateds.append(truncated)
        self.dones.append(done)
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.ids.append(self.episode_id)
        self.timesteps.append(timestep)

    def reset(self):
        self.states = {}
        for k in self.state_keys:
            self.states[k] = []
        self.proprios = []
        self.truncateds = []
        self.dones = []
        self.actions = []
        self.rewards = []
        self.ids = []
        self.timesteps = []
        self.episode_id = str(uuid.uuid4())

    def _add_to_dataset(self, group, name, data):
        if len(data) > 0:
            if name in group:
                dset_len = len(group[name])
                dset = group[name]
                dset.resize(len(dset) + len(data), 0)
                dset[dset_len:] = data
            else:
                if isinstance(data[0], np.ndarray):
                    group.create_dataset(name, data=data, maxshape=(None, *data[0].shape))
                else:
                    group.create_dataset(name, data=data, maxshape=(None,))

    def save(self, group_name='data_group'):
        os.makedirs(self.folderpath, exist_ok=True)
        for k in self.state_keys:
            state_folder = k[k.find(":") + 1:]
            os.makedirs(f'{self.folderpath}/{state_folder}', exist_ok=True)
            for t, state in zip(self.timesteps, self.states[k]):
                img = Image.fromarray(state)
                if k == "robot0:eyes_Camera_sensor_rgb":
                    img.convert('RGB').save(f'{self.folderpath}/{state_folder}/{self.episode_id}_{t}.jpeg')
                elif k == "robot0:eyes_Camera_sensor_depth_linear":
                    img = np.array(img) * 1000
                    img = Image.fromarray(img.astype(np.int32))
                    img.save(f'{self.folderpath}/{state_folder}/{self.episode_id}_{t}.png')
                else:
                    img.save(f'{self.folderpath}/{state_folder}/{self.episode_id}_{t}.png')
        h5file = h5py.File(f'{self.folderpath}/data.h5', 'a')
        group = h5file[group_name] if group_name in h5file else h5file.create_group(group_name)
        self._add_to_dataset(group, "proprio", self.proprios)
        self._add_to_dataset(group, "truncated", self.truncateds)
        self._add_to_dataset(group, "done", self.dones)
        self._add_to_dataset(group, "actions", self.actions)
        self._add_to_dataset(group, "rewards", self.rewards)
        self._add_to_dataset(group, "timesteps", self.timesteps)
        self._add_to_dataset(group, "ids", self.ids)
        h5file.close()

def main(folder, iterations):
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
                "proprio_obs": ["robot_pose", "joint_qpos", "joint_qvel", "eef_left_pos", "eef_left_quat", "grasp_left"],
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
                        "use_single_command": True
                    },
                    "gripper_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        "use_single_command": True
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
    recorder = Recorder(folder)

    for i in tqdm(range(int(iterations))):
        try:
            obs = env.reset()
            recorder.add(-1, obs, np.zeros_like(env.action_space.sample()), 0, False, False, obs['robot0']['proprio'])
            timestep = 0
            for action in controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj, track_object=True):
                action = action[0]
                obs, reward, done, truncated, info = env.step(action)
                truncated = True if timestep >= 400 else truncated
                recorder.add(timestep, obs, env.transform_action(action), reward, done, truncated, obs['robot0']['proprio'])
                timestep += 1
                if done or timestep >= 400:
                    for action in controller._execute_release():
                        action = action[0]
                        env.step(action)
                    break
        except Exception as e:
            print("Error in iteration: ", i)
            print(e)
            print('--------------------')
        # recorder.save()
        recorder.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run worker")
    # parser.add_argument("folder")
    parser.add_argument("iterations")
    
    args = parser.parse_args()
    folder_name = str(uuid.uuid4())
    main(folder_name, args.iterations)

    # seg semantic - 224 x 224
    # seg instance - 224 x 224
    # depth - 224 x 224
    # rgb - 224 x 224 x 4