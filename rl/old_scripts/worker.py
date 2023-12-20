import argparse
from datetime import datetime
import math
import uuid
import numpy as np
import matplotlib.pyplot as plt
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.sensors.scan_sensor import ScanSensor
from omnigibson.sensors.vision_sensor import VisionSensor
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
import h5py


def set_start_pose(robot):
    reset_pose_tiago = np.array([
        -1.78029833e-04,  3.20231302e-05, -1.85759447e-07, -1.16488536e-07,
        4.55182843e-08,  2.36128806e-04,  1.50000000e-01,  9.40000000e-01,
        -1.10000000e+00,  0.00000000e+00, -0.90000000e+00,  1.47000000e+00,
        0.00000000e+00,  2.10000000e+00,  2.71000000e+00,  1.50000000e+00,
        1.71000000e+00,  1.30000000e+00, -1.57000000e+00, -1.40000000e+00,
        1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
        4.50000000e-02,  4.50000000e-02,  4.50000000e-02,
    ])
    robot.set_joint_positions(reset_pose_tiago)
    og.sim.step()

def step_sim(time):
    for _ in range(int(time*100)):
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action[0])

def reset_env(env, initial_poses):
    objs = ["cologne", "coffee_table_fqluyq_0", "robot0"]
    for o in objs:
        env.scene.object_registry("name", o).set_position_orientation(*initial_poses[o])
    env.reset()
    og.sim.step()

class Recorder():
    def __init__(self, filepath):
        self.filepath = filepath
        self.state_keys = ["robot0:eyes_Camera_sensor_rgb", "robot0:eyes_Camera_sensor_depth"]
        self.reset()

    def add(self, state, action, reward):
        for k in self.state_keys:
            self.states[k].append(state['robot0'][k])
        self.actions.append(action)
        self.rewards.append(reward)
        self.ids.append(self.episode_id)

    def reset(self):
        self.states = {}
        for k in self.state_keys:
            self.states[k] = []
        self.actions = []
        self.rewards = []
        self.ids = []
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

    def save(self, group_name):
        h5file = h5py.File(self.filepath, 'a')
        group = h5file[group_name] if group_name in h5file else h5file.create_group(group_name)
        for k in self.state_keys:
            self._add_to_dataset(group, k[k.find(":") + 1:], self.states[k])
        self._add_to_dataset(group, "actions", self.actions)
        self._add_to_dataset(group, "rewards", self.rewards)
        self._add_to_dataset(group, "ids", self.ids)
        h5file.close()

def main(policy_path, rollouts_path, iterations):
    DIST_COEFF = 0.1
    GRASP_REWARD = 0.3
    h5py.get_config().track_order = True

    cfg = {
        "env": {
            "action_timestep": 1 / 10.,
            "physics_timestep": 1 / 60.,
            "flatten_obs_space": True,
            "flatten_action_space": True,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "coffee_table"],
        },
        "robots": [
            {
                "type": "Tiago",
                "obs_modalities": ["rgb", "depth"],
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
                        "modalities": ["rgb", "depth"],
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
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None, 
                        "use_delta_commands": False
                    },
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
                        "motor_type": "velocity",
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

    # Create the environment
    env = og.Environment(configs=cfg)
    scene = env.scene
    robot = env.robots[0]
    og.sim.step()

    # load policy
    # with open(policy_path, 'wb') as f:
    #     pass

    controller = StarterSemanticActionPrimitives(None, scene, robot)
    env._primitive_controller = controller
    initial_poses = {}
    for o in env.scene.objects:
        initial_poses[o.name] = o.get_position_orientation()
    obj = env.scene.object_registry("name", "cologne")
    recorder = Recorder(rollouts_path)
    group_name = datetime.now().strftime('rl_results')

    for i in range(int(iterations)):
        try:
            reset_env(env, initial_poses)
            for action in controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj):
                action = action[0]
                state, reward, done, info = env.step(action)
                recorder.add(state, action, reward)
                if done:
                    for action in controller._execute_release():
                        state, reward, done, info = env.step(action)
                        recorder.add(state, action, reward)
                    break
        except:
            print("Error in iteration: ", i)
        recorder.save(group_name)
        recorder.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run worker")
    parser.add_argument("policy_path")
    parser.add_argument("rollouts_path")
    parser.add_argument("iterations")
    
    args = parser.parse_args()
    main(args.policy_path, args.rollouts_path, args.iterations)