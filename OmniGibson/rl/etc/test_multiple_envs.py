import argparse
import cProfile
import io
import os
import pstats
import time

import numpy as np
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.envs.sb3_vec_env import SB3VectorEnvironment
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision, set_base_and_detect_collision


def pause(time):
    for _ in range(int(time * 100)):
        og.sim.render()


def pause_step(time):
    for _ in range(int(time * 100)):
        og.sim.step()


def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action)
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)


def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)


def _get_env_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "rl.yaml"))
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return config


def main():
    # Load the config
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.HEADLESS = False
    # gm.DEBUG_VISUALIZATION = True
    config = _get_env_config()

    reset_poses_path = os.path.dirname(__file__) + "/reset_poses.json"
    config["task"]["precached_reset_pose_path"] = reset_poses_path
    del config["env"]["external_sensors"]

    # Load the environment
    n_envs = 1
    vec_env = SB3VectorEnvironment(n_envs, config, render_on_step=True)

    while True:
        start_time = time.time()
        for _ in range(100):
            a = vec_env.action_space.sample()
            obs, _, _, _ = vec_env.step([a for _ in range(n_envs)])
        
        fps = 100 / (time.time() - start_time)
        print("fps", fps)
        print("effective fps", fps * n_envs)


if __name__ == "__main__":
    main()
