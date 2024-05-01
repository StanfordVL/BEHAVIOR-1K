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


def main():
    # Load the config
    gm.RENDER_VIEWER_CAMERA = True
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    config_filename = "rl.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    reset_poses_path =  os.path.dirname(__file__) + "/reset_poses.json"
    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]
    config['task']['precached_reset_pose_path'] = reset_poses_path

    # Load the environment
    vec_env = og.VectorEnvironment(1, config)
    import time

    while True:
        actions = []
        start_time = time.time()
        for e in vec_env.envs:
            actions.append(e.action_space.sample())
        vec_env.step(actions)
        print("fps", 1 / (time.time() - start_time))


if __name__ == "__main__":
    main()
