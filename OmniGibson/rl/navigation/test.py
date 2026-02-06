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


def _get_env_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "nav.yaml"))
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return config


def main():
    # Load the config
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.HEADLESS = False
    config = _get_env_config()

    # Load the environment
    n_envs = 2
    # vec_env = og.Environment(config)
    vec_env = SB3VectorEnvironment(n_envs, config, render_on_step=True)

    while True:
        start_time = time.time()
        for _ in range(100):
            a = vec_env.action_space.sample()
            # [a for _ in range(n_envs)]
            obs, reward, done, info = vec_env.step([a for _ in range(n_envs)])
            # from IPython import embed; embed()
        fps = 100 / (time.time() - start_time)
        print("fps", fps)
        print("effective fps", fps * n_envs)


if __name__ == "__main__":
    main()
