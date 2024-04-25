import argparse
import cProfile
import io
import os
import pstats
import time

import numpy as np
import yaml

import omnigibson as og
from omnigibson.macros import gm


def main():
    # Load the config
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    vec_env = og.VectorEnvironment(5, config)
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
