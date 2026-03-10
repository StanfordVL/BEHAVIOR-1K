import os
import time

import torch as th
import yaml

import omnigibson as og
from omnigibson.macros import gm

NUM_STEPS = 100


def main(random_selection=False, headless=False, short_exec=False):
    # Load the config
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_OBJECT_STATES = False

    config_filename = os.path.join(og.example_config_path, "franka_vector_env.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]
    config["env"] = {"num_envs": 5}

    # Load the environment
    env = og.Environment(configs=config)

    max_iterations = 10 if not short_exec else 1
    for _ in range(max_iterations):
        start_time = time.time()
        for _ in range(NUM_STEPS):
            actions = th.stack([
                th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float()
                for i in range(env.num_envs)
            ])
            env.step(actions)

        step_time = time.time() - start_time
        fps = NUM_STEPS / step_time
        effective_fps = NUM_STEPS * env.num_envs / step_time
        print("fps", fps)
        print("effective fps", effective_fps)

    # Always close the environment at the end
    og.shutdown()


if __name__ == "__main__":
    main()
