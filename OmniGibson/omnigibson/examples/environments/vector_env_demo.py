<<<<<<< HEAD:OmniGibson/omnigibson/examples/environments/vector_env_demo.py
import os
import time

=======
import argparse
import cProfile
import io
import os
import pstats
import time

import numpy as np
>>>>>>> rl-experiments:omnigibson/examples/environments/vector_env_demo.py
import yaml

import omnigibson as og
from omnigibson.macros import gm

NUM_STEPS = 100


<<<<<<< HEAD:OmniGibson/omnigibson/examples/environments/vector_env_demo.py
def main(random_selection=False, headless=False, short_exec=False):
=======
def main():
>>>>>>> rl-experiments:omnigibson/examples/environments/vector_env_demo.py
    # Load the config
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
<<<<<<< HEAD:OmniGibson/omnigibson/examples/environments/vector_env_demo.py
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_OBJECT_STATES = False

    config_filename = os.path.join(og.example_config_path, "franka_vector_env.yaml")
=======
    config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
>>>>>>> rl-experiments:omnigibson/examples/environments/vector_env_demo.py
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    vec_env = og.VectorEnvironment(5, config)
<<<<<<< HEAD:OmniGibson/omnigibson/examples/environments/vector_env_demo.py

    max_iterations = 10 if not short_exec else 1
    for _ in range(max_iterations):
=======
    import time

    while True:
>>>>>>> rl-experiments:omnigibson/examples/environments/vector_env_demo.py
        start_time = time.time()
        for _ in range(NUM_STEPS):
            actions = []
            for e in vec_env.envs:
                actions.append(e.action_space.sample())
            vec_env.step(actions)

        step_time = time.time() - start_time
        fps = NUM_STEPS / step_time
        effective_fps = NUM_STEPS * len(vec_env.envs) / step_time
        print("fps", fps)
        print("effective fps", effective_fps)

<<<<<<< HEAD:OmniGibson/omnigibson/examples/environments/vector_env_demo.py
    # Always close the environment at the end
    og.shutdown()

=======
>>>>>>> rl-experiments:omnigibson/examples/environments/vector_env_demo.py

if __name__ == "__main__":
    main()
