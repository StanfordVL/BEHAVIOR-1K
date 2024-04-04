import os
import yaml
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
import time

gm.USE_FLATCACHE = True

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "./service/omni_grpc.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    env = og.Environment(configs=config)

    while True:
        start_time = time.time()
        env.step(env.action_space.sample())
        print("fps: ", 1 / (time.time() - start_time))
