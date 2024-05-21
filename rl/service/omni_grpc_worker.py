import argparse
import os

import numpy as np
import wandb
import yaml
from telegym import serve_env_over_grpc

import omnigibson as og
from omnigibson.macros import gm

gm.USE_FLATCACHE = True


def main(local_addr, learner_addr, render):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "omni_grpc.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    if not render:
        gm.ENABLE_RENDERING = False
        del config["env"]["external_sensors"]

    env = og.Environment(configs=config)

    # Calculate fps
    # import time
    # while True:
    #     start_time = time.time()
    #     env.step(env.action_space.sample())
    #     print("fps", 1/(time.time() - start_time))

    wandb.init(entity="behavior-rl", project="sb3", group="worker")

    # Now start servicing!
    serve_env_over_grpc(env, local_addr, learner_addr)

if __name__ == "__main__":
    import socket
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("learner_addr", type=str)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Obtain an unused port
    if args.port is not None:
        local_port = args.port
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        local_port = s.getsockname()[1]
        s.close()

    main("0.0.0.0:" + str(local_port), args.learner_addr, args.render)
