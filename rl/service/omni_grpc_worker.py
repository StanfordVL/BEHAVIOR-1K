import os
import yaml
import numpy as np
import omnigibson as og
from omnigibson.macros import gm

from telegym import serve_env_over_grpc

gm.USE_FLATCACHE = True

def main(local_addr, learner_addr):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "omni_grpc.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    env = og.Environment(configs=config)

    # Now start servicing!
    serve_env_over_grpc(env, local_addr, learner_addr)

if __name__ == "__main__":
    import sys, socket

    # Obtain an unused port
    if len(sys.argv) > 2:
        local_port = int(sys.argv[2])
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        local_port = s.getsockname()[1]
        s.close()

    main("0.0.0.0:" + str(local_port), sys.argv[1])
