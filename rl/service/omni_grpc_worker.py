import yaml
import numpy as np
import omnigibson as og
from omnigibson.macros import gm

from telegym import serve_env_over_grpc

gm.USE_FLATCACHE = True

def main(local_addr, learner_addr):
    config_filename = "omni_grpc.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    env = og.Environment(configs=config)

    # Now start servicing!
    serve_env_over_grpc(env, local_addr, learner_addr)

if __name__ == "__main__":
    import sys
    local_port = int(sys.argv[1])
    main("localhost:" + str(local_port), "localhost:50051")