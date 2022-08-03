import os

from IPython import embed

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.config_utils import parse_config

activity = "assembling_gift_baskets"
scene_id = "Rs_int"

config_file = os.path.join(igibson.example_config_path, "behavior_vr.yaml")
env_config = parse_config(config_file)
env_config["scene_id"] = scene_id
env_config["task"] = activity
env_config["task_id"] = 0
env_config["online_sampling"] = True
env_config["load_clutter"] = False
env_config["not_load_object_categories"] = ["ceilings"]


env = iGibsonEnv(
    config_file=env_config,
    mode="headless",
    use_pb_gui=True,
)

print("success", env.task.initialized)
embed()

while True:
    env.simulator.step()
    success, sorted_conditions = env.task.check_success()
    print("TASK SUCCESS:", success)
    if not success:
        print("FAILED CONDITIONS:", sorted_conditions["unsatisfied"])
    else:
        pass
