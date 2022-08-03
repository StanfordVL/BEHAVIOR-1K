import os

from IPython import embed

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.config_utils import parse_config

task_choices = [
    "packing_lunches_filtered",
    "assembling_gift_baskets_filtered",
    "organizing_school_stuff_filtered",
    "re-shelving_library_books_filtered",
    "serving_hors_d_oeuvres_filtered",
    "putting_away_toys_filtered",
    "putting_away_Christmas_decorations_filtered",
    "putting_dishes_away_after_cleaning_filtered",
    "cleaning_out_drawers_filtered",
]
task = "assembling_gift_baskets"
task_id = 0
scene = "Rs_int"
num_init = 0

config_file = os.path.join(igibson.example_config_path, "behavior_vr.yaml")
env_config = parse_config(config_file)
env_config["scene_id"] = scene
env_config["task"] = task
env_config["task_id"] = task_id
env_config["instance_id"] = num_init
urdf_file = "{}_task_{}_{}_{}".format(scene, task, task_id, num_init)
env_config["urdf_file"] = urdf_file
env_config["online_sampling"] = False
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
