import argparse
import json
import os
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.tasks import BehaviorTask
from omnigibson.macros import gm

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, default=None, help="Scene model to sample tasks in")
parser.add_argument(
    "--activity",
    type=str,
    default=None,
    help="Activity to be postprocessed.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Whether to forcibly overwrite any pre-existing files",
)


def main():
    args = parser.parse_args()
    task_name = BehaviorTask.get_cached_activity_scene_filename(
        scene_model=args.scene_model,
        activity_name=args.activity,
        activity_definition_id=0,
        activity_instance_id=0,
    )
    json_dir = os.path.join(get_dataset_path("behavior-1k-assets"), f"scenes/{args.scene_model}/json")
    full_scene_full_json = f"{json_dir}/{args.scene_model}_stable.json"
    sampled_scene_partial_json = f"{json_dir}/{task_name}-partial_rooms.json"
    with open(full_scene_full_json, "r") as f:
        scene_a = json.load(f)
    with open(sampled_scene_partial_json, "r") as f:
        scene_b = json.load(f)
    sampled_scene_full_dict = merge_scene_files(scene_a, scene_b, keep_robot_from="b")
    out_path = sampled_scene_partial_json.replace("-partial_rooms.json", ".json")
    if os.path.exists(out_path) and not args.overwrite:
        raise ValueError(f"args.overwrite=False and file already exists at: {out_path}!")
    with open(out_path, "w+") as f:
        json.dump(sampled_scene_full_dict, f, indent=4)


if __name__ == "__main__":
    main()
