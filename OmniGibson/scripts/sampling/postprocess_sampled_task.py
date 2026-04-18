import argparse
import json
import os
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.tasks import BehaviorTask

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--activity",
    type=str,
    default=None,
    required=True,
    help="Activity to be postprocessed.",
)
parser.add_argument("-s", "--scene_model", type=str, default=None, required=True, help="Scene model to sample tasks in")
parser.add_argument(
    "-w",
    "--overwrite",
    action="store_true",
    help="Whether to forcibly overwrite any pre-existing files",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default=None,
    help="Output directory for sampled tasks (default: gm.DATA_PATH/2026-challenge-task-instances)",
)


def main():
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", args.scene_model, "json")

    task_name = BehaviorTask.get_cached_activity_scene_filename(
        scene_model=args.scene_model,
        activity_name=args.activity,
        activity_definition_id=0,
        activity_instance_id=0,
    )

    sampled_scene_partial_json = os.path.join(args.output_dir, args.activity, f"{task_name}-partial_rooms.json")
    full_scene_json_dir = os.path.join(get_dataset_path("behavior-1k-assets"), f"scenes/{args.scene_model}/json")
    full_scene_full_json = os.path.join(full_scene_json_dir, f"{args.scene_model}_stable.json")

    with open(full_scene_full_json, "r") as f:
        scene_a = json.load(f)
    with open(sampled_scene_partial_json, "r") as f:
        scene_b = json.load(f)
    sampled_scene_full_dict = merge_scene_files(scene_a, scene_b, keep_robot_from="b")
    out_path = sampled_scene_partial_json.replace("-partial_rooms.json", ".json")
    if os.path.exists(out_path) and not args.overwrite:
        raise ValueError(f"File already exists at {out_path}, use --overwrite to overwrite.")
    with open(out_path, "w+") as f:
        json.dump(sampled_scene_full_dict, f, indent=4)
    print(f"Postprocessed sampled scene saved to: {out_path}")


if __name__ == "__main__":
    main()
