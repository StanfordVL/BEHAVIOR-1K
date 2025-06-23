import os
import argparse
import sys
import json
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.config_utils import TorchEncoder
import bddl
from gello.utils.qa_utils import *
from gello.utils.b1k_utils import aggregate_episode_validation


def get_valid_tasks():
    return set(activity for activity in os.listdir(os.path.join(bddl.__path__[0], "activity_definitions")))


def evaluate_qa_metrics(fpath, task):
    # Make sure the task is a valid task
    assert task in get_valid_tasks(), f"Got invalid task: {task}!"

    dir_path = os.path.dirname(fpath)
    fname = os.path.basename(fpath)
    results_fpath = os.path.join(dir_path, fname.replace(".json", "_qa_results.json"))
    with open(fpath, "r") as f:
        all_episodes_metrics = json.load(f)
    all_episodes_metrics = recursively_convert_to_torch(all_episodes_metrics)

    all_episodes_results = dict()
    for episode_id, all_episode_metrics in all_episodes_metrics.items():
        success, results = aggregate_episode_validation(task=task, all_episode_metrics=all_episode_metrics)
        all_episodes_results[episode_id] = {
            "success": success,
            "results": results,
        }

    with open(results_fpath, "w+") as f:
        json.dump(all_episodes_results, f, cls=TorchEncoder, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Compute Success / Failure based on aggregated QA metrics")
    parser.add_argument("--task", required=True, help="Name of the task to check")
    parser.add_argument("--files", required=True, nargs="*", help="Individual aggregated episode metric file(s) to process")
    args = parser.parse_args()

    # Process each file
    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"Error: File {fpath} does not exist", file=sys.stderr)
            continue

        evaluate_qa_metrics(fpath, args.task)


if __name__ == "__main__":
    main()
