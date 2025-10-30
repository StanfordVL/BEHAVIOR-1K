import argparse
import csv
import json
import os
from omnigibson.macros import gm
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES


def compute_final_q_score(input_dir: str, output_dir: str, final_score_only: bool = True) -> None:
    """
    Compute the final Q score from the evaluation result json files stored in the given path.
    Args:
        input_dir (str): Path to the directory containing evaluation result json files.
            Should be of the form <track>.<team>.<affiliation>.<date>/ containing all result json files.
        output_dir (str): Path to save the computed final scores json file.
        final_score_only (bool): Whether to only save the final scores, or also per-rollout scores.
    """
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)
    # assert path exists
    assert os.path.exists(input_dir), f"Input path {input_dir} does not exist"
    # get the root of the input dir to extract team, affiliation, date
    base_name = os.path.basename(os.path.normpath(input_dir))
    track, team, affiliation, date = base_name.split(".")
    # load test instance files
    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, "r") as f:
        lines = list(csv.reader(f))[1:]
    # get all possible filenames:
    possible_filenames = set()
    for task_name, task_idx in TASK_NAMES_TO_INDICES.items():
        test_instances = [int(x) for x in lines[task_idx][2].strip().split(",")][:10]
        for instance_id in test_instances:
            for rollout_id in range(1):  # 1 rollout per instance
                filename = f"{task_name}_{instance_id}_{rollout_id}.json"
                possible_filenames.add(filename)
    # Initialize score dictionaries
    q_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    time_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    base_distance_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    left_distance_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    right_distance_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    # Load results
    n_rollouts = 0
    for file in os.listdir(input_dir):
        if file.endswith(".json") is False:
            print(f"Skipping non-json file {file} in input directory")
            continue
        assert file in possible_filenames, f"File {file} is not a valid evaluation result file"
        # get file name without extension
        file_name = os.path.splitext(file)[0]
        task_name, instance_id, rollout_id = file_name.rsplit("_", 2)
        with open(os.path.join(input_dir, file), "r") as f:
            result = json.load(f)
        # get score
        q_score[task_name][f"{instance_id}_{rollout_id}"] = result["q_score"]["final"]
        time_score[task_name][f"{instance_id}_{rollout_id}"] = result["time"]["normalized_time"]
        base_distance_score[task_name][f"{instance_id}_{rollout_id}"] = result["normalized_agent_distance"]["base"]
        left_distance_score[task_name][f"{instance_id}_{rollout_id}"] = result["normalized_agent_distance"]["left"]
        right_distance_score[task_name][f"{instance_id}_{rollout_id}"] = result["normalized_agent_distance"]["right"]
        n_rollouts += 1

    # Now, compute averaged task score
    q_score_avg, time_score_avg, base_distance_score_avg, left_distance_score_avg, right_distance_score_avg = (
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
    )
    for task_name in TASK_NAMES_TO_INDICES.keys():
        q_score_avg[task_name] = sum(q_score[task_name].values()) / 10
        time_score_avg[task_name] = sum(time_score[task_name].values()) / 10
        base_distance_score_avg[task_name] = sum(base_distance_score[task_name].values()) / 10
        left_distance_score_avg[task_name] = sum(left_distance_score[task_name].values()) / 10
        right_distance_score_avg[task_name] = sum(right_distance_score[task_name].values()) / 10

    # Now, compute overall score across tasks
    overall_q_score = sum(q_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_time_score = sum(time_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_base_distance_score = sum(base_distance_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_left_distance_score = sum(left_distance_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_right_distance_score = sum(right_distance_score_avg.values()) / len(TASK_NAMES_TO_INDICES)

    output_json = {
        "team": team.replace("_", " "),
        "affiliation": affiliation.replace("_", " "),
        "date": date,
        "track": track,
        "overall_scores": {
            "q_score": overall_q_score,
            "time_score": overall_time_score,
            "base_distance_score": overall_base_distance_score,
            "left_distance_score": overall_left_distance_score,
            "right_distance_score": overall_right_distance_score,
        },
    }
    if not final_score_only:
        output_json["per_task_scores"] = {
            "q_score": q_score_avg,
            "time_score": time_score_avg,
            "base_distance_score": base_distance_score_avg,
            "left_distance_score": left_distance_score_avg,
            "right_distance_score": right_distance_score_avg,
        }
        output_json["per_rollout_scores"] = {
            "q_score": q_score,
            "time_score": time_score,
            "base_distance_score": base_distance_score,
            "left_distance_score": left_distance_score,
            "right_distance_score": right_distance_score,
        }
    with open(f"{output_dir}/{track}/{team}.{affiliation}.{date}.json", "w") as f:
        json.dump(output_json, f, indent=4)

    print("Total rollouts:", n_rollouts)
    print("Final Q Score:", overall_q_score)
    print("Final Time Score:", overall_time_score)
    print("Final Base Distance Score:", overall_base_distance_score)
    print("Final Left Distance Score:", overall_left_distance_score)
    print("Final Right Distance Score:", overall_right_distance_score)
    print(f"Final scores saved to {output_dir}/{track}/{team}.{affiliation}.{date}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Path to the directory containing evaluation result json files.",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Path to save the computed final scores json file."
    )
    args = parser.parse_args()
    compute_final_q_score(args.input_dir, args.output_dir, final_score_only=True)
