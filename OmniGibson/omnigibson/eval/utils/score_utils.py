import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from omnigibson.macros import gm
from omnigibson.eval.utils.eval_utils import (
    EVAL_TIMEOUT_MULTIPLIER,
    NUM_HIDDEN_TEST_INSTANCES,
    NUM_PUBLIC_TEST_INSTANCES,
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    TEST_INSTANCE_IDS,
)


HUMAN_STATS_KEYS = (
    "length",
    "distance_traveled",
    "left_eef_displacement",
    "right_eef_displacement",
)


def default_task_jsonl_path() -> str:
    return os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "task.jsonl")


def load_human_stats(task_name: str, task_jsonl_path: str | None = None) -> dict[str, float]:
    """
    Load task-level human statistics for evaluator metrics.

    The file is one JSON object per task and should contain:
    task_name, task_index, length, distance_traveled, left_eef_displacement, right_eef_displacement.
    """
    task_jsonl_path = os.path.expanduser(task_jsonl_path or default_task_jsonl_path())
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    with open(task_jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            task_stats = json.loads(line)
            if task_stats.get("task_name") == task_name or task_stats.get("task_index") == task_idx:
                return {key: task_stats[key] for key in HUMAN_STATS_KEYS}

    raise ValueError(f"Could not find human stats for task {task_name} in {task_jsonl_path}")


def generate_task_jsonl_from_lerobot_v3(
    data_dir: str,
    output_path: str,
    robot_type: str = "R1Pro",
    num_tasks: int | None = None,
) -> None:
    """
    Aggregate LeRobot v3 per-task datasets into the task-level task.jsonl format.

    Supports both layouts:
        data_dir/<task_name>/{meta,data}/...
        data_dir/{meta,data}/... where each chunk is one task.
    """
    import pandas as pd

    data_dir = Path(os.path.expanduser(data_dir))
    task_names_by_idx = {task_idx: task_name for task_name, task_idx in TASK_NAMES_TO_INDICES.items()}
    totals = {}
    task_idx_filter = set(range(num_tasks)) if num_tasks is not None else None

    if (data_dir / "meta" / "tasks.parquet").exists():
        with open(data_dir / "meta" / "info.json", "r") as f:
            fps = json.load(f).get("fps", 30)
        episode_chunk_dirs = sorted((data_dir / "meta" / "episodes").glob("chunk-*"))
        for episode_chunk_dir in episode_chunk_dirs:
            task_idx = int(episode_chunk_dir.name.split("-")[-1])
            if task_idx_filter is not None and task_idx not in task_idx_filter:
                continue
            task_totals = _compute_lerobot_v3_task_totals(
                episode_paths=sorted(episode_chunk_dir.glob("file-*.parquet")),
                data_paths=sorted((data_dir / "data" / f"chunk-{task_idx:03d}").glob("file-*.parquet")),
                task_name=task_names_by_idx.get(task_idx, str(task_idx)),
                fps=fps,
                robot_type=robot_type,
            )
            if task_totals is not None:
                totals[task_idx] = task_totals
        _write_task_jsonl(totals=totals, task_names_by_idx=task_names_by_idx, output_path=output_path)
        return

    for task_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        task_info_path = task_dir / "meta" / "tasks.parquet"
        if not task_info_path.exists():
            continue

        task_info = pd.read_parquet(task_info_path)
        task_idx = int(task_info["task_index"].iloc[0])
        if task_idx_filter is not None and task_idx not in task_idx_filter:
            continue
        task_name = task_names_by_idx.get(task_idx, task_dir.name)
        with open(task_dir / "meta" / "info.json", "r") as f:
            fps = json.load(f).get("fps", 30)

        task_totals = _compute_lerobot_v3_task_totals(
            episode_paths=sorted((task_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")),
            data_paths=sorted((task_dir / "data").glob("chunk-*/file-*.parquet")),
            task_name=task_name,
            fps=fps,
            robot_type=robot_type,
        )
        if task_totals is not None:
            totals[task_idx] = task_totals

    _write_task_jsonl(totals=totals, task_names_by_idx=task_names_by_idx, output_path=output_path)


def _compute_lerobot_v3_task_totals(
    episode_paths: list[Path],
    data_paths: list[Path],
    task_name: str,
    fps: int,
    robot_type: str,
) -> dict | None:
    import numpy as np
    import pandas as pd

    if not episode_paths or not data_paths:
        return None

    episode_lengths = {}
    for path in episode_paths:
        episodes = pd.read_parquet(path, columns=["episode_index", "length"])
        episode_lengths.update({int(row.episode_index): float(row.length) for row in episodes.itertuples(index=False)})

    data_paths = sorted(data_paths)
    state_sample = pd.read_parquet(data_paths[0], columns=["observation.state"]).iloc[0]["observation.state"]
    base_position_idx, base_velocity_idx, eef_indices = _resolve_lerobot_state_indices(
        robot_type=robot_type,
        state_dim=len(state_sample),
    )

    episode_totals = defaultdict(lambda: {key: 0.0 for key in HUMAN_STATS_KEYS if key != "length"})
    previous_state = {}
    for path in data_paths:
        data = pd.read_parquet(path, columns=["episode_index", "observation.state"])
        for episode_idx, group in data.groupby("episode_index", sort=False):
            episode_idx = int(episode_idx)
            current_values = np.stack(group["observation.state"].to_numpy())
            if base_position_idx is None:
                assert base_velocity_idx is not None, "Need either robot_pos or base_qvel to compute base travel"
                base_qvel = current_values[:, base_velocity_idx]
                episode_totals[episode_idx]["distance_traveled"] += (
                    np.linalg.norm(base_qvel[:, :2], axis=-1).sum() / fps
                )

            values = current_values
            if episode_idx in previous_state:
                values = np.concatenate([previous_state[episode_idx][None, :], values], axis=0)
            if len(values) > 1:
                if base_position_idx is not None:
                    robot_positions = values[:, base_position_idx]
                    episode_totals[episode_idx]["distance_traveled"] += np.linalg.norm(
                        robot_positions[1:] - robot_positions[:-1], axis=-1
                    ).sum()
                for key, state_idx in eef_indices.items():
                    positions = values[:, state_idx]
                    episode_totals[episode_idx][key] += np.linalg.norm(positions[1:] - positions[:-1], axis=-1).sum()
            previous_state[episode_idx] = values[-1]

    num_episodes = len(episode_lengths)
    if num_episodes == 0:
        return None

    return {
        "task_name": task_name,
        "num_episodes": num_episodes,
        "length": sum(episode_lengths.values()),
        "distance_traveled": sum(stats["distance_traveled"] for stats in episode_totals.values()),
        "left_eef_displacement": sum(stats["left_eef_displacement"] for stats in episode_totals.values()),
        "right_eef_displacement": sum(stats["right_eef_displacement"] for stats in episode_totals.values()),
    }


def _resolve_lerobot_state_indices(
    robot_type: str,
    state_dim: int,
) -> tuple[slice | None, slice | None, dict[str, slice]]:
    if robot_type == "R1Pro" and state_dim >= 256:
        return (
            slice(140, 143),
            None,
            {
                "left_eef_displacement": slice(186, 189),
                "right_eef_displacement": slice(225, 228),
            },
        )

    proprio_indices = PROPRIOCEPTION_INDICES[robot_type]
    base_position_idx = proprio_indices.get("robot_pos")
    base_velocity_idx = proprio_indices.get("base_qvel")
    if base_position_idx is None and base_velocity_idx is None:
        raise ValueError(
            "This LeRobot dataset does not expose robot_pos or base_qvel in observation.state, "
            "so human base distance cannot be computed."
        )

    return (
        base_position_idx,
        base_velocity_idx,
        {
            "left_eef_displacement": proprio_indices["eef_left_pos"],
            "right_eef_displacement": proprio_indices["eef_right_pos"],
        },
    )


def _write_task_jsonl(totals: dict, task_names_by_idx: dict[int, str], output_path: str) -> None:
    output_path = os.path.expanduser(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for task_idx in sorted(totals):
            num_episodes = totals[task_idx]["num_episodes"]
            if num_episodes == 0:
                continue
            task_stats = {
                "task_index": task_idx,
                "task_name": totals[task_idx].get("task_name", task_names_by_idx.get(task_idx, str(task_idx))),
                "num_episodes": num_episodes,
            }
            for key in HUMAN_STATS_KEYS:
                task_stats[key] = round(totals[task_idx][key] / num_episodes, 4)
            json.dump(task_stats, f)
            f.write("\n")


def sanity_check_filename(input_dir: str) -> None:
    """
    Sanity check whether the input directory name follows the expected format:
    input_dir/
        <track>.<testset>.<team_name>.<affiliation_name>.<date>/
            json/
                <task_name>_<instance_id>_<rollout_id>.json
    In particular, append 0 to rollout_id if missing from json file.
    Args:
        input_dir (str): Path to the directory containing evaluation result json files.
    """
    input_dir = os.path.expanduser(input_dir)
    base_name = os.path.basename(os.path.normpath(input_dir))
    track, testset, team, affiliation, date = base_name.split(".")
    assert track in ["standard", "privileged"], f"Track {track} not recognized"
    assert testset in ["public", "hidden"], f"Testset {testset} not recognized"
    assert team != "", "Team name cannot be empty"
    assert affiliation != "", "Affiliation name cannot be empty"
    assert date != "", "Date cannot be empty"
    # Now, check all json files in the json folder
    assert os.path.exists(f"{input_dir}/json"), f"Input path {input_dir}/json does not exist"
    for file in os.listdir(f"{input_dir}/json"):
        if file.endswith(".json") is False:
            print(f"Skipping non-json file {file} in input directory")
            continue
        file_name = os.path.splitext(file)[0]
        if not file_name.split("_")[-2].isdigit():
            new_file_name = f"{file_name}_0.json"
            os.rename(
                os.path.join(f"{input_dir}/json", file),
                os.path.join(f"{input_dir}/json", new_file_name),
            )
            print(f"Renamed {file} to {new_file_name}")


def compute_final_q_score(
    input_dir: str, output_dir: str, final_score_only: bool = True, verbose: bool = False
) -> None:
    """
    Compute the final Q score from the evaluation result json files stored in the given path.
    Args:
        input_dir (str): Path to the directory containing evaluation result json files.
            Should be of the form <track>.<testset>.<team>.<affiliation>.<date>/ containing a json folder for all results.
        output_dir (str): Path to save the computed final scores json file.
        final_score_only (bool): Whether to only save the final scores, or also per-rollout scores.
        verbose (bool): Whether to print verbose output.
    """
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)
    # get the root of the input dir to extract team, affiliation, date
    base_name = os.path.basename(os.path.normpath(input_dir))
    track, testset, team, affiliation, date = base_name.split(".")
    # get all possible filenames:
    possible_filenames = set()
    for task_name in TASK_NAMES_TO_INDICES:
        test_instances = (
            TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES]
            if testset == "public"
            else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
        )
        for instance_id in test_instances:
            for rollout_id in range(1):  # 1 rollout per instance
                filename = f"{task_name}_{instance_id}_{rollout_id}.json"
                possible_filenames.add(filename)
    n_instances_per_task = NUM_PUBLIC_TEST_INSTANCES if testset == "public" else NUM_HIDDEN_TEST_INSTANCES
    # Initialize score dictionaries
    q_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    time_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    base_distance_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    left_distance_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    right_distance_score = {task_name: dict() for task_name in TASK_NAMES_TO_INDICES.keys()}
    # Load results
    n_rollouts = 0
    for file in os.listdir(f"{input_dir}/json"):
        if file.endswith(".json") is False:
            print(f"Skipping non-json file {file} in input directory")
            continue
        assert file in possible_filenames, f"File {file} is not a valid evaluation result file"
        # get file name without extension
        file_name = os.path.splitext(file)[0]
        task_name, instance_id, rollout_id = file_name.rsplit("_", 2)
        with open(os.path.join(f"{input_dir}/json", file), "r") as f:
            result = json.load(f)
        # get score
        q_score[task_name][f"{instance_id}_{rollout_id}"] = result["q_score"]["final"]
        normalized_time = result["time"]["normalized_time"]
        time_score[task_name][f"{instance_id}_{rollout_id}"] = EVAL_TIMEOUT_MULTIPLIER / (
            EVAL_TIMEOUT_MULTIPLIER - 1.0
        ) - 1 / ((EVAL_TIMEOUT_MULTIPLIER - 1.0) * normalized_time)
        base_distance_score[task_name][f"{instance_id}_{rollout_id}"] = result["normalized_agent_distance"]["base"]
        left_distance_score[task_name][f"{instance_id}_{rollout_id}"] = result["normalized_agent_distance"]["left"]
        right_distance_score[task_name][f"{instance_id}_{rollout_id}"] = result["normalized_agent_distance"]["right"]
        n_rollouts += 1

    # Now, compute averaged task score
    q_score_avg, task_sr, time_score_avg, base_distance_score_avg, left_distance_score_avg, right_distance_score_avg = (
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
    )
    for task_name in TASK_NAMES_TO_INDICES.keys():
        q_score_avg[task_name] = sum(q_score[task_name].values()) / n_instances_per_task
        task_sr[task_name] = sum(1 for v in q_score[task_name].values() if v == 1) / n_instances_per_task
        time_score_avg[task_name] = sum(time_score[task_name].values()) / n_instances_per_task
        base_distance_score_avg[task_name] = sum(base_distance_score[task_name].values()) / n_instances_per_task
        left_distance_score_avg[task_name] = sum(left_distance_score[task_name].values()) / n_instances_per_task
        right_distance_score_avg[task_name] = sum(right_distance_score[task_name].values()) / n_instances_per_task

    # Now, compute overall score across tasks
    overall_q_score = sum(q_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_task_sr = sum(task_sr.values()) / len(TASK_NAMES_TO_INDICES)
    overall_time_score = sum(time_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_base_distance_score = sum(base_distance_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_left_distance_score = sum(left_distance_score_avg.values()) / len(TASK_NAMES_TO_INDICES)
    overall_right_distance_score = sum(right_distance_score_avg.values()) / len(TASK_NAMES_TO_INDICES)

    output_json = {
        "team": team.replace("_", " "),
        "affiliation": affiliation.replace("_", " "),
        "date": date,
        "track": track,
        "testset": testset,
        "overall_scores": {
            "q_score": overall_q_score,
            "task_sr": overall_task_sr,
            "time_score": overall_time_score,
            "base_distance_score": overall_base_distance_score,
            "left_distance_score": overall_left_distance_score,
            "right_distance_score": overall_right_distance_score,
        },
    }
    if not final_score_only:
        output_json["per_task_scores"] = {
            "q_score": q_score_avg,
            "task_sr": task_sr,
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
    with open(f"{output_dir}/{track}.{testset}.{team}.{affiliation}.{date}.json", "w") as f:
        json.dump(output_json, f, indent=4)

    print("Total rollouts:", n_rollouts)
    print("Final Q Score:", overall_q_score)
    print("Final Task Success Rate:", overall_task_sr)
    if verbose:
        print("Final Time Score:", overall_time_score)
        print("Final Base Distance Score:", overall_base_distance_score)
        print("Final Left Distance Score:", overall_left_distance_score)
        print("Final Right Distance Score:", overall_right_distance_score)
        print(f"Final scores saved to {output_dir}/{track}.{team}.{affiliation}.{date}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_task_jsonl",
        action="store_true",
        help="Generate task-level human stats JSONL instead of scoring submissions.",
    )
    parser.add_argument(
        "--lerobot_data_dir",
        type=str,
        default=None,
        help="LeRobot v3 directory, either one subdirectory per task or one aggregated dataset with one chunk per task.",
    )
    parser.add_argument(
        "--task_jsonl",
        type=str,
        default=default_task_jsonl_path(),
        help="Output path for generated task-level human stats, or evaluator input path.",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="R1Pro",
        help="Robot type used when aggregating LeRobot v3 data.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="Only generate stats for task indices 0 through num_tasks - 1.",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=None,
        help="Path to the directory containing evaluation result json files.",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None, help="Path to save the computed final scores json file."
    )
    parser.add_argument(
        "--final_score_only",
        "-f",
        action="store_true",
        help="Whether to only save the final scores, or also per-rollout scores.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output.")
    args = parser.parse_args()

    if args.generate_task_jsonl:
        assert args.lerobot_data_dir is not None, "--lerobot_data_dir is required with --generate_task_jsonl"
        generate_task_jsonl_from_lerobot_v3(
            args.lerobot_data_dir,
            args.task_jsonl,
            robot_type=args.robot_type,
            num_tasks=args.num_tasks,
        )
        print(f"Generated task-level human stats at {args.task_jsonl}")
        raise SystemExit(0)

    assert args.input_dir is not None, "--input_dir is required when scoring submissions"
    assert args.output_dir is not None, "--output_dir is required when scoring submissions"
    assert os.path.exists(args.input_dir), f"Input path {args.input_dir} does not exist"
    os.makedirs(args.output_dir, exist_ok=True)
    # Iterate through all submission folders in the input directory
    for submission in os.listdir(args.input_dir):
        submission_path = os.path.join(args.input_dir, submission)
        print(f"Processing submission folder: {submission_path}")
        sanity_check_filename(submission_path)
        compute_final_q_score(
            submission_path, args.output_dir, final_score_only=args.final_score_only, verbose=args.verbose
        )
