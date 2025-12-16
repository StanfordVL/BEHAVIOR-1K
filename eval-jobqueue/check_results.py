#!/usr/bin/env python3
"""
Utility for checking evaluation results against challenge submissions.

For every challenge submission, this script loads per-team and per-task results,
summarizes evaluation scores, and prints a summary of results for each team and task.

This script does not generate jobs or resources files.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence
from slugify import slugify


log = logging.getLogger("check_results")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-instances-dir",
        type=Path,
        default=repo_root / "test_instances",
        help="Path to the directory that groups task instance files (default: %(default)s)",
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=repo_root / "challenge_submissions",
        help="Path to the directory containing team submission JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--resources-raw-file",
        type=Path,
        default=repo_root / "resources-raw.json",
        help="Path to the resources-raw.json file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-jobs-file",
        type=Path,
        default=repo_root / "jobs.json",
        help="Destination path for the generated jobs JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-resources-file",
        type=Path,
        default=repo_root / "resources.json",
        help="Destination path for the generated resources JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write the jobs file—just print the summary",
    )
    return parser.parse_args()


def load_task_instances(root: Path) -> Dict[str, List[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Test instances directory not found: {root}")

    task_instances: Dict[str, List[Path]] = {}
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(
            f for f in task_dir.iterdir() if f.is_file() and f.suffix.lower() == ".json"
        )
        if not files:
            log.debug("No instance files found in %s; skipping", task_dir)
            continue
        task_instances[task_dir.name] = files
    if not task_instances:
        raise RuntimeError(f"No task instances discovered under {root}")
    return task_instances


def check_jobs(
    submissions: Sequence[Path],
    task_instances: Dict[str, List[Path]],
) -> List[dict]:
    results = {}

    submission_data = {}
    for submission_path in sorted(submissions):
        if submission_path.suffix.lower() != ".json":
            continue

        data = json.loads(submission_path.read_text())
        submission_data[submission_path] = data

    # Sort by combined q_score and limit to the top 5 submissions
    for submission_path, data in submission_data.items():
        team = data["team"]
        team_slug = slugify(team, separator="_")

        if team_slug not in [
            "simpleai_robot",
            "robot_learning_collective",
            "comet",
            "the_north_star",
        ]:
            continue

        per_task_scores = data["per_task_scores"]
        q_scores = per_task_scores["q_score"]
        results[team_slug] = {}

        for task, instances in sorted(task_instances.items()):
            entries = []
            for instance_path in instances:
                instance_name = instance_path.name

                # Process the instance basename to get the instance ID
                instance_id = int(
                    instance_name.replace("_template-tro_state.json", "").rsplit(
                        "_", 1
                    )[-1]
                )
                result_path = (
                    Path("/vision/group/behavior/eval-results/")
                    / team_slug
                    / "metrics"
                    / f"{task}_{instance_id}.json"
                )
                video_path = (
                    Path("/vision/group/behavior/eval-results/")
                    / team_slug
                    / "videos"
                    / f"{task}_{instance_id}.mp4"
                )

                q_score = 0.0
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    q_score = result["q_score"]["final"]

                entries.append(
                    {
                        "finished_eval": int(result_path.exists()),
                        "started_eval": int(video_path.exists()),
                        "q_score": q_score,
                    }
                )

            totals = {
                "sr_q_score": q_scores[task] if task in q_scores else 0.0,
                "eval_q_score": sum(e["q_score"] for e in entries)
                / sum(e["finished_eval"] for e in entries)
                if sum(e["finished_eval"] for e in entries) > 0
                else 0.0,
                "finished_eval": sum(e["finished_eval"] for e in entries),
                "started_eval": sum(e["started_eval"] for e in entries),
            }
            results[team_slug][task] = totals

    return results


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()

    test_instances = load_task_instances(args.test_instances_dir.resolve())
    submissions = sorted(args.submissions_dir.resolve().glob("*.json"))
    results = check_jobs(submissions, test_instances)

    for team, task_results in results.items():
        for task, totals in task_results.items():
            print(
                f"{team}, {task}, {totals['sr_q_score']:.2f}, {totals['eval_q_score']:.2f}, {totals['finished_eval']}, {totals['started_eval']}"
            )


if __name__ == "__main__":
    main()
