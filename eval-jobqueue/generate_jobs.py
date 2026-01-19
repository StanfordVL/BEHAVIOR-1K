#!/usr/bin/env python3
"""
Utility for producing a jobs.json file that `eval_jobqueue.py` can load.

For every challenge submission, this script inspects the per-task q_score
and schedules jobs for every (team, task, test instance) triple where the
team achieved non-zero q_score for that task.

This script also reads resources-raw.json to create pseudo-resource-groups
based on task compatibility constraints, and generates resources.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
from slugify import slugify


log = logging.getLogger("generate_jobs")


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


def create_pseudo_resource_groups(
    resources_raw_path: Path,
) -> Tuple[Dict[str, List[dict]], Dict[str, str]]:
    """
    Parse resources-raw.json and create pseudo-resource-groups.

    Returns:
        - A dict mapping pseudo-group names to lists of resources
        - A dict mapping (team_slug, task) to the appropriate pseudo-group name
    """
    if not resources_raw_path.exists():
        raise FileNotFoundError(f"Resources file not found: {resources_raw_path}")

    resources_raw = json.loads(resources_raw_path.read_text())

    pseudo_groups: Dict[str, List[dict]] = {}
    task_to_group: Dict[str, str] = {}

    for team_slug, resources in resources_raw.items():
        # Group resources by their (compatible_task, not_compatible_task) constraints
        constraint_groups: Dict[Tuple[Optional[str], Optional[str]], List[dict]] = {}

        for resource in resources:
            compatible = resource.get("compatible_task")
            not_compatible = resource.get("not_compatible_task")

            # Convert lists to tuples for hashability
            compatible_key = (
                tuple(sorted(set(compatible)))
                if isinstance(compatible, list)
                else (compatible,)
                if compatible
                else None
            )
            not_compatible_key = (
                tuple(sorted(set(not_compatible)))
                if isinstance(not_compatible, list)
                else (not_compatible,)
                if not_compatible
                else None
            )

            constraint_key = (compatible_key, not_compatible_key)

            if constraint_key not in constraint_groups:
                constraint_groups[constraint_key] = []
            constraint_groups[constraint_key].append(resource)

        # Create pseudo-groups with unique names
        group_idx = 1
        for (
            compatible_key,
            not_compatible_key,
        ), group_resources in constraint_groups.items():
            pseudo_group_name = f"{team_slug}_{group_idx}"
            pseudo_groups[pseudo_group_name] = group_resources

            # Determine which tasks this pseudo-group supports
            # and create mappings from (team_slug, task) to pseudo-group
            # We'll populate task_to_group during job creation based on task compatibility

            log.info(
                f"Created pseudo-group {pseudo_group_name} with {len(group_resources)} resources"
            )
            log.info(f"  Compatible tasks: {compatible_key}")
            log.info(f"  Not compatible tasks: {not_compatible_key}")

            group_idx += 1

    return pseudo_groups, task_to_group


def find_compatible_pseudo_group(
    team_slug: str, task: str, pseudo_groups: Dict[str, List[dict]]
) -> Optional[str]:
    """
    Find the pseudo-group that supports the given task for the given team.

    Returns the pseudo-group name, or None if no compatible group is found.
    """
    for pseudo_group_name, resources in pseudo_groups.items():
        # Check if this pseudo-group belongs to the team
        if not pseudo_group_name.startswith(f"{team_slug}_"):
            continue

        # Check the first resource's constraints (all resources in a pseudo-group have same constraints)
        if not resources:
            continue

        resource = resources[0]
        compatible = resource.get("compatible_task")
        not_compatible = resource.get("not_compatible_task")

        # Check if task is in not_compatible list
        if not_compatible:
            not_compatible_list = (
                not_compatible if isinstance(not_compatible, list) else [not_compatible]
            )
            if task in not_compatible_list:
                continue

        # Check if task is in compatible list (or if compatible is None, meaning all tasks are compatible)
        if compatible is None:
            return pseudo_group_name

        compatible_list = compatible if isinstance(compatible, list) else [compatible]
        if task in compatible_list:
            return pseudo_group_name

    return None


def build_jobs(
    submissions: Sequence[Path],
    task_instances: Dict[str, List[Path]],
    pseudo_groups: Dict[str, List[dict]],
) -> List[dict]:
    jobs: List[dict] = []
    next_job_id = 0

    submission_data = {}
    for submission_path in sorted(submissions):
        if submission_path.suffix.lower() != ".json":
            continue

        data = json.loads(submission_path.read_text())
        submission_data[submission_path] = data

    # Sort by combined q_score and limit to the top 5 submissions
    by_q_score = sorted(
        submission_data.items(),
        key=lambda x: x[1]["overall_scores"]["q_score"],
        reverse=True,
    )
    top_5_standard = set(
        [x for x, y in by_q_score if x.name.startswith("standard.")][:5]
    )
    top_5_privileged = set([x for x, y in by_q_score][:5])
    top_5_submissions_filenames = top_5_standard | top_5_privileged
    top_5_submissions = {x: submission_data[x] for x in top_5_submissions_filenames}

    for submission_path, data in top_5_submissions.items():
        team = data["team"]
        team_slug = slugify(team, separator="_")
        per_task_scores = data["per_task_scores"]
        q_scores = per_task_scores["q_score"]
        successful_tasks = [
            task
            for task, score in q_scores.items()
            if score > 0 or team_slug in ["robot_learning_collective", "comet"]
        ]
        print(
            f"Processing submission {submission_path.name} for team {team} with {len(successful_tasks)} successful tasks"
        )

        for task in sorted(successful_tasks):
            instances = task_instances[task]

            # Find the compatible pseudo-group for this team and task
            pseudo_group = find_compatible_pseudo_group(team_slug, task, pseudo_groups)
            if pseudo_group is None:
                log.warning(
                    f"No compatible pseudo-group found for team {team_slug} and task {task}; skipping"
                )
                continue

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
                if result_path.exists():
                    log.debug(
                        f"Skipping instance {instance_name} for team {team_slug} and task {task} because it already has a result"
                    )
                    continue

                payload = {
                    "team": team,
                    "team_slug": team_slug,
                    "task": task,
                    "instance_basename": instance_name,
                }

                job = {
                    "id": next_job_id,
                    "payload": payload,
                    "resource_type": pseudo_group,
                    "status": "pending",
                    "attempts": 0,
                }
                jobs.append(job)
                next_job_id += 1
    return jobs


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()

    test_instances = load_task_instances(args.test_instances_dir.resolve())
    submissions = sorted(args.submissions_dir.resolve().glob("*.json"))
    if not submissions:
        raise RuntimeError(f"No submission files found in {args.submissions_dir}")

    # Load resources-raw.json and create pseudo-groups
    pseudo_groups, _ = create_pseudo_resource_groups(args.resources_raw_file.resolve())

    jobs = build_jobs(submissions, test_instances, pseudo_groups)
    if not jobs:
        log.warning("No jobs were generated. Check input data.")

    # Print all the unique team slugs and resource types
    print(f"Unique team slugs: {set([job['payload']['team_slug'] for job in jobs])}")
    print(f"Unique resource types: {set([job['resource_type'] for job in jobs])}")

    if args.dry_run:
        log.info("[DRY RUN] Generated %d jobs; skipping write", len(jobs))
        return

    output_jobs_path = args.output_jobs_file.resolve()
    output_jobs_path.write_text(json.dumps(jobs, indent=2))
    log.info("Wrote %d jobs to %s", len(jobs), output_jobs_path)

    output_resources_path = args.output_resources_file.resolve()
    output_resources_path.write_text(json.dumps(pseudo_groups, indent=2))
    log.info("Wrote %d pseudo-groups to %s", len(pseudo_groups), output_resources_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        log.error("Failed to generate jobs: %s", exc)
        sys.exit(1)
