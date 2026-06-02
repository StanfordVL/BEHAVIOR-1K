#!/usr/bin/env python3
"""Aggregate completed LeRobot v3 replay shards into one dataset per task."""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def load_task_mapping(repo_root: Path) -> dict[str, int]:
    eval_utils = repo_root / "OmniGibson" / "omnigibson" / "learning" / "utils" / "eval_utils.py"
    tree = ast.parse(eval_utils.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TASK_NAMES_TO_INDICES":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"TASK_NAMES_TO_INDICES not found in {eval_utils}")


def parse_task_names(args: argparse.Namespace, task_mapping: dict[str, int]) -> list[str]:
    if args.task_names_file:
        names = [
            line.strip()
            for line in Path(args.task_names_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.tasks:
        names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    else:
        names = [name for name, _ in sorted(task_mapping.items(), key=lambda item: item[1])]

    unknown = sorted(set(names) - set(task_mapping))
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    return names


def shard_repo_id(task_name: str, shard_index: int) -> str:
    return f"b1k_shards/{task_name}/shard-{shard_index:03d}"


def shard_complete(lerobot_shards_root: Path, task_name: str, shard_index: int) -> bool:
    return (lerobot_shards_root / ".replay_status" / task_name / f"shard-{shard_index:03d}" / "complete").exists()


def aggregate_task(args: argparse.Namespace, task_name: str) -> None:
    lerobot_shards_root = Path(args.lerobot_shards_root).resolve()
    output_root = Path(args.output_root).resolve() / "b1k" / task_name
    repo_ids = [shard_repo_id(task_name, shard_index) for shard_index in range(args.num_shards)]
    roots = [lerobot_shards_root / repo_id for repo_id in repo_ids]

    missing = [
        shard_index
        for shard_index, root in enumerate(roots)
        if not root.exists()
        or (args.require_complete and not shard_complete(lerobot_shards_root, task_name, shard_index))
    ]
    if missing:
        print(f"skip {task_name}: missing/incomplete shards {missing}")
        return

    print(f"{'aggregate' if args.run else 'dry-run aggregate'} {task_name}: {len(repo_ids)} shards -> {output_root}")
    if not args.run:
        return

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        shutil.rmtree(output_root)

    aggregate_datasets(
        repo_ids=repo_ids,
        roots=roots,
        aggr_repo_id=f"b1k/{task_name}",
        aggr_root=output_root,
        data_files_size_in_mb=args.data_files_size_in_mb,
        video_files_size_in_mb=args.video_files_size_in_mb,
    )


def main() -> None:
    default_repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(default_repo_root))
    parser.add_argument("--lerobot_shards_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tasks", default="", help="Comma-separated task names. Defaults to all tasks.")
    parser.add_argument("--task_names_file", default="")
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--data_files_size_in_mb", type=int, default=None)
    parser.add_argument("--video_files_size_in_mb", type=int, default=None)
    parser.add_argument("--require_complete", action="store_true", default=True)
    parser.add_argument("--allow_incomplete", action="store_false", dest="require_complete")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true", help="Actually aggregate. Default is dry-run.")
    args = parser.parse_args()

    task_mapping = load_task_mapping(Path(args.repo_root).resolve())
    for task_name in parse_task_names(args, task_mapping):
        aggregate_task(args, task_name)


if __name__ == "__main__":
    main()
