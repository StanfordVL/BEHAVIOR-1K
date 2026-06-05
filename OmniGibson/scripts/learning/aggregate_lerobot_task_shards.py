#!/usr/bin/env python3
"""Aggregate completed LeRobot v3 replay shards into one dataset per task."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def ensure_shared_permissions(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to("/vision/group/behavior")
    except ValueError:
        return

    shutil.chown(resolved, group="behavior")
    for child in resolved.rglob("*"):
        shutil.chown(child, group="behavior")
        mode = child.stat().st_mode
        if child.is_dir():
            child.chmod(mode | 0o2770)
        else:
            child.chmod(mode | 0o660)
    resolved.chmod(resolved.stat().st_mode | 0o2770)


def load_task_mapping(data_root: Path) -> dict[str, int]:
    task_mapping = {}
    for year, filename in (("2025", "B50_task_misc.csv"), ("2026", "B100_task_misc.csv")):
        task_misc_path = data_root / f"{year}-challenge-task-instances" / "metadata" / filename
        if not task_misc_path.exists():
            continue
        with task_misc_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                task_mapping[row["Task"]] = int(row["Task ID"])

    if not task_mapping:
        raise RuntimeError(f"No challenge task metadata found under {data_root}")
    return task_mapping


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
    ensure_shared_permissions(output_root)


def main() -> None:
    os.umask(0o002)
    default_repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(default_repo_root))
    parser.add_argument("--data_root", default=os.environ.get("OMNIGIBSON_DATA_PATH", "/vision/group/behavior"))
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

    task_mapping = load_task_mapping(Path(args.data_root).resolve())
    for task_name in parse_task_names(args, task_mapping):
        aggregate_task(args, task_name)


if __name__ == "__main__":
    main()
