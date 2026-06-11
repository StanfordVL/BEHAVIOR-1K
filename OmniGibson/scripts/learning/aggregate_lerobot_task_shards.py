#!/usr/bin/env python3
"""Aggregate completed LeRobot v3 replay shards into one dataset per task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import pandas as pd
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
    if task_mapping and unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    if not task_mapping and not names:
        raise RuntimeError("No task names provided and no challenge task metadata found")
    return names


def shard_repo_id(task_name: str, shard_index: int) -> str:
    return f"b1k_shards/{task_name}/shard-{shard_index:03d}"


def shard_complete(lerobot_shards_root: Path, task_name: str, shard_index: int) -> bool:
    return (lerobot_shards_root / ".replay_status" / task_name / f"shard-{shard_index:03d}" / "complete").exists()


def episode_complete(lerobot_shards_root: Path, task_name: str, episode_root: Path) -> bool:
    demo_id = episode_root.name.removeprefix("episode_")
    return (lerobot_shards_root / ".replay_status" / task_name / f"shard-ep-{demo_id}" / "complete").exists()


def discover_inputs(args: argparse.Namespace, task_name: str) -> tuple[list[str], list[Path], list[str]]:
    lerobot_shards_root = Path(args.lerobot_shards_root).resolve()

    if args.per_episode:
        shard_root = lerobot_shards_root / "b1k_shards" / task_name
        roots = sorted(shard_root.glob("episode_*"))
        missing = [
            root.name
            for root in roots
            if args.require_complete and not episode_complete(lerobot_shards_root, task_name, root)
        ]
        repo_ids = [f"b1k_shards/{task_name}/{root.name}" for root in roots]
    else:
        repo_ids = [shard_repo_id(task_name, shard_index) for shard_index in range(args.num_shards)]
        roots = [lerobot_shards_root / repo_id for repo_id in repo_ids]
        missing = [
            str(shard_index)
            for shard_index, root in enumerate(roots)
            if not root.exists()
            or (args.require_complete and not shard_complete(lerobot_shards_root, task_name, shard_index))
        ]

    if args.expected_episodes is not None and len(roots) != args.expected_episodes:
        missing.append(f"expected {args.expected_episodes} inputs, found {len(roots)}")

    missing.extend(str(root) for root in roots if not root.exists())
    return repo_ids, roots, missing


def verify_aggregate(output_root: Path, expected_episodes: int | None, require_packed: bool) -> None:
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    info = json.loads(info_path.read_text())
    total_episodes = int(info.get("total_episodes", -1))
    if expected_episodes is not None and total_episodes != expected_episodes:
        raise RuntimeError(f"Expected {expected_episodes} episodes, got {total_episodes}")

    episodes_path = output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")
    episodes = pd.read_parquet(episodes_path)
    if expected_episodes is not None and len(episodes) != expected_episodes:
        raise RuntimeError(f"Expected {expected_episodes} episode rows, got {len(episodes)}")

    data_files = list((output_root / "data").rglob("*.parquet"))
    video_files = list((output_root / "videos").rglob("*.mp4"))
    if not data_files:
        raise RuntimeError("Aggregate has no data parquet files")
    video_keys = sorted(
        {
            column.removeprefix("videos/").removesuffix("/chunk_index")
            for column in episodes.columns
            if column.startswith("videos/") and column.endswith("/chunk_index")
        }
    )
    if video_keys and not video_files:
        raise RuntimeError("Aggregate has video keys but no video files")

    def format_ref(pattern: str, **values: int | str) -> str:
        return pattern.format(**values)

    data_refs = set()
    if "data/file_path" in episodes.columns:
        data_refs = set(episodes["data/file_path"].dropna().astype(str))
    elif {"data/chunk_index", "data/file_index"}.issubset(episodes.columns):
        data_pattern = info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        for _, row in episodes[["data/chunk_index", "data/file_index"]].drop_duplicates().iterrows():
            data_refs.add(
                format_ref(
                    data_pattern,
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
            )
    else:
        raise RuntimeError("Episode metadata does not contain data file references")

    missing_refs = [ref for ref in sorted(data_refs) if not (output_root / ref).exists()]

    video_refs = set()
    file_path_cols = [col for col in episodes.columns if col.startswith("videos/") and col.endswith("/file_path")]
    for col in file_path_cols:
        video_refs.update(episodes[col].dropna().astype(str))

    video_pattern = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        if not {chunk_col, file_col}.issubset(episodes.columns):
            continue
        for _, row in episodes[[chunk_col, file_col]].dropna().drop_duplicates().iterrows():
            video_refs.add(
                format_ref(
                    video_pattern,
                    video_key=video_key,
                    chunk_index=int(row[chunk_col]),
                    file_index=int(row[file_col]),
                )
            )

    missing_refs.extend(ref for ref in sorted(video_refs) if not (output_root / ref).exists())
    if missing_refs:
        raise RuntimeError(f"Missing {len(missing_refs)} referenced files, examples: {missing_refs[:5]}")

    if require_packed and expected_episodes is not None:
        if len(data_files) >= expected_episodes:
            raise RuntimeError(f"Data files are not packed: {len(data_files)} files for {expected_episodes} episodes")
        if video_keys and len(video_files) >= expected_episodes * len(video_keys):
            raise RuntimeError(
                f"Video files are not packed: {len(video_files)} files for {expected_episodes} episodes "
                f"and {len(video_keys)} video keys"
            )


def aggregate_task(args: argparse.Namespace, task_name: str) -> None:
    output_root = Path(args.output_root).resolve() / "b1k" / task_name
    repo_ids, roots, missing = discover_inputs(args, task_name)
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
    if args.verify:
        verify_aggregate(output_root, args.expected_episodes, args.require_packed)
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
    parser.add_argument("--per_episode", action="store_true", help="Aggregate b1k_shards/<task>/episode_* roots.")
    parser.add_argument("--expected_episodes", type=int, default=None)
    parser.add_argument("--data_files_size_in_mb", type=int, default=None)
    parser.add_argument("--video_files_size_in_mb", type=int, default=None)
    parser.add_argument("--verify", action="store_true", help="Verify metadata references after aggregation.")
    parser.add_argument(
        "--require_packed", action="store_true", help="Fail verification if output remains one-file-per-episode."
    )
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
