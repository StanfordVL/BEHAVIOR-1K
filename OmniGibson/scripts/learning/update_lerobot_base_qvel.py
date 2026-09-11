#!/usr/bin/env python3
"""Recover robot-local base qvel from raw HDF5 demos and patch LeRobot parquets.

This script is intended for repairing LeRobot datasets that were generated from
replay observations before holonomic ``base_qvel`` was recorded in the robot
base frame.

It is intentionally non-destructive:

- ``validate`` reads raw HDF5 + LeRobot parquet files and writes JSONL reports.
- ``write-staging`` writes updated data parquets and metadata under a separate
  staging root. It does not copy videos.

The recovered value is direct simulator state. For each raw HDF5 state frame, it
reads holonomic base joint velocities ``[x, y, rz]`` and rotates x/y by the base
yaw into the robot-local frame. For correspondence checks, ``action[t]`` is
compared to recovered ``base_qvel[t + 1]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from omnigibson.eval.utils.eval_utils import holonomic_base_qvel_to_robot_frame


BASE_QPOS_INDICES = np.asarray([0, 1, 5], dtype=np.int64)
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    task_index: int
    raw_episode_id: int
    length: int
    data_chunk_index: int
    data_file_index: int


@dataclass
class EpisodeResult:
    episode: EpisodeMeta
    demo_name: str | None = None
    robot_name: str | None = None
    ok: bool = False
    hard_fail: bool = False
    warnings: list[str] | None = None
    error: str | None = None
    frames: int = 0
    state_frames: int = 0
    action_max_abs_diff: float | None = None
    corr: list[float | None] | None = None
    mae: list[float] | None = None
    rmse: list[float] | None = None
    action_std: list[float] | None = None
    slope: list[float | None] | None = None

    def to_json(self) -> str:
        rec = {
            "episode_index": self.episode.episode_index,
            "task_index": self.episode.task_index,
            "raw_episode_id": self.episode.raw_episode_id,
            "length": self.episode.length,
            "data_chunk_index": self.episode.data_chunk_index,
            "data_file_index": self.episode.data_file_index,
            "demo_name": self.demo_name,
            "robot_name": self.robot_name,
            "ok": self.ok,
            "hard_fail": self.hard_fail,
            "warnings": self.warnings or [],
            "error": self.error,
            "frames": self.frames,
            "state_frames": self.state_frames,
            "action_max_abs_diff": self.action_max_abs_diff,
            "corr": self.corr,
            "mae": self.mae,
            "rmse": self.rmse,
            "action_std": self.action_std,
            "slope": self.slope,
        }
        return json.dumps(rec, sort_keys=True)


def get_uuid(name: str) -> int:
    val = int(hashlib.md5(name.encode()).hexdigest(), 16) % (10**8)
    return int(np.float32(val).item())


def find_robot_offsets(states: np.ndarray, uuid: np.float32, n_joints: int) -> np.ndarray:
    rows, cols = np.where(states == uuid)
    if len(rows) == 0:
        raise ValueError(f"robot uuid not found in state frames={states.shape[0]}")

    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    cols = cols[order]
    if len(rows) == states.shape[0] and np.array_equal(rows, np.arange(states.shape[0])):
        return cols

    offsets = np.empty(states.shape[0], dtype=np.int64)
    out_row = 0
    i = 0
    min_width = 15 + 2 * n_joints

    while i < len(rows):
        row = int(rows[i])
        if row != out_row:
            raise ValueError(f"robot uuid missing at state row {out_row}")

        j = i + 1
        while j < len(rows) and rows[j] == row:
            j += 1

        valid_cols = []
        for col in cols[i:j]:
            col = int(col)
            if col + min_width > states.shape[1]:
                continue
            is_asleep = states[row, col + 1]
            if is_asleep not in (0, 1):
                continue
            root_ori = states[row, col + 5 : col + 9]
            if not np.all(np.isfinite(root_ori)):
                continue
            ori_norm = float(np.linalg.norm(root_ori))
            if 0.95 <= ori_norm <= 1.05:
                valid_cols.append(col)

        if len(valid_cols) != 1:
            raise ValueError(f"robot uuid row={row} candidates={cols[i:j].tolist()} valid={valid_cols}")

        offsets[row] = valid_cols[0]
        out_row += 1
        i = j

    if out_row != states.shape[0]:
        raise ValueError(f"robot uuid missing after row {out_row - 1}; state_frames={states.shape[0]}")

    return offsets


def natural_demo_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"demo_(\d+)", name)
    return (int(match.group(1)) if match else -1, name)


def select_last_demo(data_group: h5py.Group) -> str:
    demos = sorted((name for name in data_group.keys() if name.startswith("demo_")), key=natural_demo_key)
    if not demos:
        raise ValueError("raw HDF5 has no data/demo_* groups")
    return demos[-1]


def find_robot_name(scene: dict) -> str:
    init_info = scene.get("objects_info", {}).get("init_info", {})
    candidates = [
        name
        for name, info in init_info.items()
        if isinstance(info, dict)
        and ("robot" in info.get("class_module", "") or info.get("class_name", "").lower().startswith("r1"))
    ]
    if not candidates:
        candidates = [
            name for name in scene["state"]["registry"].get("object_registry", {}) if name.startswith("robot")
        ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one robot candidate, got {candidates}")
    return candidates[0]


def list_column_to_matrix(column: pa.ChunkedArray, expected_width: int | None = None) -> np.ndarray:
    arr = column.combine_chunks()
    if not pa.types.is_list(arr.type) and not pa.types.is_large_list(arr.type):
        raise TypeError(f"expected list column, got {arr.type}")
    offsets = arr.offsets.to_numpy(zero_copy_only=False)
    lengths = np.diff(offsets)
    if len(lengths) == 0:
        width = expected_width or 0
    else:
        width = int(lengths[0])
        if not np.all(lengths == width):
            raise ValueError("ragged list column is not supported")
    if expected_width is not None and width != expected_width:
        raise ValueError(f"expected list width {expected_width}, got {width}")
    values = arr.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(lengths), width)


def replace_state_first3(table: pa.Table, first3: np.ndarray) -> pa.Table:
    col_idx = table.schema.get_field_index("observation.state")
    if col_idx < 0:
        raise KeyError("observation.state column missing")
    state_col = table.column(col_idx).combine_chunks()
    offsets = state_col.offsets
    offsets_np = offsets.to_numpy(zero_copy_only=False)
    lengths = np.diff(offsets_np)
    if len(lengths) != len(first3):
        raise ValueError(f"state rows {len(lengths)} != replacement rows {len(first3)}")
    if not np.all(lengths >= 3):
        raise ValueError("some observation.state rows have fewer than 3 values")

    values = state_col.values.to_numpy(zero_copy_only=False).copy()
    positions = offsets_np[:-1, None] + np.asarray([0, 1, 2], dtype=offsets_np.dtype)[None, :]
    values[positions] = first3.astype(values.dtype, copy=False)
    new_values = pa.array(values, type=state_col.type.value_type)
    new_state = pa.ListArray.from_arrays(offsets, new_values)
    return table.set_column(col_idx, table.schema.field(col_idx), new_state)


def corr_1d(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def slope_1d(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-6)
    if mask.sum() < 10 or np.std(x[mask]) == 0:
        return None
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def validate_correspondence(
    action: np.ndarray,
    qvel: np.ndarray,
) -> tuple[list[float | None], list[float], list[float], list[float], list[float | None], list[str]]:
    """Compare qvel[t + 1] against action[t]."""
    if len(action) < 2 or len(qvel) < 2:
        raise ValueError("episode too short for t+1 correspondence check")
    lhs = action[:-1, :3].astype(np.float64, copy=False)
    rhs = qvel[1 : len(lhs) + 1, :3].astype(np.float64, copy=False)
    err = rhs - lhs
    corr = [corr_1d(lhs[:, axis], rhs[:, axis]) for axis in range(3)]
    mae = np.mean(np.abs(err), axis=0).tolist()
    rmse = np.sqrt(np.mean(err * err, axis=0)).tolist()
    action_std = np.std(lhs, axis=0).tolist()
    slopes = [slope_1d(lhs[:, axis], rhs[:, axis]) for axis in range(3)]

    warnings = []
    for axis, name in enumerate(("x", "y", "yaw")):
        if action_std[axis] >= 0.02:
            axis_corr = corr[axis]
            if axis_corr is None:
                warnings.append(f"{name}: active axis has undefined correlation")
            elif axis_corr < 0.80 and mae[axis] > 0.03:
                warnings.append(f"{name}: low corr {axis_corr:.4f} with mae {mae[axis]:.5f}")
            elif axis_corr < 0.90:
                warnings.append(f"{name}: marginal corr {axis_corr:.4f}")
    return corr, mae, rmse, action_std, slopes, warnings


def format_raw_path(raw_root: Path, pattern: str, episode: EpisodeMeta) -> Path:
    return raw_root / pattern.format(
        task_index=episode.task_index,
        raw_episode_id=episode.raw_episode_id,
        episode_index=episode.episode_index,
    )


def decode_raw_episode(
    raw_root: Path,
    raw_path_pattern: str,
    episode: EpisodeMeta,
) -> tuple[str, str, np.ndarray, np.ndarray, int]:
    raw_path = format_raw_path(raw_root, raw_path_pattern, episode)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    with h5py.File(raw_path, "r") as f:
        scene = json.loads(f["data"].attrs["scene_file"])
        demo_name = select_last_demo(f["data"])
        demo = f["data"][demo_name]
        action = demo["action"][:]
        states = demo["state"][:]

    if action.shape[0] != episode.length:
        raise ValueError(f"last demo {demo_name} action length {action.shape[0]} != meta length {episode.length}")
    if states.shape[0] < episode.length:
        raise ValueError(f"state length {states.shape[0]} < meta length {episode.length}")

    robot_name = find_robot_name(scene)
    robot_state = scene["state"]["registry"]["object_registry"][robot_name]
    n_joints = len(robot_state["joint_pos"])
    uuid = np.float32(get_uuid(robot_name))
    offsets = find_robot_offsets(states, uuid, n_joints)
    row_idx = np.arange(states.shape[0])[:, None]
    base_qpos_cols = offsets[:, None] + 15 + BASE_QPOS_INDICES[None, :]
    base_qvel_cols = offsets[:, None] + 15 + n_joints + BASE_QPOS_INDICES[None, :]
    base_qpos = states[row_idx, base_qpos_cols].astype(np.float64)
    base_qvel = states[row_idx, base_qvel_cols].astype(np.float64)
    qvel_local = holonomic_base_qvel_to_robot_frame(base_qpos, base_qvel).astype(np.float32)

    return demo_name, robot_name, action.astype(np.float32, copy=False), qvel_local[: episode.length], states.shape[0]


def parse_task_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    task_ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            task_ids.extend(range(int(start), int(end) + 1))
        else:
            task_ids.append(int(part))
    return sorted(set(task_ids))


def load_episode_metadata(dataset_root: Path, task_ids: Iterable[int] | None = None) -> list[EpisodeMeta]:
    selected = set(task_ids) if task_ids is not None else None
    episodes = []
    for ep_path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(ep_path)
        if selected is not None:
            df = df[df["task_index"].isin(selected)]
        for _, row in df.iterrows():
            episodes.append(
                EpisodeMeta(
                    episode_index=int(row["episode_index"]),
                    task_index=int(row["task_index"]),
                    raw_episode_id=int(row["raw_episode_id"]),
                    length=int(row["length"]),
                    data_chunk_index=int(row["data/chunk_index"]),
                    data_file_index=int(row["data/file_index"]),
                )
            )
    episodes.sort(key=lambda ep: ep.episode_index)
    return episodes


def data_path(root: Path, chunk_index: int, file_index: int) -> Path:
    return root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"


def atomic_write_table(table: pa.Table, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def update_stats_json(source_stats: Path, output_stats: Path, memmap_path: Path, total_frames: int) -> None:
    stats = json.loads(source_stats.read_text())
    arr = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(total_frames, 3))
    obs_stats = stats["observation.state"]
    obs_stats["count"] = [int(total_frames)]
    for axis in range(3):
        col = np.asarray(arr[:, axis], dtype=np.float64)
        obs_stats["min"][axis] = float(np.min(col))
        obs_stats["max"][axis] = float(np.max(col))
        mean = float(np.mean(col))
        obs_stats["mean"][axis] = mean
        obs_stats["std"][axis] = float(np.std(col))
        for key, q in QUANTILES.items():
            obs_stats[key][axis] = float(np.quantile(col, q))
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    output_stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


def copy_metadata(dataset_root: Path, staging_root: Path) -> None:
    for rel in [".gitattributes", "LICENSE", "README.md", "meta/info.json", "meta/tasks.parquet", "meta/tasks.jsonl"]:
        src = dataset_root / rel
        if src.exists():
            dst = staging_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    src_episodes = dataset_root / "meta" / "episodes"
    dst_episodes = staging_root / "meta" / "episodes"
    if dst_episodes.exists():
        shutil.rmtree(dst_episodes)
    shutil.copytree(src_episodes, dst_episodes)


def summarize_results(results: list[EpisodeResult]) -> dict:
    hard_failures = [r for r in results if r.hard_fail]
    warnings = [r for r in results if r.warnings]
    ok = [r for r in results if r.ok]
    summary = {
        "episodes": len(results),
        "ok": len(ok),
        "hard_failures": len(hard_failures),
        "warnings": len(warnings),
        "total_frames": int(sum(r.frames for r in results if r.ok)),
    }
    for axis, name in enumerate(("x", "y", "yaw")):
        corr_vals = [r.corr[axis] for r in ok if r.corr and r.corr[axis] is not None]
        if corr_vals:
            summary[f"corr_{name}_mean"] = float(np.mean(corr_vals))
            summary[f"corr_{name}_median"] = float(np.median(corr_vals))
            summary[f"corr_{name}_min"] = float(np.min(corr_vals))
        mae_vals = [r.mae[axis] for r in ok if r.mae]
        if mae_vals:
            summary[f"mae_{name}_mean"] = float(np.mean(mae_vals))
            summary[f"mae_{name}_median"] = float(np.median(mae_vals))
            summary[f"mae_{name}_max"] = float(np.max(mae_vals))
    return summary


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    report_dir = output_root / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    task_ids = parse_task_ids(args.tasks)
    if args.mode == "write-staging" and task_ids is not None:
        raise ValueError("--tasks is only supported for validate; write-staging must process the full dataset")

    episodes = load_episode_metadata(dataset_root, task_ids=task_ids)
    if not episodes:
        raise RuntimeError("no episodes selected")
    if args.expected_episodes is not None and len(episodes) != args.expected_episodes:
        raise RuntimeError(f"expected {args.expected_episodes} episodes, found {len(episodes)}")

    by_file: dict[tuple[int, int], list[EpisodeMeta]] = {}
    for ep in episodes:
        by_file.setdefault((ep.data_chunk_index, ep.data_file_index), []).append(ep)

    staging_root = Path(args.staging_root) if args.staging_root else None
    total_frames = sum(ep.length for ep in episodes)
    memmap = None
    memmap_pos = 0
    if args.mode == "write-staging":
        if staging_root is None:
            raise ValueError("--staging-root is required for write-staging")
        staging_root.mkdir(parents=True, exist_ok=True)
        memmap = np.memmap(
            report_dir / "base_qvel_first3.float32.memmap",
            dtype=np.float32,
            mode="w+",
            shape=(total_frames, 3),
        )

    results = []
    report_path = report_dir / f"{args.mode}_episodes.jsonl"
    with report_path.open("w") as report:
        for file_idx, ((chunk_idx, part_idx), file_episodes) in enumerate(sorted(by_file.items()), start=1):
            src_data_path = data_path(dataset_root, chunk_idx, part_idx)
            if not src_data_path.exists():
                raise FileNotFoundError(src_data_path)

            read_columns = ["episode_index", "frame_index", "action"]
            if args.mode == "write-staging":
                table = pq.read_table(src_data_path)
                action_table = table.select(read_columns)
            else:
                table = None
                action_table = pq.read_table(src_data_path, columns=read_columns)

            ep_indices = action_table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
            frame_indices = action_table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False)
            action_matrix = list_column_to_matrix(action_table["action"], expected_width=args.action_width)
            replacements = np.empty((len(ep_indices), 3), dtype=np.float32) if args.mode == "write-staging" else None

            for ep in sorted(file_episodes, key=lambda item: item.episode_index):
                result = EpisodeResult(episode=ep, warnings=[])
                try:
                    positions = np.flatnonzero(ep_indices == ep.episode_index)
                    if len(positions) != ep.length:
                        raise ValueError(f"parquet row count {len(positions)} != meta length {ep.length}")
                    if not np.array_equal(frame_indices[positions], np.arange(ep.length)):
                        raise ValueError("parquet frame_index is not contiguous 0..length-1")

                    demo_name, robot_name, raw_action, qvel_local, state_frames = decode_raw_episode(
                        raw_root,
                        args.raw_path_pattern,
                        ep,
                    )
                    result.demo_name = demo_name
                    result.robot_name = robot_name
                    result.frames = ep.length
                    result.state_frames = state_frames

                    parquet_action = action_matrix[positions]
                    max_abs_diff = float(np.max(np.abs(raw_action[:, : args.action_width] - parquet_action)))
                    result.action_max_abs_diff = max_abs_diff
                    if max_abs_diff > args.action_atol:
                        raise ValueError(f"raw action and parquet action differ: max_abs={max_abs_diff}")

                    corr, mae, rmse, action_std, slopes, warnings_ = validate_correspondence(
                        parquet_action[:, :3],
                        qvel_local,
                    )
                    result.corr = corr
                    result.mae = mae
                    result.rmse = rmse
                    result.action_std = action_std
                    result.slope = slopes
                    result.warnings = warnings_
                    result.ok = True

                    if replacements is not None:
                        replacements[positions] = qvel_local[: ep.length]
                        assert memmap is not None
                        memmap[memmap_pos : memmap_pos + ep.length] = qvel_local[: ep.length]
                        memmap_pos += ep.length
                except Exception as exc:
                    result.error = repr(exc)
                    result.hard_fail = True
                results.append(result)
                report.write(result.to_json() + "\n")

            if args.mode == "write-staging":
                assert table is not None and replacements is not None and staging_root is not None
                atomic_write_table(
                    replace_state_first3(table, replacements),
                    data_path(staging_root, chunk_idx, part_idx),
                )

            if file_idx % args.progress_every == 0 or file_idx == len(by_file):
                summary = summarize_results(results)
                print(
                    f"processed_files={file_idx}/{len(by_file)} episodes={summary['episodes']} "
                    f"ok={summary['ok']} hard_failures={summary['hard_failures']} warnings={summary['warnings']}",
                    flush=True,
                )

    summary = summarize_results(results)
    (report_dir / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    hard_failures = [r for r in results if r.hard_fail]
    if hard_failures:
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise RuntimeError(f"{len(hard_failures)} hard failures; see {report_path}")

    if args.mode == "write-staging":
        assert staging_root is not None and memmap is not None
        memmap.flush()
        copy_metadata(dataset_root, staging_root)
        update_stats_json(
            dataset_root / "meta" / "stats.json",
            staging_root / "meta" / "stats.json",
            report_dir / "base_qvel_first3.float32.memmap",
            total_frames,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report_dir={report_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "write-staging"])
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root containing data/ and meta/")
    parser.add_argument("--raw-root", required=True, help="Root containing raw HDF5 files")
    parser.add_argument("--output-root", required=True, help="Directory for reports and temporary memmaps")
    parser.add_argument("--staging-root", help="Output LeRobot staging root for write-staging mode")
    parser.add_argument("--run-name", default="base_qvel_update")
    parser.add_argument("--tasks", help="Comma-separated task IDs or ranges, e.g. 0-9,20")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--raw-path-pattern", default="task-{task_index:04d}/episode_{raw_episode_id:08d}.hdf5")
    parser.add_argument("--action-width", type=int, default=23)
    parser.add_argument("--action-atol", type=float, default=1e-5)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
