"""
B1K LeRobot benchmark for the current upstream-depth integration path.

This is a focused counterpart to ``outputs/benchmark.py``.  The original script
benchmarks many historical encoding variants laid out in the challenge-demo
``videos/task-*`` tree.  This script targets the current B1K integration combo:

- RGB:   LeRobotDataset video features encoded by ``VideoEncoderConfig``
- Depth: LeRobotDataset depth video features encoded by ``DepthEncoderConfig``

It uses the official ``lerobot.datasets.LeRobotDataset`` read path for random
access speed and compares decoded depth frames against the original replay HDF5
observations for reconstruction error.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets import LeRobotDataset
from tqdm import tqdm


CAMERA_PREFIXES: dict[str, str] = {
    "observation.depth_linear.left_realsense_link_camera_0": "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "observation.depth_linear.right_realsense_link_camera_0": "robot_r1::robot_r1:right_realsense_link:Camera:0",
    "observation.depth_linear.zed_link_camera_0": "robot_r1::robot_r1:zed_link:Camera:0",
}

STAT_KEYS = ("mean", "median", "std", "min", "max", "p01", "p05", "p25", "p75", "p95", "p99")


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _as_numpy_frame(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value.astype(np.float32, copy=False)


def _depth_unit(dataset: LeRobotDataset, depth_key: str) -> str:
    feature_info = dataset.meta.features[depth_key].get("info") or {}
    unit = feature_info.get("video.output_unit")
    if unit is None:
        raise ValueError(f"Depth feature {depth_key} is missing info['video.output_unit']")
    if unit not in {"m", "mm"}:
        raise ValueError(f"Unsupported depth unit for {depth_key}: {unit!r}")
    return unit


def _to_meters(depth: np.ndarray, unit: str) -> np.ndarray:
    if unit == "m":
        return depth
    return depth / 1000.0


def _resolve_hdf5_path(args: argparse.Namespace) -> Path:
    if args.hdf5_path is not None:
        return Path(args.hdf5_path).expanduser()

    data_root = Path(args.data_folder).expanduser()
    replayed = data_root / "replayed" / f"episode_{args.demo_id:08d}.hdf5"
    if replayed.exists():
        return replayed

    raw_root = data_root / "2026-challenge-rawdata"
    if args.task_id is not None:
        return raw_root / f"task-{args.task_id:04d}" / f"episode_{args.demo_id:08d}.hdf5"

    matches = sorted(raw_root.glob(f"task-*/episode_{args.demo_id:08d}.hdf5"))
    if not matches:
        raise FileNotFoundError(f"No episode_{args.demo_id:08d}.hdf5 found under {raw_root}/task-*")
    if len(matches) > 1:
        task_hint = args.task_name.replace("_", " ") if args.task_name else None
        raise ValueError(
            f"Found multiple HDF5 matches for demo_id={args.demo_id}: {matches}. "
            f"Pass --task_id or --hdf5_path explicitly. task_name hint was {task_hint!r}."
        )
    return matches[0]


def _select_hdf5_demo(f: h5py.File) -> str:
    demos = list(f["data"].keys())
    return max(demos, key=lambda key: f["data"][key].attrs.get("num_samples", 0))


def _valid_depth_indices(obs: h5py.Group, hdf5_keys: list[str], max_len: int) -> np.ndarray:
    valid_indices = []
    for idx in range(max_len):
        valid = True
        for hdf5_key in hdf5_keys:
            frame = np.asarray(obs[hdf5_key][idx])
            if not np.isfinite(frame).all() or float(np.abs(frame).max()) == 0.0:
                valid = False
                break
        if valid:
            valid_indices.append(idx)

    if not valid_indices:
        raise ValueError("No valid nonzero HDF5 depth frames found for reconstruction benchmark.")
    return np.asarray(valid_indices, dtype=np.int64)


def _dataset_summary(dataset: LeRobotDataset) -> None:
    print("=" * 100)
    print("Dataset")
    print("=" * 100)
    print(f"root        : {dataset.root}")
    print(f"frames      : {dataset.num_frames}")
    print(f"episodes    : {dataset.num_episodes}")
    print(f"video keys  : {len(dataset.meta.video_keys)}")
    print(f"depth keys  : {dataset.meta.depth_keys}")
    print(f"git hash    : {dataset.meta.info.get('omnigibson_git_hash')}")
    print(f"intrinsics  : {sorted((dataset.meta.info.get('cam_intrinsics') or {}).keys())}")

    print("\nVideo feature metadata:")
    for key in dataset.meta.video_keys:
        info = dataset.meta.features[key].get("info") or {}
        print(
            f"  {key}: codec={info.get('video.codec')} pix_fmt={info.get('video.pix_fmt')} "
            f"g={info.get('video.g')} crf={info.get('video.crf')} depth={info.get('is_depth_map', False)} "
            f"output_unit={info.get('video.output_unit')}"
        )


def benchmark_random_access_speed(dataset: LeRobotDataset, sample_count: int, seed: int) -> dict[str, float]:
    print("\n" + "=" * 100)
    print("Benchmark 1: LeRobotDataset Random-Access Decode Speed")
    print("=" * 100)
    rng = np.random.default_rng(seed)
    n = min(sample_count, len(dataset))
    indices = rng.permutation(len(dataset))[:n]

    # Warm one item so any lazy decoder state is initialized outside the timed loop.
    _ = dataset[int(indices[0])]

    t0 = time.perf_counter()
    for idx in tqdm(indices, desc="LeRobotDataset __getitem__", leave=False):
        _ = dataset[int(idx)]
    elapsed = time.perf_counter() - t0
    fps = n / elapsed
    print(f"samples     : {n}")
    print(f"elapsed_s   : {elapsed:.3f}")
    print(f"samples/s   : {fps:.2f}")
    return {"samples": float(n), "elapsed_s": elapsed, "samples_per_s": fps}


def benchmark_depth_reconstruction(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    sample_count: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    print("\n" + "=" * 130)
    print("Benchmark 2: Depth Reconstruction Error vs HDF5 Ground Truth")
    print("=" * 130)
    print(f"hdf5        : {hdf5_path}")

    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, float]] = {}
    with h5py.File(hdf5_path, "r") as f:
        demo_key = _select_hdf5_demo(f)
        if "obs" not in f["data"][demo_key]:
            raise KeyError(
                f"{hdf5_path} has no data/{demo_key}/obs group. "
                "Use a replayed observation HDF5, e.g. data_folder/replayed/episode_*.hdf5."
            )
        obs = f["data"][demo_key]["obs"]
        print(f"demo        : {demo_key}")

        depth_to_hdf5_key = {}
        for depth_key in dataset.meta.depth_keys:
            prefix = CAMERA_PREFIXES.get(depth_key)
            if prefix is None:
                print(f"[WARN] No HDF5 camera mapping for {depth_key}; skipping")
                continue
            hdf5_key = f"{prefix}::depth_linear"
            if hdf5_key not in obs:
                print(f"[WARN] Missing HDF5 key {hdf5_key}; skipping")
                continue
            depth_to_hdf5_key[depth_key] = hdf5_key

        valid_indices = _valid_depth_indices(obs, list(depth_to_hdf5_key.values()), len(dataset))
        n = min(sample_count, len(valid_indices))
        indices = rng.permutation(valid_indices)[:n]
        print(f"valid frames: {len(valid_indices)} / {len(dataset)}")
        print(f"samples     : {n}")
        results["_summary"] = {
            "valid_frames": float(len(valid_indices)),
            "total_frames": float(len(dataset)),
            "samples": float(n),
        }

        header = f"{'depth_key':<58} {'samples':>8}"
        for stat in STAT_KEYS:
            header += f" {stat:>10}"
        print(header)
        print("-" * len(header))

        for depth_key, hdf5_key in depth_to_hdf5_key.items():
            errors: list[np.ndarray] = []
            for idx in tqdm(indices, desc=depth_key, leave=False):
                idx = int(idx)
                item = dataset[idx]
                decoded = _to_meters(_as_numpy_frame(item[depth_key]), _depth_unit(dataset, depth_key))
                gt = np.asarray(obs[hdf5_key][idx], dtype=np.float32)
                errors.append(np.abs(decoded - gt).ravel())

            err = np.concatenate(errors)
            stats = _stats(err)
            results[depth_key] = stats
            row = f"{depth_key:<58} {n:>8}"
            for stat in STAT_KEYS:
                row += f" {stats[stat]:>10.6f}"
            print(row)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark current B1K LeRobot streaming decode and depth reconstruction."
    )
    parser.add_argument("--root", default="/home/wsai/Documents/Files/behavior/lerobot/b1k/turning_on_radio")
    parser.add_argument("--repo_id", default="b1k/turning_on_radio")
    parser.add_argument("--data_folder", default="/home/wsai/Documents/Files/behavior")
    parser.add_argument("--task_name", default="turning_on_radio")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--demo_id", type=int, default=10)
    parser.add_argument("--hdf5_path", default=None)
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument("--speed_samples", type=int, default=256)
    parser.add_argument("--error_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None, help="Optional path to write machine-readable results.")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=Path(args.root).expanduser(), video_backend=args.video_backend)
    hdf5_path = _resolve_hdf5_path(args)
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    _dataset_summary(dataset)
    speed = benchmark_random_access_speed(dataset, args.speed_samples, args.seed)
    depth = benchmark_depth_reconstruction(dataset, hdf5_path, args.error_samples, args.seed)

    result = {"speed": speed, "depth_reconstruction": depth}
    if args.json is not None:
        out = Path(args.json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nWrote JSON results to {out}")


if __name__ == "__main__":
    main()
