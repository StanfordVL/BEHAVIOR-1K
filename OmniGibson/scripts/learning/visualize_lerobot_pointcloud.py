"""Visualize RGB-D point clouds from a B1K LeRobot dataset.

The script reads decoded RGB/depth frames and robot-relative camera poses via
``LeRobotDataset``, projects depth with ``obs_utils.depth_to_pcd``, then writes
or displays a colored point cloud in the robot base frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch as th
from lerobot.datasets import LeRobotDataset

from omnigibson.eval.utils.eval_utils import CAMERA_INTRINSICS
from omnigibson.eval.utils.obs_utils import depth_to_pcd, color_pcd_vis


CAMERA_KEY_TO_INTRINSIC_NAME = {
    "left_realsense_link_camera_0": "left_wrist",
    "right_realsense_link_camera_0": "right_wrist",
    "zed_link_camera_0": "head",
}


def _infer_repo_id(root: Path) -> str:
    if root.parent.name:
        return f"{root.parent.name}/{root.name}"
    raise ValueError("Could not infer repo_id from --root. Please pass --repo_id explicitly.")


def _as_hwc_rgb(value) -> th.Tensor:
    rgb = value.detach().cpu() if isinstance(value, th.Tensor) else th.as_tensor(value)
    if rgb.ndim == 3 and rgb.shape[0] in {3, 4}:
        rgb = rgb[:3].permute(1, 2, 0)
    elif rgb.ndim == 3 and rgb.shape[-1] in {3, 4}:
        rgb = rgb[..., :3]
    else:
        raise ValueError(f"Expected RGB frame with shape CxHxW or HxWxC, got {tuple(rgb.shape)}")
    return rgb.float()


def _as_hw_depth(value, unit: str) -> th.Tensor:
    depth = value.detach().cpu() if isinstance(value, th.Tensor) else th.as_tensor(value)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim != 2:
        raise ValueError(f"Expected depth frame with shape 1xHxW, HxWx1, or HxW, got {tuple(depth.shape)}")

    depth = depth.float()
    if unit == "mm":
        depth = depth / 1000.0
    elif unit != "m":
        raise ValueError(f"Unsupported depth unit {unit!r}; expected 'm' or 'mm'")
    return depth


def _depth_unit(dataset: LeRobotDataset, key: str) -> str:
    info = dataset.meta.features[key].get("info") or {}
    unit = info.get("video.output_unit")
    if unit is None:
        raise ValueError(f"Depth feature {key} is missing info['video.output_unit']")
    return unit


def _load_intrinsics(args: argparse.Namespace, dataset: LeRobotDataset) -> dict[str, np.ndarray]:
    if args.intrinsics_json is not None:
        with Path(args.intrinsics_json).expanduser().open("r") as f:
            raw_intrinsics = json.load(f)
        return {name: np.asarray(K, dtype=np.float32) for name, K in raw_intrinsics.items()}

    meta_intrinsics = dataset.meta.info.get("cam_intrinsics") or {}
    if meta_intrinsics:
        return {name: np.asarray(K, dtype=np.float32) for name, K in meta_intrinsics.items()}

    return CAMERA_INTRINSICS[args.robot_type]


def _camera_names(dataset: LeRobotDataset, requested: list[str] | None) -> list[str]:
    names = []
    for key in dataset.meta.depth_keys:
        prefix = "observation.depth_linear."
        if not key.startswith(prefix):
            continue
        camera_name = key[len(prefix) :]
        if f"observation.rgb.{camera_name}" in dataset.meta.video_keys:
            names.append(camera_name)

    if requested is not None:
        requested_set = set(requested)
        names = [name for name in names if name in requested_set]

    if not names:
        raise ValueError("No cameras with matching RGB and depth features were found.")
    return names


def _build_camera_pcd(
    item: dict,
    dataset: LeRobotDataset,
    camera_name: str,
    K: np.ndarray,
    max_depth: float,
) -> np.ndarray:
    rgb_key = f"observation.rgb.{camera_name}"
    depth_key = f"observation.depth_linear.{camera_name}"
    pose_key = f"observation.robot2cam_pose.{camera_name}"

    rgb = _as_hwc_rgb(item[rgb_key])
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    depth = _as_hw_depth(item[depth_key], unit=_depth_unit(dataset, depth_key))
    rel_pose_value = item[pose_key]
    rel_pose = (
        rel_pose_value.detach().cpu().float()
        if isinstance(rel_pose_value, th.Tensor)
        else th.as_tensor(rel_pose_value).float()
    )

    xyz = depth_to_pcd(
        depth.unsqueeze(0),
        rel_pose.unsqueeze(0),
        th.as_tensor(K, dtype=th.float32),
    )[0]

    mask = th.isfinite(depth) & th.isfinite(xyz).all(dim=-1) & (depth > 0.0) & (depth <= max_depth)
    color_xyz = th.cat([rgb, xyz], dim=-1)[mask]
    return color_xyz.numpy()


def _filter_range(color_pcd: np.ndarray, pcd_range: list[float] | None) -> np.ndarray:
    if pcd_range is None:
        return color_pcd

    x_min, x_max, y_min, y_max, z_min, z_max = pcd_range
    xyz = color_pcd[:, 3:6]
    mask = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] <= z_max)
    )
    return color_pcd[mask]


def _downsample(color_pcd: np.ndarray, num_points: int | None, seed: int) -> np.ndarray:
    if num_points is None or len(color_pcd) <= num_points:
        return color_pcd
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(color_pcd), size=num_points, replace=False)
    return color_pcd[indices]


def _sample_fixed(color_pcd: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    if len(color_pcd) == 0:
        raise ValueError("Point cloud is empty after filtering; loosen --pcd_range or --max_depth.")

    rng = np.random.default_rng(seed)
    replace = len(color_pcd) < num_points
    indices = rng.choice(len(color_pcd), size=num_points, replace=replace)
    return color_pcd[indices]


def _build_frame_pcd(
    item: dict,
    dataset: LeRobotDataset,
    camera_names: list[str],
    intrinsics: dict[str, np.ndarray],
    max_depth: float,
    pcd_range: list[float] | None,
) -> np.ndarray:
    color_pcds = []
    for camera_name in camera_names:
        intrinsic_name = CAMERA_KEY_TO_INTRINSIC_NAME.get(camera_name, camera_name)
        if camera_name in intrinsics:
            K = intrinsics[camera_name]
        elif intrinsic_name in intrinsics:
            K = intrinsics[intrinsic_name]
        else:
            raise KeyError(
                f"No intrinsics found for camera {camera_name!r}. " f"Available keys: {sorted(intrinsics.keys())}"
            )
        color_pcds.append(_build_camera_pcd(item, dataset, camera_name, K, max_depth=max_depth))

    color_pcd = np.concatenate(color_pcds, axis=0)
    return _filter_range(color_pcd, pcd_range)


def _write_ply(color_pcd: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:, 3:6])
    o3d.io.write_point_cloud(str(output), pcd)


def _iter_video_pcds(
    dataset: LeRobotDataset,
    start_frame: int,
    first_item: dict,
    camera_names: list[str],
    intrinsics: dict[str, np.ndarray],
    max_depth: float,
    pcd_range: list[float] | None,
    num_points: int,
    seed: int,
    stride: int,
    max_frames: int | None,
):
    episode_index = int(first_item["episode_index"])
    frame_idx = start_frame
    frame_count = 0
    while frame_idx < len(dataset) and (max_frames is None or frame_count < max_frames):
        item = first_item if frame_idx == start_frame else dataset[frame_idx]
        if int(item["episode_index"]) != episode_index:
            break

        color_pcd = _build_frame_pcd(
            item=item,
            dataset=dataset,
            camera_names=camera_names,
            intrinsics=intrinsics,
            max_depth=max_depth,
            pcd_range=pcd_range,
        )
        yield _sample_fixed(color_pcd, num_points, seed=seed + frame_idx)

        frame_count += 1
        frame_idx += stride


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("~/Downloads/2026-challenge-demos/b1k/turning_on_radio"),
        help="Path to one LeRobot dataset root, e.g. .../b1k/turning_on_radio.",
    )
    parser.add_argument("--repo_id", default=None, help="LeRobot repo id. Defaults to '<parent>/<root-name>'.")
    parser.add_argument("--frame", type=int, default=0, help="Global dataset frame index to visualize.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera names to fuse. Defaults to all RGB-D cameras in the dataset.",
    )
    parser.add_argument("--robot_type", default="R1Pro", help="Robot type for default intrinsics.")
    parser.add_argument(
        "--intrinsics_json",
        default=None,
        help="Optional JSON mapping camera or intrinsic names to 3x3 matrices.",
    )
    parser.add_argument("--max_depth", type=float, default=10.0, help="Drop pixels beyond this depth in meters.")
    parser.add_argument(
        "--pcd_range",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Optional robot-frame crop box.",
    )
    parser.add_argument("--num_points", type=int, default=200000, help="Randomly downsample to this many points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lerobot_pointcloud.ply"))
    parser.add_argument("--visualize", action="store_true", help="Open an interactive Open3D visualizer.")
    parser.add_argument(
        "--visualize_video",
        action="store_true",
        help="Stream consecutive frames from the current episode through color_pcd_vis.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Maximum frames for --visualize_video. Use 0 to stream until the episode ends.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for --visualize_video.")
    args = parser.parse_args()

    root = args.root.expanduser()
    repo_id = args.repo_id or _infer_repo_id(root)
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    intrinsics = _load_intrinsics(args, dataset)
    camera_names = _camera_names(dataset, args.cameras)

    if args.visualize_video:
        if args.num_points is None:
            raise ValueError("--visualize_video requires --num_points so every frame has the same shape.")
        if args.stride <= 0:
            raise ValueError("--stride must be positive.")

        first_item = dataset[args.frame]
        max_frames = None if args.max_frames == 0 else args.max_frames
        video_pcds = list(
            _iter_video_pcds(
                dataset=dataset,
                start_frame=args.frame,
                first_item=first_item,
                camera_names=camera_names,
                intrinsics=intrinsics,
                max_depth=args.max_depth,
                pcd_range=args.pcd_range,
                num_points=args.num_points,
                seed=args.seed,
                stride=args.stride,
                max_frames=max_frames,
            )
        )
        if not video_pcds:
            raise ValueError("No frames were collected for video visualization.")

        color_pcd_video = np.stack(video_pcds, axis=0)
        _write_ply(color_pcd_video[0], args.output.expanduser())
        print(
            f"Streaming {len(color_pcd_video)} frames with {color_pcd_video.shape[1]} points each. "
            f"Wrote first frame to {args.output.expanduser()}"
        )
        color_pcd_vis(color_pcd_video)
        return

    item = dataset[args.frame]
    color_pcd = _build_frame_pcd(
        item=item,
        dataset=dataset,
        camera_names=camera_names,
        intrinsics=intrinsics,
        max_depth=args.max_depth,
        pcd_range=args.pcd_range,
    )
    color_pcd = _downsample(color_pcd, args.num_points, args.seed)

    _write_ply(color_pcd, args.output.expanduser())
    print(f"Wrote {len(color_pcd)} points to {args.output.expanduser()}")

    if args.visualize:
        color_pcd_vis(color_pcd)


if __name__ == "__main__":
    main()
