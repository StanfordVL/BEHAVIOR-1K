import argparse
import av
import json
import logging
import numpy as np
import os
import pandas as pd
import torch as th
import torchvision
from lerobot.datasets.compute_stats import _assert_type_and_shape
from omnigibson.learning.utils.dataset_utils import get_credentials
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    PROPRIOCEPTION_INDICES,
)
from omnigibson.learning.utils.obs_utils import (
    dequantize_depth,
    MIN_DEPTH,
    MAX_DEPTH,
    DEPTH_SHIFT,
)
from pathlib import Path
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.io import VideoReader
from typing import Any, Dict, Tuple
from tqdm import tqdm


def hf_transform_to_torch(items_dict: dict[th.Tensor | None]):
    """
    Adapted from lerobot.datasets.utils.hf_transform_to_torch
    Preserve float64 for timestamp to avoid precision issues
    Below is the original docstring:
    Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        if key == "timestamp":
            items_dict[key] = [x if isinstance(x, str) else th.tensor(x, dtype=th.float64) for x in items_dict[key]]
        else:
            first_item = items_dict[key][0]
            if isinstance(first_item, PILImage.Image):
                to_tensor = transforms.ToTensor()
                items_dict[key] = [to_tensor(img) for img in items_dict[key]]
            elif first_item is None:
                pass
            else:
                items_dict[key] = [x if isinstance(x, str) else th.tensor(x) for x in items_dict[key]]
    return items_dict


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> th.Tensor:
    return decode_video_frames_torchvision(
        video_path=video_path, timestamps=timestamps, tolerance_s=tolerance_s, backend=backend
    )


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    backend: str | None = None,
) -> th.Tensor:
    """
    Adapted from decode_video_frames_vision to handle depth decoding
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    if "depth" in video_path:
        backend = "pyav"
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    if "depth" in video_path:
        reader = DepthVideoReader(video_path, "video")
    else:
        reader = VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps) - 5  # a little backward to account for timestamp mismatch
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    reader.container.close()

    reader = None

    query_ts = th.tensor(timestamps)
    loaded_ts = th.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = th.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = th.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(th.float32)
    if "depth" not in video_path:
        closest_frames = closest_frames / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


class DepthVideoReader(VideoReader):
    """
    Adapted from torchvision.io.VideoReader to support gray16le decoding for depth
    """

    def __next__(self) -> Dict[str, Any]:
        """Decodes and returns the next frame of the current stream.
        Frames are encoded as a dict with mandatory
        data and pts fields, where data is a tensor, and pts is a
        presentation timestamp of the frame expressed in seconds
        as a float.

        Returns:
            (dict): a dictionary and containing decoded frame (``data``)
            and corresponding timestamp (``pts``) in seconds

        """
        try:
            frame = next(self._c)
            pts = float(frame.pts * frame.time_base)
            if "video" in self.pyav_stream:
                frame = th.as_tensor(
                    dequantize_depth(
                        frame.reformat(format="gray16le").to_ndarray(),
                        min_depth=MIN_DEPTH,
                        max_depth=MAX_DEPTH,
                        shift=DEPTH_SHIFT,
                    )
                )
            elif "audio" in self.pyav_stream:
                frame = th.as_tensor(frame.to_ndarray()).permute(1, 0)
            else:
                frame = None
        except av.error.EOFError:
            raise StopIteration

        if frame.numel() == 0:
            raise StopIteration

        return {"data": frame, "pts": pts}


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    q01 = np.stack([s["q01"] for s in stats_ft_list])
    q99 = np.stack([s["q99"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    # Compute weighted quantiles
    weighted_q01 = np.percentile(q01, 1, axis=0)
    weighted_q99 = np.percentile(q99, 99, axis=0)

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "q01": weighted_q01,
        "q99": weighted_q99,
        "count": total_count,
    }


def generate_task_json(data_dir: str, credentials_path: str) -> int:
    num_tasks = len(TASK_NAMES_TO_INDICES)
    gc = get_credentials(credentials_path=credentials_path)[0]
    spreadsheet = gc.open("B50 Task Misc")
    ws = spreadsheet.worksheet("Task natural language instruction")
    rows = ws.get_all_values()
    with open(f"{data_dir}/meta/tasks.jsonl", "w") as f:
        for task_name, task_index in tqdm(TASK_NAMES_TO_INDICES.items()):
            # find the corresponding row in the google sheet
            prompt = None
            for row in rows:
                if row[0] == task_name:
                    prompt = row[1].replace("\u2019", "'")
                    break
            assert prompt is not None, f"Natural language instruction not found for task: {task_name}"
            json.dump(
                {
                    "task_index": task_index,
                    "task_name": task_name,
                    "task": prompt,
                },
                f,
            )
            f.write("\n")
    print(f"Generated task JSON for {num_tasks} tasks.")
    return num_tasks


def generate_episode_json(data_dir: str, robot_type: str = "R1Pro") -> Tuple[int, int]:
    assert os.path.exists(f"{data_dir}/meta/tasks.jsonl"), "Task JSON does not exist!"
    assert os.path.exists(f"{data_dir}/meta/episodes"), "Episode metadata directory does not exist!"
    with open(f"{data_dir}/meta/tasks.jsonl", "r") as f:
        task_json = [json.loads(line) for line in f]
    num_frames = 0
    num_episodes = 0
    with open(f"{data_dir}/meta/episodes.jsonl", "w") as out_f:
        with open(f"{data_dir}/meta/episodes_stats.jsonl", "w") as out_stats_f:
            for task_info in tqdm(task_json):
                task_index = task_info["task_index"]
                task_name = task_info["task"]
                if not os.path.exists(f"{data_dir}/meta/episodes/task-{task_index:04d}"):
                    continue
                for episode_name in tqdm(sorted(os.listdir(f"{data_dir}/meta/episodes/task-{task_index:04d}"))):
                    with open(f"{data_dir}/meta/episodes/task-{task_index:04d}/{episode_name}", "r") as f:
                        episode_info = json.load(f)
                        episode_index = int(episode_name.split(".")[0].split("_")[-1])
                        episode_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "length": episode_info["num_samples"],
                        }
                        # load the corresponding parquet file
                        episode_df = pd.read_parquet(
                            f"{data_dir}/data/task-{task_index:04d}/episode_{episode_index:08d}.parquet"
                        )
                        episode_stats = {}
                        for key in ["action", "observation.state", "observation.cam_rel_poses"]:
                            if key not in episode_stats:
                                episode_stats[key] = {}
                            values = np.stack(episode_df[key].values)
                            if len(values.shape) == 1:
                                values = values[:, np.newaxis]
                            episode_stats[key]["min"] = values.min(axis=0).tolist()
                            episode_stats[key]["max"] = values.max(axis=0).tolist()
                            episode_stats[key]["mean"] = values.mean(axis=0).tolist()
                            episode_stats[key]["std"] = values.std(axis=0).tolist()
                            episode_stats[key]["q01"] = np.quantile(values, 0.01, axis=0).tolist()
                            episode_stats[key]["q99"] = np.quantile(values, 0.99, axis=0).tolist()
                            episode_stats[key]["count"] = [values.shape[0]]
                            if key == "observation.state":
                                robot_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["robot_pos"]]
                                episode_json["distance_traveled"] = round(
                                    np.sum(np.linalg.norm(robot_pos[1:, :] - robot_pos[:-1, :], axis=-1)).item(), 4
                                )
                                left_eef_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["eef_left_pos"]]
                                right_eef_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["eef_right_pos"]]
                                episode_json["left_eef_displacement"] = round(
                                    np.sum(np.linalg.norm(left_eef_pos[1:, :] - left_eef_pos[:-1, :], axis=-1)).item(),
                                    4,
                                )
                                episode_json["right_eef_displacement"] = round(
                                    np.sum(
                                        np.linalg.norm(right_eef_pos[1:, :] - right_eef_pos[:-1, :], axis=-1)
                                    ).item(),
                                    4,
                                )
                        episode_stats_json = {
                            "episode_index": episode_index,
                            "stats": episode_stats,
                        }
                        num_episodes += 1
                        num_frames += episode_info["num_samples"]
                    json.dump(episode_json, out_f)
                    out_f.write("\n")
                    json.dump(episode_stats_json, out_stats_f)
                    out_stats_f.write("\n")
    print(f"Generated episode JSON for {num_episodes} episodes and {num_frames} frames.")
    return num_episodes, num_frames


def generate_info_json(
    data_dir: str,
    fps: int = 30,
    total_episodes: int = 50,
    total_tasks: int = 50,
    total_frames: int = 50,
):
    info = {
        "codebase_version": "v2.1",
        "robot_type": "R1Pro",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 9,
        "chunks_size": 10000,
        "fps": fps,
        "splits": {
            "train": "0:" + str(total_episodes),
        },
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
        "metainfo_path": "meta/episodes/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "annotation_path": "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "features": {
            "observation.images.rgb.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.depth.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "action": {"dtype": "float32", "shape": [23], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.cam_rel_poses": {"dtype": "float32", "shape": [21], "names": None},
            "observation.state": {"dtype": "float32", "shape": [256], "names": None},
            "observation.task_info": {"dtype": "float32", "shape": [None], "names": None},
        },
    }

    with open(f"{data_dir}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"Generated info JSON for {len(info)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="/scr/behavior/2025-challenge-demos")
    args = parser.parse_args()

    # expand root
    data_dir = os.path.expanduser(args.data_dir)
    print("Generating task JSON...")
    num_tasks = generate_task_json(data_dir, credentials_path="~/Documents/credentials")
    print("Generating episode JSON...")
    num_episodes, num_frames = generate_episode_json(data_dir)
    print(num_tasks, num_episodes, num_frames)
    print("Generating info JSON...")
    generate_info_json(data_dir, fps=30, total_episodes=num_episodes, total_tasks=num_tasks, total_frames=num_frames)
