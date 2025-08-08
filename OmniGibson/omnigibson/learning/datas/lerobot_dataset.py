import argparse
import json
import os
import packaging.version
import numpy as np
import torch as th
from collections.abc import Callable
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES, ROBOT_CAMERA_NAMES
from pathlib import Path
from torch.utils.data import Dataset
from typing import Iterable, Tuple

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, CODEBASE_VERSION
from lerobot.datasets.utils import (
    EPISODES_PATH,
    check_delta_timestamps,
    check_timestamps_sync,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
    load_jsonlines,
)
from lerobot.datasets.video_utils import get_safe_default_codec


class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    LerobotDatasetMetadata with the following customizations:
        1. Custom task names mapping to indices.
    """

    def __init__(self, *args, modalities: Iterable[str] = None, cameras: Iterable[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            {"rgb", "depth", "seg_instance_id"}
        ), f"Modalities must be a subset of ['rgb', 'depth', 'seg_instance_id'], but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        features = dict()
        # pop not required features
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features


class BehaviorLeRobotDataset(LeRobotDataset):
    """
    LeRobotDataset with the following customizations:
        1. Custom chunking logic based on
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
    ):
        """
        Custom args:
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            video_backend: video backend to use for decoding videos.
        """
        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # ========== Customizations ==========
        # Load metadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            modalities=modalities,
            cameras=cameras,
        )
        self.tasks_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        # overwrite episode based on task
        if episodes is None:
            episodes = load_jsonlines(self.root / EPISODES_PATH)
            self.episodes = sorted([item["episode_index"] for item in episodes if item["tasks"][0] in self.tasks_names])
        # ====================================

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths


def generate_task_json(data_dir: str) -> int:
    num_tasks = len(TASK_NAMES_TO_INDICES)

    with open(f"{data_dir}/meta/tasks.jsonl", "w") as f:
        for task_name, task_index in TASK_NAMES_TO_INDICES.items():
            json.dump({"task_index": task_index, "task": task_name}, f)
            f.write("\n")
    return num_tasks


def generate_episode_json(data_dir: str) -> Tuple[int, int]:
    assert os.path.exists(f"{data_dir}/meta/tasks.jsonl"), "Task JSON does not exist!"
    assert os.path.exists(f"{data_dir}/meta/episodes"), "Episode metadata directory does not exist!"
    with open(f"{data_dir}/meta/tasks.jsonl", "r") as f:
        task_json = [json.loads(line) for line in f]
    num_frames = 0
    num_episodes = 0
    with open(f"{data_dir}/meta/episodes.jsonl", "w") as out_f:
        with open(f"{data_dir}/meta/episodes_stats.jsonl", "w") as out_stats_f:
            for task_info in task_json:
                task_index = task_info["task_index"]
                task_name = task_info["task"]
                for episode_name in sorted(os.listdir(f"{data_dir}/meta/episodes/task-{task_index:04d}")):
                    with open(f"{data_dir}/meta/episodes/task-{task_index:04d}/{episode_name}", "r") as f:
                        episode_info = json.load(f)
                        episode_index = int(episode_name.split(".")[0].split("_")[-1])
                        episode_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "length": episode_info["num_samples"],
                        }
                        episode_stats_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "stats": {
                                "obs": {
                                    "min": np.array([episode_info["num_samples"]]).tolist(),
                                    "max": np.array([episode_info["num_samples"]]).tolist(),
                                    "mean": np.array([episode_info["num_samples"]]).tolist(),
                                    "std": np.array([episode_info["num_samples"]]).tolist(),
                                    "count": np.array([episode_info["num_samples"]]).tolist(),
                                }
                            },
                        }
                        num_episodes += 1
                        num_frames += episode_info["num_samples"]
                    json.dump(episode_json, out_f)
                    out_f.write("\n")
                    json.dump(episode_stats_json, out_stats_f)
                    out_stats_f.write("\n")
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
        "total_videos": total_episodes * 6,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": "0:" + str(total_episodes),
        },
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:07d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:07d}.mp4",
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
                "names": ["height", "width", "seg_instance_id"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "seg_instance_id"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "seg_instance_id"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "action": {"dtype": "float32", "shape": [23], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.cam_rel_poses": {"dtype": "float32", "shape": [21], "names": None},
            "observation.state": {"dtype": "float32", "shape": [None], "names": None},
        },
    }

    with open(f"{data_dir}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="/home/svl/Documents/Files/behavior")
    args = parser.parse_args()

    num_tasks = generate_task_json(args.data_dir)
    num_episodes, num_frames = generate_episode_json(args.data_dir)
    print(num_tasks, num_episodes, num_frames)

    generate_info_json(
        args.data_dir, fps=30, total_episodes=num_episodes, total_tasks=num_tasks, total_frames=num_frames
    )
