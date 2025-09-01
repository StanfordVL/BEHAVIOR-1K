import os
import numpy as np
import packaging.version
import torch as th
from collections import defaultdict
from collections.abc import Callable
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES, ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.lerobot_utils import decode_video_frames, aggregate_stats
from omnigibson.utils.ui_utils import create_module_logger
from pathlib import Path
from torch.utils.data import Dataset
from typing import Iterable

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, CODEBASE_VERSION
from lerobot.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    STATS_PATH,
    TASKS_PATH,
    cast_stats_to_numpy,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
    backward_compatible_episodes_stats,
    load_json,
    load_jsonlines,
    load_info,
    is_valid_version,
)
from lerobot.datasets.video_utils import get_safe_default_codec


logger = create_module_logger("BehaviorLeRobotDataset")


class BehaviorLeRobotDataset(LeRobotDataset):
    """
    BehaviorLeRobotDataset is a customized dataset class for loading and managing LeRobot datasets,
    with additional filtering and loading options tailored for the BEHAVIOR-1K benchmark.
    This class extends LeRobotDataset and introduces the following customizations:
        - Task-based filtering: Load only episodes corresponding to specific tasks.
        - Modality and camera selection: Load only specified modalities (e.g., "rgb", "depth", "seg_instance_id")
          and cameras (e.g., "left_wrist", "right_wrist", "head").
        - Local-only mode: Optionally restrict dataset usage to local files, disabling downloads.
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
        batch_encoding_size: int = 1,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
        local_only: bool = False,
    ):
        """
        Custom args:
            episodes (List[int]): list of episodes to use PER TASK.
                NOTE: This is different from the actual episode indices in the dataset.
                Rather, this is meant to be used for train/val split, or loading a specific amount of partial data.
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
                Note: only one of episodes or tasks can be specified. If both are None, will load everything.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            local_only (bool): whether to only use local data (not download from HuggingFace).
        """
        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # ========== Customizations ==========
        if "depth" in modalities:
            assert self.video_backend == "pyav", (
                "Depth videos can only be decoded with the 'pyav' backend. "
                "Please set video_backend='pyav' when initializing the dataset."
            )
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        self.task_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.task_indices = [TASK_NAMES_TO_INDICES[task] for task in self.task_names]
        # Load metadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names,
            modalities=modalities,
            cameras=cameras,
        )
        # overwrite episode based on task
        all_episodes = load_jsonlines(self.root / EPISODES_PATH)
        # get the episodes grouped by task
        epi_by_task = defaultdict(list)
        for item in all_episodes:
            if item["tasks"][0] in self.task_names:
                epi_by_task[item["tasks"][0]].append(item["episode_index"])
        # sort and cherrypick episodes within each task
        for task, ep_indices in epi_by_task.items():
            epi_by_task[task] = sorted(ep_indices)
            if episodes is not None:
                epi_by_task[task] = [epi_by_task[task][i] for i in episodes if i < len(epi_by_task[task])]
        # now put episodes back together
        self.episodes = sorted([ep for eps in epi_by_task.values() for ep in eps])
        # record the positional index of each episode index within self.episodes
        self.episode_data_index_pos = {ep_idx: i for i, ep_idx in enumerate(self.episodes)}
        logger.info(f"Total episodes: {len(self.episodes)}")
        # ====================================

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            for fpath in self.get_episodes_file_paths():
                assert (self.root / fpath).is_file(), f"Missing file: {self.root / fpath}"
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as e:
            if local_only:
                raise e
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
        # append language and metainfo annotations
        fpaths += [str(self.meta.get_annotation_path(ep_idx)) for ep_idx in episodes]
        fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]

        return fpaths

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_idx = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": th.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            # post-process seg instance id:
            # if "seg_instance_id" in vid_key:
            #     palette = th.from_numpy(generate_yuv_palette(len(self.id_list))).float()
            #     N, H, W, C = frames.shape
            #     rgb_flat = frames.reshape(N, -1, C)  # (H*W, 3)
            #     # For each rgb pixel, find the index of the nearest color in the equidistant bins
            #     distances = th.cdist(rgb_flat, palette.unsqueeze(0).expand(N, -1, -1), p=2)
            #     ids = th.argmin(distances, dim=-1)  # (N, H*W)
            #     frames = self.id_list[ids].reshape(N, H, W)  # (N, H, W)
            item[vid_key] = frames.squeeze(0)

        return item


class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    BehaviorLerobotDatasetMetadata extends LeRobotDatasetMetadata with the following customizations:
        1. Restricts the set of allowed modalities to {"rgb", "depth", "seg_instance_id"}.
        2. Restricts the set of allowed camera names to those defined in ROBOT_CAMERA_NAMES["R1Pro"].
        3. Provides a filtered view of dataset features, including only those corresponding to the selected modalities and camera names.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
    ):
        # ========== Customizations ==========
        self.task_name_candidates = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            {"rgb", "depth", "seg_instance_id"}
        ), f"Modalities must be a subset of ['rgb', 'depth', 'seg_instance_id'], but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"
        # ===================================

        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/**", ignore_patterns="episodes/**")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(self.root)
        # filter based on self.task_name_candidates
        valid_task_indices = [idx for idx, name in self.task_names.items() if name in self.task_name_candidates]
        self.task_names = set([self.task_names[idx] for idx in valid_task_indices])
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_to_task_index = {k: v for k, v in self.tasks.items()}

        self.episodes = self.load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = self.load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def load_tasks(self, local_dir: Path) -> tuple[dict, dict]:
        tasks = load_jsonlines(local_dir / TASKS_PATH)
        task_names = {item["task_index"]: item["task_name"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        task_to_task_index = {task: task_index for task_index, task in tasks.items()}
        return tasks, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path) -> dict:
        episodes = load_jsonlines(local_dir / EPISODES_PATH)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["tasks"][0] in self.task_names
        }

    def load_stats(self, local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path) -> dict:
        episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def get_annotation_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Formattable string for the annotation files."""
        return self.info["annotation_path"]

    @property
    def metainfo_path(self) -> str | None:
        """Formattable string for the metainfo files."""
        return self.info["metainfo_path"]

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
