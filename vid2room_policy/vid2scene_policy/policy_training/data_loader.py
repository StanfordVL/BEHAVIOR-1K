import sys
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vid2scene_policy.data_collection.lerobot_datasets.datasets.lerobot_dataset import LeRobotDataset
from tqdm.auto import tqdm

class PickDataset(Dataset):
    OBSERVATION_KEYS = (
        "observation.images.wrist",
        "observation.images.head",
        "observation.images.wrist_seg_depth",
        "observation.images.head_seg_depth",
    )

    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        video_backend: str = "pyav",
        return_debug_indices: bool = False,
        frame_cache_size: int = 256,
    ):
        self.dataset_path = dataset_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.return_debug_indices = return_debug_indices
        self.frame_cache_size = frame_cache_size

        absolute_dataset_path = Path(dataset_path).resolve()
        self.dataset = LeRobotDataset(
            repo_id=absolute_dataset_path.name,
            root=absolute_dataset_path,
            force_cache_sync=False,
            revision="v3.0",
            video_backend=video_backend,
        )

        self.total_num_episodes = self.dataset.num_episodes
        self.total_num_frames = self.dataset.num_frames
        print(f"Number of episodes selected: {self.total_num_episodes}")
        print(f"Number of frames selected: {self.total_num_frames}")

        first_sample = self.dataset[0]
        self.action_pad = torch.zeros_like(first_sample["action"])
        if self.action_pad.numel() > 0:
            self.action_pad[-1] = -1.0

        self._frame_cache = OrderedDict()

        self.sample_index = []
        for episode_index in range(self.total_num_episodes):
            from_idx = int(self.dataset.meta.episodes["dataset_from_index"][episode_index])
            to_idx = int(self.dataset.meta.episodes["dataset_to_index"][episode_index])
            episode_length = to_idx - from_idx
            for i in range(episode_length):
                self.sample_index.append((from_idx, to_idx, i))

        print("Preloading all frames into RAM...")
        self._frame_cache = {}
        max_frame = max(to_idx for _, to_idx, _ in self.sample_index)
        min_frame = min(from_idx for from_idx, _, _ in self.sample_index)
        for frame_idx in tqdm(range(min_frame, max_frame)):
            self._frame_cache[frame_idx] = self.dataset[frame_idx]
        print(f"Preloaded {len(self._frame_cache)} frames.")

    def __len__(self):
        return len(self.sample_index)

    def _get_frame(self, frame_idx: int):
        return self._frame_cache.get(frame_idx)

    def __getitem__(self, idx: int):
        from_idx, to_idx, i = self.sample_index[idx]

        obs_start = from_idx + i - (self.obs_horizon - 1)
        obs_indices = [max(from_idx, obs_start + k) for k in range(self.obs_horizon)]

        obs_samples = [self._get_frame(obs_idx) for obs_idx in obs_indices]
        action_indices = [from_idx + i + j for j in range(self.pred_horizon)]
        valid_action_mask = [action_idx < to_idx for action_idx in action_indices]
        action_samples = []

        for action_idx, is_valid in zip(action_indices, valid_action_mask):
            if is_valid:
                action_samples.append(self._get_frame(action_idx)["action"])
            else:
                action_samples.append(self.action_pad)

        sample = {
            "action": torch.stack(action_samples, dim=0),
            "action_mask": torch.tensor(valid_action_mask, dtype=torch.float32),
        }
        for key in self.OBSERVATION_KEYS:
            sample[key] = torch.stack([frame[key] for frame in obs_samples], dim=0)

        # Per camera: concatenate RGB and seg_depth channels.
        sample["wrist"] = torch.cat(
            [sample["observation.images.wrist"], sample["observation.images.wrist_seg_depth"]],
            dim=1,
        )
        sample["head"] = torch.cat(
            [sample["observation.images.head"], sample["observation.images.head_seg_depth"]],
            dim=1,
        )

        if self.return_debug_indices:
            sample["obs_indices"] = torch.tensor(obs_indices, dtype=torch.long)
            sample["action_indices"] = torch.tensor(
                [action_idx if action_idx < to_idx else -1 for action_idx in action_indices],
                dtype=torch.long,
            )

        return sample

    def compute_action_stats(self, max_samples: int = 50_000):
        self.dataset._ensure_hf_dataset_loaded()
        hf = self.dataset.hf_dataset
        action_col = hf["action"]
        step = max(1, len(self.sample_index) // max_samples)
        all_actions = []
        for idx in range(0, len(self.sample_index), step):
            from_idx, to_idx, i = self.sample_index[idx]
            for j in range(self.pred_horizon):
                if from_idx + i + j < to_idx:
                    a = action_col[from_idx + i + j]
                    if isinstance(a, torch.Tensor):
                        pass
                    elif hasattr(a, "numpy"):
                        a = torch.from_numpy(a)
                    else:
                        a = torch.tensor(a)
                    a = a.flatten()[..., :-1]  # drop gripper, ensure 1d
                    all_actions.append(a)
        all_actions = torch.stack(all_actions, dim=0)
        mean = all_actions.mean(0)
        std = all_actions.std(0).clamp(min=1e-5)
        return mean, std
