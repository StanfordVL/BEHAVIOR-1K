import h5py
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset
from omnigibson.learning.utils.array_tensor_utils import any_concat, any_ones_like, any_slice, any_stack, get_batch_size
from typing import Optional

# Action indices
ACTION_QPOS_INDICES = {
    "mobile_base": np.s_[0:3],
    "torso": np.s_[3:7],
    "left_arm": np.s_[7:14],
    "left_gripper": np.s_[14:15],
    "right_arm": np.s_[15:22],
    "right_gripper": np.s_[22:23],
}
# Proprioception indices
PROPRIO_QPOS_INDICES = {
    "torso": np.s_[6:10],
    "left_arm": np.s_[10:24:2],
    "right_arm": np.s_[11:24:2],
    "left_gripper": np.s_[24:26],
    "right_gripper": np.s_[26:28],
}
PROPRIO_BASED_VEL_INDICES = np.s_[-3:]


class BehaviorDataset(Dataset):
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    gripper_strike_low = np.array([0])
    gripper_strike_high = np.array([100])

    def __init__(
        self,
        data_path: str,
        obs_window_size: int,
        action_prediction_horizon: int,
        max_num_demos: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert os.path.exists(data_path)
        self._data_path = data_path
        self._obs_window_size = obs_window_size
        self._action_prediction_horizon = action_prediction_horizon
        self._max_num_demos = max_num_demos
        self._random_state = np.random.RandomState(seed)

        # open hdf files
        self._demos = []
        for file_name in tqdm(sorted(os.listdir(data_path))):
            if file_name.endswith(".hdf5"):
                f = h5py.File(os.path.join(data_path, file_name), "r", swmr=True, libver="latest")
                for demo_name in f["data"]:
                    self._demos.append(f["data"][demo_name])
        self._random_state.shuffle(self._demos)
        # limit number of demos
        if self._max_num_demos is not None:
            self._demos = self._demos[: self._max_num_demos]

        self._len = 0
        # compute length
        for demo in self._demos:
            L = get_batch_size(demo["obs"], strict=True)
            N_chunks = L - self._obs_window_size + 1
            self._len += N_chunks
        # give a random starting demo
        self._demo_ptr = np.random.randint(len(self._demos))
        self._demo_chunk_ptr = 0
        self._data_chunk, self._chunk_idxs = self._chunk_demo(self._demos[self._demo_ptr])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # decide if we need to move to the next demo
        if self._demo_chunk_ptr >= len(self._data_chunk):
            self._demo_chunk_ptr = 0
            self._demo_ptr += 1
            if self._demo_ptr >= len(self._demos):
                self._demo_ptr = 0
            self._data_chunk, self._chunk_idxs = self._chunk_demo(self._demos[self._demo_ptr])
        data = self._data_chunk[self._demo_chunk_ptr]
        self._demo_chunk_ptr += 1

        return data

    def _chunk_demo(self, demo):
        # make actions from (T, A) to (T, L_pred_horizon, A)
        # need to construct a mask
        action_chunks = []
        action_chunk_masks = []
        action = demo["action"][:]
        action_structure = deepcopy(any_slice(action, np.s_[0:1]))  # (1, A)
        for t in range(get_batch_size(action, strict=True)):
            action_chunk = any_slice(action, np.s_[t : t + self._action_prediction_horizon])
            action_chunk_size = get_batch_size(action_chunk, strict=True)
            pad_size = self._action_prediction_horizon - action_chunk_size
            mask = any_concat(
                [
                    np.ones((action_chunk_size,), dtype=bool),
                    np.zeros((pad_size,), dtype=bool),
                ],
                dim=0,
            )  # (L_pred_horizon,)
            action_chunk = any_concat(
                [
                    action_chunk,
                ]
                + [any_ones_like(action_structure)] * pad_size,
                dim=0,
            )  # (L_pred_horizon, A)
            action_chunks.append(action_chunk)
            action_chunk_masks.append(mask)
        action_chunks = any_stack(action_chunks, dim=0)  # (T, L_pred_horizon, A)
        action_chunk_masks = np.stack(action_chunk_masks, axis=0)  # (T, L_pred_horizon)

        data = dict()

        data["action_chunks"] = action_chunks
        data["action_chunk_masks"] = action_chunk_masks
        # store observations
        data["obs"] = self._get_obs_from_demo(demo)

        # Now, chunk data
        data_chunks = []
        chunk_idxs = []
        L = get_batch_size(data, strict=True)
        assert L >= self._obs_window_size >= 1
        N_chunks = L - self._obs_window_size + 1
        # split obs into chunks
        for chunk_idx in range(N_chunks):
            s = np.s_[chunk_idx : chunk_idx + self._obs_window_size]
            chunk_idxs.append(chunk_idx)
            data_chunks.append(any_slice(data, s))
        return data_chunks, chunk_idxs

    def _get_obs_from_demo(self, demo: h5py.Group) -> dict:
        """
        Extracts observations from the demo.
        Args:
            demo: h5py group containing the demo data.
        Returns:
            A dictionary of observations.
        """
        obs = dict()
        for key in demo["obs"]:
            # remove the last observation, since we don't have action for it
            obs[key] = demo["obs"][key][:-1]
        return obs
