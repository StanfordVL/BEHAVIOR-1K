import numpy as np
import torch
from collections import deque
from hydra.utils import instantiate
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, PROPRIO_QPOS_INDICES
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.array_tensor_utils import any_concat, any_slice
from OmniGibson.omnigibson.learning.utils.obs_utils import process_fused_point_cloud
from omegaconf import DictConfig


class WBVIMA(BasePolicy):
    """
    WB-VIMA policy from Jiang et al. https://arxiv.org/abs/2503.05652
    """

    def __init__(
        self,
        # ====== policy model ======
        policy: DictConfig,
        action_prediction_horizon: int,
        obs_window_size: int = 1,
        # ====== other args for base class ======
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.policy = instantiate(policy)
        self.action_prediction_horizon = action_prediction_horizon
        self.obs_window_size = obs_window_size
        self._obs_history = deque(maxlen=obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0

    @classmethod
    def load(cls, *args, **kwargs) -> "BasePolicy":
        """
        Load the policy (e.g. from a checkpoint given a file path).
        """
        return super().load_from_checkpoint(checkpoint_path=kwargs["ckpt_path"], strict=kwargs.get("strict", True))

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        obs = self.process_data(data_batch=obs, extract_action=False)
        if len(self._obs_history) == 0:
            for _ in range(self.obs_window_size):
                self._obs_history.append(obs)
        else:
            self._obs_history.append(obs)
        obs = any_concat(self._obs_history, dim=1)

        need_inference = self._action_idx % self.action_prediction_horizon == 0
        if need_inference:
            self._action_traj_pred = self.policy.act(obs)  # dict of (B = 1, T_A, ...)
            self._action_traj_pred = {
                k: v[0].detach().cpu() for k, v in self._action_traj_pred.items()
            }  # dict of (T_A, ...)
            self._action_idx = 0
        action = any_slice(self._action_traj_pred, np.s_[self._action_idx])
        self._action_idx += 1
        return torch.cat(list(action.values()))

    def reset(self) -> None:
        pass

    def process_obs(self, obs: dict) -> dict:
        # process observation data
        proprio = obs["robot_r1::proprio"]
        if proprio.ndim == 1:
            # if proprio is 1D, we need to expand it to 3D
            proprio = proprio[None, None, :].to(self.device)
        if "robot_r1::fused_pcd" in obs:
            fused_pcd = obs["robot_r1::fused_pcd"]
        else:
            fused_pcd = process_fused_point_cloud(obs)
        # if fused_pcd is 1D, we need to expand it to 3D
        if fused_pcd.ndim == 2:
            fused_pcd = torch.from_numpy(fused_pcd[None, None, :]).to(self.device)
        processed_obs = {
            "pointcloud": {
                "rgb": fused_pcd[..., :3],
                "xyz": fused_pcd[..., 3:],
            },
            "qpos": {key: proprio[..., PROPRIO_QPOS_INDICES["R1Pro"][key]] for key in PROPRIO_QPOS_INDICES["R1Pro"]},
            "odom": {"base_velocity": proprio[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]},
        }
        return processed_obs
