import cv2
import logging
import torch as th
import torch.nn.functional as F
from collections import deque
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.array_tensor_utils import any_concat
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, PROPRIO_QPOS_INDICES, JOINT_RANGE, ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from typing import Dict, Optional, List


class VisionActionILPolicy(BasePolicy):
    """
    Simple vision-action imitation learning policy
    """

    def __init__(
        self,
        *args,
        # ====== policy model ======
        use_websocket: bool = True,
        host: Optional[str] = None,
        port: Optional[int] = None,
        deployed_action_steps: int,
        obs_window_size: int = 1,
        obs_output_size: Dict[str, List[int]],
        # ====== other args for base class ======
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if use_websocket:
            self.policy = WebsocketClientPolicy(host=host, port=port)
            logging.info(f"Server metadata: {self.policy.get_server_metadata()}")
        else:
            self.policy = None
            logging.info(f"Skipped creating websocket client policy")
        # post-processing func
        if use_websocket:
            # convert to numpy
            self._post_processing_fn = lambda x: x.cpu().numpy()
        else:
            # move all tensor to self.device
            self._post_processing_fn = lambda x: x.to(self.device)
        self.deployed_action_steps = deployed_action_steps
        self.obs_window_size = obs_window_size
        self.obs_output_size = {k: tuple(v) for k, v in obs_output_size.items()}
        self._obs_history = deque(maxlen=obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0
        self.robot_type = "R1Pro"
        self.joint_range = JOINT_RANGE[self.robot_type]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        obs = self.process_obs(obs=obs)
        if len(self._obs_history) == 0:
            for _ in range(self.obs_window_size):
                self._obs_history.append(obs)
        else:
            self._obs_history.append(obs)
        obs = any_concat(self._obs_history, dim=1)

        need_inference = self._action_idx % self.deployed_action_steps == 0
        if need_inference:
            self._action_traj_pred = self.policy.act({"obs": obs}).squeeze(0).detach().cpu() # (T_A, A)
            self._action_idx = 0
        action = self._action_traj_pred[self._action_idx]
        self._action_idx += 1
        return action

    def reset(self) -> None:
        self.policy.reset()
        self._obs_history = deque(maxlen=self.obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0

    def process_obs(self, obs: dict) -> dict:
        # Expand twice to get B and T_A dimensions
        processed_obs = {}
        proprio = obs["robot_r1::proprio"].unsqueeze(0).unsqueeze(0)
        processed_obs.update({
            "qpos": {
                key: self._post_processing_fn(
                    (proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]] - self.joint_range[key][0]) / 
                    (self.joint_range[key][1] - self.joint_range[key][0])
                )
                for key in PROPRIO_QPOS_INDICES[self.robot_type]
            },
            "odom": {
                "base_velocity": self._post_processing_fn(
                    (proprio[..., PROPRIOCEPTION_INDICES[self.robot_type]["base_qvel"]] - self.joint_range["base"][0]) / 
                    (self.joint_range["base"][1] - self.joint_range["base"][0])
                ),
            },
        })
        for camera_id, camera in ROBOT_CAMERA_NAMES.items():
            processed_obs[f"{camera}::rgb"] = self._post_processing_fn(
                F.interpolate(
                    obs[f"{camera}::rgb"][..., :3].unsqueeze(0).movedim(-1, -3).to(th.float32), 
                    self.obs_output_size[camera_id], 
                    mode="area"
                ).unsqueeze(0)
            )
            processed_obs[f"{camera}::depth_linear"] = self._post_processing_fn(
                F.interpolate(
                    obs[f"{camera}::depth_linear"].unsqueeze(0).unsqueeze(0).to(th.float32), 
                    self.obs_output_size[camera_id], 
                    mode="area"
                )
            )
            processed_obs["cam_rel_poses"] = self._post_processing_fn(
                obs["robot_r1::cam_rel_poses"].unsqueeze(0).unsqueeze(0).to(th.float32)
            )
        return processed_obs
