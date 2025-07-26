import logging
import torch as th
import torch.nn.functional as F
from collections import deque
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.array_tensor_utils import any_concat
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES, 
    PROPRIO_QPOS_INDICES, 
    JOINT_RANGE, 
    ROBOT_CAMERA_NAMES, 
    CAMERA_INTRINSICS
)
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from omnigibson.learning.utils.obs_utils import process_fused_point_cloud, color_pcd_vis
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
        visual_obs_types: List[str],
        pcd_range: List[float],
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
        assert set(visual_obs_types).issubset({"rgb", "depth_linear", "seg_instance_id", "pcd"}), \
            "visual_obs_types must be a subset of {'rgb', 'depth_linear', 'seg_instance_id', 'pcd'}!"
        self.visual_obs_types = visual_obs_types
        # store camera intrinsics
        self.camera_intrinsics = dict()
        for camera_id, camera_name in ROBOT_CAMERA_NAMES.items():
            camera_intrinsics = th.from_numpy(CAMERA_INTRINSICS[camera_id]) / 4.0
            camera_intrinsics[-1, -1] = 1.0  # make it homogeneous
            self.camera_intrinsics[camera_name] = camera_intrinsics
        self._pcd_range = tuple(pcd_range)
        # action steps for deployed policy
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
                    2 * (proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]] - self.joint_range[key][0]) / 
                    (self.joint_range[key][1] - self.joint_range[key][0]) - 1
                )
                for key in PROPRIO_QPOS_INDICES[self.robot_type]
            },
            "odom": {
                "base_velocity": self._post_processing_fn(
                    2 * (proprio[..., PROPRIOCEPTION_INDICES[self.robot_type]["base_qvel"]] - self.joint_range["base"][0]) / 
                    (self.joint_range["base"][1] - self.joint_range["base"][0]) - 1
                ),
            },
        })
        for camera_id, camera in ROBOT_CAMERA_NAMES.items():
            if "rgb" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                processed_obs[f"{camera}::rgb"] = self._post_processing_fn(
                    F.interpolate(
                        obs[f"{camera}::rgb"][..., :3].unsqueeze(0).movedim(-1, -3).to(th.float32), 
                        self.obs_output_size[camera_id], 
                        mode="nearest-exact"
                    ).unsqueeze(0)
                )
                if "pcd" in self.visual_obs_types:
                    # move rgb dim back
                    processed_obs[f"{camera}::rgb"] = processed_obs[f"{camera}::rgb"].movedim(-3, -1)
            if "depth_linear" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                processed_obs[f"{camera}::depth_linear"] = self._post_processing_fn(
                    F.interpolate(
                        obs[f"{camera}::depth_linear"].unsqueeze(0).unsqueeze(0).to(th.float32), 
                        self.obs_output_size[camera_id], 
                        mode="nearest-exact"
                    )
            )
            if "seg_instance_id" in self.visual_obs_types:
                processed_obs[f"{camera}::seg_instance_id"] = self._post_processing_fn(
                    F.interpolate(
                        obs[f"{camera}::seg_instance_id"].unsqueeze(0).unsqueeze(0).to(th.float32), 
                        self.obs_output_size[camera_id], 
                        mode="nearest-exact"
                    )
                )
        if "pcd" in self.visual_obs_types:
            processed_obs["cam_rel_poses"] = self._post_processing_fn(
                obs["robot_r1::cam_rel_poses"].unsqueeze(0).unsqueeze(0).to(th.float32)
            )
            fused_pcd = process_fused_point_cloud(
                obs=processed_obs,
                camera_intrinsics=self.camera_intrinsics,
                pcd_range=self._pcd_range,
                pcd_num_points=4096,
                use_fps=True,

            )[0]
            processed_obs["pcd"] = fused_pcd
            for camera_id, camera in ROBOT_CAMERA_NAMES.items():
                if "rgb" not in self.visual_obs_types:
                    # pop rgb if not needed
                    processed_obs.pop(f"{camera}::rgb", None)
                else:
                    # move rgb dim forward
                    processed_obs[f"{camera}::rgb"] = processed_obs[f"{camera}::rgb"].movedim(-1, -3)
                if "depth_linear" not in self.visual_obs_types:
                    # pop depth_linear if not needed
                    processed_obs.pop(f"{camera}::depth_linear", None)
            
        return processed_obs
