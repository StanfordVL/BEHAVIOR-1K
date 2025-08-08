import logging
import torch as th
import torch.nn.functional as F
from collections import deque
from omegaconf import OmegaConf, ListConfig
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.array_tensor_utils import any_concat
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    PROPRIO_QPOS_INDICES,
    JOINT_RANGE,
    ROBOT_CAMERA_NAMES,
    CAMERA_INTRINSICS,
    EEF_POSITION_RANGE,
)
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from omnigibson.learning.utils.obs_utils import process_fused_point_cloud
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
        use_task_info: bool = False,
        task_info_range: Optional[ListConfig] = None,
        pcd_range: List[float],
        robot_type: str = "R1Pro",
        # ====== other args for base class ======
        **kwargs,
    ) -> None:
        super().__init__(robot_type=robot_type, *args, **kwargs)
        if use_websocket:
            # create websocket client policy
            assert host is not None, "Host must be specified when using websocket client policy!"
            assert port is not None, "Port must be specified when using websocket client policy!"
            logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
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
        assert set(visual_obs_types).issubset(
            {"rgb", "depth_linear", "seg_instance_id", "pcd"}
        ), "visual_obs_types must be a subset of {'rgb', 'depth_linear', 'seg_instance_id', 'pcd'}!"
        self.visual_obs_types = visual_obs_types
        self._use_task_info = use_task_info
        self._task_info_range = (
            th.tensor(OmegaConf.to_container(task_info_range)) if task_info_range is not None else None
        )
        # store camera intrinsics
        self.camera_intrinsics = dict()
        for camera_id, camera_name in ROBOT_CAMERA_NAMES[self.robot_type].items():
            camera_intrinsics = th.from_numpy(CAMERA_INTRINSICS[self.robot_type][camera_id]) / 4.0
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
        self._robot_name = None
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
            self._action_traj_pred = self.policy.act({"obs": obs}).squeeze(0).detach().cpu()  # (T_A, A)
            self._action_idx = 0
        action = self._action_traj_pred[self._action_idx]
        self._action_idx += 1
        return action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self._obs_history = deque(maxlen=self.obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0

    def process_obs(self, obs: dict) -> dict:
        # Expand twice to get B and T_A dimensions
        processed_obs = {"qpos": dict(), "eef": dict()}
        if self._robot_name is None:
            for key in obs:
                if "proprio" in key:
                    self._robot_name = key.split("::")[0]
                    break
        proprio = obs[f"{self._robot_name}::proprio"].unsqueeze(0).unsqueeze(0)
        if "base_qvel" in PROPRIOCEPTION_INDICES[self.robot_type]:
            processed_obs["odom"] = {
                "base_velocity": self._post_processing_fn(
                    2
                    * (proprio[..., PROPRIOCEPTION_INDICES[self.robot_type]["base_qvel"]] - self.joint_range["base"][0])
                    / (self.joint_range["base"][1] - self.joint_range["base"][0])
                    - 1
                ),
            }
        for key in PROPRIO_QPOS_INDICES[self.robot_type]:
            if "gripper" in key:
                # rectify gripper actions to {-1, 1}
                processed_obs["qpos"][key] = th.mean(
                    proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]], dim=-1, keepdim=True
                )
                processed_obs["qpos"][key] = self._post_processing_fn(
                    th.where(
                        processed_obs["qpos"][key]
                        > (JOINT_RANGE[self.robot_type][key][0] + JOINT_RANGE[self.robot_type][key][1]) / 2,
                        1.0,
                        -1.0,
                    )
                )
            else:
                # normalize the qpos to [-1, 1]
                processed_obs["qpos"][key] = self._post_processing_fn(
                    2
                    * (proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]] - JOINT_RANGE[self.robot_type][key][0])
                    / (JOINT_RANGE[self.robot_type][key][1] - JOINT_RANGE[self.robot_type][key][0])
                    - 1.0
                )
        for key in EEF_POSITION_RANGE[self.robot_type]:
            processed_obs["eef"][f"{key}_pos"] = self._post_processing_fn(
                2
                * (
                    proprio[..., PROPRIOCEPTION_INDICES[self.robot_type][f"eef_{key}_pos"]]
                    - EEF_POSITION_RANGE[self.robot_type][key][0]
                )
                / (EEF_POSITION_RANGE[self.robot_type][key][1] - EEF_POSITION_RANGE[self.robot_type][key][0])
                - 1.0
            )
            # don't normalize the eef orientation
            processed_obs["eef"][f"{key}_quat"] = self._post_processing_fn(
                proprio[..., PROPRIOCEPTION_INDICES[self.robot_type][f"eef_{key}_quat"]]
            )
        if "pcd" in self.visual_obs_types:
            pcd_obs = dict()
        for camera_id, camera in ROBOT_CAMERA_NAMES[self.robot_type].items():
            if "rgb" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                rgb_obs = F.interpolate(
                    obs[f"{camera}::rgb"][..., :3].unsqueeze(0).movedim(-1, -3).to(th.float32),
                    self.obs_output_size[camera_id],
                    mode="nearest-exact",
                ).unsqueeze(0)
                if "pcd" in self.visual_obs_types:
                    # move rgb dim back
                    pcd_obs[f"{camera}::rgb"] = rgb_obs.movedim(-3, -1).to(self.device)
                else:
                    processed_obs[f"{camera}::rgb"] = self._post_processing_fn(rgb_obs)
            if "depth_linear" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                depth_obs = F.interpolate(
                    obs[f"{camera}::depth_linear"].unsqueeze(0).unsqueeze(0).to(th.float32),
                    self.obs_output_size[camera_id],
                    mode="nearest-exact",
                )
                if "pcd" in self.visual_obs_types:
                    # move depth_linear dim back
                    pcd_obs[f"{camera}::depth_linear"] = depth_obs.to(self.device)
                else:
                    processed_obs[f"{camera}::depth_linear"] = self._post_processing_fn(depth_obs)
            if "seg_instance_id" in self.visual_obs_types:
                processed_obs[f"{camera}::seg_instance_id"] = self._post_processing_fn(
                    F.interpolate(
                        obs[f"{camera}::seg_instance_id"].unsqueeze(0).unsqueeze(0).to(th.float32),
                        self.obs_output_size[camera_id],
                        mode="nearest-exact",
                    )
                )
        if "pcd" in self.visual_obs_types:
            pcd_obs["cam_rel_poses"] = (
                obs["robot_r1::cam_rel_poses"].unsqueeze(0).unsqueeze(0).to(th.float32).to(self.device)
            )
            processed_obs["pcd"] = self._post_processing_fn(
                process_fused_point_cloud(
                    obs=pcd_obs,
                    camera_intrinsics=self.camera_intrinsics,
                    pcd_range=self._pcd_range,
                    pcd_num_points=4096,
                    use_fps=True,
                )[0]
            )
        if self._use_task_info:
            processed_obs["task"] = dict()
            for key in obs:
                if key.startswith("task::"):
                    if self._task_info_range is not None:
                        # Normalize task info to [-1, 1]
                        processed_obs["task"] = (
                            self._post_processing_fn(
                                2
                                * (obs[key] - self._task_info_range[0])
                                / (self._task_info_range[1] - self._task_info_range[0])
                                - 1.0
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(th.float32)
                        )
                    else:
                        # If no range is provided, just use the raw data
                        processed_obs["task"] = self._post_processing_fn(
                            obs[key].unsqueeze(0).unsqueeze(0).to(th.float32)
                        )
                    break
        return processed_obs
