import logging
import torch as th
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from typing import Optional

from omnigibson.learning.utils.obs_utils import process_fused_point_cloud, color_pcd_vis
from omnigibson.learning.utils.eval_utils import CAMERA_INTRINSICS, ROBOT_CAMERA_NAMES

__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
]


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(self, *args, action_dim: Optional[int] = None, **kwargs) -> None:
        self.policy = None  # To be set later
        self.action_dim = action_dim

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        # ===== JUST FOR TESTING, REMOVE BEFORE MERGING PR =======
        cam_intrinsics = {}
        for camera_id, camera_names in ROBOT_CAMERA_NAMES["R1Pro"].items():
            cam_intrinsics[camera_names] = th.from_numpy(CAMERA_INTRINSICS["R1Pro"][camera_id])
        obs["cam_rel_poses"] = obs["robot_r1::cam_rel_poses"]
        pcd = process_fused_point_cloud(
            obs=obs,
            camera_intrinsics=cam_intrinsics,
            pcd_range=(
                -10,
                10,
                -10,
                10,
                0,
                2,
            ),  # TODO: this range is now in world frame if we are using native point cloud
            pcd_num_points=100000,
            use_fps=False,
        ).cpu()
        color_pcd_vis(pcd)
        # ============================================================
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            # ===== JUST FOR TESTING, REVERT BEFORE MERGING PR =======
            return th.randn(self.action_dim, dtype=th.float32) * 0.5
            # return th.zeros(self.action_dim, dtype=th.float32)
            # ============================================================

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.policy = WebsocketClientPolicy(host=host, port=port)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        # convert observation to numpy
        obs = torch_to_numpy(obs)
        return self.policy.act(obs).detach().cpu()

    def reset(self) -> None:
        self.policy.reset()
