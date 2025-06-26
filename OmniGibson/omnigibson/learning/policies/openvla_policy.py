import torch
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES


class OpenVLA(BasePolicy):
    """
    OpenVLA-OFT policy from Kim et al. https://arxiv.org/pdf/2502.19645
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs) -> "BasePolicy":
        """
        Load the dummy policy (not applicable for this policy).
        For this policy, it simply returns an instance of DummyPolicy.
        """
        return cls(*args, **kwargs)

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        return torch.zeros(self.action_dim, dtype=torch.float32)

    def reset(self) -> None:
        pass

    def process_obs(self, obs: dict) -> dict:
        """
        Prepare observation for policy input.
        """
        # Get preprocessed images
        img = obs["external::external_sensor0::rgb"]
        left_wrist_img = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"]
        right_wrist_img = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"]

        # Prepare observations dict
        return {
            "full_image": img,
            "left_wrist_image": left_wrist_img,
            "right_wrist_image": right_wrist_img,
            "state": obs["robot_r1::proprio"][PROPRIOCEPTION_INDICES["joint_qpos"]],
        }
