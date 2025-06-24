import torch
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from typing import Any
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import MultiStepLR


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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            params=[self.dummy_param],
            lr=5e-4,
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=[100000],  # Number of steps after which LR will change
            gamma=0.1,  # Multiplicative factor of learning rate decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update the scheduler every step
                "frequency": 1,  # Frequency of updating the scheduler
            },
        }

    def policy_training_step(self, batch, batch_idx) -> Any:
        # TODO
        pass

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        # TODO
        pass

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        """
        Prepare observation for policy input.
        """
        # Get preprocessed images
        img = data_batch["obs"]["external::external_sensor0::rgb"]
        left_wrist_img = data_batch["obs"]["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"]
        right_wrist_img = data_batch["obs"]["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"]

        # Prepare observations dict
        return {
            "full_image": img,
            "left_wrist_image": left_wrist_img,
            "right_wrist_image": right_wrist_img,
            "state": data_batch["obs"]["robot_r1::proprio"][PROPRIOCEPTION_INDICES["joint_qpos"]],
        }
