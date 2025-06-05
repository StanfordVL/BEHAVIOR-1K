import torch
import torch.nn as nn
from omnigibson.learning.policies.policy_base import BasePolicy
from typing import Any
from pytorch_lightning.utilities.types import OptimizerLRScheduler


class DummyPolicy(BasePolicy):
    """
    Dummy policy that always outputs zero delta action
    """

    def __init__(self, action_dim: int = 23, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # setup a dummy parameter to avoid errors in the optimizer
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.action_dim = action_dim

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        return torch.zeros(self.action_dim, dtype=torch.float32)

    def reset(self) -> None:
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            params=[self.dummy_param],
            lr=1e-9,  # very small learning rate to avoid updates
        )

    def policy_training_step(self, batch, batch_idx) -> Any:
        pass

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        pass

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        """
        Directly return the observation without processing.
        """
        return data_batch
