import torch
import torch.nn as nn
from omnigibson.learning.policies.policy_base import BasePolicy


class DummyPolicy(BasePolicy):
    """
    Dummy policy that always outputs zero delta action
    """

    def __init__(self, action_dim: int = 21, robot_type: str = "R1Pro", *args, **kwargs) -> None:
        super().__init__(robot_type=robot_type, *args, **kwargs)
        # setup a dummy parameter to avoid errors in the optimizer
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.action_dim = action_dim

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
        Directly return the observation without processing.
        """
        return obs
