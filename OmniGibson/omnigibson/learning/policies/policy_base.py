import logging
import torch as th
from abc import ABC, abstractmethod


logger = logging.getLogger("BasePolicy")


class BasePolicy(ABC):
    """
    Base class for policies that is used for training and rollout
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs) -> "BasePolicy":
        """
        Load the policy (e.g. from a checkpoint given a file path).
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Forward pass of the policy.
        This is used for inference and should return the action.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the policy
        """
        raise NotImplementedError

    @abstractmethod
    def process_obs(self, obs: dict) -> dict:
        """
        Process the observation dictionary to match the expected input format for the model.
        Args:
            obs (dict): Observation data from the environment.
        Returns:
            dict: Processed obs that can be used by the policy.
        """
        raise NotImplementedError
