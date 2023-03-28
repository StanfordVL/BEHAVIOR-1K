import gym
from abc import ABCMeta, abstractmethod
import numpy as np

from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


def recursively_generate_flat_dict(dic, prefix=None):
    """
    Helper function to recursively iterate through dictionary / gym.spaces.Dict @dic and flatten any nested elements,
    such that the result is a flat dictionary mapping keys to values

    Args:
        dic (dict or gym.spaces.Dict): (Potentially nested) dictionary to convert into a flattened dictionary
        prefix (None or str): Prefix to append to the beginning of all strings in the flattened dictionary. None results
            in no prefix being applied

    Returns:
        dict: Flattened version of @dic
    """
    out = dict()
    prefix = "" if prefix is None else f"{prefix}::"
    for k, v in dic.items():
        if isinstance(v, gym.spaces.Dict) or isinstance(v, dict):
            out.update(recursively_generate_flat_dict(dic=v, prefix=f"{prefix}{k}"))
        elif isinstance(v, gym.spaces.Tuple) or isinstance(v, tuple):
            for i, vv in enumerate(v):
                # Assume no dicts are nested within tuples
                out[f"{prefix}{k}::{i}"] = vv
        else:
            # Add to out dict
            out[f"{prefix}{k}"] = v

    return out


class GymObservable(metaclass=ABCMeta):
    """
    Simple class interface for observable objects. These objects should implement a way to grab observations,
    (get_obs()), and should define an observation space that is created when load_observation_space() is called

    Args:
        kwargs: dict, does nothing, used to sink any extraneous arguments during initialization
    """
    def __init__(self, *args, **kwargs):
        # Initialize variables that we will fill in later
        self.observation_space = None

        # Call any super methods
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_obs(self, **kwargs):
        """
        Get observations for the object. Note that the shape / nested structure should match that
        of @self.observation_space!

        Args:
            kwargs (dict): Any keyword args necessary for grabbing observations

        Returns:
            dict: Keyword-mapped observations mapping observation names to nested observations
        """
        raise NotImplementedError()

    @staticmethod
    def _build_obs_box_space(shape, low, high, dtype=np.float32):
        """
        Helper function that builds individual observation box spaces.

        Args:
            shape (n-array): Shape of the space
            low (float): Lower bound of the space
            high (float): Upper bound of the space

        Returns:
            gym.spaces.Box: Generated gym box observation space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

    @abstractmethod
    def _load_observation_space(self):
        """
        Create the observation space for this object. Should be implemented by subclass

        Returns:
            dict: Keyword-mapped observation space for this object mapping observation name to observation space
        """
        raise NotImplementedError()

    def load_observation_space(self):
        """
        Load the observation space internally, and also return this value

        Returns:
            gym.spaces.Dict: Loaded observation space for this object
        """
        # Load the observation space and convert it into a gym-compatible dictionary
        self.observation_space = gym.spaces.Dict(self._load_observation_space())
        log.debug(f"Loaded obs space dictionary for: {self.__class__.__name__}")

        return self.observation_space
