import collections
import json
import os
import random

import numpy as np

import scipy
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transforms3d import quaternions

from omnigibson import example_config_path

# File I/O related


def parse_config(config):

    """
    Parse OmniGibson config file / object
    """
    if isinstance(config, collections.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data


def parse_str_config(config):
    """
    Parse string config
    """
    return yaml.safe_load(config)


def dump_config(config):
    """
    Converts YML config into a string
    """
    return yaml.dump(config)


def load_default_config():
    """
    Loads a default configuration to use for OmniGibson

    Returns:
        dict: Loaded default configuration file
    """
    return parse_config(f"{example_config_path}/default_cfg.yaml")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)