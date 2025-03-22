import collections.abc
import json
import os

import torch as th
import yaml

# File I/O related


def parse_config(config):
    """
    Parse OmniGibson config file / object

    Args:
        config (dict or str): Either config dictionary or path to yaml config to load

    Returns:
        dict: Parsed config
    """
    if isinstance(config, collections.abc.Mapping):
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

    Args:
        config (str): Yaml cfg as a string to load

    Returns:
        dict: Parsed config
    """
    return yaml.safe_load(config)


def dump_config(config):
    """
    Converts YML config into a string

    Args:
        config (dict): Config to dump

    Returns:
        str: Config as a string
    """
    return yaml.dump(config)


def load_default_config():
    """
    Loads a default configuration to use for OmniGibson

    Returns:
        dict: Loaded default configuration file
    """
    from omnigibson import example_config_path

    return parse_config(f"{example_config_path}/default_cfg.yaml")


class TorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, th.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
