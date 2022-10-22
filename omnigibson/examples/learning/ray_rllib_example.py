"""
Example showing how to wrap the OmniGibson class using ray for rllib.
Multiple environments are only supported on Linux. If issues arise, please ensure torch/numpy
are installed *without* MKL support.

This example requires ray to be installed with rllib support, and pytorch to be installed:
    `pip install torch "ray[rllib]"`

Note: rllib only supports a single observation modality:
"""
import argparse
import functools
import logging
import operator
import os
from sys import platform

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env

import omnigibson
from omnigibson.envs.omnigibson_env import OmniGibsonEnv

# ray.init(local_mode=True)
ray.init(ignore_reinit_error=True)


class ConvNet1D(nn.Module):
    def __init__(self):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.conv2 = nn.Conv1d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 5)
        # If the size is a square, you can specify with a single number
        x = F.max_pool1d(F.relu(self.conv2(x)), 5)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet2D(nn.Module):
    def __init__(self, channels=1, input_dim=[128, 128, 3]):
        super(ConvNet2D, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        num_features = functools.reduce(
            operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim).permute(0, 3, 1, 2)).shape)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FC(nn.Module):
    def __init__(self, input_size, output_size=10):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OmniGibsonPPOModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        rgb_input_shape = obs_space.original_space["rgb"].shape
        depth_input_shape = obs_space.original_space["depth"].shape
        self.rgb_encoder = ConvNet2D(channels=3, input_dim=rgb_input_shape)
        self.depth_encoder = ConvNet2D(channels=1, input_dim=depth_input_shape)
        self.scan_encoder = ConvNet1D()
        self.task_encoder = FC(4)

        self.value_head = FC(40, 1)
        self.action_head = FC(40, 2)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        obs["task_obs"] = obs["task_obs"].float()
        obs["rgb"] = obs["rgb"].float().permute(0, 3, 1, 2)
        obs["depth"] = obs["depth"].float().permute(0, 3, 1, 2)
        obs["scan"] = obs["scan"].float().permute(0, 2, 1)

        rgb_obs_encoding = self.rgb_encoder(obs["rgb"])
        depth_obs_encoding = self.depth_encoder(obs["depth"])
        scan_obs_encoding = self.scan_encoder(obs["scan"])
        task_obs_encoding = self.task_encoder(obs["task_obs"])

        policy_input = torch.cat([rgb_obs_encoding, depth_obs_encoding, scan_obs_encoding, task_obs_encoding], dim=1)

        self._value_out = torch.flatten(self.value_head(policy_input))
        action_out = self.action_head(policy_input)

        return action_out, []

    def value_function(self):
        return self._value_out


class OmniGibsonRLLibEnv(OmniGibsonEnv):
    def __init__(self, env_config):
        super().__init__(
            config_file=env_config["config_file"],
            mode=env_config["mode"],
            action_timestep=env_config["action_timestep"],
            physics_timestep=env_config["physics_timestep"],
        )


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of how to train a robot with parallelization provided by Ray using RLLib
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests and we
    # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    if not (random_selection and headless and short_exec):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            "-c",
            default=os.path.join(omnigibson.root_path, "configs", "turtlebot_nav.yaml"),
            help="which config file to use [default: use yaml files in examples/configs]",
        )
        parser.add_argument(
            "--local_dir", default=None, help="Directory where to save model logs and default checkpoints"
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            choices=[None, "PROMPT"],
            help="Whether to resume the experiment. Note this does *not* restore the checkpoint, just re-uses the same config/log.",
        )
        parser.add_argument("--restore_checkpoint", default=None, help="Checkpoint to force restore")
        parser.add_argument(
            "--exp_name", default="my_omnigibson_run", help="which mode for simulation (default: headless)"
        )
        parser.add_argument(
            "--mode",
            "-m",
            choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
            default="headless",
            help="which mode for simulation (default: headless)",
        )

        args = parser.parse_args()
        config = args.config
        local_dir = args.local_dir
        resume = args.resume
        restore_checkpoint = args.restore_checkpoint
        exp_name = args.exp_name
        mode = args.mode
    else:
        config = os.path.join(omnigibson.root_path, "configs", "turtlebot_nav.yaml")
        local_dir = None
        resume = None
        restore_checkpoint = None
        exp_name = "my_omnigibson_run"
        mode = "headless"

    ModelCatalog.register_custom_model("OmniGibsonPPOModel", OmniGibsonPPOModel)
    register_env("OmniGibsonEnv", lambda c: OmniGibsonRLLibEnv(c))
    # Note, some things you may want to change
    # See: https://docs.ray.io/en/master/rllib-training.html#common-parameters for more details
    # num_gpus -- number of GPUs used for the driver (trainer)
    # num_cpus_per_driver -- number of cpus used on the driver (trainer)
    # num_workers -- defines the number of workers collecting iG trials
    # num_envs_per_worker -- number of OmniGibson instances per worker
    # num_cpus_per_driver -- number of cpus used on the worker
    # remote_worker_envs -- this parallelizes the data collection loop *on the worker* into its own process with IPC overhead, not recommended
    # train_batch_size -- the total timesteps per SGD (affects RAM usage)
    # rollout_fragment_legnth is the fragment collected from each rollout worker, concatenated across workers to meet size of train_batch
    # sgd_minibatch_size -- Total SGD batch size across all devices (affects GPU VRAM)
    #
    # train_batch size should be divisble by sgd_minibatch_size **and** (rollout fragment length * num_workers)
    # Good config: train_batch_size = 1000, rollout_fragment_length = 200, num_workers = 5, sgd_minibatch_size = largest your GPU can support that train_batch_size is divisble by (500 for Titan X)
    config_filename = os.path.join(omnigibson.example_config_path, config)
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config_data["image_width"] = 160
    config_data["image_height"] = 120

    # Reduce texture scale and reqs of GPU/CPU for Mac
    if platform == "darwin":
        short_exec = True  # Force a short training process. Remove this for real training
        config_data["texture_scale"] = 0.5
        config = {
            "env": "OmniGibsonEnv",
            "env_config": {
                "config_file": config_data,
                "mode": mode,
                "action_timestep": 1.0 / 10.0,
                "physics_timestep": 1.0 / 40.0,
            },
            "num_gpus": 0,  # 1
            "num_cpus_for_driver": 1,
            # "remote_worker_envs": True,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "num_cpus_per_worker": 1,
            "rollout_fragment_length": 200 if not short_exec else 20,
            "train_batch_size": 1000 if not short_exec else 10,
            "sgd_minibatch_size": 100 if not short_exec else 1,
            "model": {
                "custom_model": "OmniGibsonPPOModel",
            },
            "framework": "torch",
        }
    else:  # Server or Linux
        config = {
            "env": "OmniGibsonEnv",
            "env_config": {
                "config_file": config_data,
                "mode": mode,
                "action_timestep": 1.0 / 10.0,
                "physics_timestep": 1.0 / 40.0,
            },
            "num_gpus": 1,
            "num_cpus_for_driver": 1,
            # "remote_worker_envs": True,
            "num_workers": 5,
            "num_envs_per_worker": 1,
            "num_cpus_per_worker": 10,
            "rollout_fragment_length": 200 if not short_exec else 20,
            "train_batch_size": 1000 if not short_exec else 10,
            "sgd_minibatch_size": 100 if not short_exec else 1,
            "model": {
                "custom_model": "OmniGibsonPPOModel",
            },
            "framework": "torch",
        }
    stop = {"training_iteration": 100000 if not short_exec else 2}
    if resume is not None:
        assert restore_checkpoint is not None, "Error: When resuming must provide explicit path to checkpoint"

    results = tune.run(
        "PPO",
        config=config,
        verbose=2,
        restore=restore_checkpoint,
        name=exp_name,
        local_dir=local_dir,
        checkpoint_freq=100,
        resume=resume,
        stop=stop,
    )


if __name__ == "__main__":
    main()
