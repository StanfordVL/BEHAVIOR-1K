import argparse
import logging
import socket
import os

import yaml

log = logging.getLogger(__name__)

from telegym import GRPCClientVecEnv

import gymnasium as gym
import torch as th
import torch.nn as nn
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, StopTrainingOnNoModelImprovement
from wandb.integration.sb3 import WandbCallback 
from wandb import AlertLevel

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "mean_grasps"},
    "parameters": {
        "grasp_reward": {"min": 0.0, "max": 1.0, 'distribution': 'uniform'},
        "collision_penalty": {"min": 0.0, "max": 1.0, 'distribution': 'uniform'},
        "eef_position_penality_coef": {"min": 0.0, "max": 1.0, 'distribution': 'uniform'},
        "eef_orientation_penalty_coef": {"min": 0.0, "max": 1.0, 'distribution': 'uniform'},
        "eef_orientation_penalty_coef": {"min": 0.0, "max": 1.0, 'distribution': 'uniform'},
        "regularization_coef": {"min": 0.0,  "max": 1.0, 'distribution': 'uniform'},
    },
}


sweep_id = wandb.sweep(sweep=sweep_configuration, entity="behavior-rl", project="sweep")
print(sweep_id)