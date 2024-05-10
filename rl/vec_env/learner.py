import argparse
import logging
import os
import socket
import sys

import yaml

from omnigibson.envs.sb3_vec_env import SB3VectorEnvironment

log = logging.getLogger(__name__)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import torch as th
import torch.nn as nn
from service.telegym import GRPCClientVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import omnigibson as og
import wandb
from omnigibson.macros import gm
from wandb import AlertLevel

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False

# Parse args
parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR")
parser.add_argument(
    "--n_envs", type=int, default=0, help="Number of parallel environments to wait for. 0 to run a local environment."
)
parser.add_argument("--port", type=int, default=None, help="The port to listen at. Defaults to a random port.")
parser.add_argument("--eval_port", type=int, default=None, help="Port to listen at for evaluation.")
parser.add_argument(
    "--eval", type=bool, default=False, help="Whether to evaluate a policy instead of training. Fixes n_envs at 0."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Absolute path to desired PPO checkpoint to load for evaluation",
)
parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to run.")
args = parser.parse_args()


def _get_env_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "rl.yaml"))
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return config


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


EVAL_EVERY_N_EPISODES = 10
NUM_EVAL_EPISODES = 5
STEPS_PER_EPISODE = _get_env_config()["task"]["termination_config"]["max_steps"]
reset_poses_path = os.path.dirname(__file__) + "/../reset_poses.json"


def instantiate_envs():
    # Decide whether to use a local environment or remote
    n_envs = args.n_envs if args.n_envs else 5
    config = _get_env_config()
    del config["env"]["external_sensors"]
    config["task"]["precached_reset_pose_path"] = reset_poses_path
    env = SB3VectorEnvironment(n_envs, config, render_on_step=False)
    env = VecFrameStack(env, n_stack=5)
    env = VecMonitor(env, info_keywords=("is_success",))
    eval_env = SB3VectorEnvironment(1, config, render_on_step=True)
    eval_env = VecFrameStack(eval_env, n_stack=5)
    eval_env = VecMonitor(eval_env, info_keywords=("is_success",))
    return env, eval_env


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         super().__init__(observation_space, features_dim=1)
#         extractors = {}
#         self.step_index = 0
#         total_concat_size = 0
#         feature_size = 128
#         time_dimension = None
#         for key, subspace in observation_space.spaces.items():
#             # For now, only keep RGB observations
#             log.info(f"obs {key} shape: {subspace.shape}")
#             if time_dimension is None:
#                 time_dimension = subspace.shape[0]
#             else:
#                 assert time_dimension == subspace.shape[0], f"All observation subspaces must have the same time dimension. Expected {time_dimension}, found {subspace.shape[0]}"
#             if key == "rgb":
#                 assert len(subspace.shape) == 4, "Expected frame-stacked (f, c, h, w) RGB observations"
#                 n_input_channels = subspace.shape[1]  # channel
#                 cnn = nn.Sequential(
#                     nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),
#                     nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),
#                     nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
#                     nn.ReLU(),
#                     nn.Flatten(),
#                 )
#                 test_tensor = th.zeros(subspace.shape)
#                 with th.no_grad():
#                     n_flatten = cnn(test_tensor[None]).shape[1]
#                 fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#                 extractors[key] = nn.Sequential(cnn, fc)
#                 total_concat_size += feature_size
#             elif key == "proprio":
#                 assert len(subspace.shape) == 2, "Expected frame-stacked (f, n) proprio observations"
#                 mlp = nn.Sequential(
#                     nn.Linear(subspace.shape[0], feature_size),
#                     nn.ReLU(),
#                     nn.Linear(subspace.shape[0], feature_size),
#                     nn.ReLU(),
#                 )
#                 extractors[key] = mlp
#                 total_concat_size += feature_size

#         self._across_time = nn.Sequential(
#             nn.Linear(total_concat_size * 5, feature_size),
#             nn.ReLU(),
#             nn.Linear(subspace.shape[0], feature_size),
#             nn.ReLU(),
#         )

#         self.extractors = nn.ModuleDict(extractors)

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []
#         self.step_index += 1

#         # self.extractors contain nn.Modules that do all the processing.
#         for i in range(next(iter(observations.values())).shape[0]):
#             for key, extractor in self.extractors.items():
#                 encoded_tensor_list.append(extractor(observations[key][i]))

#         feature = th.cat(encoded_tensor_list, dim=1)
#         feature = self._across_time(feature)
#         return feature


def train(env, eval_env):
    prefix = ""
    seed = 0
    if args.sweep_id:
        run = wandb.init(sync_tensorboard=True, monitor_gym=True)
        task_config = _get_env_config()["task"]
        task_config["reward_config"]["dist_coeff"] = wandb.config.dist_coeff
        task_config["reward_config"]["grasp_reward"] = wandb.config.grasp_reward
        task_config["reward_config"]["collision_penalty"] = wandb.config.collision_penalty
        task_config["reward_config"]["eef_position_penalty_coef"] = wandb.config.eef_position_penalty_coef
        task_config["reward_config"]["eef_orientation_penalty_coef"] = (
            wandb.config.eef_orientation_penalty_coef_relative * wandb.config.eef_position_penalty_coef
        )
        task_config["reward_config"]["regularization_coef"] = wandb.config.regularization_coef
        env.env_method("update_task", task_config)
        # eval_env.env_method("update_task", task_config)
    else:
        run = wandb.init(
            entity="behavior-rl",
            project="sb3",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            # save_code=True,  # optional
        )

    # eval_env = VecVideoRecorder(
    #     eval_env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % (NUM_EVAL_EPISODES * STEPS_PER_EPISODE) == 0,
    #     video_length=STEPS_PER_EPISODE,
    # )
    # Set the set
    set_random_seed(seed)
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCombinedExtractor,
    # )
    if args.eval:
        # Need to enable rendering in simulator.step and something else
        assert args.checkpoint is not None, "If evaluating a PPO policy, @checkpoint argument must be specified!"
        ceiling = env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        model = PPO.load(args.checkpoint)
        log.info("Starting evaluation...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        log.info("Finished evaluation!")
        log.info(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
    else:
        config = {
            "policy": "MultiInputPolicy",
            "n_steps": 512,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "n_epochs": 20,
            "ent_coef": 0.0,
            "sde_sample_freq": 4,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "learning_rate": 3e-5,
            "use_sde": True,
            "clip_range": 0.4,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [512, 512], "vf": [512, 512]},
            },
        }
        tensorboard_log_dir = f"runs/{run.id}"
        if args.checkpoint is None:
            model = PPO(
                env=env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                device="cuda",
                **config,
            )
        else:
            model = PPO.load(args.checkpoint, env=env)
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        wandb_callback = WandbCallback(
            model_save_path=tensorboard_log_dir,
            verbose=2,
        )
        # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)
        # eval_callback = EvalCallback(
        #     eval_env,
        #     eval_freq=EVAL_EVERY_N_EPISODES * STEPS_PER_EPISODE,
        #     callback_after_eval=None,
        #     verbose=1,
        #     best_model_save_path="logs/best_model",
        #     n_eval_episodes=NUM_EVAL_EPISODES,
        #     deterministic=True,
        #     render=False,
        # )
        callback = CallbackList(
            [
                # wandb_callback,
                # checkpoint_callback,
                # eval_callback,
            ]
        )
        print(callback.callbacks)

        log.debug(model.policy)
        log.info(f"model: {model}")
        log.info("Starting training...")
        wandb.alert(title="Run launched", text=f"Run ID: {wandb.run.id}", level=AlertLevel.INFO)
        model.learn(
            total_timesteps=4_000_000,
            callback=callback,
        )
        log.info("Finished training!")


if __name__ == "__main__":
    env, eval_env = instantiate_envs()
    _train = lambda: train(env, eval_env)
    if args.sweep_id:
        wandb.agent(args.sweep_id, entity="behavior-rl", project="sb3", count=20, function=_train)
    else:
        _train()
