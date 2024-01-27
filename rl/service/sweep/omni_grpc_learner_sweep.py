import argparse
import logging
import socket
import os
import sys
import yaml
import numpy as np

log = logging.getLogger(__name__)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, StopTrainingOnNoModelImprovement, BaseCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback 
from wandb import AlertLevel
import omnigibson as og

class MetricsCallback(BaseCallback):
    """
    A custom metrics callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        envs_num_grasps = list(map(lambda x: x['reward']['grasp']['grasp_success'], self.locals['infos']))
        grasp_success_rate = np.mean(envs_num_grasps)
        wandb.log({"grasp_success_rate": grasp_success_rate})

# Parse args
parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR")
parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments to wait for. 0 to run a local environment.")
parser.add_argument("--port", type=int, default=None, help="The port to listen at. Defaults to a random port.")
parser.add_argument("--eval", type=bool, default=False, help="Whether to evaluate a policy instead of training. Fixes n_envs at 0.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Absolute path to desired PPO checkpoint to load for evaluation",
)
parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to run.")
args = parser.parse_args()

def instantiate_envs():
    # Decide whether to use a local environment or remote
    n_envs = args.n_envs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "../omni_grpc.yaml"))
    env_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    if args.port is not None:
        local_port = int(args.port)
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        local_port = s.getsockname()[1]
        s.close()
    print(f"Listening on port {local_port}")
    env = GRPCClientVecEnv(f"0.0.0.0:{local_port}", n_envs)
    env = VecFrameStack(env, n_stack=5)
    env = VecMonitor(env)

    # Manually specify port for eval env
    # eval_env = GRPCClientVecEnv(f"0.0.0.0:50064", 1)
    # eval_env = VecFrameStack(eval_env, n_stack=5)

    eval_env = og.Environment(configs=env_config)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(env, n_stack=5)
    return env, eval_env

def train(env, eval_env):
    prefix = ''
    seed = 0
    run = wandb.init(
        sync_tensorboard=True,
        monitor_gym=True
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "../omni_grpc.yaml"))
    env_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    task_config = env_config['task']
    task_config['dist_coeff'] = wandb.config.dist_coeff
    task_config['grasp_reward'] = wandb.config.grasp_reward
    task_config['collision_penalty'] = wandb.config.collision_penalty
    task_config['eef_position_penalty_coef'] = wandb.config.eef_position_penalty_coef
    task_config['eef_orientation_penalty_coef'] = wandb.config.eef_orientation_penalty_coef_relative * wandb.config.eef_position_penalty_coef
    task_config['regularization_coef'] = wandb.config.regularization_coef
    env.env_method('update_task', task_config)

    # eval_env.env_method('update_task', task_config)
    eval_env.update_task(task_config)
    eval_env = VecVideoRecorder(
        eval_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )
    # Set the set
    set_random_seed(seed)
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCombinedExtractor,
    # )
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
            "net_arch": {"pi": [512, 512], "vf": [512, 512]}
        },
    }
    tensorboard_log_dir = f"runs/{run.id}"
    model = PPO(
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        device='cuda',
        **config,
    )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
    wandb_callback = WandbCallback(
        model_save_path=tensorboard_log_dir,
        verbose=2,
    )
    metrics_callback = MetricsCallback()
    # Add with eval call back https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#stoptrainingonnomodelimprovement
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=10, verbose=1)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=100, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=2000, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path='logs/best_model')
    callback = CallbackList([
        wandb_callback,
        checkpoint_callback,
        metrics_callback,
        eval_callback,
    ])
    print(callback.callbacks)

    log.debug(model.policy)
    log.info(f"model: {model}")

    log.info("Starting training...")

    USER = os.environ['USER']
    policy_save_path = wandb.run.dir.split("/")[2:-3]
    policy_save_path.insert(0, f"/cvgl2/u/{USER}/OmniGibson")
    policy_save_path.append("runs")
    policy_save_path.append(wandb.run.id)
    policy_save_path = "/".join(policy_save_path)
    text = f"Saved policy path: {policy_save_path}"
    wandb.alert(title="Run launched", text=text, level=AlertLevel.INFO)
    model.learn(
        total_timesteps=2_000_000,
        callback=callback,
    )
    log.info("Finished training!")


if __name__ == "__main__":
    # print(args.sweep_id)
    env, eval_env = instantiate_envs()
    def _train():
        return train(env, eval_env)
    wandb.agent(args.sweep_id, entity="behavior-rl", project="sb3", count=20, function=_train)
