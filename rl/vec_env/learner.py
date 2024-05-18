import argparse
from collections import defaultdict
import logging
import os
import socket
import sys

import numpy as np
import yaml

log = logging.getLogger(__name__)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import torch as th
import torch.nn as nn
import wandb
from service.telegym import GRPCClientVecEnv
from stable_baselines3 import A2C, PPO, SAC
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
from wandb import AlertLevel
from wandb.integration.sb3 import WandbCallback

# Parse args
# parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR")
# parser.add_argument(
#     "--n_envs", type=int, default=5, help="Number of parallel environments to wait for. 0 to run a local environment."
# )
# parser.add_argument("--port", type=int, default=None, help="The port to listen at. Defaults to a random port.")
# parser.add_argument("--eval_port", type=int, default=None, help="Port to listen at for evaluation.")
# parser.add_argument(
#     "--eval", type=bool, default=False, help="Whether to evaluate a policy instead of training. Fixes n_envs at 0."
# )
# parser.add_argument(
#     "--checkpoint",
#     type=str,
#     default=None,
#     help="Absolute path to desired PPO checkpoint to load for evaluation",
# )
# parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to run.")
# args = parser.parse_args()


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


class ReportInfosCallback(BaseCallback):
    # Get the monitoring keys. These are reward_ + the keys in the info
    # dict in grasp_reward.py
    MONITORING_KEYS = [
        "reward_regularization_penalty_factor",
        "reward_regularization_penalty",
        "reward_position_penalty_factor",
        "reward_position_penalty",
        "reward_rotation_penalty_factor",
        "reward_rotation_penalty",
        "reward_collision_penalty_factor",
        "reward_collision_penalty",
        "reward_grasp_reward_factor",
        "reward_grasp_reward",
        "reward_pregrasp_dist",
        "reward_pregrasp_dist_reward_factor",
        "reward_pregrasp_dist_reward",
        "reward_postgrasp_dist",
        "reward_postgrasp_dist_reward_factor",
        "reward_postgrasp_dist_reward",
    ]

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        assert "infos" in self.locals, "`infos` variable is not defined, please check your code next to `callback.on_step()`"
        by_key = defaultdict(list)
        for info in self.locals["infos"]:
            for key, value in info.items():
                if key in self.MONITORING_KEYS:
                    by_key[key].append(value)

        assert len(by_key) == len(self.MONITORING_KEYS), "Some keys are missing in the monitoring keys"
        for key, values in by_key.items():
            updated_key = "reward/" + key[len("reward_"):]
            self.logger.record(updated_key, np.mean(values))

        return True


class AfterEvalCallback(BaseCallback):
    def __init__(self, env, eval_env, verbose=0):
        super(AfterEvalCallback, self).__init__(verbose)
        self.env = env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.env.reset()

        return True

def train():

    import omnigibson as og
    from omnigibson.envs.sb3_vec_env import SB3VectorEnvironment
    from omnigibson.macros import gm

    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.HEADLESS = True

    # Decide whether to use a local environment or remote
    # n_envs = args.n_envs
    n_envs = 5
    env_config = _get_env_config()
    env_config["task"]["precached_reset_pose_path"] = reset_poses_path
    del env_config["env"]["external_sensors"]
    env = SB3VectorEnvironment(n_envs, env_config, render_on_step=False)
    env = VecFrameStack(env, n_stack=5)
    env = VecMonitor(env, info_keywords=("is_success",))

    eval_env_config = _get_env_config()
    eval_env_config["task"]["precached_reset_pose_path"] = reset_poses_path
    eval_env = SB3VectorEnvironment(1, eval_env_config, render_on_step=True)
    eval_env = VecFrameStack(eval_env, n_stack=5)
    eval_env = VecMonitor(eval_env, info_keywords=("is_success",))

    prefix = ""
    seed = 0
    # run = wandb.init(sync_tensorboard=True, monitor_gym=True)
    run = wandb.init(
        entity="behavior-rl",
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    if "dist_coeff" in wandb.config:
        task_config = _get_env_config()["task"]
        task_config["precached_reset_pose_path"] = reset_poses_path
        task_config["reward_config"]["dist_coeff"] = wandb.config.dist_coeff
        task_config["reward_config"]["grasp_reward"] = wandb.config.grasp_reward
        task_config["reward_config"]["collision_penalty"] = wandb.config.collision_penalty
        task_config["reward_config"]["eef_position_penalty_coef"] = wandb.config.eef_position_penalty_coef
        task_config["reward_config"]["eef_orientation_penalty_coef"] = (
            wandb.config.eef_orientation_penalty_coef_relative * wandb.config.eef_position_penalty_coef
        )
        task_config["reward_config"]["regularization_coef"] = wandb.config.regularization_coef
        env.env_method("update_task", task_config)
        eval_env.env_method("update_task", task_config)

    eval_env = VecVideoRecorder(
        eval_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % (NUM_EVAL_EPISODES * STEPS_PER_EPISODE) == 0,
        video_length=STEPS_PER_EPISODE,
    )

    eval_env.render()

    # Set the set
    set_random_seed(seed)
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCombinedExtractor,
    # )
    # if args.eval:
    if False:
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
        algo_config = {
            "policy": "MultiInputPolicy",
            "n_steps": STEPS_PER_EPISODE,
            "batch_size": STEPS_PER_EPISODE,
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
        a2c_config = {
            "policy": "MultiInputPolicy",
            "n_steps": STEPS_PER_EPISODE,
            "gamma": 0.99,
            "gae_lambda": 0.9,
            "ent_coef": 0.0,
            "sde_sample_freq": 4,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "learning_rate": 3e-5,
            "use_sde": True,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": nn.ReLU,
                "net_arch": {"pi": [512, 512], "vf": [512, 512]},
            },
        }
        sac_config = {
            "policy": "MultiInputPolicy",
            "batch_size": STEPS_PER_EPISODE,
            "gamma": 0.99,
            "ent_coef": 0.0,
            "sde_sample_freq": 4,
            "learning_rate": 3e-5,
            "use_sde": True,
            "policy_kwargs": {
                "n_critics": 1,
                "net_arch": [128, 64],
            },
        }
        tensorboard_log_dir = f"runs/{run.id}"
        # if args.checkpoint is None:
        if True:
            # model = PPO(
            #     env=env,
            #     verbose=1,
            #     tensorboard_log=tensorboard_log_dir,
            #     device="cuda",
            #     **algo_config,
            # )
            # model = A2C(
            #     env=env,
            #     verbose=1,
            #     tensorboard_log=tensorboard_log_dir,
            #     device="cuda",
            #     **a2c_config,
            # )
            model = SAC(
                env=env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                device="cuda",
                **sac_config,
            )
        else:
            model = PPO.load(args.checkpoint, env=env)
        report_infos_callback = ReportInfosCallback()
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        wandb_callback = WandbCallback(
            model_save_path=tensorboard_log_dir,
            verbose=2,
        )
        # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)
        after_eval_callback = AfterEvalCallback(env, eval_env)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=EVAL_EVERY_N_EPISODES * STEPS_PER_EPISODE,
            callback_after_eval=after_eval_callback,
            verbose=1,
            best_model_save_path="logs/best_model",
            n_eval_episodes=NUM_EVAL_EPISODES,
            deterministic=True,
            render=False,
        )
        callback = CallbackList(
            [
                report_infos_callback,
                wandb_callback,
                checkpoint_callback,
                eval_callback,
            ]
        )
        print(callback.callbacks)

        log.debug(model.policy)
        log.info(f"model: {model}")
        log.info("Starting training...")
        wandb.alert(title="Run launched", text=f"Run ID: {wandb.run.id}", level=AlertLevel.INFO)
        model.learn(
            total_timesteps=10_000_000,
            callback=callback,
            log_interval=4
        )
        log.info("Finished training!")


if __name__ == "__main__":
    train()
