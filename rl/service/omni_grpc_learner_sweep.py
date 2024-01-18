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

def train():
    prefix = ''
    seed = 0
    # Decide whether to use a local environment or remote
    n_envs = args.n_envs
    config_filename = "omni_grpc.yaml"
    env_config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    if n_envs > 0:
        if args.port is not None:
            local_port = int(args.port)
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            local_port = s.getsockname()[1]
            s.close()
        print(f"Listening on port {local_port}")
        env = GRPCClientVecEnv(f"0.0.0.0:{local_port}", n_envs)
        task_config = env_config['task']
        task_config['dist_coeff'] = wandb.config.dist_coeff
        task_config['grasp_reward'] = wandb.config.grasp_reward
        task_config['collision_penalty'] = wandb.config.collision_penalty
        task_config['eef_position_penalty_coef'] = wandb.config.eef_position_penalty_coef
        task_config['eef_orientation_penalty_coef'] = wandb.config.eef_orientation_penalty_coef
        task_config['regularization_coef'] = wandb.config.regularization_coef
        env.env_method('update_task', task_config)

    else:
        import omnigibson as og
        from omnigibson.macros import gm

        gm.USE_FLATCACHE = True
        env_config['task']['dist_coeff'] = wandb.config.dist_coeff
        env_config['task']['grasp_reward'] = wandb.config.grasp_reward
        env_config['task']['collision_penalty'] = wandb.config.collision_penalty
        env_config['task']['eef_position_penalty_coef'] = wandb.config.eef_position_penalty_coef
        env_config['task']['eef_orientation_penalty_coef'] = wandb.config.eef_orientation_penalty_coef
        env_config['task']['regularization_coef'] = wandb.config.regularization_coef
        og_env = og.Environment(configs=env_config)
        env = DummyVecEnv([lambda: og_env])
        env = VecFrameStack(env, n_stack=5)
        n_envs = 1

    # import IPython; IPython.embed()

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        ceiling = og_env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    # Set the set
    set_random_seed(seed)
    env.reset()

    # policy_kwargs = dict(
    #     features_extractor_class=CustomCombinedExtractor,
    # )

    if args.eval:    
        assert args.checkpoint is not None, "If evaluating a PPO policy, @checkpoint argument must be specified!"
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
                "net_arch": {"pi": [512, 512], "vf": [512, 512]}
            },
        }
        run = wandb.init()
        env = VecFrameStack(env, n_stack=5)
        env = VecMonitor(env)
        env = VecVideoRecorder(
            env,
            f"videos/{run.id}",
            record_video_trigger=lambda x: x % 2000 == 0,
            video_length=400,
        )
        tensorboard_log_dir = f"runs/{run.id}"
        if args.checkpoint is None:
            model = PPO(
                env=env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                device='cuda',
                **config,
            )
        else:
            model = PPO.load(args.checkpoint, env=env)
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        wandb_callback = WandbCallback(
            model_save_path=tensorboard_log_dir,
            verbose=2,
        )
        # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=10, verbose=1)
        callback = CallbackList([
            wandb_callback,
            checkpoint_callback,
            # stop_train_callback,
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
        # wandb.alert(title="Run launched", text=text, level=AlertLevel.INFO)
        model.learn(
            total_timesteps=2_000_000,
            callback=callback,
        )
        log.info("Finished training!")



if __name__ == "__main__":
    # print(args.sweep_id)
    wandb.agent(args.sweep_id, entity="behavior-rl", project="sweep", count=3, function=train)
