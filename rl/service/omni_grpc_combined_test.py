import argparse
import logging

log = logging.getLogger(__name__)

import gymnasium as gym
import torch as th
import torch.nn as nn
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback 

import numpy as np
import omnigibson as og
from omnigibson.macros import gm

gm.USE_FLATCACHE = True

def create_env():
    DIST_COEFF = 0.1
    GRASP_REWARD = 0.3

    cfg = {
        "env": {
            "action_timestep": 1 / 10.,
            "physics_timestep": 1 / 60.,
            "flatten_obs_space": True,
            "flatten_action_space": True,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "coffee_table"],
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb", "proprio"],
                "proprio_obs": ["joint_qpos", "joint_qvel", "eef_0_pos", "eef_0_quat", "grasp_0"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "sensor_config": {
                    "VisionSensor": {
                        "modalities": ["rgb"],
                        "sensor_kwargs": {
                            "image_width": 224,
                            "image_height": 224
                        }
                    }
                },
                "controller_config": {
                    "base": {
                        "name": "DifferentialDriveController",
                    },
                    "arm_0": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": "default",
                        "command_output_limits": None,
                        "use_delta_commands": True,
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "mode": "ternary",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": "default",
                        "command_output_limits": None,
                        "use_delta_commands": False
                    }
                }
            }
        ],
        "task": {
            "type": "GraspTask",
            "obj_name": "cologne",
            "termination_config": {
                "max_steps": 400,
            },
            "reward_config": {
                "r_dist_coeff": DIST_COEFF,
                "r_grasp": GRASP_REWARD
            }
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
            },
        ]
    }

    env = og.Environment(configs=cfg)
    return env

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        self.step_index = 0
        total_concat_size = 0
        feature_size = 128
        time_dimension = None
        for key, subspace in observation_space.spaces.items():
            # For now, only keep RGB observations
            log.info(f"obs {key} shape: {subspace.shape}")
            if time_dimension is None:
                time_dimension = subspace.shape[0]
            else:
                assert time_dimension == subspace.shape[0], f"All observation subspaces must have the same time dimension. Expected {time_dimension}, found {subspace.shape[0]}"
            if key == "rgb":
                assert len(subspace.shape) == 4, "Expected frame-stacked (f, c, h, w) RGB observations"
                n_input_channels = subspace.shape[1]  # channel 
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros(subspace.shape)
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
                total_concat_size += feature_size
            elif key == "proprio":
                assert len(subspace.shape) == 2, "Expected frame-stacked (f, n) proprio observations"
                mlp = nn.Sequential(
                    nn.Linear(subspace.shape[0], feature_size),
                    nn.ReLU(),
                    nn.Linear(subspace.shape[0], feature_size),
                    nn.ReLU(),
                )
                extractors[key] = mlp
                total_concat_size += feature_size

        self._across_time = nn.Sequential(
            nn.Linear(total_concat_size * 5, feature_size),
            nn.ReLU(),
            nn.Linear(subspace.shape[0], feature_size),
            nn.ReLU(),
        )

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        self.step_index += 1

        # self.extractors contain nn.Modules that do all the processing.
        for i in range(next(iter(observations.values())).shape[0]):
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key][i]))

        feature = th.cat(encoded_tensor_list, dim=1)
        feature = self._across_time(feature)
        return feature


def main():
    prefix = ''
    seed = 0

    env = DummyVecEnv([create_env])

    # TODO: None of this stuff works: make it work by running env locally and connecting to it.
    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    # if args.eval:
    #     ceiling = env.scene.object_registry("name", "ceilings")
    #     ceiling.visible = False
    #     og.sim.enable_viewer_camera_teleoperation()

    # Set the set
    set_random_seed(seed)
    env.reset()

    # policy_kwargs = dict(
    #     features_extractor_class=CustomCombinedExtractor,
    # )

    config = {
        "policy_type": "MultiInputPolicy",
        "n_steps": 30 * 10,
        "batch_size": 8,
        "total_timesteps": 10_000_000,
    }
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    env = VecFrameStack(env, n_stack=5, channels_order="first")
    env = VecMonitor(env)
    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % 2000 == 0,
    #     video_length=200,
    # )
    tensorboard_log_dir = f"runs/{run.id}"
    model = PPO(
        config["policy_type"],
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        # policy_kwargs=policy_kwargs,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        device='cuda',
    )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
    # eval_callback = EvalCallback(eval_env=env, eval_freq=1000, n_eval_episodes=20)
    wandb_callback = WandbCallback(
        model_save_path=tensorboard_log_dir,
        verbose=2,
    )
    callback = CallbackList([wandb_callback, checkpoint_callback])
    print(callback.callbacks)

    log.debug(model.policy)
    log.info(f"model: {model}")

    log.info("Starting training...")
    model.learn(
        total_timesteps=10000000,
        callback=callback,
    )
    log.info("Finished training!")


if __name__ == "__main__":
    main()
