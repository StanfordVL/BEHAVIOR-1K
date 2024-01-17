import argparse
import logging
import socket

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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback 


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

    args = parser.parse_args()
    prefix = ''
    seed = 0

    # Decide whether to use a local environment or remote
    n_envs = args.n_envs
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
    else:
        import omnigibson as og
        from omnigibson.macros import gm

        gm.USE_FLATCACHE = True

        config_filename = "omni_grpc.yaml"
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        og_env = og.Environment(configs=config)
        env = DummyVecEnv([lambda: og_env])
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
        run = wandb.init(
            entity="behavior-rl",
            project="sb3",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            # save_code=True,  # optional
        )
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
        callback = CallbackList([
            wandb_callback,
            checkpoint_callback,
        ])
        print(callback.callbacks)

        log.debug(model.policy)
        log.info(f"model: {model}")

        log.info("Starting training...")
        model.learn(
            total_timesteps=1_500_000,
            callback=callback,
        )
        log.info("Finished training!")


if __name__ == "__main__":
    main()
