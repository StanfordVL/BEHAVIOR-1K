"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

import argparse
import os, time, cv2

import omnigibson as og
from omnigibson import example_config_path

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

except ModuleNotFoundError:
    og.log.error("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        self.debug_length = 10
        self.debug_mode = True
        extractors = {}
        self.step_index = 0
        self.img_save_dir = 'img_save_dir'
        os.makedirs(self.img_save_dir, exist_ok=True)
        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["rgb", "ins_seg"]:
                og.log.info(f"obs {key} shape: {subspace.shape}")
                n_input_channels = subspace.shape[2]  # channel last
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
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["accum_reward", 'obj_joint']:
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size))
            else:
                continue
            total_concat_size += feature_size
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        self.step_index += 1

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key in ["rgb",]:
                if self.debug_mode:
                    cv2.imwrite(os.path.join(self.img_save_dir, '{0:06d}.png'.format(self.step_index % self.debug_length)), cv2.cvtColor((observations[key][0].cpu().numpy()*255).astype('uint8'), cv2.COLOR_RGB2BGR))
                observations[key] = observations[key].permute((0, 3, 1, 2))
            elif key in ["ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2)) / 500.
            elif key in ['accum_reward', 'obj_joint']:
                if len(observations[key].shape) == 3:
                    observations[key] = observations[key].squeeze(-1)  # [:, :, 0]
            else:
                continue

            encoded_tensor_list.append(extractor(observations[key]))

        feature = th.cat(encoded_tensor_list, dim=1)
        return feature


def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent in BEHAVIOR")
    parser.add_argument(
        "--config",
        type=str,
        default=f"{example_config_path}/fetch_behavior.yaml",
        help="Absolute path to desired OmniGibson environment config to load",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Absolute path to desired PPO checkpoint to load for evaluation",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will evaluate the PPO agent found from --checkpoint",
    )

    args = parser.parse_args()
    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ''
    seed = 0

    env = og.Environment(configs=args.config, action_timestep=1 / 60., physics_timestep=1 / 60.)

    # If we're evaluating, hide the ceilings and enable camera teleoperation so the user can easily
    # visualize the rollouts dynamically
    if args.eval:
        ceiling = env.scene.object_registry("name", "ceilings")
        ceiling.visible = False
        og.sim.enable_viewer_camera_teleoperation()

    # Set the set
    set_random_seed(seed)
    env.reset()

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    os.makedirs(tensorboard_log_dir, exist_ok=True)

    if args.eval:
        assert args.checkpoint is not None, "If evaluating a PPO policy, @checkpoint argument must be specified!"
        model = PPO.load(args.checkpoint)
        og.log.info("Starting evaluation...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        og.log.info("Finished evaluation!")
        og.log.info(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            policy_kwargs=policy_kwargs,
            n_steps=20 * 10,
            batch_size=8,
            device='cuda',
        )
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        og.log.debug(model.policy)
        og.log.info(f"model: {model}")

        og.log.info("Starting training...")
        model.learn(total_timesteps=10000000, callback=checkpoint_callback,
                    eval_env=env, eval_freq=1000,
                    n_eval_episodes=20)
        og.log.info("Finished training!")


if __name__ == "__main__":
    main()
