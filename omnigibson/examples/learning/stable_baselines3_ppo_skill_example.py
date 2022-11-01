import logging
import os, time, cv2
from typing import Callable
import pdb
import bddl

import omnigibson as ig
from omnigibson.wrappers import ActionPrimitiveWrapper
from omnigibson import example_config_path

log = logging.getLogger(__name__)

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
    log.error("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

#
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
                print(subspace.shape)
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
    config_file = f"{example_config_path}/behavior_mp_tiago.yaml"
    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    prefix = ''
    seed = 0

    env = ig.Environment(configs=config_file, action_timestep=1 / 60., physics_timestep=1 / 60.)
    env = ActionPrimitiveWrapper(env=env, action_generator="BehaviorActionPrimitives", mode='baseline')
    set_random_seed(seed)
    env.reset()

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    os.makedirs(tensorboard_log_dir, exist_ok=True)

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
    '''
    load_path = 'log_dir/20221016-193306/_5000_steps'
    model = PPO.load(load_path)
    print(model.policy)
    for name, param in model.policy.named_parameters():
        print(name, param)
    model.set_env(env)
    print('Successfully RESUME from {}'.format(load_path))
    '''
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
    log.debug(model.policy)
    print(model)

    model.learn(total_timesteps=10000000, callback=checkpoint_callback,
                eval_env=env, eval_freq=1000,
                n_eval_episodes=20)

    model = PPO.load(os.path.join(tensorboard_log_dir, "ckpt_{}".format(prefix)))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    log.info(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()