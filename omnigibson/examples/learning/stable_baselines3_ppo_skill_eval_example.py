# https://github.com/StanfordVL/behavior/blob/main/behavior/baselines/rl/stable_baselines3_ppo_training.py

import logging
import os
from typing import Callable
import pdb
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
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3_ppo_skill_example import CustomCombinedExtractor

except ModuleNotFoundError:
    log.error("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""


def main():
    config_file = f"{example_config_path}/behavior_mp_tiago.yaml"  # os.path.join(igibson.configs_path, "fetch_rl_cleaning_microwave_oven.yaml")
    tensorboard_log_dir = "log_dir"

    seed = 0
    env = ig.Environment(configs=config_file, action_timestep=1 / 60., physics_timestep=1 / 60.)
    env = ActionPrimitiveWrapper(env=env, action_generator="BehaviorActionPrimitives", mode='baseline')
    ceiling = env.scene.object_registry("name", "ceilings")
    ceiling.visible = False
    env.seed(seed)
    set_random_seed(seed)
    env.reset()

    os.makedirs(tensorboard_log_dir, exist_ok=True)
    load_path = 'log_dir/20220604-153916/_43000_steps'
    model = PPO.load(load_path)
    print(model.policy)
    pdb.set_trace()
    for name, param in model.policy.named_parameters():
        print(name, param)
    print('Successfully loaded from {}'.format(load_path))
    log.debug(model.policy)
    print('Evaluating Started ...')
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print('Evaluating Finished ...')
    log.info(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
