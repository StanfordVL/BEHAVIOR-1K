import copy

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv

import omnigibson as og
from omnigibson.envs.env_base import Environment


class SB3VectorEnvironment(DummyVecEnv):
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        self.env_fns = [lambda: Environment(configs=copy.deepcopy(config), num_env=i) for i in range(num_envs)]
        super().__init__(self.env_fns)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        for i, action in enumerate(actions):
            self.envs[i]._pre_step(action)

    def step_wait(self) -> VecEnvStepReturn:
        # Step the entire simulation
        og.sim.step()

        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx]._post_step(
                self.actions[env_idx]
            )
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), copy.deepcopy(self.buf_infos))
