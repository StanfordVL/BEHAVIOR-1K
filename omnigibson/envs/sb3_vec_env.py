import copy

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from tqdm import trange

import omnigibson as og
from omnigibson.envs.env_base import Environment
import time


class SB3VectorEnvironment(DummyVecEnv):
    def __init__(self, num_envs, config):
        self.num_envs = num_envs

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        tmp_envs = [
            Environment(configs=copy.deepcopy(config), in_vec_env=True)
            for _ in trange(num_envs, desc="Loading environments")
        ]

        # Play, and finish loading all the envs
        og.sim.play()
        for env in tmp_envs:
            env.post_play_load()

        # Now produce some functions that will make DummyVecEnv think it's creating these envs itself
        env_fns = [lambda env_=env: env_ for env in tmp_envs]
        super().__init__(env_fns)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        for i, action in enumerate(actions):
            self.envs[i]._pre_step(action)

    def step_wait(self) -> VecEnvStepReturn:
        start_time = time.time()
        # Step the entire simulation
        og.sim.step()
        fps = 1 / (time.time() - start_time)
        print("internal fps:", fps)

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

    def reset(self):
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            self.envs[env_idx].reset(get_obs=False, seed=self._seeds[env_idx], **maybe_options)

        # Settle the robots
        for _ in range(30):
            og.sim.step()

        # Get the new obs
        for env_idx in range(self.num_envs):
            obs, info = self.envs[env_idx].get_obs()
            self._save_obs(env_idx, obs)

        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()
