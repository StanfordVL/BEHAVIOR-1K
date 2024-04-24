import copy

import omnigibson as og
import warnings
from copy import deepcopy
from typing import Any, List, Type

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


# TODO: Figure out if there is a good interface to implement in Gymnasium
class VectorEnvironment(DummyVecEnv):

    # def __init__(self, num_envs, config):
    #     self.num_envs = num_envs
    #     self.envs = [og.Environment(configs=copy.deepcopy(config)) for _ in range(num_envs)]

    # def step(self, actions):
    #     try:
    #         observations, rewards, dones, infos = [], [], [], []
    #         for i, action in enumerate(actions):
    #             self.envs[i]._pre_step(action)
    #         # Run simulation step
    #         og.sim.step()
    #         for i, action in enumerate(actions):
    #             obs, reward, done, info = self.envs[i]._post_step(action)
    #             observations.append(obs)
    #             rewards.append(reward)
    #             dones.append(done)
    #             infos.append(info)
    #         return observations, rewards, dones, infos
    #     except Exception as e:
    #         print(e)


    def __init__(self, num_envs, config):
        self.waiting = False
        self.render_mode = "rgb_array"

        self.num_envs = num_envs
        self.envs = [og.Environment(configs=copy.deepcopy(config)) for _ in range(num_envs)]

        super().__init__([(lambda x=x: x) for x in self.envs])

    def step_wait(self) -> VecEnvStepReturn:
        for i, action in enumerate(self.actions):
            self.envs[i]._pre_step(action)

        # Run simulation step
        og.sim.step()

        step = lambda env_i, act_i: env_i._post_step(act_i)
        for env_idx, result in enumerate(map(step, self.envs, self.actions)):
            obs, reward, terminated, truncated, info = result

            done = terminated or truncated
            info["TimeLimit.truncated"] = truncated and not terminated

            reset_infos = {}
            if done:
                info["terminal_observation"] = obs
                obs, reset_infos = self.envs[env_idx].reset()
            
            self._save_obs(env_idx, obs)
            self.buf_rews[env_idx] = reward
            self.buf_dones[env_idx] = done
            self.buf_infos[env_idx] = info
            self.reset_infos[env_idx] = reset_infos
        
        # from IPython import embed; embed()

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    # def reset(self):
    #     reset = lambda env_i, seed_i, opts_i: env_i.reset(seed=seed_i, **opts_i)

    #     maybe_options = [
    #         {"options": self._options[env_idx]} if self._options[env_idx] else {} for env_idx in range(self.num_envs)
    #     ]
    #     for env_idx, (obs, reset_info) in enumerate(map(reset, self.envs, self._seeds, maybe_options)):
    #         self.reset_infos[env_idx] = reset_info
    #         self._save_obs(env_idx, obs)

    #     self._reset_seeds()
    #     self._reset_options()
    #     return self._obs_from_buf()
