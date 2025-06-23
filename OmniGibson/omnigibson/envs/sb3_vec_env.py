import copy
import time

import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from tqdm import trange

import omnigibson as og
from omnigibson.envs.env_base import Environment

# Keep track of the last used env and what time, to require that others be reset before getting used
last_stepped_env = None
last_stepped_time = None


class SB3VectorEnvironment(DummyVecEnv):
    def __init__(self, num_envs, config, render_on_step):
        self.num_envs = num_envs
        self.render_on_step = render_on_step

        if og.sim is not None:
            og.sim.stop()

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

        # Keep track of our last reset time
        self.last_reset_time = time.time()

    def step_async(self, actions: th.tensor) -> None:
        # We go into this context in case the pre-step tries to call step / render
        with og.sim.render_on_step(self.render_on_step):
            global last_stepped_env, last_stepped_time

            if last_stepped_env != self:
                # If another environment was used after us, we need to check that we have been reset after that.
                # Consider the common setup where you have a train env and an eval env in the same process.
                # When you step the eval env, the physics state of the train env also gets stepped,
                # despite the train env not taking new actions or outputting new observations.
                # By the time you next step the train env your state has drastically changed.
                # To avoid this from happening, we add a requirement: you can only be stepping
                # one vector env at a time - if you want to step another one, you need to reset it first.
                assert (
                    last_stepped_time is None or self.last_reset_time > last_stepped_time
                ), "You must call reset() before using a different environment."
                last_stepped_env = self
                last_stepped_time = time.time()

            self.actions = actions
            for i, action in enumerate(actions):
                self.envs[i]._pre_step(action)

    def step_wait(self) -> VecEnvStepReturn:
        with og.sim.render_on_step(self.render_on_step):
            # Step the entire simulation
            og.sim.step()

            for env_idx in range(self.num_envs):
                obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[
                    env_idx
                ]._post_step(self.actions[env_idx])
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

            return (
                self._obs_from_buf(),
                th.clone(self.buf_rews),
                th.clone(self.buf_dones),
                copy.deepcopy(self.buf_infos),
            )

    def reset(self):
        with og.sim.render_on_step(self.render_on_step):
            self.last_reset_time = time.time()

            for env_idx in range(self.num_envs):
                maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
                self.envs[env_idx].reset(get_obs=False, seed=self._seeds[env_idx], **maybe_options)

            # Settle the environments
            # TODO: fix this once we make the task classes etc. vectorized
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
