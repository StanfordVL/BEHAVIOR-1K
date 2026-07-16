"""Newton-native OmniGibson Environment entry point."""

from __future__ import annotations


class Environment:
    """Canonical OmniGibson environment facade backed by the Newton simulator."""

    def __init__(self, configs, in_vec_env=False):
        from omnigibson.simulator import Simulator

        self.configs = configs if isinstance(configs, (list, tuple)) else [configs]
        self.in_vec_env = in_vec_env
        self.render_mode = "human"
        self.metadata = {"render.modes": ["human"]}
        self.action_space = None
        self.observation_space = None

        env_cfg = self._merged_env_config(self.configs)
        self._flatten_action_space = bool(env_cfg.get("flatten_action_space", False))
        self._flatten_obs_space = bool(env_cfg.get("flatten_obs_space", False))
        self._current_episode = 0
        self._current_step = 0

        self.simulator = Simulator.from_environment_configs(self.configs)
        self._built = False
        self._closed = False
        self._initial_state = None

        if self.simulator.should_auto_build_environment(self.configs):
            self._ensure_built()

    @staticmethod
    def _merged_env_config(configs):
        from omnigibson.newton.config import load_newton_config

        env_cfg = {}
        for config in configs:
            env_cfg.update(load_newton_config(config).get("env") or {})
        return env_cfg

    def reset(self, get_obs=True, *, seed=None, options=None):
        if options:
            raise NotImplementedError("Newton Environment reset options are not implemented yet.")
        self._ensure_built()
        if seed is not None:
            self._seed(seed)
        self.simulator.load_state(self._initial_state)
        self._current_episode += 1
        self._current_step = 0
        if not get_obs:
            return None, {}
        return self.get_obs()

    @staticmethod
    def _seed(seed):
        import random

        import numpy as np
        import torch as th

        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

    def step(self, action=None, n_render_iterations=1):
        self._ensure_built()
        self.simulator.apply_environment_action(action)
        self.simulator.step()
        if self.simulator.viewer is not None:
            for _ in range(n_render_iterations):
                self.simulator.render()
        self._current_step += 1
        obs, info = self.get_obs()
        return obs, 0.0, False, False, info

    def get_obs(self):
        from omnigibson.utils.gym_utils import maxdim, recursively_generate_flat_dict

        self._ensure_built()
        obs = {}
        info = {}
        for robot in self.robots:
            if maxdim(robot.observation_space) > 0:
                obs[robot.name], info[robot.name] = robot.get_obs()
        if self._flatten_obs_space:
            obs = recursively_generate_flat_dict(dic=obs)
        return obs, info

    def load_observation_space(self):
        import gymnasium as gym

        from omnigibson.utils.gym_utils import maxdim, recursively_generate_flat_dict

        obs_space = {}
        for robot in self.robots:
            robot_obs = robot.load_observation_space()
            if maxdim(robot_obs) > 0:
                obs_space[robot.name] = robot_obs
        space = gym.spaces.Dict(obs_space)
        if self._flatten_obs_space:
            space = gym.spaces.Dict(recursively_generate_flat_dict(dic=space))
        self.observation_space = space
        return self.observation_space

    def _load_action_space(self):
        import gymnasium as gym
        import numpy as np

        action_space = gym.spaces.Dict({robot.name: robot.action_space for robot in self.robots})
        if self._flatten_action_space:
            lows = []
            highs = []
            for space in action_space.values():
                assert isinstance(
                    space, gym.spaces.Box
                ), "Can only flatten action space where all individual spaces are gym.space.Box instances!"
                lows.append(space.low)
                highs.append(space.high)
            if lows:
                action_space = gym.spaces.Box(np.concatenate(lows), np.concatenate(highs), dtype=np.float32)
            else:
                action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(0,), dtype=np.float32)
        self.action_space = action_space

    def render(self):
        self._ensure_built()
        if self.simulator.viewer is None:
            self.simulator.attach_viewer()
        self.simulator.render()

    def close(self):
        if not self._closed:
            self.simulator.close()
            self._closed = True
            self._built = False

    def summary(self):
        self._ensure_built()
        return self.simulator.summary()

    @property
    def scene(self):
        self._ensure_built()
        return self.simulator.scene

    @property
    def robots(self):
        self._ensure_built()
        return list(self.simulator.robots)

    @property
    def objects(self):
        self._ensure_built()
        return list(self.simulator.objects)

    def _ensure_built(self):
        if self._closed:
            raise RuntimeError("Cannot use a closed Environment.")
        if not self._built:
            self.simulator.build_environment()
            self._built = True
            # The reset target is the freshly built environment state.
            self._initial_state = self.simulator.dump_state()
            self.load_observation_space()
            self._load_action_space()

    def __enter__(self):
        self._ensure_built()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


__all__ = ["Environment"]
