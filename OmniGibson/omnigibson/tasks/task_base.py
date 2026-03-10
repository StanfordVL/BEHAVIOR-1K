from abc import ABCMeta, abstractmethod
from copy import deepcopy

import torch as th

from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import Registerable, classproperty

REGISTERED_TASKS = dict()


class BaseTask(GymObservable, Registerable, metaclass=ABCMeta):
    """
    Base Task class.
    Task-specific reset_scene, reset_agent, step methods are implemented in subclasses

    Args:
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
        include_obs (bool): Whether to include observations or not for this task
    """

    def __init__(self, termination_config=None, reward_config=None, include_obs=True):
        # Make sure configs are dictionaries
        termination_config = dict() if termination_config is None else termination_config
        reward_config = dict() if reward_config is None else reward_config

        # Sanity check termination and reward conditions -- any keys found in the inputted config but NOT
        # found in the default config should raise an error
        unknown_termination_keys = set(termination_config.keys()) - set(self.default_termination_config.keys())
        assert (
            len(unknown_termination_keys) == 0
        ), f"Got unknown termination config keys inputted: {unknown_termination_keys}"
        unknown_reward_keys = set(reward_config.keys()) - set(self.default_reward_config.keys())
        assert len(unknown_reward_keys) == 0, f"Got unknown reward config keys inputted: {unknown_reward_keys}"

        # Combine with defaults and store internally
        self._termination_config = self.default_termination_config
        self._termination_config.update(termination_config)
        self._reward_config = self.default_reward_config
        self._reward_config.update(reward_config)

        # Generate reward and termination functions
        self._termination_conditions = self._create_termination_conditions()
        self._reward_functions = self._create_reward_functions()

        # Store other internal vars that will be populated at runtime
        self._loaded = False
        self._num_envs = None
        self._reward = None
        self._done = None
        self._success = None
        self._info = None
        self._low_dim_obs_dim = None
        self._low_dim_obs_keys = None
        self._include_obs = include_obs

        # Run super init
        super().__init__()

    @abstractmethod
    def _load(self, env):
        """
        Load this task. Should be implemented by subclass. Can include functionality, e.g.: loading dynamic objects
        into the environment
        """
        raise NotImplementedError()

    @abstractmethod
    def _load_non_low_dim_observation_space(self):
        """
        Loads any non-low dim observation spaces for this task.

        Returns:
            dict: Keyword-mapped observation space for this object mapping non low dim task observation name to
                observation space
        """
        raise NotImplementedError()

    @classmethod
    def verify_scene_and_task_config(cls, scene_cfg, task_cfg):
        """
        Runs any necessary sanity checks on the scene and task configs passed; and possibly modifies them in-place

        Args:
            scene_cfg (dict): Scene configuration
            task_cfg (dict): Task configuration
        """
        # Default is no-op
        pass

    def _load_observation_space(self):
        # Create the non low dim obs space
        obs_space = self._load_non_low_dim_observation_space()

        # Create the low dim obs space and add to the main obs space dict -- make sure we're flattening low dim obs
        if self._low_dim_obs_dim > 0:
            obs_space["low_dim"] = self._build_obs_box_space(
                shape=(self._low_dim_obs_dim,), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
            )

        return obs_space

    def load(self, env):
        """
        Load this task
        """
        self._num_envs = env.num_envs

        # Make sure all scenes are of the correct type!
        for scene in env.scenes:
            assert any([issubclass(scene.__class__, valid_cls) for valid_cls in self.valid_scene_types]), (
                f"Got incompatible scene type {scene.__class__.__name__} for task {self.__class__.__name__}! "
                f"Scene class must be a subclass of at least one of: "
                f"{[cls_type.__name__ for cls_type in self.valid_scene_types]}"
            )

        # Run internal method
        self._load(env=env)

        # We're now initialized
        self._loaded = True

    def post_play_load(self, env):
        """
        Complete any loading tasks that require the simulator to be playing

        Args:
            env (Environment): environment instance
        """
        # Reset all scenes to their initial stored configuration
        for scene in env.scenes:
            scene.reset(hard=False)

        # Compute the low dimensional observation dimension (use env_idx=0 as representative)
        obs = self.get_obs(env=env, env_idx=0, flatten_low_dim=False)
        if "low_dim" in obs:
            self._low_dim_obs_keys = list(obs["low_dim"].keys())
            self._low_dim_obs_dim = len(self._flatten_low_dim_obs(obs=obs["low_dim"]))
        else:
            self._low_dim_obs_keys = []
            self._low_dim_obs_dim = 0

    @property
    def low_dim_obs_keys(self):
        """
        Returns:
            list of str: List of low-dimensional observation keys for this task
        """
        # Make sure we're loaded
        assert self._loaded, "Task must be loaded using load() before accessing low_dim_obs_keys!"
        return self._low_dim_obs_keys

    @abstractmethod
    def _create_termination_conditions(self):
        """
        Creates the termination functions in the environment

        Returns:
            dict of BaseTerminationCondition: Termination functions created for this task
        """
        raise NotImplementedError()

    @abstractmethod
    def _create_reward_functions(self):
        """
        Creates the reward functions in the environment

        Returns:
            dict of BaseRewardFunction: Reward functions created for this task
        """
        raise NotImplementedError()

    def _reset_scene(self, env, env_indices):
        """
        Task-specific scene reset. Default is resetting specified scenes.

        Args:
            env (Environment): environment instance
            env_indices (th.Tensor): Indices of environments to reset
        """
        for idx in env_indices:
            env.scenes[idx].reset()

    def _reset_agent(self, env, env_indices):
        """
        Task-specific agent reset

        Args:
            env (Environment): environment instance
            env_indices (th.Tensor): Indices of environments to reset
        """
        # Default is no-op
        pass

    def _reset_variables(self, env, env_indices):
        """
        Task-specific internal variable reset

        Args:
            env (Environment): environment instance
            env_indices (th.Tensor): Indices of environments to reset
        """
        if self._reward is None:
            # First-time init
            self._reward = th.zeros(self._num_envs, dtype=th.float32)
            self._done = th.zeros(self._num_envs, dtype=th.bool)
            self._success = th.zeros(self._num_envs, dtype=th.bool)
        self._reward[env_indices] = 0.0
        self._done[env_indices] = False
        self._success[env_indices] = False
        self._info = None  # info is rebuilt each step

    def reset(self, env, env_indices=None):
        """
        Resets this task in the environment

        Args:
            env (Environment): environment instance to reset
            env_indices (None or th.Tensor): Indices of environments to reset. If None, resets all.
        """
        if env_indices is None:
            env_indices = th.arange(self._num_envs)

        # Reset the scene, agent, and variables
        self._reset_scene(env, env_indices)
        self._reset_agent(env, env_indices)
        self._reset_variables(env, env_indices)

        # Also reset all termination conditions and reward functions
        for termination_condition in self._termination_conditions.values():
            termination_condition.reset(self, env, env_indices)
        for reward_function in self._reward_functions.values():
            reward_function.reset(self, env, env_indices)

    def _step_termination(self, env, action, infos=None):
        """
        Step and aggregate termination conditions

        Args:
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment
            infos (None or list[dict]): If provided, a list of per-env info dicts to accumulate into.
                If None, fresh info dicts are created.

        Returns:
            2-tuple:
                - th.Tensor: (num_envs,) bool tensor of dones
                - list[dict]: per-env info dicts
        """
        # Get all dones and successes from individual termination conditions
        if infos is not None:
            for i in range(self._num_envs):
                if "termination_conditions" not in infos[i]:
                    infos[i]["termination_conditions"] = dict()
        else:
            infos = [{"termination_conditions": dict()} for _ in range(self._num_envs)]
        dones = th.zeros(self._num_envs, dtype=th.bool)
        successes = th.zeros(self._num_envs, dtype=th.bool)

        for name, termination_condition in self._termination_conditions.items():
            d, s = termination_condition.step(self, env, action)
            dones = dones | d
            successes = successes | s
            for env_idx in range(self._num_envs):
                infos[env_idx]["termination_conditions"][name] = {
                    "done": d[env_idx].item(),
                    "success": s[env_idx].item(),
                }

        for env_idx in range(self._num_envs):
            infos[env_idx]["success"] = successes[env_idx].item()

        return dones, infos

    def _step_reward(self, env, action, infos=None):
        """
        Step and aggregate reward functions

        Args:
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment
            infos (None or list[dict]): If provided, a list of per-env info dicts to accumulate into.
                If None, fresh info dicts are created.

        Returns:
            2-tuple:
                - th.Tensor: (num_envs,) float tensor of total rewards
                - list[dict]: per-env info dicts
        """
        # Aggregate rewards over all reward functions
        infos = infos if infos is not None else [dict() for _ in range(self._num_envs)]
        total_rewards = th.zeros(self._num_envs, dtype=th.float32)

        for reward_name, reward_function in self._reward_functions.items():
            rewards, reward_infos = reward_function.step(self, env, action)
            total_rewards += rewards
            for env_idx in range(self._num_envs):
                infos[env_idx][reward_name] = reward_infos[env_idx]
                if "reward_breakdown" not in infos[env_idx]:
                    infos[env_idx]["reward_breakdown"] = dict()
                infos[env_idx]["reward_breakdown"][reward_name] = rewards[env_idx].item()

        return total_rewards, infos

    @abstractmethod
    def _get_obs(self, env, env_idx):
        """
        Get task-specific observation for a specific environment

        Args:
            env (Environment): Environment instance
            env_idx (int): Index of the environment

        Returns:
            2-tuple:
                - dict: Keyword-mapped low dimensional observations from this task
                - dict: All other keyword-mapped observations from this task
        """
        raise NotImplementedError()

    def _flatten_low_dim_obs(self, obs):
        """
        Flattens dictionary containing low-dimensional observations @obs and converts it from a dictionary into a
        1D numpy array

        Args:
            obs (dict): Low-dim observation dictionary where each value is a 1D array

        Returns:
            n-array: 1D-numpy array of flattened low-dim observations
        """
        # By default, we simply concatenate all values in our obs dict
        return th.cat([ob for ob in obs.values()]) if len(obs.values()) > 0 else th.empty(0)

    def get_obs(self, env, env_idx=None, flatten_low_dim=True):
        """
        Get task observations.

        Args:
            env (Environment): environment instance
            env_idx (int or None): If specified, return obs for one env. If None, return list for all envs.
            flatten_low_dim (bool): Whether to flatten low-dimensional observations
        """
        if not self._include_obs:
            return dict() if env_idx is not None else [dict()] * self._num_envs

        if env_idx is not None:
            low_dim_obs, obs = self._get_obs(env=env, env_idx=env_idx)
            if low_dim_obs:
                obs["low_dim"] = self._flatten_low_dim_obs(obs=low_dim_obs) if flatten_low_dim else low_dim_obs
            return obs
        else:
            return [self.get_obs(env, env_idx=i, flatten_low_dim=flatten_low_dim) for i in range(self._num_envs)]

    def step(self, env, action):
        """
        Perform task-specific step for every timestep

        Args:
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment

        Returns:
            3-tuple:
                - th.Tensor: (num_envs,) reward tensor
                - th.Tensor: (num_envs,) done bool tensor
                - list[dict]: per-env info dicts
        """
        # Make sure we're initialized
        assert self._loaded, "Task must be loaded using load() before calling step()!"

        # We calculate termination conditions first and then rewards
        # (since some rewards can rely on termination conditions to update)
        dones, done_infos = self._step_termination(env=env, action=action)
        rewards, reward_infos = self._step_reward(env=env, action=action)

        # Update the internal state of this task
        self._reward = rewards
        self._done = dones
        self._success = th.tensor([di["success"] for di in done_infos], dtype=th.bool)
        self._info = [
            {"reward": reward_infos[i], "done": done_infos[i]}
            for i in range(self._num_envs)
        ]

        return self._reward, self._done, deepcopy(self._info)

    @property
    def name(self):
        """
        Returns:
            str: Name of this task. Defaults to class name
        """
        return self.__class__.__name__

    @property
    def reward(self):
        """
        Returns:
            th.Tensor: (num_envs,) current reward tensor for this task
        """
        assert self._reward is not None, "At least one step() must occur before reward can be calculated!"
        return self._reward

    @property
    def done(self):
        """
        Returns:
            th.Tensor: (num_envs,) bool tensor of whether each env is done
        """
        assert self._done is not None, "At least one step() must occur before done can be calculated!"
        return self._done

    @property
    def success(self):
        """
        Returns:
            th.Tensor: (num_envs,) bool tensor of whether each env has succeeded
        """
        assert self._success is not None, "At least one step() must occur before success can be calculated!"
        return self._success

    @property
    def info(self):
        """
        Returns:
            list[dict]: Per-env nested dictionary of information for this task, including reward- and done-specific information
        """
        assert self._info is not None, "At least one step() must occur before info can be calculated!"
        return self._info

    @classproperty
    def valid_scene_types(cls):
        """
        Returns:
            set of Scene: Scene type(s) that are valid (i.e.: compatible) with this specific task. This will be
                used to sanity check the task + scene combination at runtime
        """
        raise NotImplementedError()

    @classproperty
    def default_reward_config(cls):
        """
        Returns:
            dict: Default reward configuration for this class. Should include any kwargs necessary for
                any of the reward classes generated in self._create_rewards(). Note: this default config
                should be fully verbose -- any keys inputted in the constructor but NOT found in this default config
                will raise an error!
        """
        raise NotImplementedError()

    @classproperty
    def default_termination_config(cls):
        """
        Returns:
            dict: Default termination configuration for this class. Should include any kwargs necessary for
                any of the termination classes generated in self._create_terminations(). Note: this default config
                should be fully verbose -- any keys inputted in the constructor but NOT found in this default config
                will raise an error!
        """
        raise NotImplementedError()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseTask")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_TASKS
        return REGISTERED_TASKS
