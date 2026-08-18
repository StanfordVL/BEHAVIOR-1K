import torch as th

from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class PotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)

    Args:
        potential_fcn (method): function for calculating potential. Function signature should be:

            potential = potential_fcn(env, env_idx)

            where @env is an Environment instance, @env_idx is the environment index,
            and @potential is a float value representing the calculated potential

        r_potential (float): Reward weighting to give proportional to the potential difference calculated
            in between env timesteps
    """

    def __init__(self, potential_fcn, r_potential=1.0):
        # Store internal vars
        self._potential_fcn = potential_fcn
        self._r_potential = r_potential

        # Store internal vars that will be filled in at runtime
        self._potential = None

        # Run super
        super().__init__()

    def reset(self, task, env, env_indices):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        :param env_indices: indices of environments to reset
        """
        super().reset(task, env, env_indices)
        # Reset potential
        if self._potential is None:
            self._potential = th.zeros(env.num_envs, dtype=th.float32)
        for idx in env_indices:
            self._potential[idx] = self._potential_fcn(env, idx)

    def _step(self, task, env, action):
        # Reward is proportional to the potential difference between the current and previous timestep
        rewards = th.zeros(env.num_envs, dtype=th.float32)
        infos = [dict() for _ in range(env.num_envs)]
        for env_idx in range(env.num_envs):
            new_potential = self._potential_fcn(env, env_idx)
            rewards[env_idx] = (self._potential[env_idx] - new_potential) * self._r_potential
            # Update internal potential
            self._potential[env_idx] = new_potential
        return rewards, infos
