from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class PotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)

    Args:
        potential_fcn (method): function for calculating potential. Function signature should be:

            potential = potential_fcn(env)

            where @env is a Environment instance, and @potential is a float value representing the calculated potential

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

    def reset(self, task, env):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        """
        # Reset potential
        self._potential = self._potential_fcn(env)

    def _step(self, task, env, action):
        # Reward is proportional to the potential difference between the current and previous timestep
        new_potential = self._potential_fcn(env)
        reward = (self._potential - new_potential) * self._r_potential

        # Update internal potential
        self._potential = new_potential

        return reward, {}
