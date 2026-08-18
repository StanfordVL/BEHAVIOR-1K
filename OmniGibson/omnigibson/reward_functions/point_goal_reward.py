import torch as th

from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class PointGoalReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base

    Args:
        pointgoal (PointGoal): Termination condition for checking whether a point goal is reached
        r_pointgoal (float): Reward for reaching the point goal
    """

    def __init__(self, pointgoal, r_pointgoal=10.0):
        # Store internal vars
        self._pointgoal = pointgoal
        self._r_pointgoal = r_pointgoal

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Reward received when the pointgoal success condition is met
        # self._pointgoal.success is (num_envs,) bool tensor
        rewards = th.where(self._pointgoal.success, self._r_pointgoal, 0.0)
        infos = [dict() for _ in range(env.num_envs)]
        return rewards, infos
