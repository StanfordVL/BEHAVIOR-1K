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
        # Reward received the pointgoal success condition is met
        reward = self._r_pointgoal if self._pointgoal.success else 0.0

        return reward, {}
