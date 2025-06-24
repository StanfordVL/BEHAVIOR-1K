import omnigibson.utils.transform_utils as T
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction


class ReachingGoalReward(BaseRewardFunction):
    """
    Reaching goal reward
    Success reward for reaching the goal with the robot's end-effector

    Args:
        robot_idn (int): robot identifier to evaluate point goal with. Default is 0, corresponding to the first
            robot added to the scene
        r_reach (float): reward for succeeding to reach the goal
        distance_tol (float): Distance (m) tolerance between goal position and @robot_idn's robot eef position
            that is accepted as a success
    """

    def __init__(self, robot_idn=0, r_reach=10.0, distance_tol=0.1):
        # Store internal vars
        self._robot_idn = robot_idn
        self._r_reach = r_reach
        self._distance_tol = distance_tol

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Sparse reward is received if distance between robot_idn robot's eef and goal is below the distance threshold
        success = T.l2_distance(env.robots[self._robot_idn].get_eef_position(), task.goal_pos) < self._distance_tol
        reward = self._r_reach if success else 0.0

        return reward, {}
