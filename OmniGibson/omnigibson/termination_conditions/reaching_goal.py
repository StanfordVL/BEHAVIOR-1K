import omnigibson.utils.transform_utils as T
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition


class ReachingGoal(SuccessCondition):
    """
    ReachingGoal (success condition) used for reaching-type tasks
    Episode terminates if reaching goal is reached within @distance_tol by the @robot_idn robot's end effector

    Args:

    Args:
        robot_idn (int): robot identifier to evaluate point goal with. Default is 0, corresponding to the first
            robot added to the scene
        distance_tol (float): Distance (m) tolerance between goal position and @robot_idn's robot eef position
            that is accepted as a success
    """

    def __init__(self, robot_idn=0, distance_tol=0.5):
        self._robot_idn = robot_idn
        self._distance_tol = distance_tol

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        # Terminate if point goal is reached (distance below threshold)
        return T.l2_distance(env.scene.robots[self._robot_idn].get_eef_position(), task.goal_pos) < self._distance_tol
