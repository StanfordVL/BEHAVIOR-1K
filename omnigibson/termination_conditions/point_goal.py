import omnigibson.utils.transform_utils as T
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition


class PointGoal(SuccessCondition):
    """
    PointGoal (success condition) used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached within @distance_tol by the @robot_idn robot's base

    Args:
        robot_idn (int): robot identifier to evaluate point goal with. Default is 0, corresponding to the first
            robot added to the scene
        distance_tol (float): Distance (m) tolerance between goal position and @robot_idn's robot base position
            that is accepted as a success
        distance_axes (str): Which axes to calculate distances when calculating the goal. Any combination of "x",
            "y", and "z" is valid (e.g.: "xy" or "xyz" or "y")
    """

    def __init__(self, robot_idn=0, distance_tol=0.5, distance_axes="xyz"):
        self._robot_idn = robot_idn
        self._distance_tol = distance_tol
        self._distance_axes = [i for i, axis in enumerate("xyz") if axis in distance_axes]

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        # Make sure task is of type PointNavigation -- we import at runtime to avoid circular imports
        from omnigibson.tasks.point_navigation_task import PointNavigationTask

        assert isinstance(
            task, PointNavigationTask
        ), f"Cannot use {self.__class__.__name__} with a non-PointNavigationTask task instance!"
        # Terminate if point goal is reached (distance below threshold)
        return (
            T.l2_distance(task.get_current_pos(env)[self._distance_axes], task.get_goal_pos()[self._distance_axes])
            < self._distance_tol
        )
