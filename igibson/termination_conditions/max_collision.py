from igibson.termination_conditions.termination_condition_base import FailureCondition


class MaxCollision(FailureCondition):
    """
    MaxCollision (failure condition) used for navigation tasks
    Episode terminates if the robot has collided more than
    max_collisions_allowed times

    Args:
        max_collisions (int): Maximum number of collisions allowed for any robots in the scene before a termination
            is triggered
    """

    def __init__(self, max_collisions=500):
        self._max_collisions = max_collisions

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        # Terminate if the robot has collided more than self._max_collisions times
        return env.collision_step > self._max_collisions
