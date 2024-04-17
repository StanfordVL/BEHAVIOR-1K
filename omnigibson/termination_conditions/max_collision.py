from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.termination_conditions.termination_condition_base import FailureCondition


class MaxCollision(FailureCondition):
    """
    MaxCollision (failure condition) used for navigation tasks
    Episode terminates if the robot has collided more than max_collisions_allowed times
    Note that we ignore collisions with any floor objects.

    Args:
        robot_idn (int): robot identifier to evaluate collision checking with. Default is 0, corresponding to the first
            robot added to the scene
        ignore_self_collisions (bool): Whether to ignore robot self-collisions or not
        max_collisions (int): Maximum number of collisions allowed for any robots in the scene before a termination
            is triggered
    """

    def __init__(self, robot_idn=0, ignore_self_collisions=True, max_collisions=500):
        self._robot_idn = robot_idn
        self._ignore_self_collisions = ignore_self_collisions
        self._max_collisions = max_collisions
        self._n_collisions = 0

        # Run super init
        super().__init__()

    def reset(self, task, env):
        # Call super first
        super().reset(task, env)

        # Also reset collision counter
        self._n_collisions = 0

    def _step(self, task, env, action):
        # Terminate if the robot has collided more than self._max_collisions times
        robot = env.robots[self._robot_idn]
        floors = list(env.scene.object_registry("category", "floors", []))
        ignore_objs = floors if self._ignore_self_collisions is None else floors + [robot]
        in_contact = (
            len(env.robots[self._robot_idn].states[ContactBodies].get_value(ignore_objs=tuple(ignore_objs))) > 0
        )
        self._n_collisions += int(in_contact)
        return self._n_collisions > self._max_collisions
