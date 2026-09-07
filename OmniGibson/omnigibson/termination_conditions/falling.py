import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.termination_conditions.termination_condition_base import FailureCondition


class Falling(FailureCondition):
    """
    Falling (failure condition) used for any navigation-type tasks
    Episode terminates if the robot falls out of the world (i.e.: falls below the floor height by at least
    @fall_height

    Args:
        robot_idn (int): robot identifier to evaluate condition with. Default is 0, corresponding to the first
            robot added to the scene
        fall_height (float): distance (m) > 0 below the scene's floor height under which the the robot is considered
            to be falling out of the world
        topple (bool): whether to also consider the robot to be falling if it is toppling over (i.e.: if it is
            no longer upright
    """

    def __init__(self, robot_idn=0, fall_height=0.03, topple=True, tilt_tolerance=0.75):
        # Store internal vars
        self._robot_idn = robot_idn
        self._fall_height = fall_height
        self._topple = topple
        self._tilt_tolerance = tilt_tolerance

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        results = th.zeros(env.num_envs, dtype=th.bool)
        for env_idx in range(env.num_envs):
            robot = env.scenes[env_idx].robots[self._robot_idn]
            pos, quat = robot.get_position_orientation()
            robot_z = pos[2]
            floor_height = env.scenes[env_idx].get_floor_height()
            # Terminate if the specified robot is falling out of the scene
            fell = robot_z < (floor_height - self._fall_height)

            # Terminate if the robot has toppled over
            if not fell and self._topple:
                robot_up = T.quat_apply(quat, th.tensor([0, 0, 1], dtype=th.float32))
                fell = robot_up[2] < self._tilt_tolerance

            results[env_idx] = fell
        return results
