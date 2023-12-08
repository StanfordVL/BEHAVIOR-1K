import numpy as np
from scipy.spatial.transform import Rotation as R
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

    def __init__(self, robot_idn=0, fall_height=0.03, topple=True):
        # Store internal vars
        self._robot_idn = robot_idn
        self._fall_height = fall_height
        self._topple = topple

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        # Terminate if the specified robot is falling out of the scene
        robot_z = env.scene.robots[self._robot_idn].get_position()[2]
        if robot_z < (env.scene.get_floor_height() - self._fall_height):
            return True
        
        # Terminate if the robot has toppled over
        if self._topple:
            rotation = R.from_quat(env.scene.robots[self._robot_idn].get_orientation())
            robot_up = rotation.apply(np.array([0, 0, 1]))
            if robot_up[2] < 0.75:
                return True
            
        return False
