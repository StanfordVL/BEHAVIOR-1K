from functools import cached_property

import torch as th

from omnigibson.robots.two_wheel_robot import TwoWheelRobot


class Turtlebot(TwoWheelRobot):
    """
    Turtlebot robot
    Reference: http://wiki.ros.org/Robots/TurtleBot
    Uses joint velocity control
    """

    @property
    def wheel_radius(self):
        return 0.038

    @property
    def wheel_axle_length(self):
        return 0.23

    @cached_property
    def base_joint_names(self):
        return ["wheel_left_joint", "wheel_right_joint"]

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)
