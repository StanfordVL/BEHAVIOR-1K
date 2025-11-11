from functools import cached_property

import torch as th

from omnigibson.robots.two_wheel_robot import TwoWheelRobot


class Locobot(TwoWheelRobot):
    """
    Locobot robot
    Reference: https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
    """

    @property
    def wheel_radius(self):
        return 0.038

    @property
    def wheel_axle_length(self):
        return 0.230

    @cached_property
    def base_joint_names(self):
        return ["wheel_left_joint", "wheel_right_joint"]

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)
