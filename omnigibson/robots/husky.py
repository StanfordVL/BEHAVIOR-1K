from functools import cached_property

import torch as th

from omnigibson.robots.locomotion_robot import LocomotionRobot


class Husky(LocomotionRobot):
    """
    Husky robot
    Reference: https://clearpathrobotics.com/, http://wiki.ros.org/Robots/Husky
    """

    def _create_discrete_action_space(self):
        raise ValueError("Husky does not support discrete actions!")

    @property
    def wheel_radius(self):
        return 0.165

    @property
    def wheel_axle_length(self):
        return 0.670

    @cached_property
    def base_joint_names(self):
        return ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)
