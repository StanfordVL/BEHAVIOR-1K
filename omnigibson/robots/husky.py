import os

import torch as th

from omnigibson.macros import gm
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

    @property
    def base_control_idx(self):
        return th.Tensor([0, 1, 2, 3])

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/husky/husky/husky.usd")

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/husky/husky.urdf")
