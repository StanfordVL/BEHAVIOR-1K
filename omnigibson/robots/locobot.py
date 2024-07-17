import os

import torch as th

from omnigibson.macros import gm
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

    @property
    def base_joint_names(self):
        return ["wheel_right_joint", "wheel_left_joint"]

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/locobot/locobot/locobot.usd")

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/locobot/locobot.urdf")
