import os

import numpy as np

import omnigibson
from omnigibson.robots.two_wheel_robot import TwoWheelRobot


class Locobot(TwoWheelRobot):
    """
    Locobot robot
    Reference: https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx
    """

    @property
    def model_name(self):
        return "Locobot"

    @property
    def wheel_radius(self):
        return 0.038

    @property
    def wheel_axle_length(self):
        return 0.230

    @property
    def base_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([1, 0])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def usd_path(self):
        return os.path.join(omnigibson.assets_path, "models/locobot/locobot/locobot.usd")

    @property
    def urdf_path(self):
        return os.path.join(omnigibson.assets_path, "models/locobot/locobot.urdf")
