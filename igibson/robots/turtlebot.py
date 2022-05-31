import os

import numpy as np

import igibson
from igibson.robots.two_wheel_robot import TwoWheelRobot


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

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([1, 0])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/turtlebot/turtlebot/turtlebot.usd")
