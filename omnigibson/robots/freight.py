import os
import numpy as np

from omnigibson.macros import gm
from omnigibson.robots.two_wheel_robot import TwoWheelRobot


class Freight(TwoWheelRobot):
    """
    Freight Robot
    Reference: https://fetchrobotics.com/robotics-platforms/freight-base/
    Uses joint velocity control
    """

    @property
    def model_name(self):
        return "Freight"

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def base_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/freight/freight.usd")

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/freight.urdf")
