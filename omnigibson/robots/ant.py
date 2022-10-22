import os

import numpy as np

import omnigibson
from omnigibson.robots.locomotion_robot import LocomotionRobot


class Ant(LocomotionRobot):
    """
    OpenAI Ant Robot
    """

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Ant"

    def _create_discrete_action_space(self):
        raise ValueError("Ant does not support discrete actions!")

    @property
    def base_control_idx(self):
        return np.array([0, 1, 2, 3, 4, 5, 6, 7])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def usd_path(self):
        return os.path.join(omnigibson.assets_path, "models/ant/ant.xml")

    @property
    def urdf_path(self):
        return os.path.join(omnigibson.assets_path, "models/ant/ant.xml")
