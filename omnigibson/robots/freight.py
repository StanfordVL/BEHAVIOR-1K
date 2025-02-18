from functools import cached_property

import torch as th

from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


class Freight(TwoWheelRobot):
    """
    Freight Robot
    Reference: https://fetchrobotics.com/robotics-platforms/freight-base/
    Uses joint velocity control
    """

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @cached_property
    def base_joint_names(self):
        return ["l_wheel_joint", "r_wheel_joint"]

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)
