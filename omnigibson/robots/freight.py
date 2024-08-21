import os

import torch as th

from omnigibson.macros import gm
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


class Freight(TwoWheelRobot):
    """
    Freight Robot
    Reference: https://fetchrobotics.com/robotics-platforms/freight-base/
    Uses joint velocity control
    """

    def _post_load(self):
        super()._post_load()

        # Set the wheels back to using sphere approximations
        for wheel_name in ["l_wheel_link", "r_wheel_link"]:
            log.warning(
                "Freight wheel links are post-processed to use sphere approximation collision meshes. "
                "Please ignore any previous errors about these collision meshes."
            )
            wheel_link = self.links[wheel_name]
            assert set(wheel_link.collision_meshes) == {"collisions"}, "Wheel link should only have 1 collision!"
            wheel_link.collision_meshes["collisions"].set_collision_approximation("boundingSphere")

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def base_joint_names(self):
        return ["r_wheel_joint", "l_wheel_joint"]

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/freight/freight.usd")

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/freight.urdf")
