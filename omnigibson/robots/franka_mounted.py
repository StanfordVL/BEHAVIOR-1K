import os

import torch as th

from omnigibson.macros import gm
from omnigibson.robots.franka import FrankaPanda
from omnigibson.robots.manipulation_robot import GraspingPoint


class FrankaMounted(FrankaPanda):
    """
    The Franka Emika Panda robot mounted on a custom chassis with a custom gripper
    """

    @property
    def _raw_controller_order(self):
        return [f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers[f"arm_{self.default_arm}"] = "InverseKinematicsController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"
        return controllers

    @property
    def eef_to_fingertip_lengths(self):
        return {arm: {name: 0.15 for name in names} for arm, names in self.finger_link_names.items()}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_mounted.usd")

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_mounted.urdf")

    @property
    def curobo_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_mounted_description_curobo.yaml")

    @property
    def _assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="panda_rightfinger", position=th.tensor([0.0, 0.001, 0.045])),
            ]
        }

    @property
    def _assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="panda_leftfinger", position=th.tensor([0.0, 0.001, 0.045])),
            ]
        }
