import os

from omnigibson.macros import gm
from omnigibson.robots.franka import FrankaPanda
from omnigibson.robots.manipulation_robot import GraspingPoint


class FrankaMounted(FrankaPanda):
    """
    The Franka Emika Panda robot mounted on a custom chassis with a custom gripper
    """

    @property
    def controller_order(self):
        return ["arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.15}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_mounted.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/franka/franka_mounted_description.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_mounted.urdf")

    @property
    def curobo_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_mounted_description_curobo.yaml")

    @property
    def eef_usd_path(self):
        # TODO: Update!
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/franka/franka_panda_eef.usd")}

    @property
    def assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="panda_rightfinger", position=th.tensor([0.0, 0.001, 0.045])),
            ]
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="panda_leftfinger", position=th.tensor([0.0, 0.001, 0.045])),
            ]
        }
