import os
from omnigibson.macros import gm
from omnigibson.robots.franka import FrankaPanda
from omnigibson.utils.asset_utils import get_dataset_path


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
    def usd_path(self):
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"), "models/franka/franka_mounted/usd/franka_mounted.usda"
        )

    @property
    def urdf_path(self):
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"), "models/franka/franka_mounted/urdf/franka_mounted.urdf"
        )

    @property
    def curobo_path(self):
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"),
            "models/franka/franka_mounted/curobo/franka_mounted_description_curobo_default.yaml",
        )

    @property
    def _assisted_grasp_start_points(self):
        return None  # automatically inferred with this gripper

    @property
    def _assisted_grasp_end_points(self):
        return None  # automatically inferred with this gripper
