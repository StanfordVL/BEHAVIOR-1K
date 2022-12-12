from abc import abstractmethod
import numpy as np

from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.python_utils import classproperty


class ActiveCameraRobot(BaseRobot):
    """
    Robot that is is equipped with an onboard camera that can be moved independently from the robot's other kinematic
    joints (e.g.: independent of base and arm for a mobile manipulator).

    NOTE: controller_config should, at the minimum, contain:
        camera: controller specifications for the controller to control this robot's camera.
            Should include:

            - name: Controller to create
            - <other kwargs> relevant to the controller being created. Note that all values will have default
                values specified, but setting these individual kwargs will override them

    """

    def _validate_configuration(self):
        # Make sure a camera controller is specified
        assert (
            "camera" in self._controllers
        ), "Controller 'camera' must exist in controllers! Current controllers: {}".format(
            list(self._controllers.keys())
        )

        # run super
        super()._validate_configuration()

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add camera pos info
        joint_positions = self.get_joint_positions(normalized=False)
        joint_velocities = self.get_joint_velocities(normalized=False)
        dic["camera_qpos"] = joint_positions[self.camera_control_idx]
        dic["camera_qpos_sin"] = np.sin(joint_positions[self.camera_control_idx])
        dic["camera_qpos_cos"] = np.cos(joint_positions[self.camera_control_idx])
        dic["camera_qvel"] = joint_velocities[self.camera_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["camera_qpos_sin", "camera_qpos_cos"]

    @property
    def controller_order(self):
        # By default, only camera is supported
        return ["camera"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # For best generalizability use, joint controller as default
        controllers["camera"] = "JointController"

        return controllers

    @property
    def _default_camera_joint_controller_config(self):
        """
        Returns:
            dict: Default camera joint controller config to control this robot's camera
        """
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "motor_type": "velocity",
            "control_limits": self.control_limits,
            "dof_idx": self.camera_control_idx,
            "command_output_limits": "default",
            "use_delta_commands": False,
        }

    @property
    def _default_camera_null_joint_controller_config(self):
        """
        Returns:
            dict: Default null joint controller config to control this robot's camera i.e. dummy controller
        """
        return {
            "name": "NullJointController",
            "control_freq": self._control_freq,
            "motor_type": "velocity",
            "control_limits": self.control_limits,
            "dof_idx": self.camera_control_idx,
        }

    @property
    def _default_controller_config(self):
        # Always run super method first
        cfg = super()._default_controller_config

        # We additionally add in camera default
        cfg["camera"] = {
            self._default_camera_joint_controller_config["name"]: self._default_camera_joint_controller_config,
            self._default_camera_null_joint_controller_config["name"]: self._default_camera_null_joint_controller_config,
        }

        return cfg

    @property
    @abstractmethod
    def camera_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to camera joints.
        """
        raise NotImplementedError

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ActiveCameraRobot")
        return classes
