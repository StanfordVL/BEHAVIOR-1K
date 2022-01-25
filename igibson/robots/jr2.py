import os

import numpy as np


import igibson
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.robots.manipulation_robot import ManipulationRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class JR2(ManipulationRobot, TwoWheelRobot):
    """
    JR2 Kinova robot
    Reference: https://cvgl.stanford.edu/projects/jackrabbot/
    """

    def __init__(
        self,
        name=None,
        control_freq=None,
        action_type="continuous",
        action_normalize=True,
        proprio_obs="default",
        reset_joint_pos=None,
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=True,
        class_id=SemanticClass.ROBOTS,
        rendering_params=None,
        grasping_mode="physical",
    ):
        """
        :param name: None or str, name of the robot object
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_robot will automatically set the control frequency to be 1 / render_timestep by default.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or str or Array[float], if specified, should be the joint positions that the robot
            should be set to during a reset. If str, should be one of {tuck, untuck}, corresponds to default
            configurations for un/tucked modes. If None (default), self.default_joint_pos (untuck mode) will be used
            instead.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific controller
            configurations for this robot. This will override any default values specified by this class.
        :param base_name: None or str, robot link name that will represent the entire robot's frame of reference. If not None,
            this should correspond to one of the link names found in this robot's corresponding URDF / MJCF file.
            None defaults to the first link name used in @model_file
        :param scale: int, scaling factor for model (default is 1)
        :param self_collision: bool, whether to enable self collision
        :param class_id: SemanticClass, semantic class this robot belongs to. Default is SemanticClass.ROBOTS.
        :param rendering_params: None or Dict[str, Any], If not None, should be keyword-mapped rendering options to set.
            See DEFAULT_RENDERING_PARAMS for the values passed by default.
        :param grasping_mode: None or str, One of {"physical", "assisted", "sticky"}.
            If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
            If "assisted", will magnetize any object touching and within the gripper's fingers.
            If "sticky", will magnetize any object touching the gripper's fingers.
        """
        # Parse reset joint pos if specifying special string
        if isinstance(reset_joint_pos, str):
            assert (
                reset_joint_pos in RESET_JOINT_OPTIONS
            ), "reset_joint_pos should be one of {} if using a string!".format(RESET_JOINT_OPTIONS)
            reset_joint_pos = (
                self.tucked_default_joint_pos if reset_joint_pos == "tuck" else self.untucked_default_joint_pos
            )

        # Run super init
        super().__init__(
            name=name,
            control_freq=control_freq,
            action_type=action_type,
            action_normalize=action_normalize,
            proprio_obs=proprio_obs,
            reset_joint_pos=reset_joint_pos,
            controller_config=controller_config,
            base_name=base_name,
            scale=scale,
            self_collision=self_collision,
            class_id=class_id,
            rendering_params=rendering_params,
            grasping_mode=grasping_mode,
        )

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "JR2"

    def _create_discrete_action_space(self):
        # JR2 does not support discrete actions if we're controlling the arm as well
        raise ValueError("Full JR2 does not support discrete actions!")

    def _validate_configuration(self):
        # Make sure we're not using assisted grasping
        assert self.grasping_mode == "physical", "Cannot use assisted grasping modes for JR2 since gripper is disabled!"

        # Make sure we're using a null controller for the gripper
        assert (
            self.controller_config["gripper_{}".format(self.default_arm)]["name"] == "NullGripperController"
        ), "JR2 robot has its gripper disabled, so cannot use any controller other than NullGripperController!"

        # run super
        super()._validate_configuration()

    def tuck(self):
        """
        Immediately set this robot's configuration to be in tucked mode
        """
        self.set_joint_positions(self.tucked_default_joint_pos)

    def untuck(self):
        """
        Immediately set this robot's configuration to be in untucked mode
        """
        self.set_joint_positions(self.untucked_default_joint_pos)

    def reset(self):
        # In addition to normal reset, reset the joint configuration to be in default mode
        super().reset()
        joints = self.default_joint_pos
        set_joint_positions(self.get_body_ids()[0], self.joint_ids, joints)

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use differential drive with joint controller for arm, since arm only has 5DOF
        controllers["base"] = "DifferentialDriveController"
        controllers["arm_{}".format(self.default_arm)] = "JointController"
        controllers["gripper_{}".format(self.default_arm)] = "NullGripperController"

        return controllers

    @property
    def tucked_default_joint_pos(self):
        # todo: tune values
        return np.array([0.0, 0.0, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0])

    @property
    def untucked_default_joint_pos(self):
        return np.array([0.0, 0.0, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0])

    @property
    def default_joint_pos(self):
        return self.untucked_default_joint_pos

    @property
    def wheel_radius(self):
        return 0.2405

    @property
    def wheel_axle_length(self):
        return 0.5421

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([1, 0])

    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        return {self.default_arm: np.array([2, 3, 4, 5, 6])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        return {self.default_arm: np.array([], dtype=np.int)}

    @property
    def disabled_collision_pairs(self):
        return [
            ["base_chassis_link", "pan_link"],
            ["base_chassis_link", "tilt_link"],
            ["base_chassis_link", "camera_link"],
            ["jr2_fixed_body_link", "pan_link"],
            ["jr2_fixed_body_link", "tilt_link"],
            ["jr2_fixed_body_link", "camera_link"],
        ]

    @property
    def eef_link_names(self):
        return {self.default_arm: "m1n6s200_end_effector"}

    @property
    def finger_link_names(self):
        return {self.default_arm: []}

    @property
    def finger_joint_names(self):
        return {self.default_arm: []}

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/jr2_urdf/jr2_kinova.urdf")
