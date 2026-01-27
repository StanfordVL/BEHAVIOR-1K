import math
from functools import cached_property

import torch as th

from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


class Stretch(ManipulationRobot, TwoWheelRobot, ActiveCameraRobot):
    """
    Strech Robot from Hello Robotics
    Reference: https://hello-robot.com/stretch-3-product
    """

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("Stretch does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "camera", f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers[f"arm_{self.default_arm}"] = "JointController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_joint_pos(self):
        # Custom pose: lift raised, gripper open, camera looking down
        # Order matches actual joint order from robot.joints.keys()
        return th.tensor([
            -1.5764,    # joint_head_pan
            0.0,        # joint_left_wheel
            0.6893,     # joint_lift
            0.0,        # joint_right_wheel
            -0.5714,    # joint_head_tilt
            0.0220,     # joint_arm_l3
            0.0017,     # joint_arm_l2
            0.0389,     # joint_arm_l1
            0.0062,     # joint_arm_l0
            -0.0649,    # joint_wrist_yaw
            -0.0513,    # joint_wrist_pitch
            -0.0315,    # joint_wrist_roll
            0.6,        # joint_gripper_finger_left
            0.6,        # joint_gripper_finger_right
        ])

    @property
    def wheel_radius(self):
        return 0.050

    @property
    def wheel_axle_length(self):
        return 0.330

    @property
    def disabled_collision_pairs(self):
        return [
            ["link_lift", "link_arm_l3"],
            ["link_lift", "link_arm_l2"],
            ["link_lift", "link_arm_l1"],
            ["link_lift", "link_arm_l0"],
            ["link_arm_l3", "link_arm_l1"],
            ["link_arm_l0", "link_arm_l1"],
            ["link_arm_l0", "link_arm_l2"],
            ["link_arm_l0", "link_arm_l3"],
        ]

    @cached_property
    def base_joint_names(self):
        return ["joint_left_wheel", "joint_right_wheel"]

    @cached_property
    def camera_joint_names(self):
        return ["joint_head_pan", "joint_head_tilt"]

    @cached_property
    def arm_link_names(self):
        return {
            self.default_arm: [
                "link_lift",
                "link_arm_l3",
                "link_arm_l2",
                "link_arm_l1",
                "link_arm_l0",
                "link_wrist_yaw",
                "link_wrist_pitch",
                "link_wrist_roll",
            ]
        }

    @cached_property
    def arm_joint_names(self):
        return {
            self.default_arm: [
                "joint_lift",
                "joint_arm_l3",
                "joint_arm_l2",
                "joint_arm_l1",
                "joint_arm_l0",
                "joint_wrist_yaw",
                "joint_wrist_pitch",
                "joint_wrist_roll",
            ]
        }

    @cached_property
    def eef_link_names(self):
        return {self.default_arm: "eef_link"}

    @cached_property
    def finger_link_names(self):
        return {
            self.default_arm: [
                "link_gripper_finger_left",
                "link_gripper_finger_right",
            ]
        }

    @cached_property
    def finger_joint_names(self):
        return {self.default_arm: ["joint_gripper_finger_right", "joint_gripper_finger_left"]}

    @property
    def _assisted_grasp_start_points(self):
        """Grasping start points for assisted grasping raycast detection."""
        return {
            self.default_arm: [
                # Use finger link with offset toward fingertip (along -X axis in finger frame)
                GraspingPoint(link_name="link_gripper_finger_right", position=th.tensor([-0.15, 0.0, 0.0])),
            ]
        }

    @property
    def _assisted_grasp_end_points(self):
        """Grasping end points for assisted grasping raycast detection."""
        return {
            self.default_arm: [
                # Use finger link with offset toward fingertip (along +X axis in finger frame due to mirroring)
                GraspingPoint(link_name="link_gripper_finger_left", position=th.tensor([0.15, 0.0, 0.0])),
            ]
        }
