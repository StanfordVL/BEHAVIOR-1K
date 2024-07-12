import os

import numpy as np

from omnigibson.macros import gm
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.robots.two_wheel_robot import TwoWheelRobot


class Stretch(ManipulationRobot, TwoWheelRobot, ActiveCameraRobot):
    """
    Strech Robot from Hello Robotics
    Reference: https://hello-robot.com/stretch-3-product
    """

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Stretch does not support discrete actions
        raise ValueError("Stretch does not support discrete actions!")

    @property
    def controller_order(self):
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
        return np.array([0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, np.pi / 8, np.pi / 8])

    @property
    def wheel_radius(self):
        return 0.050

    @property
    def wheel_axle_length(self):
        return 0.330

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.04}

    @property
    def assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="r_gripper_finger_link", position=[0.025, -0.012, 0.0]),
                GraspingPoint(link_name="r_gripper_finger_link", position=[-0.025, -0.012, 0.0]),
            ]
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="l_gripper_finger_link", position=[0.025, 0.012, 0.0]),
                GraspingPoint(link_name="l_gripper_finger_link", position=[-0.025, 0.012, 0.0]),
            ]
        }

    @property
    def disabled_collision_pairs(self):
        return [
            ["base_link", "caster_link"],
            ["base_link", "link_aruco_left_base"],
            ["base_link", "link_aruco_right_base"],
            ["base_link", "base_imu"],
            ["base_link", "laser"],
            ["base_link", "link_left_wheel"],
            ["base_link", "link_right_wheel"],
            ["base_link", "link_mast"],
            ["link_mast", "link_head"],
            ["link_head", "link_head_pan"],
            ["link_head_pan", "link_head_tilt"],
            ["camera_link", "link_head_tilt"],
            ["camera_link", "link_head_pan"],
            ["link_head_nav_cam", "link_head_tilt"],
            ["link_head_nav_cam", "link_head_pan"],
            ["link_mast", "link_lift"],
            ["link_lift", "link_aruco_shoulder"],
            ["link_lift", "link_arm_l4"],
            ["link_lift", "link_arm_l3"],
            ["link_lift", "link_arm_l2"],
            ["link_lift", "link_arm_l1"],
            ["link_arm_l4", "link_arm_l3"],
            ["link_arm_l4", "link_arm_l2"],
            ["link_arm_l4", "link_arm_l1"],
            ["link_arm_l3", "link_arm_l2"],
            ["link_arm_l3", "link_arm_l1"],
            ["link_arm_l2", "link_arm_l1"],
            ["link_arm_l0", "link_arm_l1"],
            ["link_arm_l0", "link_arm_l2"],
            ["link_arm_l0", "link_arm_l3"],
            ["link_arm_l0", "link_arm_l4"],
            ["link_arm_l0", "link_arm_l1"],
            ["link_arm_l0", "link_aruco_inner_wrist"],
            ["link_arm_l0", "link_aruco_top_wrist"],
            ["link_arm_l0", "link_wrist_yaw"],
            ["link_arm_l0", "link_wrist_yaw_bottom"],
            ["link_arm_l0", "link_wrist_pitch"],
            ["link_wrist_yaw_bottom", "link_wrist_pitch"],
            ["gripper_camera_link", "link_gripper_s3_body"],
            ["link_gripper_s3_body", "link_aruco_d405"],
            ["link_gripper_s3_body", "link_gripper_finger_left"],
            ["link_gripper_finger_left", "link_aruco_fingertip_left"],
            ["link_gripper_finger_left", "link_gripper_fingertip_left"],
            ["link_gripper_s3_body", "link_gripper_finger_right"],
            ["link_gripper_finger_right", "link_aruco_fingertip_right"],
            ["link_gripper_finger_right", "link_gripper_fingertip_right"],
            ["respeaker_base", "link_head"],
            ["respeaker_base", "link_mast"],
        ]

    @property
    def base_joint_names(self):
        return ["joint_left_wheel", "joint_right_wheel"]

    @property
    def camera_joint_names(self):
        return ["joint_head_pan", "joint_head_tilt"]

    @property
    def arm_link_names(self):
        return {
            self.default_arm: [
                "link_mast",
                "link_lift",
                "link_arm_l4",
                "link_arm_l3",
                "link_arm_l2",
                "link_arm_l1",
                "link_arm_l0",
                "link_aruco_inner_wrist",
                "link_aruco_top_wrist",
                "link_wrist_yaw",
                "link_wrist_yaw_bottom",
                "link_wrist_pitch",
                "link_wrist_roll",
            ]
        }

    @property
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

    @property
    def eef_link_names(self):
        return {self.default_arm: "link_grasp_center"}

    @property
    def finger_link_names(self):
        return {
            self.default_arm: [
                "link_gripper_finger_left",
                "link_gripper_finger_right",
                "link_gripper_fingertip_left",
                "link_gripper_fingertip_right",
            ]
        }

    @property
    def finger_joint_names(self):
        return {self.default_arm: ["joint_gripper_finger_right", "joint_gripper_finger_left"]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/stretch/stretch/stretch.usd")

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/stretch/stretch.urdf")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/stretch/stretch_descriptor.yaml")}
