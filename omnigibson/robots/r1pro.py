from functools import cached_property

import torch as th

from omnigibson.robots.r1 import R1


class R1Pro(R1):
    """
    R1 Pro Robot
    """

    @property
    def tucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.05, 0.05])  # open gripper
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.05, 0.05])  # open gripper
        pos[self.arm_control_idx["left"]] = th.tensor([0.0, 1.57, 0.0, -1.57, 1.57, 0.0, 0.0])
        pos[self.arm_control_idx["right"]] = th.tensor([0.0, -1.57, 0.0, -1.57, -1.57, 0.0, 0.0])
        return pos

    @cached_property
    def floor_touching_base_link_names(self):
        return ["wheel_motor_link1", "wheel_motor_link2", "wheel_motor_link3"]

    @cached_property
    def arm_link_names(self):
        return {arm: [f"{arm}_arm_link{i}" for i in range(1, 8)] for arm in self.arm_names}

    @cached_property
    def arm_joint_names(self):
        return {arm: [f"{arm}_arm_joint{i}" for i in range(1, 8)] for arm in self.arm_names}

    @cached_property
    def gripper_link_names(self):
        return {arm: [f"{arm}_gripper_link", f"{arm}_realsense_link"] for arm in self.arm_names}

    @cached_property
    def finger_link_names(self):
        return {arm: [f"{arm}_gripper_finger_link{i}" for i in range(1, 3)] for arm in self.arm_names}

    @cached_property
    def finger_joint_names(self):
        return {arm: [f"{arm}_gripper_finger_joint{i}" for i in range(1, 3)] for arm in self.arm_names}

    @property
    def arm_workspace_range(self):
        return {arm: th.deg2rad(th.tensor([-45, 45], dtype=th.float32)) for arm in self.arm_names}

    @property
    def disabled_collision_pairs(self):
        # badly modeled gripper collision meshes
        return [
            ["left_arm_link1", "torso_link4"],
            ["left_arm_link2", "torso_link4"],
            ["right_arm_link1", "torso_link4"],
            ["right_arm_link2", "torso_link4"],
            ["left_arm_link5", "left_arm_link7"],
            ["right_arm_link5", "right_arm_link7"],
            ["left_gripper_finger_link1", "left_realsense_link"],
            ["right_gripper_finger_link1", "right_realsense_link"],
            ["left_gripper_finger_link1", "left_gripper_finger_link2"],
            ["right_gripper_finger_link1", "right_gripper_finger_link2"],
            ["base_link", "wheel_motor_link1"],
            ["base_link", "wheel_motor_link2"],
            ["base_link", "wheel_motor_link3"],
            ["torso_link2", "torso_link4"],
        ]
