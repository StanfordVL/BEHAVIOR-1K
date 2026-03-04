import math
import logging

import torch as th
import omnigibson.utils.transform_utils as T

from .config import DataCollectionConfig

logger = logging.getLogger(__name__)

class RobotContext:
    KP_LIN_VEL = 0.6
    KP_ANGLE_VEL = 2.95
    ANGLE_THRESHOLD = 0.10

    def __init__(self, env, robot, config: DataCollectionConfig):
        self.env = env
        self.robot = robot
        self.config = config
        default_arm = robot.default_arm if hasattr(robot, "default_arm") else "right"
        self.arm = self._select_active_arm(default_arm)

        self.base_idx = robot.controller_action_idx["base"]
        self.arm_idx = robot.controller_action_idx[f"arm_{self.arm}"]
        self.gripper_idx = robot.controller_action_idx[f"gripper_{self.arm}"]
        self.arm_controller_name = (
            type(self.robot.controllers[f"arm_{self.arm}"]).__name__
            if hasattr(self.robot, "controllers") and f"arm_{self.arm}" in self.robot.controllers
            else "Unknown"
        )

        self.action_dim = int(
            sum(int(self.robot.controllers[name].command_dim) for name in self.robot.controller_order)
        )
        self.controller_action_dim = int(
            sum(int(self.robot.controllers[name].command_dim) for name in self.robot.controller_order)
        )

        print(
            f"[RobotContext] arm={self.arm}, arm_ctrl={self.arm_controller_name}, "
            f"base_idx={self.base_idx}, arm_idx={self.arm_idx}, "
            f"gripper_idx={self.gripper_idx}, "
            f"action_dim={self.action_dim}, controller_action_dim={self.controller_action_dim}",
            flush=True,
        )

    def _idx_len(self, idx) -> int:
        try:
            return len(idx)
        except TypeError:
            return 0

    def _select_active_arm(self, default_arm: str) -> str:
        candidates = [default_arm, "right", "left"]
        seen = set()
        ordered = []
        for arm in candidates:
            if arm not in seen:
                seen.add(arm)
                ordered.append(arm)

        for arm in ordered:
            arm_key = f"arm_{arm}"
            grip_key = f"gripper_{arm}"
            arm_idx = self.robot.controller_action_idx.get(arm_key, [])
            gripper_idx = self.robot.controller_action_idx.get(grip_key, [])
            if self._idx_len(arm_idx) > 0 and self._idx_len(gripper_idx) > 0:
                return arm

        raise RuntimeError(
            "No active arm controller found with non-empty action indices. "
            f"Available keys: {list(self.robot.controller_action_idx.keys())}"
        )

    def empty_action(self) -> th.Tensor:
        """Return a zero action in controller-action space."""
        action = th.zeros(self.action_dim)
        return action

    @property
    def has_base_control(self) -> bool:
        return self._idx_len(self.base_idx) >= 2

    @property
    def is_ik_arm(self) -> bool:
        return "InverseKinematicsController" in self.arm_controller_name

    def get_robot_yaw(self) -> float:
        _, robot_quat = self.robot.get_position_orientation()
        robot_euler = T.quat2euler(robot_quat)
        return robot_euler[2].item()

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
