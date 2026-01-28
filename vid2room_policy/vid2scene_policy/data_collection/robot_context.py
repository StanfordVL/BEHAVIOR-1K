import math
import logging

import torch as th
import omnigibson.utils.transform_utils as T

from .config import DataCollectionConfig

logger = logging.getLogger(__name__)

# Mode flags for extra action dimension
MODE_NAVIGATION = 0.0
MODE_MANIPULATION = 1.0


class RobotContext:
    KP_LIN_VEL = 0.6
    KP_ANGLE_VEL = 2.95
    ANGLE_THRESHOLD = 0.10

    def __init__(self, env, robot, config: DataCollectionConfig):
        self.env = env
        self.robot = robot
        self.config = config
        self.arm = robot.default_arm if hasattr(robot, 'default_arm') else "right"

        self.base_idx = robot.controller_action_idx["base"]
        self.arm_idx = robot.controller_action_idx[f"arm_{self.arm}"]
        self.gripper_idx = robot.controller_action_idx[f"gripper_{self.arm}"]

        # Extra dimension index for mode flag (appended at end)
        self.mode_idx = robot.action_dim  # Will be the last dimension
        self.extended_action_dim = robot.action_dim + 1

        logger.info("RobotContext initialized: base_idx=%s, arm_idx=%s, gripper_idx=%s, mode_idx=%d",
                    self.base_idx, self.arm_idx, self.gripper_idx, self.mode_idx)

    def empty_action(self, mode: float = MODE_NAVIGATION) -> th.Tensor:
        """Create empty action with mode flag. Default is navigation mode."""
        action = th.zeros(self.extended_action_dim)
        for name, controller in self.robot._controllers.items():
            action_idx = self.robot.controller_action_idx[name]
            if len(action_idx) == 0:
                continue
            partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
            action[action_idx] = partial_action
        action[self.mode_idx] = mode
        return action

    def get_robot_yaw(self) -> float:
        _, robot_quat = self.robot.get_position_orientation()
        robot_euler = T.quat2euler(robot_quat)
        return robot_euler[2].item()

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
