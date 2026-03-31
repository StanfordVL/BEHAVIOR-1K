from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from gello.agents.agent import Agent
from gello.robots.dynamixel_robot import DynamixelRobot
from gello.utils.dynamixel_utils import OperatingMode, GainType


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]
    gripper_config: Optional[Sequence] = None

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )


class DynamixelArmAgent(Agent):
    """Minimal Dynamixel agent wrapping one or more arms.

    Wraps a DynamixelRobot with current-based impedance control.
    For bimanual setups, create one DynamixelArmAgent with all motors
    and let BimanualAgent coordinate left/right subsets via idxs.
    """

    def __init__(
        self,
        port: str,
        dynamixel_config: DynamixelRobotConfig,
        start_joints: Optional[np.ndarray] = None,
        damping_motor_kp: float = 0.0,
    ):
        super().__init__()
        self._robot = dynamixel_config.make_robot(port=port, start_joints=start_joints)
        self._reset_qpos = start_joints
        self._damping_motor_kp = damping_motor_kp
        self._damping_motor_kv = 2 * np.sqrt(damping_motor_kp) * 1.0
        self._current_enabled = False

        # Initialize hardware
        self._robot.set_operating_mode(OperatingMode.NONE)
        self._robot._driver.set_gain(GainType.P, 500)
        self._robot._driver.set_gain(GainType.I, 0)
        self._robot._driver.set_gain(GainType.D, 200)

    @property
    def robot(self) -> DynamixelRobot:
        return self._robot

    def get_joint_state(self, idxs=None) -> np.ndarray:
        jnts = self._robot.get_joint_state()
        return jnts[idxs] if idxs is not None else jnts

    def get_joint_velocities(self, idxs=None) -> np.ndarray:
        vel = self._robot.get_joint_velocities()
        return vel[idxs] if idxs is not None else vel

    def set_reset_qpos(self, qpos):
        self._reset_qpos = np.array(qpos)

    def enable_current_feedback(self):
        if not self._current_enabled:
            self._robot.set_operating_mode(OperatingMode.CURRENT)
            self._current_enabled = True

    def disable_current_feedback(self):
        if self._current_enabled:
            self._robot.set_operating_mode(OperatingMode.NONE)
            self._current_enabled = False

    def reset(self):
        if self._reset_qpos is not None:
            self._robot.set_operating_mode(OperatingMode.EXTENDED_POSITION)
            self._robot._driver.set_gain(GainType.I, 150)
            self._robot.command_joint_state(self._reset_qpos)
            import time

            time.sleep(1)

    def start(self):
        self._robot._driver.set_gain(GainType.I, 0)
        if self._damping_motor_kp != 0.0:
            self.enable_current_feedback()
        else:
            self.disable_current_feedback()

    def act(self, target_joints: Optional[np.ndarray] = None) -> np.ndarray:
        """Read joints and optionally apply impedance control toward target.

        Args:
            target_joints: Desired joint positions in GELLO format.
                If None, only reads and returns current joint state.

        Returns:
            Current joint positions (numpy array).
        """
        jnts = self._robot.get_joint_state()

        if target_joints is not None and self._current_enabled:
            current_idxs = np.where(
                self._robot.operating_mode == OperatingMode.CURRENT
            )[0]
            if len(current_idxs) > 0:
                joint_error = np.rad2deg(target_joints - jnts)
                jnts_vel = self._robot.get_joint_velocities()
                current = (
                    self._damping_motor_kp
                    * 0.2
                    * (joint_error**2)
                    * np.sign(joint_error)
                    - self._damping_motor_kv * jnts_vel * 0.1
                )
                self._robot.command_current(current[current_idxs], idxs=current_idxs)

        return jnts
