import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
from enum import Enum

import torch as th
import numpy as np
import time

from gello.agents.agent import Agent
from gello.robots.dynamixel import DynamixelRobot
from gello.dynamixel.driver import OperatingMode, GainType

from omnigibson.utils.processing_utils import ExponentialAverageFilter

class MotorFeedbackConfig(Enum):
    """
    Enum specifying different types of force feedback, which are used in
    compute_feedback_currents
    """
    NONE = -1
    JOINT_SPACE = 0
    OPERATIONAL_SPACE = 1


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of GELLO. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

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


PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8IT07C-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=[-5*np.pi/2, 2*np.pi/2, 0*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 3.5*np.pi/2],
        joint_signs=(1, 1, 1, -1, 1, -1, 1),
        gripper_config=(8, 202, 152),
    ),
    # "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISZZV-if00-port0": DynamixelRobotConfig(
    #     joint_ids=(0, 1, 2, 3, 4, 5),
    #     joint_offsets=[3*np.pi/2, 0.5*np.pi/2, 4*np.pi/2, 2*np.pi/2, 1*np.pi/2, 3*np.pi/2],
    #     joint_signs=(1, 1, 1, 1, 1, 1),
    #     gripper_config=None, #(8, 202, 152),
    # ),
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISZZV-if00-port0": DynamixelRobotConfig(
        joint_ids=(6, 7, 8, 9, 10, 11), #(0, 1, 2, 3, 4, 5),
        joint_offsets=[np.pi/2, 0 * np.pi/2, 2*np.pi/2, 3*np.pi/2, 2*np.pi/2, 3*np.pi/2],
        joint_signs=(1, 1, 1, 1, 1, 1),
        gripper_config=None, #(8, 202, 152),
    ),
}



class GelloAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
        default_joints: Optional[np.ndarray] = None,
        damping_motor_kp: Optional[float] = 0.0,
    ):
        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        else:
            assert os.path.exists(port), port
            assert port in PORT_CONFIG_MAP, f"Port {port} not in config map"

            config = PORT_CONFIG_MAP[port]
            self._robot = config.make_robot(port=port, start_joints=start_joints)

        # If using damping, make sure torque is enabled
        self._reset_qpos = start_joints if default_joints is None else np.array(default_joints)
        self._reset_cooldown_active = False
        self._damping_motor_kp = damping_motor_kp
        self._damping_motor_kv = 2 * np.sqrt(self._damping_motor_kp) * 1.0

        # Always start with current / torque disabled
        self._robot.set_operating_mode(OperatingMode.NONE)
        self._current_enabled = False

        # Set default gains
        self._robot._driver.set_gain(GainType.P, 500)
        self._robot._driver.set_gain(GainType.I, 0)
        self._robot._driver.set_gain(GainType.D, 200)
        
        # Initialize exponential average filter for smoothing
        self._smoothing_filter = ExponentialAverageFilter(
            obs_dim=self._robot.num_dofs(),
            alpha=0.2
        )

        # Call super method
        super().__init__()

    def enable_current_feedback(self):
        if not self._current_enabled:
            self._robot.set_operating_mode(mode=OperatingMode.CURRENT)
            self._current_enabled = True

    def disable_current_feedback(self):
        if self._current_enabled:
            self._robot.set_operating_mode(mode=OperatingMode.NONE)
            self._current_enabled = False

    def _gello_joints_to_obs_joints(self, joints: np.ndarray) -> np.ndarray:
        """
        Maps raw GELLO joints to observation equivalent joints from the environment

        Args:
            joints (np.ndarray): Raw GELLO joints

        Returns:
            np.ndarray: Equivalent joint configuration in environment-compatible array form
        """
        # Default simply returns the joints directly
        return joints

    def _obs_joints_to_gello_joints(self, obs: Dict) -> np.ndarray:
        """
        Maps observed joints from the environment into equivalent GELLO joint configuration

        Args:
            obs (dict): Keyword-mapped dictionary of relevant joints obtained from the environment

        Returns:
            np.ndarray: Equivalent joint configuration in GELLO-compatible array form
        """
        # Must be implemented by subclass!
        raise NotImplementedError

    def set_reset_qpos(self, qpos):
        self._reset_qpos = np.array(qpos)

    def reset(self):
        # Move arms back to their reset pose if any is specified
        if self._reset_qpos is not None:
            self._robot.set_operating_mode(OperatingMode.EXTENDED_POSITION)
            # Set non-zero I gain
            self._robot._driver.set_gain(GainType.I, 150)
            self._robot.command_joint_state(self._reset_qpos)
            # Sleep for a little bit to let motors settle and reduce chance of overload
            time.sleep(1)

    def start(self):
        # Set default gains
        self._robot._driver.set_gain(GainType.I, 0)

        # Set torque mode if we have a nonzero damping kp
        if self._damping_motor_kp != 0.0:
            self.enable_current_feedback()
        else:
            self.disable_current_feedback()

    def act(self, obs: Dict[str, np.ndarray]) -> th.tensor:
        # Resist if specified
        jnts = self._robot.get_joint_state()
        jnts_vel = self._robot.get_joint_velocities() # Values from the dynamixels
        if self._current_enabled:
            # Disable controller if we're in cooldown
            current_idxs = np.where(self._robot.operating_mode == OperatingMode.CURRENT)[0]

            self._reset_cooldown_active = (obs["in_cooldown"] > 0)

            if self._reset_cooldown_active or obs["reset_joints"]:
                # Setpoint is reset position
                joint_setpoint = self._reset_qpos
            else:
                # Setpoint is sim position, mapped to GELLO joints
                joint_setpoint = self._obs_joints_to_gello_joints(obs)

            # Compute joint errors and delegate to control function implemented by subclass
            joint_error = np.rad2deg(joint_setpoint - jnts)

            current = self.compute_feedback_currents(joint_error, jnts_vel, obs)
            self._robot.command_current(current[current_idxs], idxs=current_idxs)

        # Apply smoothing to the joint states
        jnts = self._smoothing_filter.estimate(jnts)
        
        # Return GELLO joints in environment-compatible form (for the sim)
        return th.from_numpy(self._gello_joints_to_obs_joints(joints=jnts).astype(np.float32))

    def compute_feedback_currents(self, joint_error, joint_vel, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the feedback currents (i.e. motor torques) given the current discrepancy between
        the sim joints and gello joints

        Parameters:
            joint_error:    array of joint errors (in degrees), in GELLO format
            joint_vel:      array of joint velocities, in GELLO format
            obs:            dict of observations from the sim
        Returns:
            Array of feedback currents, one for each GELLO joint

        Should be implemented by a sub-class - default to returning zero here
        """
        return np.zeros(joint_error.shape)