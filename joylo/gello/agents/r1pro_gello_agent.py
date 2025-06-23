from typing import Dict, Optional

import torch as th
import numpy as np

from gello.agents.joycon_agent import JoyconAgent
from gello.agents.gello_agent import DynamixelRobotConfig, MotorFeedbackConfig
from gello.dynamixel.driver import OperatingMode

from gello.agents.base_r1_gello_agent import BaseR1GelloAgent

class R1ProGelloAgent(BaseR1GelloAgent):
    """R1Pro Gello Agent implementation."""
    
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
        default_joints: Optional[np.ndarray] = None,
        damping_motor_kp: Optional[float] = 0.0,
        motor_feedback_type: MotorFeedbackConfig = MotorFeedbackConfig.OPERATIONAL_SPACE,
        enable_locking_joints: bool = True,
        joycon_agent: Optional[JoyconAgent] = None,
    ):
        # R1Pro JoyLo has 7 joints(9 motors) per arm
        super().__init__(
            port=port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
            default_joints=default_joints,
            damping_motor_kp=damping_motor_kp,
            motor_feedback_type=motor_feedback_type,
            enable_locking_joints=enable_locking_joints,
            joycon_agent=joycon_agent,
            motors_per_arm=9,
        )
    
    def _gello_joints_to_obs_joints(self, joints: np.ndarray) -> np.ndarray:
        # Filter out redundant motors at idxs 1, 3, 10, 12
        return (joints - self.joint_offsets)[[0, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17]]

    def _obs_joints_to_gello_joints(self, obs: Dict) -> np.ndarray:
        # Convert [left, right] arm qpos into single array
        obs_jnts = np.concatenate([obs[f"arm_{arm}_joint_positions"].detach().cpu().numpy() for arm in ["left", "right"]])
        # Duplicate values for redundant motors
        return obs_jnts[[0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12, 13]] + self.joint_offsets
    
    def _get_operational_space_jacobians(self, obs):
        # Duplicate the columns in the jacobian to handle duplicate joints
        # Only use velocity part as we are interested in position offset
        J_left = obs["arm_left_jacobian"][:3, [0, 0, 1, 1, 2, 3, 4, 5, 6]]
        J_right = obs["arm_right_jacobian"][:3, [0, 0, 1, 1, 2, 3, 4, 5, 6]]
        
        return J_left, J_right
    
    def _obs_to_gello_form(self, obs_jnts):
        # Convert observation joints to GELLO form
        return obs_jnts[[0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12, 13]]
    
    def _gello_to_obs_form(self, gello_jnts):
        # Convert GELLO joints to observation form
        return th.from_numpy(gello_jnts[[0, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17]])

    def _handle_all_arm_locking(
            self,
            arm_info,
            lock_all,
            wrist_id,
            gello_jnts,
            operating_modes,
            active_operating_mode_idxs,
            active_commanded_jnt_idxs
    ):
        """
        Handle upper arm locking for R1 Pro.
        """
        all_currently_locked = arm_info["locked"]["all"]
        if lock_all:
            if all_currently_locked:
                # Already locked, do nothing
                pass
            else:
                # Just became locked, update this arm's operating mode (first five joints should be using
                # POSITION mode)
                operating_modes[arm_info["gello_ids"]] = [OperatingMode.EXTENDED_POSITION] * 9
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Add all joints to commanded set of joint idxs
                active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"]])

                # Finally, update our lock state
                arm_info["locked"]["all"] = True

        else:
            if not all_currently_locked:
                # Already not locked, do nothing
                pass
            else:
                # Just became not locked, so update this arm's operating mode (all joints should be using
                # the default mode)
                operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Finally, update our lock state
                arm_info["locked"]["all"] = False

        return operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs

    def _handle_upper_arm_locking(
        self, 
        arm_info, 
        lock_upper,
        wrist_id, 
        gello_jnts, 
        operating_modes, 
        active_operating_mode_idxs, 
        active_commanded_jnt_idxs
    ):
        """
        Handle upper arm locking for R1 Pro.
        """
        upper_currently_locked = arm_info["locked"]["upper"]
        if lock_upper:
            if upper_currently_locked:
                # Already locked, do nothing
                pass
            else:
                # Just became locked, update this arm's operating mode (first five joints should be using
                # POSITION mode)
                operating_modes[arm_info["gello_ids"]] = [OperatingMode.EXTENDED_POSITION] * 5 + [OperatingMode.NONE] * 4
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Add upper joints to commanded set of joint idxs
                active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"][:5]])

                # Finally, update our lock state
                arm_info["locked"]["upper"] = True

        else:
            if not upper_currently_locked:
                # Already not locked, do nothing
                pass
            else:
                # Just became not locked, so update this arm's operating mode (all joints should be using
                # the default mode)
                operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Finally, update our lock state
                arm_info["locked"]["upper"] = False
        
        return operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs
    
    def _handle_lower_arm_locking(
        self, 
        arm_info, 
        lock_lower,
        operating_modes, 
        active_operating_mode_idxs, 
        active_commanded_jnt_idxs
    ):
        """
        Handle lower arm locking for R1Pro.
        """
        lower_currently_locked = arm_info["locked"]["lower"]
        if lock_lower:
            if lower_currently_locked:
                # Already locked, do nothing
                pass
            else:
                # Just became locked, update this arm's operating mode (final three joints should be using
                # POSITION mode)
                operating_modes[arm_info["gello_ids"]] = [OperatingMode.NONE] * 6 + [OperatingMode.EXTENDED_POSITION] * 3
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Add lower joints to commanded set of joint idxs
                active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"][-3:]])

                # Finally, update our lock state
                arm_info["locked"]["lower"] = True

        else:
            if not lower_currently_locked:
                # Already not locked, do nothing
                pass
            else:
                # Just became not locked, so update this arm's operating mode (all joints should be using
                # the default mode)
                operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Finally, update our lock state
                arm_info["locked"]["lower"] = False
        
        return operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs