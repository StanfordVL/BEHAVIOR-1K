from typing import Dict, Optional

import torch as th
import numpy as np

from gello.agents.joycon_agent import JoyconAgent
from gello.agents.gello_agent import DynamixelRobotConfig, MotorFeedbackConfig
from gello.dynamixel.driver import OperatingMode

from gello.agents.base_r1_gello_agent import BaseR1GelloAgent

class R1GelloAgent(BaseR1GelloAgent):
    """R1 Gello Agent implementation."""
    
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
        # R1 JoyLo has 6 joints(8 motors) per arm
        super().__init__(
            port=port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
            default_joints=default_joints,
            damping_motor_kp=damping_motor_kp,
            motor_feedback_type=motor_feedback_type,
            enable_locking_joints=enable_locking_joints,
            joycon_agent=joycon_agent,
            motors_per_arm=8,
        )
    
    def _gello_joints_to_obs_joints(self, joints: np.ndarray) -> np.ndarray:
        # Filter out redundant motors at indices 1, 3, 9, 11
        return (joints - self.joint_offsets)[[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15]]

    def _obs_joints_to_gello_joints(self, obs: Dict) -> np.ndarray:
        # Convert [left, right] arm qpos into single array
        obs_jnts = np.concatenate([obs[f"arm_{arm}_joint_positions"].detach().cpu().numpy() for arm in ["left", "right"]])
        # Duplicate values for redundant motors
        return obs_jnts[[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11]] + self.joint_offsets
    
    def _get_operational_space_jacobians(self, obs):
        # Duplicate the columns in the jacobian to handle duplicate joints
        # Only use velocity part as we are interested in position offset
        J_left = obs["arm_left_jacobian"][:3, [0, 0, 1, 1, 2, 3, 4, 5]]
        J_right = obs["arm_right_jacobian"][:3, [0, 0, 1, 1, 2, 3, 4, 5]]
        
        return J_left, J_right
    
    def _obs_to_gello_form(self, obs_jnts):
        # Convert observation joints to GELLO form
        return obs_jnts[[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11]]
    
    def _gello_to_obs_form(self, gello_jnts):
        # Convert GELLO joints to observation form
        return th.from_numpy(gello_jnts[[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15]])
    
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
        Handle upper arm locking for R1 model.
        """
        num_joints = len(arm_info["gello_ids"])
        upper_currently_locked = arm_info["locked"]["upper"]
        
        if lock_upper:
            if upper_currently_locked:
                # Already locked, do nothing
                pass
            else:
                # Just became locked, update this arm's operating mode (all joints for the arm except final
                # one should be using POSITION mode)
                operating_modes[arm_info["gello_ids"]] = [OperatingMode.EXTENDED_POSITION] * (num_joints - 1) + [OperatingMode.NONE]
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # In addition, the final wrist joint should NOT change in the environment -- so keep track of
                # the current angle so we can apply an offset as long as the upper arm is locked
                # NOTE: This value will ALREADY have any pre-existing offset applied, which is expected
                # because the "freezing upper" effect is assumed to be cumulative
                arm_info["locked_wrist_angle"] = gello_jnts[arm_info["gello_ids"][-1]]

                # Add upper joint to commanded set of joint idxs
                active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"][:-1]])

                # Finally, update our lock state
                arm_info["locked"]["upper"] = True

            # If we're locked, force the returned wrist value to be the locked value
            gello_jnts[wrist_id] = arm_info["locked_wrist_angle"]

        else:
            if not upper_currently_locked:
                # Already not locked, do nothing
                pass
            else:
                # Just became not locked, so update this arm's operating mode (all joints should be using
                # the default mode)
                operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Update the joint offset to include the difference between the locked wrist and the current
                # wrist qpos
                additional_wrist_offset = gello_jnts[wrist_id] - arm_info["locked_wrist_angle"]
                self.joint_offsets[wrist_id] += additional_wrist_offset

                # This offset hasn't been applied yet to the current joints, so apply it now (negative because
                # we're going from GELLO -> Env, whereas the offset captures the env -> GELLO delta)
                gello_jnts[wrist_id] -= additional_wrist_offset

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
        Handle lower arm locking for R1 model.
        """
        lower_currently_locked = arm_info["locked"]["lower"]
        if lock_lower:
            if lower_currently_locked:
                # Already locked, do nothing
                pass
            else:
                # Just became locked, update this arm's operating mode (final two joints should be using
                # POSITION mode)
                operating_modes[arm_info["gello_ids"]] = [OperatingMode.CURRENT] * 6 + [OperatingMode.EXTENDED_POSITION] * 2
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                # Add lower joints to commanded set of joint idxs
                active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"][-2:]])

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