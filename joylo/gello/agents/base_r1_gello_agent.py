from typing import Dict, Optional

import torch as th
import numpy as np

from gello.agents.joycon_agent import JoyconAgent
from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent, MotorFeedbackConfig
from gello.dynamixel.driver import OperatingMode


class BaseR1GelloAgent(GelloAgent):
    """Base class for R1 Gello Agents that handles common functionality."""
    
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
        motors_per_arm: int = 8,  # Default for R1
    ):
        # Create local variables
        self.arm_info = {
            "left": {
                "locked": {
                    "upper": False,
                    "lower": False,
                    "all": False,
                },
                "gello_ids": np.arange(motors_per_arm),
                "locked_wrist_angle": None,
                "colliding": False
            },
            "right": {
                "locked": {
                    "upper": False,
                    "lower": False,
                    "all": False,
                },
                "gello_ids": np.arange(motors_per_arm) + motors_per_arm,
                "locked_wrist_angle": None,
                "colliding": False
            },
        }
        self.joycon_agent = joycon_agent
        self._motor_feedback_type = motor_feedback_type

        # Total number of joints in the system
        self.total_motors = motors_per_arm * 2

        # Can only enable locking joints if we have a valid joycon agent
        self.enable_locking_joints = enable_locking_joints and self.joycon_agent is not None

        # Stores joint offsets to apply dynamically (note: including motor redundancies!)
        self.joint_offsets = np.zeros(self.total_motors)

        # Whether we're waiting to resume or not
        self._waiting_to_resume = False

        # No feedback by default
        self.default_operation_modes = np.array([OperatingMode.NONE for _ in range(self.total_motors)])

        # Run super
        super().__init__(
            port=port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
            default_joints=default_joints,
            damping_motor_kp=damping_motor_kp,
        )
    
    def _gello_joints_to_obs_joints(self, joints: np.ndarray) -> np.ndarray:
        """
        Convert GELLO joint configuration to observation joint configuration.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _obs_joints_to_gello_joints(self, obs: Dict) -> np.ndarray:
        """
        Maps observed joints from the environment into equivalent GELLO joint configuration.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
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
        """
        if self._motor_feedback_type == MotorFeedbackConfig.NONE:
            # No feedback
            current = np.zeros(joint_error.shape)

        elif self._motor_feedback_type == MotorFeedbackConfig.JOINT_SPACE:
            # Joint space feedback
            current = self._damping_motor_kp * 0.2 * (joint_error ** 2) * np.sign(joint_error) - self._damping_motor_kv * joint_vel * 0.1
        
        elif self._motor_feedback_type == MotorFeedbackConfig.OPERATIONAL_SPACE:
            # Get jacobians for each arm - this part must be implemented by subclasses
            J_left, J_right = self._get_operational_space_jacobians(obs)
            
            # Stack jacobians into block diagonal matrix so the whole computation
            # can be performed at once
            J_shape = J_left.shape
            assert J_left.shape == J_right.shape

            J = np.block([
                [J_left, np.zeros(J_shape)],
                [np.zeros(J_shape), J_right]
            ])

            eef_error = J @ np.deg2rad(joint_error)
            current = self._damping_motor_kp * J.T @ eef_error - self._damping_motor_kv * joint_vel * 0.01
        
        else:
            raise ValueError(f"Unexpected joint feedback type: {self._motor_feedback_type}!")
        
        return current
    
    def _get_operational_space_jacobians(self, obs):
        """
        Get the Jacobians for operational space control.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def start(self):
        super().start()

        # Set all joints to default operating modes
        self._robot.set_operating_mode(self.default_operation_modes)

    def act(self, obs: Dict) -> th.Tensor:
        # Run super first
        jnts = super().act(obs=obs)

        # Convert back to gello form - this method should be implemented by subclasses
        gello_jnts = self._obs_to_gello_form(jnts.numpy())

        # If we see that we're waiting to resume from the sim, reset the joints to the observed values
        if obs["waiting_to_resume"] and not self._waiting_to_resume:
            # Up signal -- track the current pose from the robot and reset to that qpos
            reset_jnts = self._obs_joints_to_gello_joints(obs=obs)
            self.set_reset_qpos(qpos=reset_jnts)
            self.reset()
            print("Waiting to resume from sim...")
            self._waiting_to_resume = True
        elif not obs["waiting_to_resume"] and self._waiting_to_resume:
            # Down signal
            self.start()
            self._waiting_to_resume = False

        # Only compute action if we're not waiting to resume
        if self._waiting_to_resume:
            action = jnts
        else:
            active_operating_mode_idxs = np.array([], dtype=int)
            operating_modes = np.zeros(len(gello_jnts), dtype=int)
            active_commanded_jnt_idxs = np.array([], dtype=int)
            commanded_jnts = gello_jnts + self.joint_offsets    # We include offsets because these are the raw commands to send to GELLO

            # If we have a joycon agent, we possibly provide additional constraints to GELLO
            if self.enable_locking_joints:
                # Handle locking joints based on joycon input
                self._handle_joint_locking(
                    obs, 
                    gello_jnts, 
                    operating_modes, 
                    active_operating_mode_idxs, 
                    active_commanded_jnt_idxs, 
                    commanded_jnts
                )

            # Convert back to observation form for the final action - this method should be implemented by subclasses
            action = self._gello_to_obs_form(gello_jnts)

        return action
    
    def _obs_to_gello_form(self, obs_jnts):
        """
        Convert observation joint values to GELLO joint form.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _gello_to_obs_form(self, gello_jnts):
        """
        Convert GELLO joint values to observation joint form.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _handle_joint_locking(
        self, 
        obs, 
        gello_jnts, 
        operating_modes, 
        active_operating_mode_idxs, 
        active_commanded_jnt_idxs, 
        commanded_jnts
    ):
        """
        Handle joint locking based on joycon input.
        """
        # Feedback case 1: - / + is pressed -- this will disable final motor motor current while locking
        # all other joints' positions to allow for positional offsetting in the final joint (NOTE: this only applies to R1)
        # Feedback case 2: L / R is pressed -- this will lock the final two/three wrist joints of the given arm
        # while freeing all the other joints to allow for easy elbow maneuvering
        for arm, (lock_all, lock_lower) in zip(
                ("left", "right"),
                (
                    (self.joycon_agent.gripper_info["-"]["status"], self.joycon_agent.jc_left.get_button_l()),
                    (self.joycon_agent.gripper_info["+"]["status"], self.joycon_agent.jc_right.get_button_r()),
                ),
        ):
            arm_info = self.arm_info[arm]
            wrist_id = arm_info["gello_ids"][-1]

            if self._motor_feedback_type != MotorFeedbackConfig.NONE:
                # If using feedback, enable current control for the arms which are colliding
                if obs[f"arm_{arm}_contact"] and not arm_info["colliding"]:
                        arm_info["colliding"] = True

                        operating_modes[arm_info["gello_ids"]] = OperatingMode.CURRENT
                        active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

                elif not obs[f"arm_{arm}_contact"] and arm_info["colliding"]:
                        arm_info["colliding"] = False

                        operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                        active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])

            # Handle entire arm locking
            operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs = self._handle_all_arm_locking(
                arm_info,
                lock_all == -1,
                wrist_id,
                gello_jnts,
                operating_modes,
                active_operating_mode_idxs,
                active_commanded_jnt_idxs
            )

            # # Handle upper arm locking
            # operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs = self._handle_upper_arm_locking(
            #     arm_info,
            #     lock_upper,
            #     wrist_id,
            #     gello_jnts,
            #     operating_modes,
            #     active_operating_mode_idxs,
            #     active_commanded_jnt_idxs
            # )

            # Handle lower arm locking
            operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs = self._handle_lower_arm_locking(
                arm_info, 
                lock_lower,
                operating_modes, 
                active_operating_mode_idxs, 
                active_commanded_jnt_idxs
            )

        # Update our operating mode if requested
        if len(active_operating_mode_idxs) > 0:
            self._robot.set_operating_mode(operating_modes[active_operating_mode_idxs], idxs=active_operating_mode_idxs)

        # Command joints if requested
        if len(active_commanded_jnt_idxs) > 0:
            self._robot.command_joint_state(commanded_jnts[active_commanded_jnt_idxs], idxs=active_commanded_jnt_idxs)

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
        Handle all arm locking. This is specific to each model and should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

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
        Handle upper arm locking. This is specific to each model and should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _handle_lower_arm_locking(
        self, 
        arm_info, 
        lock_lower,
        operating_modes, 
        active_operating_mode_idxs, 
        active_commanded_jnt_idxs
    ):
        """
        Handle lower arm locking. This is specific to each model and should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")