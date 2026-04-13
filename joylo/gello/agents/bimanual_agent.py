from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch as th

from gello.agents.agent import Agent
from gello.agents.dynamixel_arm_agent import DynamixelArmAgent
from gello.agents.joycon_agent import JoyconAgent
from gello.utils.dynamixel_utils import OperatingMode


class MotorFeedbackConfig(Enum):
    NONE = -1
    JOINT_SPACE = 0
    OPERATIONAL_SPACE = 1


@dataclass
class ArmLockConfig:
    upper_lock_count: int = 0
    lower_lock_count: int = 0
    all_lock_mode: Optional[OperatingMode] = None
    track_wrist_offset: bool = False


@dataclass
class BimanualAgentConfig:
    joints_per_arm: int
    motors_per_arm: int
    gello_to_obs_indices: List[int]
    obs_to_gello_indices: List[int]
    jacobian_columns: List[int]
    default_operation_modes: List[OperatingMode]
    left_lock: ArmLockConfig = field(default_factory=ArmLockConfig)
    right_lock: ArmLockConfig = field(default_factory=ArmLockConfig)
    start_joints: Optional[np.ndarray] = None


R1_CONFIG = BimanualAgentConfig(
    joints_per_arm=6,
    motors_per_arm=8,
    gello_to_obs_indices=[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15],
    obs_to_gello_indices=[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11],
    jacobian_columns=[0, 0, 1, 1, 2, 3, 4, 5],
    default_operation_modes=[OperatingMode.NONE] * 16,
    left_lock=ArmLockConfig(7, 2, None, track_wrist_offset=True),
    right_lock=ArmLockConfig(7, 2, None, track_wrist_offset=True),
    start_joints=np.array(
        [
            np.pi / 2,
            np.pi / 2,
            np.pi,
            np.pi,
            -np.pi,
            0,
            0,
            0,
            -np.pi / 2,
            -np.pi / 2,
            np.pi,
            np.pi,
            -np.pi,
            0,
            0,
            0,
        ]
    ),
)

R1PRO_CONFIG = BimanualAgentConfig(
    joints_per_arm=7,
    motors_per_arm=9,
    gello_to_obs_indices=[0, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17],
    obs_to_gello_indices=[0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12, 13],
    jacobian_columns=[0, 0, 1, 1, 2, 3, 4, 5, 6],
    default_operation_modes=[OperatingMode.NONE] * 18,
    left_lock=ArmLockConfig(
        5, 3, OperatingMode.EXTENDED_POSITION, track_wrist_offset=False
    ),
    right_lock=ArmLockConfig(
        5, 3, OperatingMode.EXTENDED_POSITION, track_wrist_offset=False
    ),
    start_joints=np.zeros(18),
)

ROBOT_TELEOP_CONFIGS = {
    "r1": R1_CONFIG,
    "r1pro": R1PRO_CONFIG,
}


class BimanualAgent(Agent):
    """Config-driven bimanual teleop agent.

    Owns a single DynamixelArmAgent for the full bimanual robot and
    coordinates joint locking, operational-space force feedback, and
    JoyCon input via config-driven parameters.
    """

    def __init__(
        self,
        config: BimanualAgentConfig,
        arm_agent: DynamixelArmAgent,
        joycon_agent: Optional[JoyconAgent] = None,
        motor_feedback_type: MotorFeedbackConfig = MotorFeedbackConfig.OPERATIONAL_SPACE,
    ):
        super().__init__()
        self.config = config
        self._arm = arm_agent
        self._joycon = joycon_agent
        self._feedback_type = motor_feedback_type
        self._enable_locking = joycon_agent is not None
        self._waiting_to_resume = False

        mpa = config.motors_per_arm
        self._arm_info = {}
        for i, arm in enumerate(("left", "right")):
            lock_cfg = config.left_lock if arm == "left" else config.right_lock
            self._arm_info[arm] = {
                "gello_ids": np.arange(mpa) + i * mpa,
                "lock_config": lock_cfg,
                "locked": {"upper": False, "lower": False, "all": False},
                "locked_wrist_angle": None,
                "colliding": False,
            }

        self._joint_offsets = np.zeros(mpa * 2)

    def _gello_to_obs(self, gello_jnts: np.ndarray) -> np.ndarray:
        return gello_jnts[self.config.gello_to_obs_indices]

    def _obs_to_gello(self, obs: Dict) -> np.ndarray:
        obs_jnts = np.concatenate(
            [
                obs[f"arm_{arm}_joint_positions"].detach().cpu().numpy()
                for arm in ("left", "right")
            ]
        )
        return obs_jnts[self.config.obs_to_gello_indices] + self._joint_offsets

    def _get_jacobians(self, obs: Dict):
        cols = self.config.jacobian_columns
        J_left = obs["arm_left_jacobian"][:3, cols]
        J_right = obs["arm_right_jacobian"][:3, cols]
        return J_left, J_right

    def _compute_feedback_currents(self, joint_error, joint_vel, obs) -> np.ndarray:
        if self._feedback_type == MotorFeedbackConfig.NONE:
            return np.zeros_like(joint_error)
        elif self._feedback_type == MotorFeedbackConfig.JOINT_SPACE:
            kp, kv = self._arm._damping_motor_kp, self._arm._damping_motor_kv
            return (
                kp * 0.2 * (joint_error**2) * np.sign(joint_error)
                - kv * joint_vel * 0.1
            )
        elif self._feedback_type == MotorFeedbackConfig.OPERATIONAL_SPACE:
            J_left, J_right = self._get_jacobians(obs)
            assert J_left.shape == J_right.shape
            J = np.block(
                [
                    [J_left, np.zeros(J_left.shape)],
                    [np.zeros(J_right.shape), J_right],
                ]
            )
            eef_error = J @ np.deg2rad(joint_error)
            kp, kv = self._arm._damping_motor_kp, self._arm._damping_motor_kv
            return kp * J.T @ eef_error - kv * joint_vel * 0.01
        raise ValueError(f"Unknown feedback type: {self._feedback_type}")

    def start(self):
        super().start()
        self._arm.start()
        modes = np.array(self.config.default_operation_modes)
        self._arm.robot.set_operating_mode(modes.tolist())

    def reset(self):
        super().reset()
        modes = np.array(self.config.default_operation_modes)
        self._arm.robot.set_operating_mode(modes.tolist())

        for arm in ("left", "right"):
            info = self._arm_info[arm]
            for k in ("upper", "lower", "all"):
                info["locked"][k] = False
            info["colliding"] = False

        self._arm.reset()

    def act(self, obs: Dict) -> th.Tensor:
        gello_jnts = self._obs_to_gello(obs)
        target_jnts = gello_jnts.copy()

        # Handle wait-to-resume
        if obs["waiting_to_resume"] and not self._waiting_to_resume:
            self._arm.set_reset_qpos(gello_jnts)
            self.reset()
            print("Waiting to resume from sim...")
            self._waiting_to_resume = True
        elif not obs["waiting_to_resume"] and self._waiting_to_resume:
            self.start()
            self._waiting_to_resume = False

        if not self._waiting_to_resume:
            self._apply_impedance_feedback(gello_jnts, target_jnts, obs)
            if self._enable_locking:
                self._handle_joint_locking(obs, gello_jnts)

        # Read joints and convert to obs form
        raw_jnts = self._arm.get_joint_state()
        action = th.from_numpy(self._gello_to_obs(raw_jnts).astype(np.float32))
        # append joycon input if applicable
        if self._joycon is not None:
            jc_input = self._joycon.act(obs)
            action = th.cat([action, jc_input], dim=0)
        return action

    def _apply_impedance_feedback(self, gello_jnts, target_jnts, obs):
        if self._feedback_type == MotorFeedbackConfig.NONE:
            return

        robot = self._arm.robot
        current_idxs = np.where(robot.operating_mode == OperatingMode.CURRENT)[0]
        if len(current_idxs) == 0:
            return

        joint_error = np.rad2deg(target_jnts - gello_jnts)
        jnts_vel = self._arm.get_joint_velocities()
        currents = self._compute_feedback_currents(joint_error, jnts_vel, obs)
        robot.command_current(currents[current_idxs], idxs=current_idxs)

    def _handle_joint_locking(self, obs, gello_jnts):
        robot = self._arm.robot
        mpa = self.config.motors_per_arm
        total = mpa * 2
        operating_modes = np.array(self.config.default_operation_modes[:total])
        active_mode_idxs = np.array([], dtype=int)
        active_cmd_idxs = np.array([], dtype=int)
        commanded_jnts = gello_jnts + self._joint_offsets

        for arm in ("left", "right"):
            info = self._arm_info[arm]
            lock_cfg = info["lock_config"]

            if arm == "left":
                lock_all_signal = self._joycon.gripper_info["-"]["status"]
                lock_lower_signal = self._joycon.jc_left.get_button_l()
            else:
                lock_all_signal = self._joycon.gripper_info["+"]["status"]
                lock_lower_signal = self._joycon.jc_right.get_button_r()

            # Contact-based operating mode switching
            if self._feedback_type != MotorFeedbackConfig.NONE:
                if obs[f"arm_{arm}_contact"] and not info["colliding"]:
                    info["colliding"] = True
                    operating_modes[info["gello_ids"]] = OperatingMode.CURRENT
                    active_mode_idxs = np.concatenate(
                        [active_mode_idxs, info["gello_ids"]]
                    )
                elif not obs[f"arm_{arm}_contact"] and info["colliding"]:
                    info["colliding"] = False
                    operating_modes[info["gello_ids"]] = (
                        self.config.default_operation_modes[:mpa]
                    )
                    active_mode_idxs = np.concatenate(
                        [active_mode_idxs, info["gello_ids"]]
                    )

            # All-arm locking
            if lock_cfg.all_lock_mode is not None:
                active_mode_idxs, active_cmd_idxs = self._do_all_arm_locking(
                    info,
                    lock_all_signal == -1,
                    operating_modes,
                    active_mode_idxs,
                    active_cmd_idxs,
                )

            # Lower arm locking
            active_mode_idxs, active_cmd_idxs = self._do_lower_arm_locking(
                info,
                lock_lower_signal,
                operating_modes,
                active_mode_idxs,
                active_cmd_idxs,
            )

        # Apply operating mode changes and joint commands
        if len(active_mode_idxs) > 0:
            unique_idxs = np.unique(active_mode_idxs.astype(int))
            robot.set_operating_mode(
                operating_modes[unique_idxs].tolist(), idxs=unique_idxs.tolist()
            )

        if len(active_cmd_idxs) > 0:
            unique_idxs = np.unique(active_cmd_idxs.astype(int))
            robot.command_joint_state(commanded_jnts[unique_idxs], idxs=unique_idxs)

    def _do_all_arm_locking(
        self, arm_info, lock_all, operating_modes, active_mode_idxs, active_cmd_idxs
    ):
        is_locked = arm_info["locked"]["all"]
        lock_mode = arm_info["lock_config"].all_lock_mode

        if lock_all and not is_locked:
            operating_modes[arm_info["gello_ids"]] = lock_mode
            active_mode_idxs = np.concatenate([active_mode_idxs, arm_info["gello_ids"]])
            active_cmd_idxs = np.concatenate([active_cmd_idxs, arm_info["gello_ids"]])
            arm_info["locked"]["all"] = True
        elif not lock_all and is_locked:
            mpa = self.config.motors_per_arm
            operating_modes[arm_info["gello_ids"]] = (
                self.config.default_operation_modes[:mpa]
            )
            active_mode_idxs = np.concatenate([active_mode_idxs, arm_info["gello_ids"]])
            arm_info["locked"]["all"] = False

        return active_mode_idxs, active_cmd_idxs

    def _do_lower_arm_locking(
        self, arm_info, lock_lower, operating_modes, active_mode_idxs, active_cmd_idxs
    ):
        is_locked = arm_info["locked"]["lower"]
        lock_cfg = arm_info["lock_config"]
        n = lock_cfg.lower_lock_count

        if lock_lower and not is_locked:
            n_motors = len(arm_info["gello_ids"])
            modes = [OperatingMode.NONE] * (n_motors - n) + [
                OperatingMode.EXTENDED_POSITION
            ] * n
            operating_modes[arm_info["gello_ids"]] = modes
            active_mode_idxs = np.concatenate([active_mode_idxs, arm_info["gello_ids"]])
            active_cmd_idxs = np.concatenate(
                [active_cmd_idxs, arm_info["gello_ids"][-n:]]
            )
            arm_info["locked"]["lower"] = True
        elif not lock_lower and is_locked:
            mpa = self.config.motors_per_arm
            operating_modes[arm_info["gello_ids"]] = (
                self.config.default_operation_modes[:mpa]
            )
            active_mode_idxs = np.concatenate([active_mode_idxs, arm_info["gello_ids"]])
            arm_info["locked"]["lower"] = False

        return active_mode_idxs, active_cmd_idxs
