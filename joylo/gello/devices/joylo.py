import torch as th
import numpy as np
import gello.robots.sim_robot.og_teleop_utils as utils
import omnigibson as og
from gello.devices.device_base import BaseDevice
from gello.robots.sim_robot.og_teleop_cfg import *
from gello.robots.sim_robot.zmq_server import ZMQRobotServer, ZMQServerThread
from omnigibson.robots import R1, R1Pro
from omnigibson.utils.usd_utils import GripperRigidContactAPI, ControllableObjectViewAPI
from typing import Dict


class JoyLo(BaseDevice):
    """
    A class to control the JoyLo robot using ROS2.
    """

    def __init__(self, robot, host, port, *args, **kwargs):
        """
        Initialize the JoyLo controller.
        """
        super().__init__(robot)
        self._gripper_action_signal_detectors = {arm: utils.SignalChangeDetector(debounce_time=0.5) for arm in self.robot.arm_names}
        self._current_trunk_tilt = 0.0
        # Set default active arm
        self._active_arm = "right"
        self._arm_shoulder_directions = {"left": -1.0, "right": 1.0}
        self.obs = {"in_cooldown": False, "waiting_to_resume": True}

        # Cache values
        self._reset_max_arm_delta = DEFAULT_RESET_DELTA_SPEED * (np.pi / 180) * og.sim.get_sim_step_dt()
        qpos_min, qpos_max = self.robot.joint_lower_limits, self.robot.joint_upper_limits
        self._trunk_tilt_limits = {"lower": qpos_min[self.robot.trunk_control_idx][2],
                                   "upper": qpos_max[self.robot.trunk_control_idx][2]}
        self._arm_joint_limits = dict()
        for arm in self.robot.arm_names:
            self._arm_joint_limits[arm] = {
                "lower": qpos_min[self.robot.arm_control_idx[arm]],
                "upper": qpos_max[self.robot.arm_control_idx[arm]],
            }

        # Create ZMQ server for communication
        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port, verbose=False)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

    def start(self):
        self._zmq_server_thread.start()

    def pause(self):
        """
        Pause the control of the robot.
        """
        for detector in self._gripper_action_signal_detectors.values():
            detector.reset()

    def update_observations(self):
        """
        Update the observations of the robot based on the joint positions.
        """
        # Loop over all arms and grab relevant joint info
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        finger_impulses = GripperRigidContactAPI.get_all_impulses(self.env.scene.idx) if INCLUDE_FINGER_CONTACT_OBS else None

        obs = dict()
        obs["active_arm"] = self._active_arm
        obs["base_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.non_floor_touching_base_links) if INCLUDE_BASE_CONTACT_OBS else False
        obs["trunk_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.trunk_links) if INCLUDE_TRUNK_CONTACT_OBS else False
        obs["reset_joints"] = bool(self._joint_cmd["button_y"][0].item())

        for i, arm in enumerate(self.robot.arm_names):
            arm_control_idx = self.robot.arm_control_idx[arm]
            obs[f"arm_{arm}_control_idx"] = arm_control_idx
            obs[f"arm_{arm}_joint_positions"] = joint_pos[arm_control_idx]
            # Account for tilt offset
            obs[f"arm_{arm}_joint_positions"][0] -= self._current_trunk_tilt * self._arm_shoulder_directions[arm]
            obs[f"arm_{arm}_joint_velocities"] = joint_vel[arm_control_idx]
            obs[f"arm_{arm}_gripper_positions"] = joint_pos[self.robot.gripper_control_idx[arm]]
            obs[f"arm_{arm}_ee_pos_quat"] = th.concatenate(self.robot.eef_links[arm].get_position_orientation())
            # When using VR, this expansive check makes the view glitch
            obs[f"arm_{arm}_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.arm_links[arm]) if VIEWING_MODE != ViewingMode.VR and INCLUDE_ARM_CONTACT_OBS else False
            obs[f"arm_{arm}_finger_max_contact"] = th.max(th.sum(th.square(finger_impulses[:, 2*i:2*(i+1), :]), dim=-1)).item() if INCLUDE_FINGER_CONTACT_OBS else 0.0

            obs[f"{arm}_gripper"] = self._joint_cmd[f"{arm}_gripper"].item()

        if INCLUDE_JACOBIAN_OBS:
            for arm in self.robot.arm_names:
                link_name = self.robot.eef_link_names[arm]

                start_idx = 0 if self.robot.fixed_base else 6
                link_idx = self.robot._articulation_view.get_body_index(link_name)
                jacobian = ControllableObjectViewAPI.get_relative_jacobian(
                    self.robot.articulation_root_path
                )[-(self.robot.n_links - link_idx), :, start_idx : start_idx + self.robot.n_joints]
                
                jacobian = jacobian[:, self.robot.arm_control_idx[arm]]
                obs[f"arm_{arm}_jacobian"] = jacobian
        self.obs.update(obs)

    def get_action(self, in_cooldown: bool) -> th.Tensor:
        # Start an empty action
        action = th.zeros(self.robot.action_dim)
        # Apply arm action + extra dimension from base
        if isinstance(self.robot, R1):
            # Apply arm action
            left_act = self._joint_cmd["left_arm"].clone().clip(self._arm_joint_limits["left"]["lower"], self._arm_joint_limits["left"]["upper"])
            right_act = self._joint_cmd["right_arm"].clone().clip(self._arm_joint_limits["right"]["lower"], self._arm_joint_limits["right"]["upper"])

            # If we're in cooldown, clip values based on max delta value
            if in_cooldown:
                robot_pos = self.robot.get_joint_positions()
                robot_left_pos, robot_right_pos = [robot_pos[self.robot.arm_control_idx[arm]] for arm in ("left", "right")]
                robot_left_delta = left_act - robot_left_pos
                robot_right_delta = right_act - robot_right_pos
                left_act = robot_left_pos + robot_left_delta.clip(-self._reset_max_arm_delta, self._reset_max_arm_delta)
                right_act = robot_right_pos + robot_right_delta.clip(-self._reset_max_arm_delta, self._reset_max_arm_delta)

            left_act[0] += self._current_trunk_tilt * self._arm_shoulder_directions["left"]
            right_act[0] += self._current_trunk_tilt * self._arm_shoulder_directions["right"]
            action[self.robot.arm_action_idx["left"]] = left_act
            action[self.robot.arm_action_idx["right"]] = right_act

            # Apply base action
            action[self.robot.base_action_idx] = self._joint_cmd["base"].clone()

            # Apply gripper action
            for arm in self.robot.arm_names:
                gripper_signal = self._joint_cmd[f"{arm}_gripper"].item()
                gripper_changed = self._gripper_action_signal_detectors[arm].process_sample(gripper_signal)
                if gripper_changed:
                    self.grasp_action[arm] = -self.grasp_action[arm]
                action[self.robot.gripper_action_idx[arm]] = self.grasp_action[arm]

            # Apply trunk action
            if SIMPLIFIED_TRUNK_CONTROL:
                # Update trunk translation (height)
                self.current_trunk_translate = float(th.clamp(
                    th.tensor(self.current_trunk_translate, dtype=th.float) - 
                    th.tensor(self._joint_cmd["trunk"][0].item() * og.sim.get_sim_step_dt(), dtype=th.float),
                    0.0,
                    2.0
                ))
                trunk_action = utils.infer_torso_qpos_from_trunk_translate(self.current_trunk_translate)
                
                # Update trunk tilt offset
                self.current_trunk_tilt_offset = float(th.clamp(
                    th.tensor(self.current_trunk_tilt_offset, dtype=th.float) + 
                    th.tensor(self._joint_cmd["trunk"][1].item() * og.sim.get_sim_step_dt(), dtype=th.float),
                    self._trunk_tilt_limits["lower"] - trunk_action[2],
                    self._trunk_tilt_limits["upper"] - trunk_action[2]
                ))
                trunk_action[2] = trunk_action[2] + self.current_trunk_tilt_offset

                action[self.robot.trunk_action_idx] = trunk_action
        else:
            action[self.robot.arm_action_idx[self._active_arm]] = self._joint_cmd[self._active_arm].clone()

        return action

    def is_running(self) -> bool:
        """
        Check if the controller is running.
         
        Returns:
            bool: True if the controller is running, False otherwise.
        """
        return True

    def stop(self):
        self._zmq_server_thread.terminate()
        self._zmq_server_thread.join()

    def num_dofs(self) -> int:
        """Return the number of degrees of freedom"""
        return self.robot.n_joints

    def get_joint_state(self) -> th.tensor:
        """Get the current joint state"""
        return self._joint_state
    
    def get_observations(self) -> Dict[str, th.tensor]:
        """Get the current observations"""
        return self.obs
    
    def command_joint_state(self, joint_state: th.tensor, component=None) -> None:
        """
        Command the robot to a joint state
        
        Args:
            joint_state: Target joint state
            component: Which component to control (optional)
        """
        # If R1, process manually
        state = joint_state.clone()
        if isinstance(self.robot, R1) and not isinstance(self.robot, R1Pro):
            # [ 6DOF left arm, 6DOF right arm, 3DOF base, 2DOF trunk (z, ry), 2DOF gripper, -, +, X, Y, B, A, home, left arrow, right arrow buttons]
            start_idx = 0
            for component, dim in zip(
                    ("left_arm", "right_arm", "base", "trunk", "left_gripper", "right_gripper", "button_-", "button_+", "button_x", "button_y", "button_b", "button_a", "button_capture", "button_home", "button_left", "button_right"),
                    (6, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            ):
                if start_idx >= len(state):
                    break
                self._joint_cmd[component] = state[start_idx: start_idx + dim]
                start_idx += dim
        elif isinstance(self.robot, R1Pro):
            # [ 7DOF left arm, 7DOF right arm, 3DOF base, 2DOF trunk (z, ry), 2DOF gripper, -, +, X, Y, B, A, home, left arrow, right arrow buttons]
            start_idx = 0
            for component, dim in zip(
                    ("left_arm", "right_arm", "base", "trunk", "left_gripper", "right_gripper", "button_-", "button_+", "button_x", "button_y", "button_b", "button_a", "button_capture", "button_home", "button_left", "button_right"),
                    (7, 7, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            ):
                if start_idx >= len(state):
                    break
                self._joint_cmd[component] = state[start_idx: start_idx + dim]
                start_idx += dim
        else:
            # Sort by component
            if component is None:
                component = self._active_arm
            assert component in self._joint_cmd, \
                f"Got invalid component joint cmd: {component}. Valid options: {self._joint_cmd.keys()}"
            self._joint_cmd[component] = joint_state.clone()

    def get_base_cmd(self):
        return self._joint_cmd["base"]
    
    def get_button_input_cmd(self):
        return {
            "button_x": self._joint_cmd["button_x"].item(),
            "button_y": self._joint_cmd["button_y"].item(),
            "button_b": self._joint_cmd["button_b"].item(),
            "button_a": self._joint_cmd["button_a"].item(),
            "button_capture": self._joint_cmd["button_capture"].item(),
            "button_home": self._joint_cmd["button_home"].item(),
            "button_left": self._joint_cmd["button_left"].item(),
            "button_right": self._joint_cmd["button_right"].item()
        }

    def reset(self):
        for detector in self._gripper_action_signal_detectors.values():
            detector.reset()
        self._joint_state = self.robot.reset_joint_pos
        self._joint_cmd = {
            f"{arm}_arm": self._joint_state[self.robot.arm_control_idx[arm]] for arm in self.robot.arm_names
        }
        self._current_trunk_tilt = 0.0
        if isinstance(self.robot, (R1, R1Pro)):
            for arm in self.robot.arm_names:
                self._joint_cmd[f"{arm}_gripper"] = th.ones(len(self.robot.gripper_action_idx[arm]))
                self._joint_cmd["base"] = self._joint_state[self.robot.base_control_idx]
                self._joint_cmd["trunk"] = th.zeros(2)
                self._joint_cmd["button_-"] = th.zeros(1)
                self._joint_cmd["button_+"] = th.zeros(1)
                self._joint_cmd["button_x"] = th.zeros(1)
                self._joint_cmd["button_y"] = th.zeros(1)
                self._joint_cmd["button_b"] = th.zeros(1)
                self._joint_cmd["button_a"] = th.zeros(1)
                self._joint_cmd["button_capture"] = th.zeros(1)
                self._joint_cmd["button_home"] = th.zeros(1)
                self._joint_cmd["button_left"] = th.zeros(1)
                self._joint_cmd["button_right"] = th.zeros(1)