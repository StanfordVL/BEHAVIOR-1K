import time
from typing import Iterable, Optional, Tuple

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.objects import USDObject
from omnigibson.robots.robot_base import BaseRobot

try:
    from telemoma.configs.base_config import teleop_config
    from telemoma.human_interface.teleop_core import TeleopAction, TeleopObservation
    from telemoma.human_interface.teleop_policy import TeleopPolicy
    from telemoma.utils.general_utils import AttrDict
except ImportError as e:
    raise e from ValueError("For teleoperation, install telemoma by running 'pip install telemoma'")

m = create_module_macros(module_path=__file__)
m.movement_speed = 0.2  # the speed of the robot base movement


class TeleopSystem(TeleopPolicy):
    """
    Base class for teleop policy
    """

    def __init__(self, config: AttrDict, robot: BaseRobot, show_control_marker: bool = False) -> None:
        """
        Initializes the Teleoperation System
        Args:
            config (AttrDict): configuration dictionary
            robot (BaseRobot): the robot that will be controlled.
            show_control_marker (bool): whether to show a visual marker that indicates the target pose of the control.
        """
        super().__init__(config)
        self.teleop_action: TeleopAction = TeleopAction()
        self.robot_obs: TeleopObservation = TeleopObservation()
        self.robot = robot
        self.robot_arms = ["left", "right"] if self.robot.n_arms == 2 else ["right"]
        # robot parameters
        self.movement_speed = m.movement_speed
        self.show_control_marker = show_control_marker
        self.control_markers = {}
        if show_control_marker:
            for arm in robot.arm_names:
                arm_name = "right" if arm == robot.default_arm else "left"
                self.control_markers[arm_name] = USDObject(
                    name=f"target_{arm_name}", usd_path=robot.eef_usd_path[arm], visual_only=True
                )
                self.robot.scene.add_object(self.control_markers[arm_name])

    def get_obs(self) -> TeleopObservation:
        """
        Retrieve observation data from robot
        Returns:
            TeleopObservation: dataclass containing robot observations
        """
        robot_obs = TeleopObservation()
        base_pos, base_orn = self.robot.get_position_orientation()
        robot_obs.base = th.cat((base_pos[:2], th.tensor([T.quat2euler(base_orn)[2]])))
        for i, arm in enumerate(self.robot_arms):
            abs_cur_pos, abs_cur_orn = self.robot.eef_links[
                self.robot.arm_names[self.robot_arms.index(arm)]
            ].get_position_orientation()
            rel_cur_pos, rel_cur_orn = T.relative_pose_transform(abs_cur_pos, abs_cur_orn, base_pos, base_orn)
            gripper_pos = th.mean(
                self.robot.get_joint_positions(normalized=True)[self.robot.gripper_control_idx[self.robot.arm_names[i]]]
            ).unsqueeze(0)
            # if we are grasping, we manually set the gripper position to be at most 0.5
            if self.robot.controllers[f"gripper_{self.robot.arm_names[i]}"].is_grasping():
                gripper_pos = th.min(gripper_pos, th.tensor([0.5]))
            robot_obs[arm] = th.cat((rel_cur_pos, rel_cur_orn, gripper_pos))
        return robot_obs

    def get_action(self, robot_obs: TeleopObservation) -> th.Tensor:
        """
        Generate action data from VR input for robot teleoperation
        Args:
            robot_obs (TeleopObservation): dataclass containing robot observations
        Returns:
            th.tensor: array of action data
        """
        # get teleop action
        self.teleop_action = super().get_action(robot_obs)
        # optionally update control marker
        if self.show_control_marker:
            for arm_name in self.control_markers:
                delta_pos, delta_orn = self.teleop_action[arm_name][:3], T.euler2quat(
                    th.tensor(self.teleop_action[arm_name][3:6])
                )
                rel_target_pos = robot_obs[arm_name][:3] + delta_pos
                rel_target_orn = T.quat_multiply(delta_orn, robot_obs[arm_name][3:7])
                base_pos, base_orn = self.robot.get_position_orientation()
                target_pos, target_orn = T.pose_transform(base_pos, base_orn, rel_target_pos, rel_target_orn)
                self.control_markers[arm_name].set_position_orientation(position=target_pos, orientation=target_orn)
        return self.robot.teleop_data_to_action(self.teleop_action)

    def reset(self) -> None:
        """
        Reset the teleop policy
        """
        self.teleop_action = TeleopAction()
        self.robot_obs = TeleopObservation()
        for interface in self.interfaces.values():
            interface.reset_state()


class OVXRSystem(TeleopSystem):
    """
    VR Teleoperation System build on top of Omniverse XR extension and TeleMoMa's TeleopSystem
    """

    def __init__(
        self,
        robot: BaseRobot,
        show_control_marker: bool = False,
        system: str = "SteamVR",
        disable_display_output: bool = False,
        enable_touchpad_movement: bool = False,
        align_anchor_to_robot_base: bool = False,
        use_hand_tracking: bool = False,
    ) -> None:
        """
        Initializes the VR system
        Args:
            robot (BaseRobot): the robot that VR will control.
            show_control_marker (bool): whether to show a control marker
            system (str): the VR system to use, one of ["OpenXR", "SteamVR"], default is "SteamVR".
            disable_display_output (bool): whether we will not display output to the VR headset (only use controller tracking), default is False.
            enable_touchpad_movement (bool): whether to enable VR system anchor movement by controller, default is False.
            align_anchor_to_robot_base (bool): whether to align VR anchor to robot base, default is False.
            use_hand_tracking (bool): whether to use hand tracking instead of controllers, default is False.
            show_controller (bool): whether to show the controller model in the scene, default is False.

        NOTE: enable_touchpad_movement and align_anchor_to_robot_base cannot be enabled at the same time.
            The former is to enable free movement of the VR system (i.e. the user), while the latter is constraining the VR system to the robot pose.
        """
        self.raw_data = {}
        # enable xr extension
        lazy.omni.isaac.core.utils.extensions.enable_extension("omni.kit.xr.profile.vr")
        self.xr_device_class = lazy.omni.kit.xr.core.XRDeviceClass
        # run super method
        super().__init__(teleop_config, robot, show_control_marker)
        # we want to further slow down the movement speed if we are using touchpad movement
        if enable_touchpad_movement:
            self.movement_speed *= 0.3
        # get xr core and profile
        self.xr_core = lazy.omni.kit.xr.core.XRCore.get_singleton()
        self.vr_profile = self.xr_core.get_profile("vr")
        self.disable_display_output = disable_display_output
        self.enable_touchpad_movement = enable_touchpad_movement
        self.align_anchor_to_robot_base = align_anchor_to_robot_base
        assert not (
            self.enable_touchpad_movement and self.align_anchor_to_robot_base
        ), "enable_touchpad_movement and align_anchor_to_robot_base cannot be True at the same time!"
        # set avatar
        if self.show_control_marker:
            self.vr_profile.set_avatar(
                lazy.omni.kit.xr.ui.stage.common.XRAvatarManager.get_singleton().create_avatar("basic_avatar", {})
            )
        else:
            self.vr_profile.set_avatar(
                lazy.omni.kit.xr.ui.stage.common.XRAvatarManager.get_singleton().create_avatar("empty_avatar", {})
            )
        # set anchor mode to be custom anchor
        lazy.carb.settings.get_settings().set(
            self.vr_profile.get_scene_persistent_path() + "anchorMode", "scene origin"
        )
        # set vr system
        lazy.carb.settings.get_settings().set(self.vr_profile.get_persistent_path() + "system/display", system)
        # set display mode
        lazy.carb.settings.get_settings().set(
            self.vr_profile.get_persistent_path() + "disableDisplayOutput", disable_display_output
        )
        lazy.carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
        # devices info
        self.hmd = None
        self.controllers = {}
        self.trackers = {}
        self.xr2og_orn_offset = th.tensor([0.5, -0.5, -0.5, -0.5])
        self.og2xr_orn_offset = th.tensor([-0.5, 0.5, 0.5, -0.5])
        # setup event subscriptions
        self.reset()
        self.use_hand_tracking = use_hand_tracking
        if use_hand_tracking:
            self.raw_data["hand_data"] = {}
            self.teleop_action.hand_data = {}
            self._hand_tracking_subscription = self.xr_core.get_event_stream().create_subscription_to_pop_by_type(
                lazy.omni.kit.xr.core.XRCoreEventType.hand_joints, self._update_hand_tracking_data, name="hand tracking"
            )

    def xr2og(self, transform: th.tensor) -> Tuple[th.tensor, th.tensor]:
        """
        Apply the orientation offset from the Omniverse XR coordinate system to the OmniGibson coordinate system
        Note that we have to transpose the transform matrix because Omniverse uses row-major matrices
        while OmniGibson uses column-major matrices
        Args:
            transform (th.tensor): the transform matrix in the Omniverse XR coordinate system
        Returns:
            tuple(th.tensor, th.Tensor): the position and orientation in the OmniGibson coordinate system
        """
        pos, orn = T.mat2pose(th.tensor(transform).T)
        orn = T.quat_multiply(orn, self.xr2og_orn_offset)
        return pos, orn

    def og2xr(self, pos: th.tensor, orn: th.tensor) -> th.Tensor:
        """
        Apply the orientation offset from the OmniGibson coordinate system to the Omniverse XR coordinate system
        Args:
            pos (th.tensor): the position in the OmniGibson coordinate system
            orn (th.tensor): the orientation in the OmniGibson coordinate system
        Returns:
            th.tensor: the transform matrix in the Omniverse XR coordinate system
        """
        orn = T.quat_multiply(self.og2xr_orn_offset, orn)
        return T.pose2mat((pos, orn)).T.double()

    def reset(self) -> None:
        """
        Reset the teleop policy
        """
        super().reset()
        self.raw_data = {}
        self.teleop_action.is_valid = {"left": False, "right": False, "head": False}
        self.teleop_action.reset = {"left": False, "right": False}
        self.teleop_action.head = th.zeros(6)

    @property
    def is_enabled(self) -> bool:
        """
        Checks whether the VR system is enabled
        Returns:
            bool: whether the VR system is enabled
        """
        return self.vr_profile.is_enabled()

    def start(self) -> None:
        """
        Enabling the VR profile
        """
        self.vr_profile.request_enable_profile()
        og.sim.step()
        assert self.vr_profile.is_enabled(), "[VRSys] VR profile not enabled!"
        # We want to make sure the hmd is tracking so that the whole system is ready to go
        while True:
            print("[VRSys] Waiting for VR headset to become active...")
            self._update_devices()
            if self.hmd is not None:
                break
            time.sleep(1)
            og.sim.step()

    def stop(self) -> None:
        """
        disable VR profile
        """
        self.xr_core.request_disable_profile()
        og.sim.step()
        assert not self.vr_profile.is_enabled(), "[VRSys] VR profile not disabled!"

    def update(self) -> None:
        """
        Steps the VR system and update self.teleop_action
        """
        # update raw data
        self._update_devices()
        self._update_device_transforms()
        self._update_button_data()
        # Update teleop data based on controller input if not using hand tracking
        if not self.use_hand_tracking:
            self.teleop_action.base = th.zeros(3)
            self.teleop_action.torso = 0.0
            # update right hand related info
            for arm_name, arm in zip(["left", "right"], self.robot_arms):
                if arm in self.controllers:
                    self.teleop_action[arm_name] = th.cat(
                        (
                            self.raw_data["transforms"]["controllers"][arm][0],
                            T.quat2euler(
                                T.quat_multiply(
                                    self.raw_data["transforms"]["controllers"][arm][1],
                                    self.robot.teleop_rotation_offset[self.robot.arm_names[self.robot_arms.index(arm)]],
                                )
                            ),
                            [self.raw_data["button_data"][arm]["axis"]["trigger"]],
                        )
                    )
                    self.teleop_action.is_valid[arm_name] = self._is_valid_transform(
                        self.raw_data["transforms"]["controllers"][arm]
                    )
                else:
                    self.teleop_action.is_valid[arm_name] = False
            # update base and reset info
            if "right" in self.controllers:
                self.teleop_action.reset["right"] = self.raw_data["button_data"]["right"]["press"]["grip"]
                right_axis = self.raw_data["button_data"]["right"]["axis"]
                self.teleop_action.base[0] = right_axis["touchpad_y"] * self.movement_speed
                self.teleop_action.torso = -right_axis["touchpad_x"] * self.movement_speed
            if "left" in self.controllers:
                self.teleop_action.reset["left"] = self.raw_data["button_data"]["left"]["press"]["grip"]
                left_axis = self.raw_data["button_data"]["left"]["axis"]
                self.teleop_action.base[1] = -left_axis["touchpad_x"] * self.movement_speed
                self.teleop_action.base[2] = left_axis["touchpad_y"] * self.movement_speed
        # update head related info
        self.teleop_action.head = th.cat(
            (self.raw_data["transforms"]["head"][0], th.tensor(T.quat2euler(self.raw_data["transforms"]["head"][1])))
        )
        self.teleop_action.is_valid["head"] = self._is_valid_transform(self.raw_data["transforms"]["head"])
        # Optionally move anchor
        if self.enable_touchpad_movement:
            # we use x, y from right controller for 2d movement and y from left controller for z movement
            self._move_anchor(
                pos_offset=th.cat((th.tensor([self.teleop_action.torso]), self.teleop_action.base[[0, 2]]))
            )
        if self.align_anchor_to_robot_base:
            robot_base_pos, robot_base_orn = self.robot.get_position_orientation()
            self.vr_profile.set_virtual_world_anchor_transform(self.og2xr(robot_base_pos, robot_base_orn[[0, 2, 1, 3]]))

    def teleop_data_to_action(self) -> th.Tensor:
        """
        Generate action data from VR input for robot teleoperation
        Returns:
            th.tensor: array of action data
        """
        # optionally update control marker
        if self.show_control_marker:
            self.update_control_marker()
        return self.robot.teleop_data_to_action(self.teleop_action)

    def reset_transform_mapping(self, arm: str = "right") -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            arm(str): name of the arm, one of "left" or "right". Default is "right".
        """
        robot_base_orn = self.robot.get_position_orientation()[1]
        robot_eef_pos = self.robot.eef_links[
            self.robot.arm_names[self.robot_arms.index(arm)]
        ].get_position_orientation()[0]
        target_transform = self.og2xr(pos=robot_eef_pos, orn=robot_base_orn)
        self.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(
            target_transform, self.controllers[arm]
        )

    def set_initial_transform(self, pos: Iterable[float], orn: Iterable[float] = [0, 0, 0, 1]) -> None:
        """
        Function that sets the initial transform of the VR system (w.r.t.) head
        Note that stepping the vr system multiple times is necessary here due to a bug in OVXR plugin
        Args:
            pos(Iterable[float]): initial position of the vr system
            orn(Iterable[float]): initial orientation of the vr system
        """
        for _ in range(10):
            self.update()
            og.sim.step()
        self.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(self.og2xr(pos, orn), self.hmd)

    def _move_anchor(
        self, pos_offset: Optional[Iterable[float]] = None, rot_offset: Optional[Iterable[float]] = None
    ) -> None:
        """
        Updates the anchor of the xr system in the virtual world
        Args:
            pos_offset (Iterable[float]): the position offset to apply to the anchor *in hmd frame*.
            rot_offset (Iterable[float]): the rotation offset to apply to the anchor *in hmd frame*.
        """
        if pos_offset is not None:
            # note that x is forward, y is down, z is left for ovxr, but x is forward, y is left, z is up for og
            pos_offset = th.tensor([-pos_offset[0], pos_offset[2], -pos_offset[1]]).double()
            self.vr_profile.add_move_physical_world_relative_to_device(pos_offset)
        if rot_offset is not None:
            rot_offset = th.tensor(rot_offset).double()
            self.vr_profile.add_rotate_physical_world_around_device(rot_offset)

    def _is_valid_transform(self, transform: Tuple[th.tensor, th.tensor]) -> bool:
        """
        Determine whether the transform is valid (ovxr plugin will return a zero position and rotation if not valid)
        """
        return th.any(transform[0] != th.zeros(3)) and th.any(transform[1] != self.og2xr_orn_offset)

    def _update_devices(self) -> None:
        """
        Update the VR device list
        """
        for device in self.vr_profile.get_device_list():
            if device.get_class() == self.xr_device_class.xrdisplaydevice:
                self.hmd = device
            elif device.get_class() == self.xr_device_class.xrcontroller:
                # we want the first 2 controllers to be corresponding to the left and right hand
                d_idx = device.get_index()
                controller_name = ["left", "right"][d_idx] if d_idx < 2 else f"controller_{d_idx+1}"
                self.controllers[controller_name] = device
            elif device.get_class() == self.xr_device_class.xrtracker:
                self.trackers[device.get_index()] = device

    def _update_device_transforms(self) -> None:
        """
        Get the transform matrix of each VR device *in world frame* and store in self.raw_data
        """
        transforms = {}
        transforms["head"] = self.xr2og(self.hmd.get_virtual_world_pose())
        transforms["controllers"] = {}
        transforms["trackers"] = {}
        for controller_name in self.controllers:
            transforms["controllers"][controller_name] = self.xr2og(
                self.controllers[controller_name].get_virtual_world_pose()
            )
        for tracker_index in self.trackers:
            transforms["trackers"][tracker_index] = self.xr2og(self.trackers[tracker_index].get_virtual_world_pose())
        self.raw_data["transforms"] = transforms

    def _update_button_data(self):
        """
        Get the button data for each controller and store in self.raw_data
        Returns:
            dict: a dictionary of whether each button is pressed or touched, and the axis state for touchpad and joysticks
        """
        button_data = {}
        for controller_name in self.controllers:
            button_data[controller_name] = {}
            button_data[controller_name]["press"] = self.controllers[controller_name].get_button_press_state()
            button_data[controller_name]["touch"] = self.controllers[controller_name].get_button_touch_state()
            button_data[controller_name]["axis"] = self.controllers[controller_name].get_axis_state()
        self.raw_data["button_data"] = button_data

    def _update_hand_tracking_data(self, e) -> None:
        """
        Get hand tracking data, see https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints for joint indices
        Args:
            e (carb.events.IEvent): event that contains hand tracking data as payload
        """
        e.consume()
        data_dict = e.payload
        for hand_name, hand in zip(["left, right"], self.robot_arms):
            if data_dict[f"joint_count_{hand}"] != 0:
                self.teleop_action.is_valid[hand_name] = True
                self.raw_data["hand_data"][hand] = {"pos": [], "orn": []}
                # hand_joint_matrices is an array of flattened 4x4 transform matrices for the 26 hand markers
                hand_joint_matrices = data_dict[f"joint_matrices_{hand}"]
                for i in range(26):
                    # extract the pose from the flattened transform matrix
                    pos, orn = self.xr2og(hand_joint_matrices[16 * i : 16 * (i + 1)].reshape(4, 4))
                    self.raw_data["hand_data"][hand]["pos"].append(pos)
                    self.raw_data["hand_data"][hand]["orn"].append(orn)
                    self.teleop_action[hand_name] = th.cat(
                        (
                            self.raw_data["hand_data"][hand]["pos"][0],
                            th.tensor(
                                T.quat2euler(
                                    T.quat_multiply(
                                        self.raw_data["hand_data"][hand]["orn"][0],
                                        self.robot.teleop_rotation_offset[
                                            self.robot.arm_names[self.robot_arms.index(hand)]
                                        ],
                                    )
                                )
                            ),
                            th.tensor([0]),
                        )
                    )
                # Get each finger joint's rotation angle from hand tracking data
                # joint_angles is a 5 x 3 array of joint rotations (from thumb to pinky, from base to tip)
                joint_angles = th.zeros((5, 3))
                raw_hand_data = self.raw_data["hand_data"][hand]["pos"]
                for i in range(5):
                    for j in range(3):
                        # get the 3 related joints indices
                        prev_joint_idx, cur_joint_idx, next_joint_idx = i * 5 + j + 1, i * 5 + j + 2, i * 5 + j + 3
                        # get the 3 related joints' positions
                        prev_joint_pos = raw_hand_data[prev_joint_idx]
                        cur_joint_pos = raw_hand_data[cur_joint_idx]
                        next_joint_pos = raw_hand_data[next_joint_idx]
                        # calculate the angle formed by 3 points
                        v1 = cur_joint_pos - prev_joint_pos
                        v2 = next_joint_pos - cur_joint_pos
                        v1 /= th.norm(v1)
                        v2 /= th.norm(v2)
                        joint_angles[i, j] = th.arccos(v1 @ v2)
                self.teleop_action.hand_data[hand_name] = joint_angles
