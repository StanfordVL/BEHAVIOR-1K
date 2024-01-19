import numpy as np
import omni
import time
from dataclasses import dataclass, field
from importlib import import_module
from threading import Thread
from typing import Any, Iterable, Optional, Tuple

import omnigibson as og
import omnigibson.lazy_omni as lo
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.objects import USDObject
from omnigibson.robots.robot_base import BaseRobot


m = create_module_macros(module_path=__file__)
m.movement_speed = 0.2  # the speed of the robot base movement


@dataclass
class TeleopData:
    is_valid: dict[str, bool] = field(default_factory=dict)
    transforms: dict[str, Any] = field(default_factory=dict)
    grippers: dict[str, float] = field(default_factory=dict)
    reset: dict[str, bool] = field(default_factory=dict)
    robot_attached: bool = False
    n_arms: int = 2

    def __post_init__(self) -> None:
        arms = ["left", "right"] if self.n_arms == 2 else ["right"]
        self.is_valid = {arm: False for arm in arms}
        self.transforms = {arm: (np.zeros(3), np.array([0, 0, 0, 1])) for arm in arms}
        self.grippers = {arm: 0. for arm in arms}
        self.reset = {arm: False for arm in arms}


class TeleopSystem:
    """
    Base class for teleop systems
    """

    def __init__(self, robot: BaseRobot, show_control_marker: bool = True, *args, **kwargs) -> None:
        """
        Initializes the VR system
        Args:
            robot (BaseRobot): the robot that will be controlled.
            show_control_marker (bool): whether to show a visual marker that indicates the target pose of the control.
        """
        # raw data directly obtained from the teleoperation device
        self.raw_data = {
            "transforms": {},
            "button_data": {},
        }
        self.robot = robot
        self.robot_arms = ["left", "right"] if self.robot.n_arms == 2 else ["right"]
        self.teleop_data = TeleopData(n_arms=self.robot.n_arms)
        # device_origin stores the original pose of the teleop device (e.g. VR controller) upon calling reset_transform_mapping
        # This is used to subtract with the current teleop device pose in each frame to get the delta pose
        self.device_origin = {arm: T.mat2pose(np.eye(4)) for arm in self.robot_arms}
        # robot_origin stores the relative pose of the robot eef w.r.t. to the robot base upon calling reset_transform_mapping
        # This is used to apply the delta offset of the VR controller to get the final target robot eef pose (relative to the robot base) 
        self.robot_origin = {arm: T.mat2pose(np.eye(4)) for arm in self.robot_arms}
        # robot parameters
        self.robot_attached = False
        self.movement_speed = m.movement_speed
        self.show_control_marker = show_control_marker
        self.control_markers = {}
        if show_control_marker:
            for arm in robot.arm_names:
                arm_name = "right" if arm == robot.default_arm else "left"
                self.control_markers[arm_name] = USDObject(name=f"target_{arm_name}", usd_path=robot.eef_usd_path[arm],
                                                           visual_only=True)
                og.sim.import_object(self.control_markers[arm_name])

    def start(self) -> None:
        """
        Starts the teleop system
        """
        raise NotImplementedError

    def stop(self) -> None:
        """
        Stops the teleop system
        """
        raise NotImplementedError

    def update(self) -> None:
        """
        Update self.teleop_data
        NOTE: all tracking data should be in world frame
        """
        raise NotImplementedError

    def teleop_data_to_action(self) -> np.ndarray:
        """
        Generate action data from VR input for robot teleoperation
        Returns:
            np.ndarray: array of action data
        """
        # optionally update control marker
        if self.show_control_marker:
            self.update_control_marker()
        return self.robot.teleop_data_to_action(self.teleop_data)

    def update_control_marker(self) -> None:
        """
        Update the target object's pose to the VR right controller's pose
        """
        for arm_name in self.control_markers:
            if arm_name in self.teleop_data.transforms:
                self.control_markers[arm_name].set_position_orientation(*self.teleop_data.transforms[arm_name])

    def reset_transform_mapping(self, arm: str = "right") -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            arm(str): name of the arm, one of "left" or "right". Default is "right".
        """
        robot_base_pose = self.robot.get_position_orientation()
        eef_pose = self.robot.eef_links[self.robot.arm_names[self.robot_arms.index(arm)]].get_position_orientation()
        self.robot_origin[arm] = T.relative_pose_transform(*eef_pose, *robot_base_pose)


class OVXRSystem(TeleopSystem):
    """
    VR Teleoperation System build on top of Omniverse XR extension
    """

    def __init__(
            self,
            robot: BaseRobot,
            show_control_marker: bool = True,
            system: str = "SteamVR",
            disable_display_output: bool = False,
            enable_touchpad_movement: bool = False,
            align_anchor_to_robot_base: bool = False,
            use_hand_tracking: bool = False,
            *args,
            **kwargs
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
        # enable xr extension
        lo.omni.isaac.core.utils.extensions.enable_extension("omni.kit.xr.profile.vr")
        self.xr_device_class = lo.omni.kit.xr.core.XRDeviceClass
        # run super method
        super().__init__(robot, show_control_marker)
        # we want to further slow down the movement speed if we are using touchpad movement
        if enable_touchpad_movement:
            self.movement_speed *= 0.1
        # get xr core and profile
        self.xr_core = lo.omni.kit.xr.core.XRCore.get_singleton()
        self.vr_profile = self.xr_core.get_profile("vr")
        self.disable_display_output = disable_display_output
        self.enable_touchpad_movement = enable_touchpad_movement
        self.align_anchor_to_robot_base = align_anchor_to_robot_base
        assert not (self.enable_touchpad_movement and self.align_anchor_to_robot_base), \
            "enable_touchpad_movement and align_anchor_to_robot_base cannot be True at the same time!"
        # set avatar
        if self.show_control_marker:
            self.vr_profile.set_avatar(lo.omni.kit.xr.ui.stage.common.XRAvatarManager.get_singleton().create_avatar("basic_avatar", {}))
        else:
            self.vr_profile.set_avatar(lo.omni.kit.xr.ui.stage.common.XRAvatarManager.get_singleton().create_avatar("empty_avatar", {}))
        # # set anchor mode to be custom anchor
        lo.carb.settings.get_settings().set(self.vr_profile.get_scene_persistent_path() + "anchorMode", "scene origin")
        # set vr system
        lo.carb.settings.get_settings().set(self.vr_profile.get_persistent_path() + "system/display", system)
        # set display mode
        lo.carb.settings.get_settings().set(
            self.vr_profile.get_persistent_path() + "disableDisplayOutput", disable_display_output
        )
        lo.carb.settings.get_settings().set('/rtx/rendermode', "RaytracedLighting")
        # devices info
        self.hmd = None
        self.controllers = {}
        self.trackers = {}
        self.reset_button_pressed = False
        self.xr2og_orn_offset = np.array([0.5, -0.5, -0.5, -0.5])
        self.og2xr_orn_offset = np.array([-0.5, 0.5, 0.5, -0.5])
        # setup event subscriptions
        self.use_hand_tracking = use_hand_tracking
        if use_hand_tracking:
            self.raw_data["hand_data"] = {}
            self.teleop_data.hand_data = {}
            self._hand_tracking_subscription = self.xr_core.get_event_stream().create_subscription_to_pop_by_type(
                lo.omni.kit.xr.core.XRCoreEventType.hand_joints, self._update_hand_tracking_data, name="hand tracking"
            )

    def xr2og(self, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the orientation offset from the Omniverse XR coordinate system to the OmniGibson coordinate system
        Note that we have to transpose the transform matrix because Omniverse uses row-major matrices 
        while OmniGibson uses column-major matrices
        Args:
            transform (np.ndarray): the transform matrix in the Omniverse XR coordinate system
        Returns:
            tuple(np.ndarray, np.ndarray): the position and orientation in the OmniGibson coordinate system
        """
        pos, orn = T.mat2pose(np.array(transform).T)
        orn = T.quat_multiply(orn, self.xr2og_orn_offset)
        return pos, orn

    def og2xr(self, pos: np.ndarray, orn: np.ndarray) -> np.ndarray:
        """
        Apply the orientation offset from the OmniGibson coordinate system to the Omniverse XR coordinate system
        Args:
            pos (np.ndarray): the position in the OmniGibson coordinate system
            orn (np.ndarray): the orientation in the OmniGibson coordinate system
        Returns:
            np.ndarray: the transform matrix in the Omniverse XR coordinate system
        """
        orn = T.quat_multiply(self.og2xr_orn_offset, orn)
        return T.pose2mat((pos, orn)).T.astype(np.float64)

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
        Steps the VR system and update self.teleop_data
        """
        # update raw data
        self._update_devices()
        self._update_device_transforms()
        self._update_button_data()
        # Update teleop data based on controller input if not using hand tracking
        if not self.use_hand_tracking:
            self.teleop_data.transforms["base"] = np.zeros(4)
            # update right hand related info
            for arm in self.robot_arms:
                if arm in self.controllers:
                    self.teleop_data.transforms[arm] = (
                        self.raw_data["transforms"]["controllers"][arm][0],
                        T.quat_multiply(
                            self.raw_data["transforms"]["controllers"][arm][1],
                            self.robot.teleop_rotation_offset[self.robot.arm_names[self.robot_arms.index(arm)]]
                        )
                    )
                    self.teleop_data.grippers[arm] = self.raw_data["button_data"][arm]["axis"]["trigger"]
                    self.teleop_data.is_valid[arm] = self._is_valid_transform(self.raw_data["transforms"]["controllers"][arm])
                else:
                    self.teleop_data.is_valid[arm] = False
            # update base and reset info
            if "right" in self.controllers:
                self.teleop_data.reset["right"] = self.raw_data["button_data"]["right"]["press"]["grip"]
                right_axis = self.raw_data["button_data"]["right"]["axis"]
                self.teleop_data.transforms["base"][0] = right_axis["touchpad_y"] * self.movement_speed
                self.teleop_data.transforms["base"][3] = -right_axis["touchpad_x"] * self.movement_speed
            if "left" in self.controllers:
                self.teleop_data.reset["left"] = self.raw_data["button_data"]["left"]["press"]["grip"]
                left_axis = self.raw_data["button_data"]["left"]["axis"]
                self.teleop_data.transforms["base"][1] = -left_axis["touchpad_x"] * self.movement_speed
                self.teleop_data.transforms["base"][2] = left_axis["touchpad_y"] * self.movement_speed
        # update head related info
        self.teleop_data.transforms["head"] = self.raw_data["transforms"]["head"]
        self.teleop_data.is_valid["head"] = self._is_valid_transform(self.teleop_data.transforms["head"])
        # update robot reset and attachment info
        if "right" in self.controllers and self.raw_data["button_data"]["right"]["press"]["grip"]:
            if not self.reset_button_pressed:
                self.reset_button_pressed = True
                self.robot_attached = not self.robot_attached
        else:
            self.reset_button_pressed = False
        self.teleop_data.robot_attached = self.robot_attached
        # Optionally move anchor    
        if self.enable_touchpad_movement:
            # we use x, y from right controller for 2d movement and y from left controller for z movement
            self._move_anchor(pos_offset=self.teleop_data.transforms["base"][[3, 0, 2]])
        if self.align_anchor_to_robot_base:
            robot_base_pos, robot_base_orn = self.robot.get_position_orientation()
            self.vr_profile.set_virtual_world_anchor_transform(self.og2xr(robot_base_pos, robot_base_orn[[0, 2, 1, 3]]))

    def reset_transform_mapping(self, arm: str = "right") -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            arm(str): name of the arm, one of "left" or "right". Default is "right".
        """
        robot_base_orn = self.robot.get_orientation()
        robot_eef_pos = self.robot.eef_links[self.robot.arm_names[self.robot_arms.index(arm)]].get_position()
        target_transform = self.og2xr(pos=robot_eef_pos, orn=robot_base_orn)
        self.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(target_transform, self.controllers[arm])

    def set_initial_transform(self, pos: Iterable[float], orn: Iterable[float]=[0, 0, 0, 1]) -> None:
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
        self, 
        pos_offset: Optional[Iterable[float]] = None,
        rot_offset: Optional[Iterable[float]] = None
    ) -> None:
        """
        Updates the anchor of the xr system in the virtual world
        Args:
            pos_offset (Iterable[float]): the position offset to apply to the anchor *in hmd frame*.
            rot_offset (Iterable[float]): the rotation offset to apply to the anchor *in hmd frame*. 
        """
        if pos_offset is not None:
            # note that x is forward, y is down, z is left for ovxr, but x is forward, y is left, z is up for og
            pos_offset = np.array([-pos_offset[0], pos_offset[2], -pos_offset[1]]).astype(np.float64)
            self.vr_profile.add_move_physical_world_relative_to_device(pos_offset)
        if rot_offset is not None:
            rot_offset = np.array(rot_offset).astype(np.float64)
            self.vr_profile.add_rotate_physical_world_around_device(rot_offset)

    def _is_valid_transform(self, transform: Tuple[np.ndarray, np.ndarray]) -> bool:
        """
        Determine whether the transform is valid (ovxr plugin will return a zero position and rotation if not valid)
        """
        return np.any(np.not_equal(transform[0], np.zeros(3))) \
            and np.any(np.not_equal(transform[1], self.og2xr_orn_offset))

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
                self.controllers[controller_name].get_virtual_world_pose())
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
        for hand in self.robot_arms:
            if data_dict[f"joint_count_{hand}"] != 0:
                self.teleop_data.is_valid[hand] = True
                self.raw_data["hand_data"][hand] = {"pos": [], "orn": []}
                # hand_joint_matrices is an array of flattened 4x4 transform matrices for the 26 hand markers
                hand_joint_matrices = data_dict[f"joint_matrices_{hand}"]
                for i in range(26):
                    # extract the pose from the flattened transform matrix
                    pos, orn = self.xr2og(np.reshape(hand_joint_matrices[16 * i: 16 * (i + 1)], (4, 4)))
                    self.raw_data["hand_data"][hand]["pos"].append(pos)
                    self.raw_data["hand_data"][hand]["orn"].append(orn)
                self.teleop_data.transforms[hand] = (
                    self.raw_data["hand_data"][hand]["pos"][0], 
                    T.quat_multiply(
                        self.raw_data["hand_data"][hand]["orn"][0],
                        self.robot.teleop_rotation_offset[self.robot.arm_names[self.robot_arms.index(hand)]]
                    )
                )
                # Get each finger joint's rotation angle from hand tracking data
                # joint_angles is a 5 x 3 array of joint rotations (from thumb to pinky, from base to tip)
                joint_angles = np.zeros((5, 3))
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
                        v1 /= np.linalg.norm(v1)
                        v2 /= np.linalg.norm(v2)
                        joint_angles[i, j] = np.arccos(v1 @ v2)
                self.teleop_data.hand_data[hand] = joint_angles
            else:
                self.teleop_data.is_valid[hand] = False


class OculusReaderSystem(TeleopSystem):
    """
    NOTE: The origin of the oculus system is the headset position. For orientation, x is right, y is up, z is back
    """

    def __init__(self, robot: BaseRobot, show_control_marker: bool = True, *args, **kwargs) -> None:
        try:
            import oculus_reader
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "[OculusReaderSys] Please install oculus_reader (https://github.com/rail-berkeley/oculus_reader) to use OculusReaderSystem"
            )
        super().__init__(robot, show_control_marker)
        # initialize oculus reader
        self.oculus_reader = oculus_reader.OculusReader(run=False)
        self.reset_button_pressed = False

    def oc2og(self, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the orientation offset from the OculusReader coordinate system to the OmniGibson coordinate system
        Args:
            transform (np.ndarray): the transform matrix in the OculusReader coordinate system
        Returns:
            tuple(np.ndarray, np.ndarray): the position and orientation in the OmniGibson coordinate system
        """
        return T.mat2pose(
            T.pose2mat(([0, 0, 0], T.euler2quat([np.pi / 2, 0, np.pi / 2]))) @ transform @ T.pose2mat(
                ([0, 0, 0], T.euler2quat([-np.pi / 2, np.pi / 2, 0])))
        )

    def start(self) -> None:
        """
        Start the oculus reader and the data thread
        """
        self.oculus_reader.run()
        self.data_thread = Thread(target=self._update_internal_data, daemon=True)
        self.data_thread.start()

    def stop(self) -> None:
        """
        Stop the oculus reader and the data thread
        """
        self.data_thread.join()
        self.oculus_reader.stop()

    def _update_internal_data(self, hz: float = 50.) -> None:
        """
        Thread that updates the raw data at a given frenquency
        Args:
            hz (float): the frequency to update the raw data, default is 50.
        """
        while True:
            time.sleep(1 / hz)
            transform, self.raw_data["button_data"] = self.oculus_reader.get_transformations_and_buttons()
            for hand in ["left", "right"]:
                if hand[0] in transform:
                    self.raw_data["transforms"][hand] = self.oc2og(transform[hand[0]])

    def update(self) -> None:
        """
        Steps the VR system and update self.teleop_data
        """
        # generate teleop data
        self.teleop_data.transforms["base"] = np.zeros(4)
        robot_based_pose = self.robot.get_position_orientation()
        # update transform data
        for hand in self.robot_arms:
            if hand in self.raw_data["transforms"]:
                self.teleop_data.is_valid[hand] = True
                delta_pos, delta_orn = T.relative_pose_transform(*self.raw_data["transforms"][hand],
                                                                 *self.device_origin[hand])
                target_rel_pos = self.robot_origin[hand][0] + delta_pos
                target_rel_orn = T.quat_multiply(delta_orn, self.robot_origin[hand][1])
                self.teleop_data.transforms[hand] = T.pose_transform(*robot_based_pose, target_rel_pos,
                                                                     target_rel_orn)
            if f"{hand}Trig" in self.raw_data["button_data"]:
                self.teleop_data.grippers[hand] = self.raw_data["button_data"][f"{hand}Trig"][0]
            else:
                self.teleop_data.is_valid[hand] = False
        # update button data
        if "rightJS" in self.raw_data["button_data"]:
            rightJS_data = self.raw_data["button_data"]["rightJS"]
            self.teleop_data.transforms["base"][0] = rightJS_data[1] * self.movement_speed
            self.teleop_data.transforms["base"][3] = -rightJS_data[0] * self.movement_speed
        if "leftJS" in self.raw_data["button_data"]:
            leftJS_data = self.raw_data["button_data"]["leftJS"]
            self.teleop_data.transforms["base"][1] = -leftJS_data[0] * self.movement_speed
            self.teleop_data.transforms["base"][2] = leftJS_data[1] * self.movement_speed
        # update robot attachment info
        if "rightGrip" in self.raw_data["button_data"] and self.raw_data["button_data"]["rightGrip"][0] >= 0.5:
            if not self.reset_button_pressed:
                self.reset_button_pressed = True
                self.robot_attached = not self.robot_attached
        else:
            self.reset_button_pressed = False
        self.teleop_data.robot_attached = self.robot_attached

    def reset_transform_mapping(self, arm: str = "right") -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            arm(str): name of the arm, one of "left" or "right". Default is "right".
        """
        super().reset_transform_mapping(arm)
        if arm in self.raw_data["transforms"]:
            self.device_origin[arm] = self.raw_data["transforms"][arm]


class SpaceMouseSystem(TeleopSystem):
    def __init__(self, robot: BaseRobot, show_control_marker: bool = True, *args, **kwargs) -> None:
        try:
            self.pyspacemouse = import_module('pyspacemouse')
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install pyspacemouse to use SpaceMouseSystem")
        super().__init__(robot, show_control_marker)

        self.raw_data = None
        self.data_thread = None
        # robot is always attached to the space mouse system
        self.teleop_data.robot_attached = True
        # data is always valid for space mouse
        for arm in self.robot_arms:
            self.teleop_data.is_valid[arm] = True
        self.delta_pose = {arm: [np.zeros(3), np.array([0, 0, 0, 1])] for arm in self.robot_arms}
        # tracker of which robot part we are controlling
        self.controllable_robot_parts = self.robot_arms.copy()
        if "base" in self.robot.controllers:
            self.controllable_robot_parts.append("base")
        self.cur_control_idx = 0
        # we want to scale down the movement speed to have better control for arms
        self.arm_speed_scaledown = 0.1


    def start(self) -> None:
        """
        Start the space mouse connection and the data thread
        """
        assert self.pyspacemouse.open(
            button_callback=self._button_callback), "[SpaceMouseSys] Cannot connect to space mouse!"
        for arm in self.robot_arms:
            self.reset_transform_mapping(arm)
        self.data_thread = Thread(target=self._update_internal_data, daemon=True)
        self.data_thread.start()

    def stop(self) -> None:
        """
        Stop the space mouse connection and the data thread
        """
        self.data_thread.join()
        self.pyspacemouse.close()

    def _update_internal_data(self) -> None:
        """
        Thread that stores the the spacemouse input to self.raw_data
        """
        while True:
            self.raw_data = self.pyspacemouse.read()

    def _button_callback(self, _, buttons) -> None:
        """
        Callback function for space mouse button press
        """
        if buttons[0]:
            # left button pressed, switch controlling part
            self.cur_control_idx = (self.cur_control_idx + 1) % len(self.controllable_robot_parts)
            print(f"Now controlling robot part {self.controllable_robot_parts[self.cur_control_idx]}")
        elif buttons[1]:
            # right button pressed, switch gripper open/close state if we are controlling one
            if self.controllable_robot_parts[self.cur_control_idx] != "base":
                arm = self.controllable_robot_parts[self.cur_control_idx]
                self.teleop_data.grippers[arm] = (self.teleop_data.grippers[arm] + 1) % 2

    def update(self) -> None:
        """
        Steps the VR system and update self.teleop_data
        """
        self.teleop_data.transforms["base"] = np.zeros(4)
        if self.raw_data:
            controlling_robot_part = self.controllable_robot_parts[self.cur_control_idx]
            # update controlling part pose
            if controlling_robot_part == "base":
                self.teleop_data.transforms[controlling_robot_part][0] = self.raw_data.y * self.movement_speed
                self.teleop_data.transforms[controlling_robot_part][1] = -self.raw_data.x * self.movement_speed
                self.teleop_data.transforms[controlling_robot_part][2] = self.raw_data.z * self.movement_speed
                self.teleop_data.transforms[controlling_robot_part][3] = -self.raw_data.yaw * self.movement_speed
            else:
                self.delta_pose[controlling_robot_part][0] += np.array(       
                    [self.raw_data.y, -self.raw_data.x, self.raw_data.z]
                ) * self.movement_speed * self.arm_speed_scaledown
                self.delta_pose[controlling_robot_part][1] = T.quat_multiply(
                    T.euler2quat(
                        np.array([
                            self.raw_data.roll, self.raw_data.pitch, -self.raw_data.yaw
                        ]) * self.movement_speed * self.arm_speed_scaledown
                    ),
                    self.delta_pose[controlling_robot_part][1]
                )
            # additionally update eef pose (to ensure it's up to date relative to robot base pose)
            robot_base_pose = self.robot.get_position_orientation()
            for arm in self.robot_arms:
                target_rel_pos = self.robot_origin[arm][0] + self.delta_pose[arm][0]
                target_rel_orn = T.quat_multiply(self.delta_pose[arm][1], self.robot_origin[arm][1])
                self.teleop_data.transforms[arm] = T.pose_transform(*robot_base_pose, target_rel_pos, target_rel_orn)


class KeyboardSystem(TeleopSystem):
    def __init__(self, robot: BaseRobot, show_control_marker: bool = True, *args, **kwargs) -> None:
        super().__init__(robot, show_control_marker)
        # robot is always attached to the keyboard system, gripper is open initially
        self.teleop_data.robot_attached = True
        # data is always valid for keyboard
        for arm in self.robot_arms:
            self.raw_data["transforms"][arm] = (np.zeros(3), np.zeros(3))
            self.teleop_data.is_valid[arm] = True
        self.raw_data["transforms"]["base"] = np.zeros(4)
        self.cur_control_idx = 0
        # we want to scale down the movement speed to have better control for arms
        self.arm_speed_scaledown = 0.01
        self.delta_pose = {arm: [np.zeros(3), np.array([0, 0, 0, 1])] for arm in self.robot_arms}
        
    def start(self) -> None:
        # start the keyboard subscriber
        appwindow = omni.appwindow.get_default_app_window()
        input_interface = lo.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        input_interface.subscribe_to_keyboard_events(keyboard, self._update_internal_data)   
        for arm in self.robot_arms:
            self.reset_transform_mapping(arm)  

    def stop(self) -> None:
        pass

    def _update_internal_data(self, event) -> None:
        hand = self.robot_arms[self.cur_control_idx]
        if event.type == lo.carb.input.KeyboardEventType.KEY_RELEASE:
            for arm in self.robot_arms:
                self.raw_data["transforms"][arm] = (np.zeros(3), np.zeros(3))
            self.raw_data["transforms"]["base"] = np.zeros(4)
        if event.type == lo.carb.input.KeyboardEventType.KEY_PRESS \
            or event.type == lo.carb.input.KeyboardEventType.KEY_REPEAT:
            key = event.input    
            # update gripper state
            if key == lo.carb.input.KeyboardInput.KEY_4:
                self.teleop_data.grippers[hand] = (self.teleop_data.grippers[hand] + 1) % 2            
            # update current finger positions
            elif key == lo.carb.input.KeyboardInput.W:
                self.raw_data["transforms"][hand][0][0] = self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.S:
                self.raw_data["transforms"][hand][0][0] = -self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.A:
                self.raw_data["transforms"][hand][0][1] = self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.D:
                self.raw_data["transforms"][hand][0][1] = -self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.E:
                self.raw_data["transforms"][hand][0][2] = self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.Q:
                self.raw_data["transforms"][hand][0][2] = -self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.C:
                self.raw_data["transforms"][hand][1][0] = self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.X:
                self.raw_data["transforms"][hand][1][0] = -self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.T:
                self.raw_data["transforms"][hand][1][1] = self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.B:
                self.raw_data["transforms"][hand][1][1] = -self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.Z:
                self.raw_data["transforms"][hand][1][2] = self.movement_speed * self.arm_speed_scaledown
            elif key == lo.carb.input.KeyboardInput.V:
                self.raw_data["transforms"][hand][1][2] = -self.movement_speed * self.arm_speed_scaledown
            # rotate around hands to control
            elif key == lo.carb.input.KeyboardInput.KEY_1:
                self.cur_control_idx = (self.cur_control_idx - 1) % len(self.robot_arms)
                print(f"Now controlling robot part {self.robot_arms[self.cur_control_idx]}")
            elif key == lo.carb.input.KeyboardInput.KEY_2:
                self.cur_control_idx = (self.cur_control_idx + 1) % len(self.robot_arms)
                print(f"Now controlling robot part {self.robot_arms[self.cur_control_idx]}")
            # update base positions 
            elif key == lo.carb.input.KeyboardInput.I:
                self.raw_data["transforms"]["base"][0] = self.movement_speed
            elif key == lo.carb.input.KeyboardInput.K:
                self.raw_data["transforms"]["base"][0] = -self.movement_speed
            elif key == lo.carb.input.KeyboardInput.U:
                self.raw_data["transforms"]["base"][1] = self.movement_speed
            elif key == lo.carb.input.KeyboardInput.O:
                self.raw_data["transforms"]["base"][1] = -self.movement_speed
            elif key == lo.carb.input.KeyboardInput.P:
                self.raw_data["transforms"]["base"][2] = self.movement_speed
            elif key == lo.carb.input.KeyboardInput.SEMICOLON:
                self.raw_data["transforms"]["base"][2] = -self.movement_speed
            elif key == lo.carb.input.KeyboardInput.J:
                self.raw_data["transforms"]["base"][3] = self.movement_speed
            elif key == lo.carb.input.KeyboardInput.L:
                self.raw_data["transforms"]["base"][3] = -self.movement_speed
        return True

    def update(self) -> None:
        # update robot base pose
        self.teleop_data.transforms["base"] = self.raw_data["transforms"]["base"] 
        # update robot arm poses
        cur_arm = self.robot_arms[self.cur_control_idx]
        self.delta_pose[cur_arm][0] += self.raw_data["transforms"][cur_arm][0]
        self.delta_pose[cur_arm][1] = T.quat_multiply(
            T.euler2quat(self.raw_data["transforms"][cur_arm][1]),
            self.delta_pose[cur_arm][1]
        )
        robot_base_pose = self.robot.get_position_orientation()
        for arm in self.robot_arms:
            target_rel_pos = self.robot_origin[arm][0] + self.delta_pose[arm][0]
            target_rel_orn = T.quat_multiply(self.delta_pose[arm][1], self.robot_origin[arm][1])
            self.teleop_data.transforms[arm] = T.pose_transform(*robot_base_pose, target_rel_pos, target_rel_orn)
