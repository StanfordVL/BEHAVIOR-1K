import carb
import numpy as np
import time
from importlib import import_module
from threading import Thread
from typing import Iterable, Optional, Tuple

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects import USDObject
from omnigibson.robots import BaseRobot, LocomotionRobot


class TeleopSystem():
    """
    Base class for teleop systems
    """
    def __init__(self, robot: BaseRobot, show_control_marker: bool=True, *args, **kwargs) -> None:
        """
        self.raw_data is the raw data directly obtained from the teleoperation device
            It should be in the format as follows:
            {
                "transforms": { ... },          # device/landmark transforms
                "button_data": { ... }.         # device button data
                ...                             # other raw data
            }
        self.teleop_data is the processed data that will be used to generate robot actions
            It should be in the format as follows:
            {
                "robot_attached": bool, 
                "transforms": {
                    "right": (pos, orn)          
                    "left": (pos, orn)            
                    "base": [dx, dy, dz, dth]       
                    ...                         # other trackers/landmarks
                }
                "gripper_left": float,         
                "gripper_right": float,         
                ...                             # other teleop data
            }
        """
        self.raw_data = {
            "transforms": {},
            "button_data": {},
        }
        self.teleop_data = {
            "robot_attached": False,
            "transforms": {},
            "button_data": {},
        }
        # vr_origin stores the original pose of the VR controller upon calling reset_transform_mapping
        # This is used to subtract with the current VR controller pose in each frame to get the delta pose
        self.vr_origin = {"right": T.mat2pose(np.eye(4)), "left": T.mat2pose(np.eye(4))}
        # robot_origin stores the relative pose of the robot eef w.r.t. to the robot base upon calling reset_transform_mapping
        # This is used to apply the delta offset of the VR controller to get the final target robot eef pose (relative to the robot base) 
        self.robot_origin = {"right": T.mat2pose(np.eye(4)), "left": T.mat2pose(np.eye(4))}
        # robot parameters
        self.robot = robot
        self.robot_arms = ["left", "right"] if self.robot.n_arms == 2 else ["right"]
        self.robot_attached = False
        self.base_movement_speed = 0.2
        self.show_control_marker = show_control_marker
        self.control_markers = {}
        if show_control_marker:
            for arm in robot.arm_names:
                arm_name = "right" if arm == robot.default_arm else "left"
                self.control_markers[arm_name] = USDObject(name=f"target_{arm_name}", usd_path=robot.eef_usd_path[arm], visual_only=True)
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
        # optionally update control marker
        if self.show_control_marker:
            self.update_control_marker()
        return self.robot.teleop_data_to_action(self.teleop_data)

    def update_control_marker(self):
        # update the target object's pose to the VR right controller's pose
        for arm_name in self.control_markers:
            if arm_name in self.teleop_data["transforms"]:
                self.control_markers[arm_name].set_position_orientation(*self.teleop_data["transforms"][arm_name])  

    def reset_transform_mapping(self, arm: str="right") -> None:
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
        show_control_marker: bool=True,
        system: str="SteamVR",
        disable_display_output: bool=False,
        enable_touchpad_movement: bool=False,
        align_anchor_to_robot_base: bool=False,
        use_hand_tracking: bool=False,
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
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.kit.xr.profile.vr")
        from omni.kit.xr.core import XRDeviceClass, XRCore, XRCoreEventType
        from omni.kit.xr.ui.stage.common import XRAvatarManager
        self.xr_device_class = XRDeviceClass
        # run super method
        super().__init__(robot, show_control_marker)
        # get xr core and profile
        self.xr_core = XRCore.get_singleton()
        self.vr_profile = self.xr_core.get_profile("vr")
        self.disable_display_output = disable_display_output
        self.enable_touchpad_movement = enable_touchpad_movement
        self.align_anchor_to_robot_base = align_anchor_to_robot_base
        assert not (self.enable_touchpad_movement and self.align_anchor_to_robot_base), "enable_touchpad_movement and align_anchor_to_robot_base cannot be True at the same time!"
        # set avatar
        if self.show_control_marker:
            self.vr_profile.set_avatar(XRAvatarManager.get_singleton().create_avatar("basic_avatar", {}))
        else:
            self.vr_profile.set_avatar(XRAvatarManager.get_singleton().create_avatar("empty_avatar", {}))
        # # set anchor mode to be custom anchor
        carb.settings.get_settings().set(self.vr_profile.get_scene_persistent_path() + "anchorMode", "scene origin")
        # set vr system
        carb.settings.get_settings().set(self.vr_profile.get_persistent_path() + "system/display", system)
        # set display mode
        carb.settings.get_settings().set(self.vr_profile.get_persistent_path() + "disableDisplayOutput", disable_display_output)
        carb.settings.get_settings().set('/rtx/rendermode', "RaytracedLighting")
        # devices info
        self.hmd = None
        self.controllers = {}
        self.trackers = {}
        self.reset_button_pressed = False
        # setup event subscriptions
        self.use_hand_tracking = use_hand_tracking
        if use_hand_tracking:
            self.raw_data["hand_data"] = {}
            self._hand_tracking_subscription = self.xr_core.get_event_stream().create_subscription_to_pop_by_type(
                XRCoreEventType.hand_joints, self.get_hand_tracking_data, name="hand tracking"
            )

    def xr2og(self, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the orientation offset from the Omniverse XR coordinate system to the OmniGibson coordinate system
        Args:
            transform (np.ndarray): the transform matrix in the Omniverse XR coordinate system
        Returns:
            tuple(np.ndarray, np.ndarray): the position and orientation in the OmniGibson coordinate system
        """
        pos, orn = T.mat2pose(np.array(transform).T)
        orn = T.quat_multiply(orn, np.array([0.5, -0.5, -0.5, -0.5]))
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
        orn = T.quat_multiply(np.array([-0.5, 0.5, 0.5, -0.5]), orn)
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
        for _ in range(50):
            og.sim.step()
        assert self.vr_profile.is_enabled(), "[VRSys] VR profile not enabled!"
        self._update_devices()

    def stop(self) -> None:
        """
        disable VR profile
        """
        self.xr_core.request_disable_profile()
        self.sim.step()
        assert not self.vr_profile.is_enabled(), "[VRSys] VR profile not disabled!"

    def update(self):
        """
        Steps the VR system
        """
        # Optional move anchor
        if self.enable_touchpad_movement:
            self.move_anchor(pos_offset=self.compute_anchor_offset_with_controller_input())
        if self.align_anchor_to_robot_base:
            robot_base_pos, robot_base_orn = self.robot.get_position_orientation()
            self.vr_profile.set_virtual_world_anchor_transform(self.og2xr(robot_base_pos, robot_base_orn[[0, 2, 1, 3]]))
        # update raw data
        self._update_devices()
        self._update_device_transforms()
        self._update_button_data()
        if self.use_hand_tracking:
            self._update_hand_tracking_data()
        # generate teleop data
        self.teleop_data["transforms"]["base"] = np.zeros(4)
        if 1 in self.controllers:
            self.teleop_data["transforms"]["right"] = (
                self.raw_data["transforms"]["controllers"][1][0],
                T.quat_multiply(self.raw_data["transforms"]["controllers"][1][1], self.robot.teleop_rotation_offset[self.robot.arm_names[-1]])
            )
            self.teleop_data["transforms"]["base"][0] = self.raw_data["button_data"][1]["axis"]["touchpad_y"] * self.base_movement_speed
            self.teleop_data["transforms"]["base"][3] = -self.raw_data["button_data"][1]["axis"]["touchpad_x"] * self.base_movement_speed
            self.teleop_data["gripper_right"] = self.raw_data["button_data"][1]["axis"]["trigger"]
        if 0 in self.controllers:
            self.teleop_data["transforms"]["left"] = (
                self.raw_data["transforms"]["controllers"][0][0],
                T.quat_multiply(self.raw_data["transforms"]["controllers"][0][1], self.robot.teleop_rotation_offset[self.robot.arm_names[0]])
            )
            self.teleop_data["transforms"]["base"][1] = -self.raw_data["button_data"][0]["axis"]["touchpad_x"] * self.base_movement_speed
            self.teleop_data["transforms"]["base"][2] = self.raw_data["button_data"][0]["axis"]["touchpad_y"] * self.base_movement_speed
            self.teleop_data["gripper_left"] = self.raw_data["button_data"][0]["axis"]["trigger"]
        # update robot attachment info
        if 1 in self.controllers and self.raw_data["button_data"][1]["press"]["grip"]:
            if not self.reset_button_pressed:
                self.reset_button_pressed = True
                self.robot_attached = not self.robot_attached
        else:
            self.reset_button_pressed = False
        self.teleop_data["robot_attached"] = self.robot_attached

    def compute_anchor_offset_with_controller_input(self) -> np.ndarray:
        """
        Compute the desired anchor translational offset based on controller touchpad input
        Returns:
            np.ndarray: 3d translational offset *in hmd frame*
        """
        offset = np.zeros(3)
        if 1 in self.controllers:
            right_axis_state = self.controllers[1].get_axis_state()
            offset = np.array([right_axis_state["touchpad_x"], right_axis_state["touchpad_y"], 0])
        if 0 in self.controllers:
            offset[2] = self.controllers[0].get_axis_state()["touchpad_y"]
        # normalize offset
        length = np.linalg.norm(offset)
        if length != 0:
            offset *= 0.03 / length
        return np.array(offset)

    def move_anchor(self, pos_offset: Optional[Iterable[float]]=None, rot_offset: Optional[Iterable[float]]=None) -> None:
        """
        Updates the anchor of the xr system in the virtual world
        Args:
            pos_offset (Iterable[float]): the position offset to apply to the anchor *in hmd frame*.
            rot_offset (Iterable[float]): the rotation offset to apply to the anchor *in hmd frame*. 
        """
        if pos_offset is not None:
            # note that x is forward, y is down, z is left for ovxr, but x is forward, y is left, z is up for og
            pos_offset = np.array([pos_offset[0], pos_offset[2], -pos_offset[1]]).astype(np.float64)
            self.vr_profile.add_move_physical_world_relative_to_device(pos_offset)
        if rot_offset is not None:
            rot_offset = np.array(rot_offset).astype(np.float64)
            self.vr_profile.add_rotate_physical_world_around_device(rot_offset)

    def reset_transform_mapping(self, arm: str="right") -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            arm(str): name of the arm, one of "left" or "right". Default is "right".
        """
        robot_base_orn = self.robot.get_orientation()
        robot_eef_pos = self.robot.eef_links[self.robot.arm_names[self.robot_arms.index(arm)]].get_position()
        target_transform = self.og2xr(pos=robot_eef_pos, orn=robot_base_orn)
        self.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(target_transform, self.controllers[1])

    def _update_devices(self) -> None:
        """
        Update the VR device list
        """
        for device in self.vr_profile.get_device_list():
            if device.get_class() == self.xr_device_class.xrdisplaydevice:
                self.hmd = device
            elif device.get_class() == self.xr_device_class.xrcontroller:
                self.controllers[device.get_index()] = device
            elif device.get_class() == self.xr_device_class.xrtracker:
                self.trackers[device.get_index()] = device
        assert self.hmd is not None, "[VRSys] HMD not detected! Please make sure you have a VR headset connected to your computer."

    def _update_device_transforms(self) -> None:
        """
        Get the transform matrix of each VR device *in world frame* and store in self.raw_data
        Note that we have to transpose the transform matrix because Omniverse uses row-major matrices while OmniGibson uses column-major matrices
        """
        transforms = {}
        transforms["hmd"] = self.xr2og(self.hmd.get_virtual_world_pose())
        transforms["controllers"] = {}
        transforms["trackers"] = {}
        for controller_index in self.controllers:
            transforms["controllers"][controller_index] = self.xr2og(self.controllers[controller_index].get_virtual_world_pose())
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
        for controller_index in self.controllers:
            button_data[controller_index] = {}
            button_data[controller_index]["press"] = self.controllers[controller_index].get_button_press_state()
            button_data[controller_index]["touch"] = self.controllers[controller_index].get_button_touch_state()
            button_data[controller_index]["axis"] = self.controllers[controller_index].get_axis_state()
        self.raw_data["button_data"] = button_data       
    
    def _update_hand_tracking_data(self, e: carb.events.IEvent) -> None:
        """
        Get hand tracking data, see https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints for joint indices
        Args:
            e (carb.events.IEvent): event that contains hand tracking data as payload
        """
        e.consume()
        data_dict = e.payload
        for hand in ["left", "right"]:
            self.raw_data["hand_data"][hand] = {}
            if data_dict[f"joint_count_{hand}"] != 0:
                self.raw_data["hand_data"][hand]["raw"] = {"pos": [], "orn": []}
                hand_joint_matrices = data_dict[f"joint_matrices_{hand}"]
                for i in range(26):
                    pos, orn = self.xr2og(np.reshape(hand_joint_matrices[16 * i: 16 * (i + 1)], (4, 4)))
                    self.raw_data["hand_data"][hand]["raw"]["pos"].append(pos)
                    self.raw_data["hand_data"][hand]["raw"]["orn"].append(orn)
                # Get each finger joint's rotation angle from hand tracking data
                # joint_angles is a 5 x 3 array of joint rotations (from thumb to pinky, from base to tip)
                joint_angles = np.zeros((5, 3)) 
                raw_hand_data = self.raw_data["hand_data"][hand]["raw"]["pos"]
                for i in range(5):
                    for j in range(3):
                        # get the 3 related joints
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
                self.raw_data["hand_data"][hand]["angles"] = joint_angles   


class OculusReaderSystem(TeleopSystem):
    """
    The origin of the oculus system is the headset position
    x is right, y is up, z is back
    """
    def __init__(self, robot: BaseRobot, show_control_marker: bool=True, *args, **kwargs):
        try:
            import oculus_reader
        except ModuleNotFoundError:
            raise ModuleNotFoundError("[OculusReaderSys] Please install oculus_reader (https://github.com/rail-berkeley/oculus_reader) to use OculusReaderSystem")
        super().__init__(robot, show_control_marker)
        # initialize oculus reader
        self.oculus_reader = oculus_reader.OculusReader(run=False)
        self.reset_button_pressed = False

    def oc2og(self, transform):
        return T.mat2pose(
            T.pose2mat(([0, 0, 0], T.euler2quat([np.pi / 2, 0, np.pi / 2]))) @ transform @ T.pose2mat(([0, 0, 0], T.euler2quat([-np.pi / 2, np.pi / 2, 0])))
        )

    def start(self):
        self.oculus_reader.run()
        self.data_thread = Thread(target=self._update_internal_data, daemon=True)
        self.data_thread.start()

    def stop(self):
        self.data_thread.join()
        self.oculus_reader.stop()

    def _update_internal_data(self, hz: float=50.):
        while True:
            time.sleep(1 / hz)
            transform, self.raw_data["button_data"] = self.oculus_reader.get_transformations_and_buttons()
            for hand in ["left", "right"]:
                if hand[0] in transform:
                    self.raw_data["transforms"][hand] = self.oc2og(transform[hand[0]])

    def update(self):
        # generate teleop data
        self.teleop_data["transforms"]["base"] = np.zeros(4)
        robot_based_pose = self.robot.get_position_orientation()
        for hand in ["left", "right"]:
            if hand in self.raw_data["transforms"]:
                delta_pos, delta_orn = T.relative_pose_transform(*self.raw_data["transforms"][hand], *self.vr_origin[hand])
                target_rel_pos = self.robot_origin[hand][0] + delta_pos
                target_rel_orn = T.quat_multiply(delta_orn, self.robot_origin[hand][1])
                self.teleop_data["transforms"][hand] = T.pose_transform(*robot_based_pose, target_rel_pos, target_rel_orn)
            if f"{hand}Trig" in self.raw_data["button_data"]:
                self.teleop_data[f"gripper_{hand}"] = self.raw_data["button_data"][f"{hand}Trig"][0]
        if "rightJS" in self.raw_data["button_data"]:
            self.teleop_data["transforms"]["base"][0] = self.raw_data["button_data"]["rightJS"][1] * self.base_movement_speed
            self.teleop_data["transforms"]["base"][3] = -self.raw_data["button_data"]["rightJS"][0] * self.base_movement_speed
        if "leftJS" in self.raw_data["button_data"]:
            self.teleop_data["transforms"]["base"][1] = -self.raw_data["button_data"]["leftJS"][0] * self.base_movement_speed
            self.teleop_data["transforms"]["base"][2] = self.raw_data["button_data"]["leftJS"][1] * self.base_movement_speed
        # update robot attachment info
        if "rightGrip" in self.raw_data["button_data"] and self.raw_data["button_data"]["rightGrip"][0] >= 0.5:
            if not self.reset_button_pressed:
                self.reset_button_pressed = True
                self.robot_attached = not self.robot_attached
        else:
            self.reset_button_pressed = False
        self.teleop_data["robot_attached"] = self.robot_attached

    def reset_transform_mapping(self, arm: str="right") -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            arm(str): name of the arm, one of "left" or "right". Default is "right".
        """
        super().reset_transform_mapping(arm)
        if arm in self.raw_data["transforms"]:
            self.vr_origin[arm] = self.raw_data["transforms"][arm]


class SpaceMouseSystem(TeleopSystem):
    def __init__(self, robot: BaseRobot, show_control_marker: bool=True, *args, **kwargs):
        try:
            self.pyspacemouse = import_module('pyspacemouse')
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install pyspacemouse to use SpaceMouseSystem")
        super().__init__(robot, show_control_marker)

        self.raw_data = None
        # robot is always attached to the space mouse system, gripper is open initially
        self.teleop_data["robot_attached"] = True
        for arm in self.robot_arms:
            self.teleop_data[f"gripper_{arm}"] = 0
        self.delta_pose = {arm: [np.zeros(3), np.array([0, 0, 0, 1])] for arm in self.robot_arms}
        # tracker of which robot part we are controlling
        self.controllable_robot_parts = self.robot_arms.copy()
        if isinstance(robot, LocomotionRobot):
            self.controllable_robot_parts.append("base")
        self.cur_control_idx = 0
  
    def start(self):
        assert self.pyspacemouse.open(button_callback=self._button_callback), "[SpaceMouseSys] Cannot connect to space mouse!"
        for arm in self.robot_arms:
            self.reset_transform_mapping(arm)
        self.data_thread = Thread(target=self._update_internal_data, daemon=True)
        self.data_thread.start()

    def stop(self):
        self.data_thread.join()
        self.pyspacemouse.close()

    def _update_internal_data(self):
        while True:
            self.raw_data = self.pyspacemouse.read()

    def _button_callback(self, _, buttons):
        if buttons[0]:
            self.cur_control_idx = (self.cur_control_idx + 1) % len(self.controllable_robot_parts)
            print(f"Now controlling robot part {self.controllable_robot_parts[self.cur_control_idx]}")
        elif buttons[1]:
            if self.controllable_robot_parts[self.cur_control_idx] != "base":
                gripper = f"gripper_{self.controllable_robot_parts[self.cur_control_idx]}"
                self.teleop_data[gripper] = (self.teleop_data[gripper] + 1) % 2
        
    def update(self):
        self.teleop_data["transforms"]["base"] = np.zeros(4)
        if self.raw_data:
            controlling_robot_part = self.controllable_robot_parts[self.cur_control_idx]
            # update controlling part pose
            if controlling_robot_part == "base":
                self.teleop_data["transforms"]["base"][0] = self.raw_data.y * self.base_movement_speed
                self.teleop_data["transforms"]["base"][1] = self.raw_data.x * self.base_movement_speed
                self.teleop_data["transforms"]["base"][2] = self.raw_data.z * self.base_movement_speed
                self.teleop_data["transforms"]["base"][3] = -self.raw_data.yaw * self.base_movement_speed
            else:
                self.delta_pose[controlling_robot_part][0] += np.array([self.raw_data.y, -self.raw_data.x, self.raw_data.z]) * 0.01
                self.delta_pose[controlling_robot_part][1] = T.quat_multiply(
                   T.euler2quat(np.array([self.raw_data.roll, self.raw_data.pitch, -self.raw_data.yaw]) * 0.02), self.delta_pose[controlling_robot_part][1]
                )
            # additionally update eef pose (to ensure it's up to date relative to robot base pose)
            robot_base_pose = self.robot.get_position_orientation()
            for arm in self.robot_arms:
                target_rel_pos = self.robot_origin[arm][0] + self.delta_pose[arm][0]
                target_rel_orn = T.quat_multiply(self.delta_pose[arm][1], self.robot_origin[arm][1])
                self.teleop_data["transforms"][arm] = T.pose_transform(*robot_base_pose, target_rel_pos, target_rel_orn)
