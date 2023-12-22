import carb
import numpy as np
from importlib import import_module
from typing import Iterable, List, Optional, Tuple

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.robots.robot_base import BaseRobot

# enable xr extension
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.xr.profile.vr")
from omni.kit.xr.core import XRCore, XRDeviceClass, XRCoreEventType
from omni.kit.xr.ui.stage.common import XRAvatarManager


class TeleopSystem():
    """
    Base class for teleop systems
    """
    def __init__(self, robot: BaseRobot) -> None:
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
        self.raw_data = {}
        self.teleop_data = {
            "robot_attached": False,
            "transforms": {},
            "button_data": {},
        }
        self.robot = robot
        self.robot_attached = False

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
        return self.robot.teleop_data_to_action(self.teleop_data)


class VRSystem(TeleopSystem):
    """
    VR Teleoperation System build on top of Omniverse XR extension
    """
    def __init__(
        self, 
        robot: BaseRobot,
        system: str="OpenXR",
        show_controller: bool=False,
        disable_display_output: bool=False,
        enable_touchpad_movement: bool=False,
        align_anchor_to_robot_base: bool=False,
        use_hand_tracking: bool=False,
    ) -> None:
        """
        Initializes the VR system
        Args:
            robot (BaseRobot): the robot that VR will control.
            system (str): the VR system to use, one of ["OpenXR", "SteamVR"], default is "OpenXR".
            show_controller (bool): whether to show the controller model in the scene, default is False.
            disable_display_output (bool): whether we will not display output to the VR headset (only use controller tracking), default is False.
            enable_touchpad_movement (bool): whether to enable VR system anchor movement by controller, default is False.
            align_anchor_to_robot_base (bool): whether to align VR anchor to robot base, default is False.
            use_hand_tracking (bool): whether to use hand tracking instead of controllers, default is False.

        NOTE: enable_touchpad_movement and align_anchor_to_robot_base cannot be enabled at the same time. 
            The former is to enable free movement of the VR system (i.e. the user), while the latter is constraining the VR system to the robot pose.
        """
        super().__init__(robot)
        # get xr core and profile
        self.xr_core = XRCore.get_singleton()
        self.vr_profile = self.xr_core.get_profile("vr")
        self.disable_display_output = disable_display_output
        self.enable_touchpad_movement = enable_touchpad_movement
        self.align_anchor_to_robot_base = align_anchor_to_robot_base
        assert not (self.enable_touchpad_movement and self.align_anchor_to_robot_base), "enable_touchpad_movement and align_anchor_to_robot_base cannot be True at the same time!"
        # set avatar
        if show_controller:
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
            self.teleop_data["transforms"]["right"] = self.raw_data["transforms"]["controllers"][1]
            self.teleop_data["transforms"]["base"][0] = self.raw_data["button_data"][1]["axis"]["touchpad_y"]
            self.teleop_data["transforms"]["base"][3] = self.raw_data["button_data"][1]["axis"]["touchpad_x"]
            self.teleop_data["gripper_right"] = self.raw_data["button_data"][1]["axis"]["trigger"]
        if 0 in self.controllers:
            self.teleop_data["transforms"]["left"] = self.raw_data["transforms"]["controllers"][0]
            self.teleop_data["transforms"]["base"][1] = -self.raw_data["button_data"][0]["axis"]["touchpad_x"]
            self.teleop_data["transforms"]["base"][2] = self.raw_data["button_data"][0]["axis"]["touchpad_y"]
            self.teleop_data["gripper_right"] = self.raw_data["button_data"][0]["axis"]["trigger"]
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

    def snap_device_to_robot_eef(self, robot_eef_position: Iterable[float], base_rotation: Iterable[float], device: Optional[XRDeviceClass]=None) -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            robot_eef_position (Iterable[float]): the position of the robot end effector
            base_rotation (Iterable[float]): the rotation of the robot base
            device (Optional[XRDeviceClass]): the device to snap to the robot end effector, default is the right controller
        """
        if device is None:
            device = self.controllers[1]
        target_transform = self.og2xr(pos=robot_eef_position, orn=base_rotation)
        self.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(target_transform, device)

    def _update_devices(self) -> None:
        """
        Update the VR device list
        """
        for device in self.vr_profile.get_device_list():
            if device.get_class() == XRDeviceClass.xrdisplaydevice:
                self.hmd = device
            elif device.get_class() == XRDeviceClass.xrcontroller:
                self.controllers[device.get_index()] = device
            elif device.get_class() == XRDeviceClass.xrtracker:
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
    def __init__(self, robot):
        try:
            import oculus_reader
        except ModuleNotFoundError:
            raise ModuleNotFoundError("[OculusReaderSys] Please install oculus_reader (https://github.com/rail-berkeley/oculus_reader) to use OculusReaderSystem")
        super().__init__(robot)
        # initialize oculus reader
        self.oculus_reader = oculus_reader.OculusReader(run=False)
        self.reset_button_pressed = False

    def oc2og(self, transform):
        pos, orn = T.mat2pose(np.array(transform).T)
        orn = T.quat_multiply(orn, np.array([0.5, -0.5, -0.5, -0.5]))
        return pos, orn
    
    def og2oc(self, pos, orn):
        orn = T.quat_multiply(np.array([-0.5, 0.5, 0.5, -0.5]), orn)
        return T.pose2mat((pos, orn)).T.astype(np.float64)

    def start(self):
        self.oculus_reader.run()

    def stop(self):
        self.oculus_reader.stop()

    def update(self):
        # update raw data
        self.raw_data["transforms"], self.raw_data["button_data"] = self.oculus_reader.get_transformations_and_buttons()
        # generate teleop data
        self.teleop_data["transforms"]["base"] = np.zeros(4)
        for hand in ["left", "right"]:
            if hand[0] in self.raw_data["transforms"]:
                self.teleop_data["transforms"][hand] = self.oc2og(self.raw_data["transforms"][hand[0]])
                self.teleop_data[f"gripper_{hand}"] = self.raw_data["button_data"][f"{hand}Trig"][0]
        if "rightJS" in self.raw_data["button_data"]:
            self.teleop_data["transforms"]["base"][0] = self.raw_data["button_data"]["rightJS"][1]
            self.teleop_data["transforms"]["base"][3] = self.raw_data["button_data"]["rightJS"][0]
        if "leftJS" in self.raw_data["button_data"]:
            self.teleop_data["transforms"]["base"][1] = -self.raw_data["button_data"]["leftJS"][0]
            self.teleop_data["transforms"]["base"][2] = self.raw_data["button_data"]["leftJS"][1]
        # update robot attachment info
        if "rightGrip" in self.raw_data["button_data"] and self.raw_data["button_data"]["rightGrip"][0] >= 0.5:
            if not self.reset_button_pressed:
                self.reset_button_pressed = True
                self.robot_attached = not self.robot_attached
        else:
            self.reset_button_pressed = False
        self.teleop_data["robot_attached"] = self.robot_attached


class SpaceMouseSystem(TeleopSystem):
    def __init__(self, robot):
        try:
            self.pyspacemouse = import_module('pyspacemouse')
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install pyspacemouse to use OculusReaderSystem")
        super().__init__(robot)
        if not self.pyspacemouse.open():
            raise RuntimeError("[SpaceMouseSys] Cannot connect to space mouse!")
            
    def start(self):
        pass

    def stop(self):
        pass

    def update(self):
        state = self.pyspacemouse.read()



"""
(
    {
        'l': array([[ 1.00000e+00, -8.27610e-05, -1.05649e-04,  1.00732e-04],
        [-8.27610e-05, -1.00000e+00, -7.99596e-05,  9.57251e-05],
        [-1.05709e-04,  7.99298e-05, -1.00000e+00,  1.99982e-01],
        [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]]), 
        'r': array([[ 1.00000e+00,  8.27610e-05,  1.05649e-04,  1.00732e-04],
        [-8.27909e-05,  1.00000e+00,  7.99447e-05,  9.57251e-05],
        [-1.05679e-04, -7.99596e-05,  1.00000e+00,  1.99982e-01],
        [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])
    }, 
    {
        'A': False, 'B': False, 'RThU': True, 'RJ': False, 'RG': False, 'RTr': False, 'X': False, 
        'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 'leftJS': (0.0, 0.0), 
        'leftTrig': (0.0,), 'leftGrip': (0.0,), 'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (0.0,)
    }
)
"""