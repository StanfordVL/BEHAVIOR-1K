import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Tuple

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.controllers import ControllerView
from omnigibson.robots.robot import Robot
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import KeyboardEventHandler, create_module_logger
from omnigibson.utils.usd_utils import scene_relative_prim_path_to_absolute

try:
    from telemoma.configs.base_config import teleop_config

    # TODO: the imported telemoma interfaces does not work for our purposes because
    # 1) dimensionality mismatch - telemoma torso action is 1 dim 2) telemoma strictly uses numpy
    # from telemoma.human_interface.teleop_core import TeleopAction, TeleopObservation
    from telemoma.human_interface.teleop_policy import TeleopPolicy
    from telemoma.utils.general_utils import AttrDict
except ImportError as e:
    raise e from ValueError("For teleoperation, install telemoma by running 'pip install telemoma'")

# Create module logger
log = create_module_logger(module_name=__name__)

m = create_module_macros(module_path=__file__)
m.movement_speed = 0.5  # the speed of the robot base movement
m.rotation_speed = 0.1


@dataclass
class TeleopAction(AttrDict):
    left: th.Tensor = field(default_factory=lambda: th.cat((th.zeros(6), th.ones(1))))
    right: th.Tensor = field(default_factory=lambda: th.cat((th.zeros(6), th.ones(1))))
    base: th.Tensor = field(default_factory=lambda: th.zeros(3))
    torso: float = field(default_factory=lambda: 0.0)
    extra: dict = field(default_factory=dict)


@dataclass
class TeleopObservation(AttrDict):
    left: th.Tensor = field(default_factory=lambda: th.cat((th.zeros(6), th.ones(2))))
    right: th.Tensor = field(default_factory=lambda: th.cat((th.zeros(6), th.ones(2))))
    base: th.Tensor = field(default_factory=lambda: th.zeros(3))
    torso: float = field(default_factory=lambda: 0.0)
    extra: dict = field(default_factory=dict)


class TeleopSystem(TeleopPolicy):
    """
    Base class for teleop policy
    """

    def __init__(self, config: AttrDict, robot: Optional[Robot] = None, show_control_marker: bool = False) -> None:
        """
        Initializes the Teleoperation System
        Args:
            config (AttrDict): configuration dictionary
            robot (Optional[Robot]): the robot that will be controlled. Can be None.
            show_control_marker (bool): whether to show a visual marker that indicates the target pose of the control.
        """
        super().__init__(config)
        self.teleop_action = TeleopAction()
        self.robot_obs = TeleopObservation()
        self.robot = robot
        self.robot_arms = None if not self.robot else ["left", "right"] if self.robot.n_arms == 2 else ["right"]
        # robot parameters
        self.movement_speed = m.movement_speed
        self.rotation_speed = m.rotation_speed
        self.show_control_marker = show_control_marker

    def get_obs(self) -> TeleopObservation:
        """
        Retrieve observation data from robot
        Returns:
            TeleopObservation: dataclass containing robot observations
        """
        robot_obs = TeleopObservation()

        if self.robot is None:
            return robot_obs

        base_pos, base_orn = self.robot.get_position_orientation()
        robot_obs.base = th.cat((base_pos[:2], th.tensor([T.quat2euler(base_orn)[2]])))

        if self.robot_arms:
            for i, arm in enumerate(self.robot_arms):
                abs_cur_pos, abs_cur_orn = self.robot.eef_links[
                    self.robot.arm_names[self.robot_arms.index(arm)]
                ].get_position_orientation()
                rel_cur_pos, rel_cur_orn = T.relative_pose_transform(abs_cur_pos, abs_cur_orn, base_pos, base_orn)
                gripper_pos = th.mean(
                    self.robot.get_joint_positions(normalized=True)[
                        self.robot.gripper_control_idx[self.robot.arm_names[i]]
                    ]
                ).unsqueeze(0)
                # if we are grasping, we manually set the gripper position to be at most 0.5
                gripper_group_key, gripper_ci = self.robot.controllers[f"gripper_{self.robot.arm_names[i]}"]
                if ControllerView.is_grasping(gripper_group_key, gripper_ci):
                    gripper_pos = th.min(gripper_pos, th.tensor([0.5]))
                robot_obs[arm] = th.cat((rel_cur_pos, rel_cur_orn, gripper_pos))

        return robot_obs

    def get_action(self, obs: TeleopObservation) -> th.Tensor:
        """
        Generate action data from VR input for robot teleoperation
        Args:
            robot_obs (TeleopObservation): dataclass containing robot observations
        Returns:
            th.Tensor: array of action data or None if robot is None
        """
        if self.robot is None:
            return None

        # get teleop action
        self.teleop_action = super().get_action(obs)
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
        robot: Robot,
        show_control_marker: bool = True,
        system: str = "SteamVR",
        disable_display_output: bool = False,
        eef_tracking_mode: Literal["controller", "hand", "disabled"] = "controller",
        # TODO: fix this to only take something like a prim path
        align_anchor_to: Literal["camera", "base", "touchpad"] = "camera",
        view_angle_limits: Optional[Iterable[float]] = None,
    ) -> None:
        """
        Initializes the VR system
        Args:
            robot (Robot): the robot that VR will control.
            show_control_marker (bool): whether to show a control marker
            system (str): the VR system to use, one of ["OpenXR", "SteamVR"], default is "SteamVR".
            disable_display_output (bool): whether we will not display output to the VR headset (only use controller tracking), default is False.
            eef_tracking_mode (Literal): whether to use controller tracking or hand tracking, one of ["controller", "hand", "disabled"], default is controller.
            align_anchor_to (Literal): specify where the VR view aligns to, one of ["camera", "base", "touchpad"], defaults to robot camera.
                The "touchpad" option enables free movement of the VR view (i.e. the user), while the other two constrain the VR view to the either the robot base or camera pose.
            view_angle_limits (Iterable): the view angle limits for the VR system (roll, pitch, and yaw) in degrees, default is None.
        """
        align_to_prim = isinstance(align_anchor_to, XFormPrim)
        assert (
            align_anchor_to
            in [
                "camera",
                "base",
                "touchpad",
            ]
            or align_to_prim
        ), "align_anchor_to must be one of ['camera', 'base', 'touchpad'] or a XFormPrim"
        self.align_anchor_to = align_anchor_to
        self.anchor_prim = None
        if align_to_prim:
            self.set_anchor_with_prim(self.align_anchor_to)
        self.raw_data = {}
        self.old_raw_data = {}
        # run super method
        super().__init__(teleop_config, robot, show_control_marker)
        # get xr core and profile
        self.xr_core = lazy.omni.kit.xr.core.XRCore.get_singleton()
        self.vr_profile = self.xr_core.get_profile("vr")
        self.disable_display_output = disable_display_output
        # visualize control markers
        lazy.carb.settings.get_settings().set(
            "/defaults/xr/profile/" + self.xr_core.get_current_profile_name() + "/controllers/visible",
            self.show_control_marker,
        )
        # set override leveled basis to be true (if this is false, headset would not track anchor pitch orientation)
        allow_roll = False if align_anchor_to == "touchpad" else True
        lazy.carb.settings.get_settings().set(
            self.vr_profile.get_persistent_path() + "overrideLeveledBasis", allow_roll
        )
        # set anchor mode to be custom anchor
        lazy.carb.settings.get_settings().set(
            self.vr_profile.get_scene_persistent_path() + "anchorMode", "custom_anchor"
        )
        if align_anchor_to != "touchpad":
            # set override leveled basis to be true (if this is false, headset would not track anchor pitch orientation)
            lazy.carb.settings.get_settings().set(self.vr_profile.get_persistent_path() + "overrideLeveledBasis", True)
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
        # setup event subscriptions
        self.reset()
        self.eef_tracking_mode = eef_tracking_mode
        if eef_tracking_mode == "hand":
            self.raw_data["hand_data"] = {}
            self.teleop_action.hand_data = {}
            # Isaac Sim 5.1 removed XRCoreEventType.hand_joints (event-based hand
            # tracking was replaced by polling). Guard so init doesn't crash;
            # hand-joint data is unavailable on 5.1 via this endpoint (use the
            # WebXR path in hand_teleop/ for dexterous fingers instead).
            self._hand_tracking_subscription = None
            _hand_event = getattr(lazy.omni.kit.xr.core.XRCoreEventType, "hand_joints", None)
            if _hand_event is not None:
                self._hand_tracking_subscription = self.xr_core.get_message_bus().create_subscription_to_pop_by_type(
                    _hand_event, self._update_hand_tracking_data, name="hand tracking"
                )
            else:
                log.warning(
                    "[OVXRSystem] eef_tracking_mode='hand' but Isaac Sim 5.1 has no "
                    "XRCoreEventType.hand_joints — hand tracking DISABLED. Use --mode "
                    "controller, or the WebXR hand_teleop pipeline for fingers."
                )
        self.robot_cameras = (
            [s for s in self.robot.sensors.values() if isinstance(s, VisionSensor)] if self.robot else []
        )
        # TODO: this camera id is specific to R1, because it has 3 cameras, we need to make this more general
        self.active_camera_id = 2
        if self.align_anchor_to == "camera" and len(self.robot_cameras) == 0:
            raise ValueError("No camera found on robot, cannot align anchor to camera")
        # we want to further slow down the movement speed if we are using touchpad movement
        if self.align_anchor_to == "touchpad":
            self.movement_speed *= 0.1
            self.rotation_speed *= 0.3

        self.head_canonical_transformation = None
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.R,
            callback_fn=self.register_head_canonical_transformation,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Q,
            callback_fn=self.stop,
        )
        self._update_camera_callback = self.xr_core.get_message_bus().create_subscription_to_pop_by_type(
            lazy.omni.kit.xr.core.XRCoreEventType.pre_sync_update, self._update_camera_pose, name="update camera"
        )

        self._view_blackout_prim = None
        self._view_angle_limits = (
            [T.deg2rad(limit) for limit in view_angle_limits] if view_angle_limits is not None else None
        )
        if self._view_angle_limits is not None:
            scene = self.robot.scene
            blackout_relative_path = "/view_blackout"
            blackout_prim_path = scene_relative_prim_path_to_absolute(scene, blackout_relative_path)
            blackout_sphere = lazy.pxr.UsdGeom.Sphere.Define(og.sim.stage, blackout_prim_path)
            blackout_sphere.CreateRadiusAttr().Set(0.1)
            blackout_sphere.CreateDisplayColorAttr().Set(lazy.pxr.Vt.Vec3fArray([255, 255, 255]))
            self._view_blackout_prim = GeomPrim(
                relative_prim_path=blackout_relative_path,
                name="view_blackout",
            )
            self._view_blackout_prim.load(scene)
            self._view_blackout_prim.initialize()
            self._view_blackout_prim.visible = False

    def _update_camera_pose(self, e) -> None:
        # Re-entry guard: get_position_orientation() -> PoseAPI._refresh() ->
        # og.sim.render() can fire another pre_sync_update on Isaac Sim 5.1,
        # which would recurse back into this callback (RecursionError).
        if getattr(self, "_in_update_camera_pose", False):
            return
        self._in_update_camera_pose = True
        try:
            if self.align_anchor_to == "touchpad":
                # we use x, y from right controller for 2d movement and y from left controller for z movement
                self._move_anchor(
                    pos_offset=th.cat((th.tensor([self.teleop_action.torso]), self.teleop_action.base[[0, 2]]))
                )
            else:
                if self.anchor_prim is not None:
                    reference_frame = self.anchor_prim
                elif self.align_anchor_to == "camera":
                    reference_frame = self.robot_cameras[self.active_camera_id]
                elif self.align_anchor_to == "base":
                    reference_frame = self.robot
                else:
                    raise ValueError(f"Invalid anchor: {self.align_anchor_to}")

                anchor_pos, anchor_orn = reference_frame.get_position_orientation()

                if self.head_canonical_transformation is not None:
                    current_head_physical_world_pose = self.xr2og(self.hmd.get_pose())
                    # Find the orientation change from canonical to current physical orientation
                    _, relative_orientation = T.relative_pose_transform(
                        *current_head_physical_world_pose, *self.head_canonical_transformation
                    )
                    anchor_orn = T.quat_multiply(anchor_orn, relative_orientation)

                    if self._view_blackout_prim is not None:
                        relative_ori_in_euler = T.quat2euler(relative_orientation)
                        roll_limit, pitch_limit, yaw_limit = self._view_angle_limits
                        # OVXR has a different coordinate system than OmniGibson
                        if (
                            abs(relative_ori_in_euler[0]) > pitch_limit
                            or abs(relative_ori_in_euler[1]) > yaw_limit
                            or abs(relative_ori_in_euler[2]) > roll_limit
                        ):
                            self._view_blackout_prim.set_position_orientation(anchor_pos, anchor_orn)
                            self._view_blackout_prim.visible = True
                        else:
                            self._view_blackout_prim.visible = False
                anchor_pose = self.og2xr(anchor_pos, anchor_orn)
                self.xr_core.schedule_set_camera(anchor_pose.numpy())
        finally:
            self._in_update_camera_pose = False

    def register_head_canonical_transformation(self):
        """
        Here's what we need to do:
        1) Let the user press a button to record head canonical orientation (GELLO and head facing forward)
        2) when the user turn their head, get orientation change from canonical to current physical orientation as R
        3) set the head in virtual world to robot head orientation + R, same position
        """
        if self.hmd is None:
            log.warning("No HMD found, cannot register head canonical orientation")
            return
        self.head_canonical_transformation = self.xr2og(self.hmd.get_pose())

    def set_anchor_with_prim(self, prim) -> None:
        """
        Set the anchor to a prim
        Args:
            prim (BasePrim): the prim to set the anchor to
        """
        self.anchor_prim = prim

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
        return T.pose2mat((pos, orn)).T.double()

    def reset(self) -> None:
        """
        Reset the teleop policy
        """
        super().reset()
        self.raw_data = {}
        self.old_raw_data = {}
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
        # Isaac Sim 5.1: Simulator has no `.app`; pump the kit app directly.
        # Profile enable is async, so retry a few frames before asserting.
        for _ in range(180):
            lazy.omni.kit.app.get_app().update()
            if self.vr_profile.is_enabled():
                break
        assert self.vr_profile.is_enabled(), (
            "[VRSys] VR profile not enabled after ~3s — is the headset on, tracked, "
            "and SteamVR/OpenXR running?"
        )
        # We want to make sure the hmd is tracking so that the whole system is ready to go
        required_arms = self.robot_arms if self.robot_arms else ["left", "right"]
        while True:
            print("[VRSys] Waiting for VR headset and controllers to become active...")
            lazy.omni.kit.app.get_app().update()
            self._update_devices()
            if self.hmd is not None and all(a in self.controllers for a in required_arms):
                print("[VRSys] VR headset connected, put on the headset to start")
                # When taking the first step, xr internally calls clear_controller_model(hand) which removes a prim from stage
                # note that this does not invalidate the physics sim view but instead removes the physics view from articulation views
                # og.sim.step()
                # at this point we need to reinitialize the invalidated views
                # og.sim.update_handles()
                # for idx in range(50):
                #     self.update()
                #     # somehow, internally xr calls clear_controller_model(hand) which removes a prim from stage;
                #     # note that this does not invalidate the physics sim view but instead removes the physics view from articulation views
                #     breakpoint()
                #     og.sim.step()
                #     breakpoint()
                print("[VRSys] VR system is ready")
                self.register_head_canonical_transformation()
                og.sim.step()
                self.reset_head_transform()
                og.sim.step()
                break
            time.sleep(1)

    def stop(self) -> None:
        """
        disable VR profile
        """
        self.xr_core.request_disable_profile()
        og.sim.step()
        assert not self.vr_profile.is_enabled(), "[VRSys] VR profile not disabled!"

    def update(self, optimized_for_tour=False) -> None:
        """
        Steps the VR system and update self.teleop_action
        """
        # update raw data
        self.old_raw_data = self.raw_data
        self.raw_data = {}
        if not optimized_for_tour:
            self._update_devices()
            self._update_device_transforms()
        self._update_button_data()

        # Fire the button callbacks (rising-edge detection)
        old_bd = self.old_raw_data.get("button_data", {})
        for controller_name, controller_button_datas in self.raw_data["button_data"].items():
            for button_name, button_data in controller_button_datas.items():
                button_pressed = button_data["click"]
                old_click = old_bd.get(controller_name, {}).get(button_name, {}).get("click", 0)
                if button_pressed and not old_click:
                    print(f"Button {button_name} pressed on controller {controller_name}")
                    KeyboardEventHandler.xr_callback(controller_name, button_name)

        # Update teleop data based on controller input
        if self.eef_tracking_mode == "controller":
            # --- Restored working controller mapping (lost in the Isaac 5.1 stock
            # reset; recovered from the pre-5.1 working version) -------------------
            # Head-rotation compensation: the controller's offset FROM the headset
            # in PHYSICAL space is invariant to head rotation. We map that offset
            # into the robot frame with a fixed XR->world rotation captured once at
            # startup, so turning your head doesn't drag the arm. The -90deg yaw
            # (_YAW_FIX) maps your physical "forward" onto the robot's +X
            # (arm-forward); WITHOUT it, left/right and forward/back are swapped and
            # the wrist comes out rotated sideways. The orientation is emitted as
            # AXIS-ANGLE — the script's calibrated action builder consumes axis-angle.
            hmd_phys_pos, hmd_phys_orn = self.xr2og(self.hmd.get_pose())
            _, hmd_virt_orn = self.raw_data["transforms"]["head"]

            if not hasattr(self, "_xr_to_world_rot"):
                # Fixed rotation from XR tracking space to OG world, computed once.
                self._xr_to_world_rot = T.quat_multiply(hmd_virt_orn, T.quat_inverse(hmd_phys_orn))

            _, robot_base_orn = self.robot.get_position_orientation()
            xr_to_robot_rot = T.quat_multiply(T.quat_inverse(robot_base_orn), self._xr_to_world_rot)
            _YAW_FIX = th.tensor([0.0, 0.0, -0.7071067811865476, 0.7071067811865476])
            xr_to_robot_rot = T.quat_multiply(_YAW_FIX, xr_to_robot_rot)

            for vr_hand, robot_arm in zip(self.robot_arms, self.robot.arm_names):
                if vr_hand in self.controllers:
                    try:
                        ctrl_phys_pos, ctrl_phys_orn = self.xr2og(self.controllers[vr_hand].get_pose())
                        phys_offset = ctrl_phys_pos - hmd_phys_pos
                        corrected_pos = T.quat_apply(xr_to_robot_rot, phys_offset)
                        corrected_orn = T.quat_multiply(xr_to_robot_rot, ctrl_phys_orn)
                    except Exception:
                        # Fallback if controller.get_pose() is unavailable.
                        corrected_pos, corrected_orn = self._pose_in_robot_frame(
                            self.raw_data["transforms"]["controllers"][vr_hand][0],
                            self.raw_data["transforms"]["controllers"][vr_hand][1],
                        )
                    # When trigger is pressed, this value would be 1.0, otherwise 0.0
                    # Our multi-finger gripper controller closes the gripper when the value is -1.0 and opens when > 0.0
                    # So we need to negate the value here
                    trigger_press = (
                        -self.raw_data["button_data"][vr_hand]["trigger"]["value"]
                        if (
                            "button_data" in self.raw_data
                            and vr_hand in self.raw_data["button_data"]
                            and "trigger" in self.raw_data["button_data"][vr_hand]
                        )
                        else 0.0
                    )
                    if not hasattr(self, "_corrected_orn"):
                        self._corrected_orn = {}
                    self._corrected_orn[vr_hand] = corrected_orn.clone()
                    self.teleop_action[vr_hand] = th.cat(
                        (
                            corrected_pos,
                            T.quat2axisangle(
                                T.quat_multiply(
                                    corrected_orn,
                                    self.robot.teleop_rotation_offset[robot_arm],
                                )
                            ),
                            th.tensor([trigger_press], dtype=th.float32),
                        )
                    )
                    self.teleop_action.is_valid[vr_hand] = self._is_valid_transform(
                        self.raw_data["transforms"]["controllers"][vr_hand]
                    )
                else:
                    self.teleop_action.is_valid[vr_hand] = False

        # update base, torso, and reset info
        self.teleop_action.base = th.zeros(3)
        self.teleop_action.torso = 0.0
        for controller_name in self.controllers.keys():
            if "button_data" not in self.raw_data:
                continue
            if controller_name == "right" and "right" in self.raw_data["button_data"]:
                thumbstick = self.raw_data["button_data"][controller_name]["thumbstick"]
                self.teleop_action.base[2] = -thumbstick["x"] * self.rotation_speed
                self.teleop_action.torso = -thumbstick["y"] * self.movement_speed
            elif controller_name == "left" and "left" in self.raw_data["button_data"]:
                thumbstick = self.raw_data["button_data"][controller_name]["thumbstick"]
                self.teleop_action.base[0] = thumbstick["y"] * self.movement_speed
                self.teleop_action.base[1] = -thumbstick["x"] * self.movement_speed
        if not optimized_for_tour:
            # update head related info
            self.teleop_action.head = th.cat(
                (self.raw_data["transforms"]["head"][0], T.quat2euler(self.raw_data["transforms"]["head"][1]))
            )
            self.teleop_action.is_valid["head"] = self._is_valid_transform(self.raw_data["transforms"]["head"])

    def get_robot_teleop_action(self) -> th.Tensor:
        """
        Get the robot teleop action
        Returns:
            th.tensor: the robot teleop action
        """
        return self.robot.teleop_data_to_action(self.teleop_action)

    def snap_controller_to_eef(self, arm: str = "right") -> None:
        """
        Snap device to the robot end effector (ManipulationRobot only)
        Args:
            arm(str): name of the arm, one of "left" or "right". Default is "right".
        """
        # TODO: fix this
        pass
        # robot_base_orn = self.robot.get_position_orientation()[1]
        # robot_eef_pos = self.robot.eef_links[
        #     self.robot.arm_names[self.robot_arms.index(arm)]
        # ].get_position_orientation()[0]
        # target_transform = self.og2xr(pos=robot_eef_pos, orn=robot_base_orn)
        # self.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(
        #     target_transform.numpy(), self.controllers[arm]
        # )

    def reset_head_transform(self) -> None:
        """
        Function that resets the transform of the VR system (w.r.t.) head
        """
        if self.align_anchor_to == "touchpad":
            pos = th.tensor([0.0, 0.0, 1.0])
            orn = th.tensor([0.0, 0.0, 0.0, 1.0])
        else:
            if self.anchor_prim is not None:
                reference_frame = self.anchor_prim
            elif self.align_anchor_to == "camera":
                reference_frame = self.robot_cameras[self.active_camera_id]
            elif self.align_anchor_to == "base":
                reference_frame = self.robot
            else:
                raise ValueError(f"Invalid anchor: {self.align_anchor_to}")
            pos, orn = reference_frame.get_position_orientation()
        if self.robot:
            self.robot.keep_still()
        try:
            self.xr_core.schedule_set_camera(self.og2xr(pos, orn).numpy())
        except Exception as _:
            pass

    def _pose_in_robot_frame(self, pos: th.tensor, orn: th.tensor) -> Tuple[th.tensor, th.tensor]:
        """
        Get the pose in the robot frame
        Args:
            pos (th.tensor): the position in the world frame
            orn (th.tensor): the orientation in the world frame
        Returns:
            tuple(th.tensor, th.tensor): the position and orientation in the robot frame
        """
        robot_base_pos, robot_base_orn = self.robot.get_position_orientation()
        return T.relative_pose_transform(pos, orn, robot_base_pos, robot_base_orn)

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
            pos_offset = th.tensor([-pos_offset[0], pos_offset[2], -pos_offset[1]]).double().tolist()
            # self.vr_profile.add_move_physical_world_relative_to_device(pos_offset)
            self.xr_core.schedule_move_space_origin_relative_to_camera(*pos_offset)
        if rot_offset is not None:
            return
            # TODO: Fix this later
            rot_offset = th.tensor(rot_offset).double().tolist()
            # self.vr_profile.add_rotate_physical_world_around_device(rot_offset)
            self.xr_core.schedule_rotate_space_origin_relative_to_camera(*rot_offset)  # this takes yaw and pitch

    # TODO: check if this is necessary
    def _is_valid_transform(self, transform: Tuple[th.tensor, th.tensor]) -> bool:
        """
        Determine whether the transform is valid (ovxr plugin will return a zero position and rotation if not valid)
        """
        return th.any(transform[0] != th.zeros(3))

    def _update_devices(self) -> None:
        """
        Update the VR device list
        """
        # We are looking for '/user/head', '/user/hand/left', '/user/hand/right'
        for device in self.xr_core.get_all_input_devices():
            device_name = str(device.get_name())
            if device_name == "/user/head" and self.hmd is None:
                self.hmd = device
            elif device_name == "/user/hand/left" and "left" not in self.controllers:
                self.controllers["left"] = device
            elif device_name == "/user/hand/right" and "right" not in self.controllers:
                self.controllers["right"] = device

    def _update_device_transforms(self) -> None:
        """
        Get the transform matrix of each VR device *in world frame* and store in self.raw_data
        """
        assert self.hmd is not None, "VR headset device not found"
        self.raw_data["transforms"] = {
            "head": self.xr2og(self.hmd.get_virtual_world_pose()),
            "controllers": {
                name: self.xr2og(controller.get_virtual_world_pose()) for name, controller in self.controllers.items()
            },
            "trackers": {
                index: self.xr2og(tracker.get_virtual_world_pose()) for index, tracker in self.trackers.items()
            },
        }

    def _update_button_data(self):
        """
        Get the button data for each controller and store in self.raw_data
        """
        if "button_data" not in self.raw_data:
            self.raw_data["button_data"] = {}

        for name, controller in self.controllers.items():
            if len(controller.get_output_names()) == 0 and len(controller.get_input_names()) == 0:
                continue

            # Initialize controller entry if it doesn't exist
            if name not in self.raw_data["button_data"]:
                self.raw_data["button_data"][name] = {}

            # Add input data
            for input_name in controller.get_input_names():
                input_name = str(input_name)
                # Initialize input entry if it doesn't exist
                if input_name not in self.raw_data["button_data"][name]:
                    self.raw_data["button_data"][name][input_name] = {}

                # Add gesture data
                for gesture_name in controller.get_input_gesture_names(input_name):
                    gesture_name = str(gesture_name)
                    gesture_value = controller.get_input_gesture_value(input_name, gesture_name)
                    self.raw_data["button_data"][name][input_name][gesture_name] = gesture_value

    def _update_hand_tracking_data(self, e) -> None:
        """
        Get hand tracking data, see https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints for joint indices
        Args:
            e (carb.events.IEvent): event that contains hand tracking data as payload
        """
        e.consume()
        data_dict = e.payload
        for i, hand in enumerate(self.robot_arms):
            robot_arm_name = self.robot.arm_names[i]
            if data_dict[f"joint_count_{hand}"] != 0:
                self.teleop_action.is_valid[hand] = True
                self.raw_data["hand_data"][hand] = {"pos": [], "orn": []}
                # hand_joint_matrices is an array of flattened 4x4 transform matrices for the 26 hand markers
                hand_joint_matrices = data_dict[f"joint_matrices_{hand}"]
                for j in range(26):
                    # extract the pose from the flattened transform matrix
                    pos, orn = self.xr2og(hand_joint_matrices[16 * j : 16 * (j + 1)].reshape(4, 4))
                    self.raw_data["hand_data"][hand]["pos"].append(pos)
                    self.raw_data["hand_data"][hand]["orn"].append(orn)
                self.teleop_action[hand] = th.cat(
                    (
                        self.raw_data["hand_data"][hand]["pos"][0],
                        T.quat2euler(
                            T.quat_multiply(
                                self.raw_data["hand_data"][hand]["orn"][0],
                                self.robot.teleop_rotation_offset[robot_arm_name],
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
                self.teleop_action.hand_data[hand] = joint_angles
