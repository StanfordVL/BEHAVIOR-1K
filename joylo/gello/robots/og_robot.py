import os
import time
import torch as th
import numpy as np
from typing import Dict, Optional
import json
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs import HDF5CollectionWrapper
from omnigibson.macros import macros
from omnigibson.robots import Robot, REGISTERED_ROBOTS
from omnigibson.tasks import BehaviorTask
from omnigibson.systems.system_base import BaseSystem
from omnigibson.systems.macro_particle_system import MacroVisualParticleSystem
from omnigibson.object_states import Filled
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.usd_utils import RigidContactAPI, ControllableObjectViewAPI
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import ControllerView
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.asset_utils import get_task_instance_path

import gello.utils.og_teleop_utils as utils
from gello.utils.zmq_utils import ZMQRobotServer, ZMQServerThread
from gello.utils.og_teleop_cfg import *


class OGRobotServer:
    def __init__(
        self,
        robot: str,
        robot_name: str = "robot",
        config: str = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        recording_path: Optional[str] = None,
        task_name: Optional[str] = None,
        partial_load: bool = True,
        instance_id: Optional[int] = None,
        ghosting: bool = True,
        attachment_joint_visuals: Optional[bool] = None,
    ):
        if task_name is not None:
            available_tasks = utils.load_available_tasks()
            assert task_name in available_tasks, (
                f"Task {task_name} not found in available tasks"
            )
            self.task_name = task_name
            self.task_cfg = available_tasks[
                self.task_name
            ][
                0
            ]  # Regardless of whether we have multiple instances, we always load the seed instance by default; we will handle randomization for different instances during reset
            # Case 1: Both task name and instance id are provided; this is for formal data collection with domain randomization
            if instance_id is not None:
                # Initialize instance ID, decrementing by 1 to ensure proper increment during the first reset
                self.instance_id = instance_id - 1
            # Case 2: Only task name is provided; this is for task validation and testing with the seed instance
            else:
                self.instance_id = None
        else:
            # Case 3: No task or instance specified; this is for testing in an empty environment
            self.task_name = None
            self.task_cfg = None
            self.instance_id = None

        enable_attachment_joint_visuals = (
            utils.task_requires_attached_state(self.task_name)
            if attachment_joint_visuals is None
            else attachment_joint_visuals
        )
        with macros.unlocked():
            macros.object_states.attached_to.ENABLE_ATTACHMENT_JOINT_VISUALS = enable_attachment_joint_visuals

        utils.apply_omnigibson_macros()

        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        assert robot.lower() in REGISTERED_ROBOTS, (
            f"Robot {robot} is not a registered OmniGibson robot. Registered robots: {REGISTERED_ROBOTS}"
        )

        self._robot_type = robot

        if config is None:
            cfg = utils.generate_basic_environment_config(
                robot_type=robot,
                robot_name=robot_name,
                task_name=self.task_name,
                task_cfg=self.task_cfg,
            )
        else:
            # Load config from file
            cfg = parse_config(config)

        robot_config = utils.generate_robot_config(
            robot_type=robot,
            robot_name=robot_name,
            task_name=self.task_name,
            task_cfg=self.task_cfg,
        )
        cfg["robots"] = [robot_config]

        if self.task_name is not None and partial_load:
            relevant_rooms = utils.get_task_relevant_room_types(
                activity_name=self.task_name
            )
            if self.task_cfg:
                relevant_rooms = utils.augment_rooms(
                    relevant_rooms, self.task_cfg["scene_model"], self.task_name
                )
            cfg["scene"]["load_room_types"] = relevant_rooms

        self.env = og.Environment(configs=cfg)
        self.robot = self.env.robots[0]

        assert self.robot.is_manipulation, (
            f"Robot {robot} is not a manipulation robot! Cannot use GELLO"
        )
        self.is_bimanual_mobile = (
            self.robot.is_mobile_manipulation
            and self.robot.n_arms == 2
            and self.robot.is_articulated_trunk
        )

        self._teleop_config = ROBOT_CONFIGS[robot]

        self.ghosting = ghosting
        if self.ghosting:
            self.ghost = utils.setup_ghost_robot(
                self.env.scene, robot, task_cfg=self.task_cfg
            )
            og.sim.step()  # Initialize ghost robot
            self._ghost_appear_counter = {arm: 0 for arm in self.robot.arm_names}
            self.ghost_info = utils.setup_ghost_robot_info(
                self.ghost, self.robot, self._teleop_config
            )

        # Handle fluid object if needed
        if USE_FLUID:
            obj = self.env.scene.object_registry("name", "obj")
            water = self.env.scene.get_system("water")
            obj.states[Filled].set_value(water, True)
            for _ in range(50):
                og.sim.step()
            self.env.scene.update_initial_state()

        # Set up cameras, visualizations, and UI
        self._setup_teleop_support()

        # Set up status display
        self.status_window, self.status_labels = utils.setup_status_display_ui(
            og.sim.viewer_camera._viewport
        )
        self.event_queue = []

        # Set variables that are set during reset call
        self._reset_max_arm_delta = (
            DEFAULT_RESET_DELTA_SPEED * (np.pi / 180) * og.sim.get_sim_step_dt()
        )
        self._resume_cooldown_time = None
        self._in_cooldown = False
        self._current_trunk_translate = DEFAULT_TRUNK_TRANSLATE
        self._current_trunk_tilt_offset = 0.0
        self._current_trunk_tilt = 0.0
        self._joint_state = None
        self._joint_cmd = None
        self._waiting_to_resume = True
        self._should_update_checkpoint = False
        self._rollback_checkpoint_idx = None
        self._grasp_action = {arm: 1 for arm in self.robot.arm_names}
        self._blink_frequency = 1.0  # Hz

        # Recording configuration
        self._recording_path = recording_path
        if self._recording_path is not None:
            self.env = HDF5CollectionWrapper(
                env=self.env,
                output_path=self._recording_path,
                viewport_camera_path=og.sim.viewer_camera.active_camera_path,
                only_successes=False,
                flush_every_n_traj=1,
                use_vr=VIEWING_MODE == ViewingMode.VR,
                keep_checkpoint_rollback_data=True,
            )

        # Status tracking
        self._prev_grasp_status = {arm: False for arm in self.robot.arm_names}
        self._prev_in_hand_status = {arm: False for arm in self.robot.arm_names}
        self._frame_counter = 0
        self._prev_base_motion = False
        self._cam_switched = False
        self._button_toggled_state = {
            "x": False,
            "y": False,
            "a": False,
            "b": False,
            "left": False,
            "right": False,
        }
        self._gripper_action_signal_detectors = {
            arm: utils.SignalChangeDetector(debounce_time=0.5)
            for arm in self.robot.arm_names
        }

        # Set default active arm
        self.active_arm = "right"
        self._arm_shoulder_directions = {"left": -1.0, "right": 1.0}
        self.obs = {}

        # Cache values
        qpos_min, qpos_max = (
            self.robot.joint_lower_limits,
            self.robot.joint_upper_limits,
        )
        self._trunk_tilt_limits = {
            "lower": qpos_min[self.robot.trunk_control_idx][2],
            "upper": qpos_max[self.robot.trunk_control_idx][2],
        }
        self._arm_joint_limits = dict()
        for arm in self.robot.arm_names:
            self._arm_joint_limits[arm] = {
                "lower": qpos_min[self.robot.arm_control_idx[arm]],
                "upper": qpos_max[self.robot.arm_control_idx[arm]],
            }

        og.sim.stop()

        # # Set lower position iteration count for faster sim speed
        # og.sim._physics_context._physx_scene_api.GetMaxPositionIterationCountAttr().Set(8)
        # og.sim._physics_context._physx_scene_api.GetMaxVelocityIterationCountAttr().Set(1)
        isregistry = lazy.carb.settings.acquire_settings_interface()
        isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_NUM_THREADS, 0)
        # isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, int(1 / og.sim.get_physics_dt()))
        # isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, 30)

        # Enable CCD for all task-relevant objects
        if isinstance(self.env.task, BehaviorTask):
            from omnigibson.systems.system_base import BaseSystem

            for bddl_obj in self.env.task.object_scope.values():
                if bddl_obj is not None and not isinstance(bddl_obj, BaseSystem):
                    for link in bddl_obj.links.values():
                        link.ccd_enabled = True
        # Postprocessing robot and objects
        for obj in self.env.scene.objects:
            if obj != self.robot:
                if obj.category in VISUAL_ONLY_CATEGORIES:
                    obj.visual_only = True
            else:
                if isinstance(obj, Robot) and obj.model in ("r1", "r1pro"):
                    obj.base_footprint_link.mass = 250.0

        # Update ghost robot's masses to be uniform to avoid orthonormal errors
        if self.ghosting:
            for link in self.ghost.links.values():
                link.mass = 0.1

        og.sim.play()

        # Make sure robot fingers are extra grippy
        if APPLY_EXTRA_GRIP:
            gripper_mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
                prim_path=f"{self.robot.prim_path}/Looks/gripper_mat",
                name="gripper_material",
                static_friction=2.0,
                dynamic_friction=1.0,
                restitution=None,
            )
            for _, links in self.robot.finger_links.items():
                for link in links:
                    for msh in link.collision_meshes.values():
                        msh.apply_physics_material(gripper_mat)

        # Set optimized settings
        utils.optimize_sim_settings(vr_mode=(VIEWING_MODE == ViewingMode.VR))

        # Reset environment to initialize
        self._needs_initial_file_update = True
        self.reset()

        # Take a single step
        action = self.get_action()
        self.env.step(action)
        self.robot.keep_still()

        # Set up keyboard handlers
        self._setup_keyboard_handlers()

        # Set up VR system if needed
        self._setup_vr()

        # For some reason, toggle buttons get warped in terms of their placement -- we have them snap to their original
        # locations by setting their scale
        from omnigibson.object_states import ToggledOn

        for obj in self.env.scene.objects:
            if ToggledOn in obj.states:
                scale = obj.states[ToggledOn].visual_marker.scale
                obj.states[ToggledOn].visual_marker.scale = scale

        # Create ZMQ server for communication
        self._zmq_server = ZMQRobotServer(
            robot=self, host=host, port=port, verbose=False
        )
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

    def _setup_teleop_support(self):
        """Set up cameras, visualizations, UI elements"""
        # Setup cameras
        self.camera_paths, self.viewports = utils.setup_cameras(
            self.robot, self.env.external_sensors, RESOLUTION, self._teleop_config
        )
        self.active_camera_id = 0

        # Setup camera blinking visualizers
        self.camera_blinking_visualizers = utils.setup_camera_blinking_visualizers(
            self.camera_paths, self.env.scene
        )

        # Setup visualizers
        self.vis_elements = utils.setup_robot_visualizers(self.robot, self.env.scene)
        self.eef_cylinder_geoms = self.vis_elements["eef_cylinder_geoms"]
        self.vis_mats = self.vis_elements["vis_mats"]
        self.vertical_visualizers = self.vis_elements["vertical_visualizers"]
        self.reachability_visualizers = self.vis_elements["reachability_visualizers"]

        # Setup flashlights
        self.flashlights = utils.setup_flashlights(self.robot)

        # Setup task-related elements if task is specified
        if self.task_name is not None:
            # Setup task instruction UI
            (
                self.overlay_window,
                self.text_labels,
                self.instance_id_label,
                self.bddl_goal_conditions,
            ) = utils.setup_task_instruction_ui(
                self.task_name, self.env, self.instance_id
            )

            # Initialize goal status tracking
            self._prev_goal_status = {
                "satisfied": [],
                "unsatisfied": list(range(len(self.bddl_goal_conditions))),
            }

            # Get task-relevant objects
            task_objects = [obj for obj in self.env.task.object_scope.values() if obj is not None]

            self.task_relevant_objects = [
                obj
                for obj in task_objects
                if not isinstance(obj, BaseSystem)
                and obj.category != "agent"
                and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES
            ]

            # Setup object beacons
            self.object_beacons = utils.setup_object_beacons(
                self.task_relevant_objects, self.env.scene
            )

            # Attachment guides are owned by the AttachedTo object state. Keep this as a placeholder for future
            # JoyLo-specific task visualizers.
            self.task_visualizers = {}

            # Get task-irrelevant objects
            self.task_irrelevant_objects = [
                obj
                for obj in self.env.scene.objects
                if not isinstance(obj, BaseSystem)
                and obj not in task_objects
                and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES
            ]
        else:
            self.overlay_window = None
            self.text_labels = None
            self.bddl_goal_conditions = None
            self.task_relevant_objects = []
            self.task_irrelevant_objects = []
            self.object_beacons = {}
            self.task_visualizers = {}

    def _setup_keyboard_handlers(self):
        """Set up keyboard event handlers"""

        def keyboard_event_handler(event, *args, **kwargs):
            # Check if we've received a key press or repeat
            if (
                event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS
                or event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT
            ):
                if event.input == lazy.carb.input.KeyboardInput.R:
                    self.reset()
                elif event.input == lazy.carb.input.KeyboardInput.P:
                    self.pause()
                elif event.input == lazy.carb.input.KeyboardInput.X:
                    self.resume_control()
                elif event.input == lazy.carb.input.KeyboardInput.ESCAPE:
                    self.stop()

            # Callback always needs to return True
            return True

        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_interface = lazy.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        self.sub_keyboard = input_interface.subscribe_to_keyboard_events(
            keyboard, keyboard_event_handler
        )

    def _setup_vr(self):
        """Set up VR system if needed"""
        self.vr_system = None
        self.camera_prims = []

        if VIEWING_MODE == ViewingMode.VR:
            for cam_path in self.camera_paths:
                cam_prim = XFormPrim(
                    relative_prim_path=utils.absolute_prim_path_to_scene_relative(
                        self.robot.scene, cam_path
                    ),
                    name=cam_path,
                )
                cam_prim.load(self.robot.scene)
                self.camera_prims.append(cam_prim)

            from omnigibson.utils.teleop_utils import OVXRSystem

            self.vr_system = OVXRSystem(
                robot=self.robot,
                show_control_marker=False,
                system="SteamVR",
                eef_tracking_mode="disabled",
                align_anchor_to=self.camera_prims[0],
            )
            self.vr_system.start()

    def num_dofs(self) -> int:
        """Return the number of degrees of freedom"""
        return self.robot.n_joints

    def get_joint_state(self) -> th.tensor:
        """Get the current joint state"""
        return self._joint_state

    def command_joint_state(self, joint_state: th.tensor, component=None) -> None:
        """
        Command the robot to a joint state

        Args:
            joint_state: Target joint state
            component: Which component to control (optional)
        """
        state = joint_state.clone()
        if self.is_bimanual_mobile:
            arm_dof = len(self.robot.arm_control_idx["left"])
            component_dims = (
                ("left", arm_dof),
                ("right", arm_dof),
                ("base", 3),
                ("trunk", 2),
                ("left_gripper", 1),
                ("right_gripper", 1),
                ("button_-", 1),
                ("button_+", 1),
                ("button_x", 1),
                ("button_y", 1),
                ("button_b", 1),
                ("button_a", 1),
                ("button_capture", 1),
                ("button_home", 1),
                ("button_left", 1),
                ("button_right", 1),
            )
            start_idx = 0
            for comp, dim in component_dims:
                if start_idx >= len(state):
                    break
                self._joint_cmd[comp] = state[start_idx : start_idx + dim]
                start_idx += dim
        else:
            # Sort by component
            if component is None:
                component = self.active_arm
            assert component in self._joint_cmd, (
                f"Got invalid component joint cmd: {component}. Valid options: {self._joint_cmd.keys()}"
            )
            self._joint_cmd[component] = joint_state.clone()

    def freedrive_enabled(self) -> bool:
        """Check if freedrive mode is enabled"""
        return True

    def set_freedrive_mode(self, enable: bool):
        """Set freedrive mode"""
        pass

    def get_observations(self) -> Dict[str, th.tensor]:
        """Get the current observations"""
        return self.obs

    def _update_observations(self) -> Dict[str, th.tensor]:
        """Update observations with current robot state"""
        # Loop over all arms and grab relevant joint info
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()

        obs = dict()
        obs["active_arm"] = self.active_arm
        obs["in_cooldown"] = self._in_cooldown
        obs["base_contact"] = (
            RigidContactAPI.is_in_contact(
                self.env.scene.idx,
                set(self.robot.non_floor_touching_base_links),
                with_set=None,
                ignore_set=None,
                current_only=False,
            )
            if INCLUDE_BASE_CONTACT_OBS
            else False
        )
        obs["trunk_contact"] = (
            RigidContactAPI.is_in_contact(
                self.env.scene.idx,
                set(self.robot.trunk_links),
                with_set=None,
                ignore_set=None,
                current_only=False,
            )
            if INCLUDE_TRUNK_CONTACT_OBS
            else False
        )
        obs["reset_joints"] = bool(self._joint_cmd["button_y"][0].item())
        obs["waiting_to_resume"] = self._waiting_to_resume

        for i, arm in enumerate(self.robot.arm_names):
            arm_control_idx = self.robot.arm_control_idx[arm]
            obs[f"arm_{arm}_control_idx"] = arm_control_idx
            obs[f"arm_{arm}_joint_positions"] = joint_pos[arm_control_idx]
            # Account for tilt offset
            obs[f"arm_{arm}_joint_positions"][0] -= (
                self._current_trunk_tilt * self._arm_shoulder_directions[arm]
            )
            obs[f"arm_{arm}_joint_velocities"] = joint_vel[arm_control_idx]
            obs[f"arm_{arm}_gripper_positions"] = joint_pos[
                self.robot.gripper_control_idx[arm]
            ]
            obs[f"arm_{arm}_ee_pos_quat"] = th.concatenate(
                self.robot.eef_links[arm].get_position_orientation()
            )
            # When using VR, this expansive check makes the view glitch
            obs[f"arm_{arm}_contact"] = (
                RigidContactAPI.is_in_contact(
                    self.env.scene.idx,
                    set(self.robot.arm_links[arm]),
                    with_set=None,
                    ignore_set=None,
                    current_only=False,
                )
                if VIEWING_MODE != ViewingMode.VR and INCLUDE_ARM_CONTACT_OBS
                else False
            )

            obs[f"{arm}_gripper"] = self._joint_cmd[f"{arm}_gripper"].item()

        if INCLUDE_JACOBIAN_OBS:
            for arm in self.robot.arm_names:
                link_name = self.robot.eef_link_names[arm]

                start_idx = 0 if self.robot.fixed_base else 6
                link_idx = self.robot._articulation_view.get_body_index(link_name)
                jacobian = ControllableObjectViewAPI.get_relative_jacobian(
                    self.robot.articulation_root_path
                )[
                    -(self.robot.n_links - link_idx),
                    :,
                    start_idx : start_idx + self.robot.n_joints,
                ]

                jacobian = jacobian[:, self.robot.arm_control_idx[arm]]
                obs[f"arm_{arm}_jacobian"] = jacobian

        self.obs = obs

    def resume_control(self):
        """Resume control after waiting"""
        if self._waiting_to_resume:
            self._waiting_to_resume = False
            self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
            self._in_cooldown = True
            self._rollback_checkpoint_idx = None
            utils.add_status_event(
                self.event_queue, "waiting", "Control Resumed, cooling down..."
            )

    def serve(self) -> None:
        """Main serving loop"""
        # Start the zmq server
        self._zmq_server_thread.start()

        while True:
            self._update_observations()

            # Process button inputs
            self._process_button_inputs()

            # Update status display
            self.event_queue = utils.update_status_display(
                self.status_window, self.status_labels, self.event_queue, time.time()
            )

            # Only decrement cooldown if we're not waiting to resume
            if not self._waiting_to_resume:
                if self._in_cooldown:
                    utils.print_color(f"\rIn cooldown!{' ' * 40}", end="", flush=True)
                    self._in_cooldown = time.time() < self._resume_cooldown_time
                else:
                    utils.print_color(f"\rRunning!{' ' * 40}", end="", flush=True)

            # If waiting to resume, simply step sim without updating action
            if self._waiting_to_resume:
                og.sim.render()
                utils.print_color(
                    f"\rPress X (keyboard or JoyCon) to resume sim!{' ' * 30}",
                    end="",
                    flush=True,
                )
                utils.add_status_event(
                    self.event_queue,
                    "waiting",
                    "Waiting to Resume... Press X to start",
                    persistent=True,
                )
            else:
                # Generate action and deploy
                action = self.get_action()
                _, _, _, _, info = self.env.step(action)

                # Update checkpoint if queued
                if self._should_update_checkpoint:
                    self.env.update_checkpoint()
                    print("Auto recorded checkpoint due to goal status change!")
                    if self.event_queue:
                        utils.add_status_event(
                            self.event_queue,
                            "checkpoint",
                            "Checkpoint Recorded due to goal status change",
                        )
                    self._should_update_checkpoint = False

                # Update visualizations and status
                self._update_visualization_and_status(info)

    def _process_button_inputs(self):
        """Process button inputs from controller"""
        # If X is toggled from OFF -> ON, either:
        # (a) begin receiving commands, if currently paused, or
        # (b) record checkpoint, if actively running, or
        # (c) rollback to checkpoint, if at least a single "Y" was pressed beforehand
        button_x_state = self._joint_cmd["button_x"].item() != 0.0
        if button_x_state and not self._button_toggled_state["x"]:
            if self._waiting_to_resume:
                self.resume_control()
            else:
                if self._recording_path is not None:
                    if self._rollback_checkpoint_idx is not None:
                        print(
                            "Rolling back to checkpoint...watch out, GELLO will move on its own!"
                        )
                        utils.add_status_event(
                            self.event_queue,
                            "rollback",
                            "Rolling back to checkpoint...watch out, GELLO will move on its own!",
                        )
                        self.env.rollback_to_checkpoint(
                            index=-self._rollback_checkpoint_idx
                        )
                        utils.optimize_sim_settings(
                            vr_mode=(VIEWING_MODE == ViewingMode.VR)
                        )

                        # Extract trunk position values and calculate offsets
                        trunk_qpos = self.robot.get_joint_positions()[
                            self.robot.trunk_control_idx
                        ]
                        self._current_trunk_translate = (
                            utils.infer_trunk_translate_from_torso_qpos(trunk_qpos, self._teleop_config)
                        )
                        base_trunk_pos = utils.infer_torso_qpos_from_trunk_translate(
                            self._current_trunk_translate, self._teleop_config
                        )
                        self._current_trunk_tilt_offset = float(
                            trunk_qpos[2] - base_trunk_pos[2]
                        )

                        # Handle gripper actions
                        for arm in self.robot.arm_names:
                            group_key, ctrl_idx = self.robot.controllers[f"gripper_{arm}"]
                            gripper_goal = float(
                                ControllerView.get_goal(group_key, ctrl_idx)["target"]
                            )
                            checkpoint_gripper_action = 1 if gripper_goal > 0 else -1
                            self._grasp_action[arm] = checkpoint_gripper_action

                        print("Finished rolling back!")
                        self._waiting_to_resume = True
                    else:
                        self.env.update_checkpoint()
                        print("Checkpoint Recorded manually")
                        utils.add_status_event(
                            self.event_queue,
                            "checkpoint",
                            "Checkpoint Recorded manually",
                        )
        self._button_toggled_state["x"] = button_x_state

        # If Y is toggled from OFF -> ON, rollback to checkpoint
        button_y_state = self._joint_cmd["button_y"].item() != 0.0
        if button_y_state and not self._button_toggled_state["y"]:
            if self._recording_path is not None and len(self.env.checkpoint_states) > 0:
                # Increment rollback counter -- this means that we will rollback next time "X" is pressed
                if self._rollback_checkpoint_idx is None:
                    self._rollback_checkpoint_idx = 0
                self._rollback_checkpoint_idx = (
                    self._rollback_checkpoint_idx % len(self.env.checkpoint_states)
                ) + 1
                print(
                    f"Preparing to rollback to checkpoint idx -{self._rollback_checkpoint_idx}"
                )
                utils.add_status_event(
                    self.event_queue,
                    "rollback",
                    f"Preparing to rollback to checkpoint idx -{self._rollback_checkpoint_idx}",
                )
        self._button_toggled_state["y"] = button_y_state

        # If B is toggled from OFF -> ON, toggle camera
        button_b_state = self._joint_cmd["button_b"].item() != 0.0
        if button_b_state and not self._button_toggled_state["b"]:
            self.active_camera_id = 1 - self.active_camera_id
            og.sim.viewer_camera.active_camera_path = self.camera_paths[
                self.active_camera_id
            ]
            if VIEWING_MODE == ViewingMode.VR:
                self.vr_system.set_anchor_with_prim(
                    self.camera_prims[self.active_camera_id]
                )
        self._button_toggled_state["b"] = button_b_state

        # If A is toggled from OFF -> ON, toggle task-irrelevant object visibility
        button_a_state = self._joint_cmd["button_a"].item() != 0.0
        if button_a_state and not self._button_toggled_state["a"]:
            for obj in self.task_irrelevant_objects:
                obj.visible = not obj.visible
            task_objects = [obj for obj in self.env.task.object_scope.values() if obj is not None]
            current_task_relevant_objects = [
                obj
                for obj in task_objects
                if not isinstance(obj, BaseSystem)
                and obj.category != "agent"
                and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES
            ]
            should_highlight = not any(self.object_beacons[key].visible for key in current_task_relevant_objects if key in self.object_beacons)
            for entity in self.env.task.object_scope.values():
                if entity is None:
                    continue

                # Handle objects
                if entity in current_task_relevant_objects:
                    entity.highlighted = not entity.highlighted
                    if entity in self.object_beacons:
                        beacon = self.object_beacons[entity]
                        beacon.set_position_orientation(
                            position=entity.aabb_center + th.tensor([0, 0, BEACON_LENGTH / 2.0]),
                            orientation=T.euler2quat(th.tensor([0, 0, 0])),
                            frame="world",
                        )
                        beacon.visible = not beacon.visible
                    if entity.fixed_base and entity.articulated:
                        for name, link in entity.links.items():
                            if "meta" not in name and link != entity.root_link:
                                link.visible = not entity.highlighted
                    for vis_list in self.task_visualizers.values():
                        for vis in vis_list:
                            vis.visible = entity.highlighted

                # Handle visual particle systems - infer action from beacon visibility
                elif isinstance(entity, MacroVisualParticleSystem) and entity.initialized:
                    if should_highlight:
                        entity.particle_object.material.enable_highlight(highlight_color=[1.0, 0.1, 0.92], highlight_intensity=10000.0)
                    else:
                        entity.particle_object.material.disable_highlight()
        self._button_toggled_state["a"] = button_a_state

        # If capture is toggled from OFF -> ON, save and stop
        if self._joint_cmd["button_capture"].item() != 0.0:
            if not self._in_cooldown:
                self.stop()

        # If home is toggled from OFF -> ON, reset env
        if self._joint_cmd["button_home"].item() != 0.0:
            if not self._in_cooldown:
                self.reset()

        # If left arrow is toggled from OFF -> ON, toggle flashlight on left eef
        button_left_arrow_state = self._joint_cmd["button_left"].item() != 0.0
        if button_left_arrow_state and not self._button_toggled_state["left"]:
            with og.sim.editing_usd():
                if self.flashlights["left"].GetVisibilityAttr().Get() == "invisible":
                    self.flashlights["left"].MakeVisible()
                else:
                    self.flashlights["left"].MakeInvisible()
        self._button_toggled_state["left"] = button_left_arrow_state

        # If right arrow is toggled from OFF -> ON, toggle flashlight on right eef
        button_right_arrow_state = self._joint_cmd["button_right"].item() != 0.0
        if button_right_arrow_state and not self._button_toggled_state["right"]:
            with og.sim.editing_usd():
                if self.flashlights["right"].GetVisibilityAttr().Get() == "invisible":
                    self.flashlights["right"].MakeVisible()
                else:
                    self.flashlights["right"].MakeInvisible()
        self._button_toggled_state["right"] = button_right_arrow_state

    def _update_visualization_and_status(self, info):
        """Update visualization and status based on new information"""
        # Update task goal status if task is active
        if self.task_name is not None and "done" in info:
            current_goal_status = utils.update_goal_status(
                self.text_labels,
                info["done"]["goal_status"],
                self._prev_goal_status,
                self.env,
                self._recording_path,
                self.event_queue,
            )

            # Update checkpoint if new goals are satisfied
            if AUTO_CHECKPOINTING and len(current_goal_status["satisfied"]) > len(
                self._prev_goal_status["satisfied"]
            ):
                if self._recording_path is not None:
                    self._should_update_checkpoint = True

            self._prev_goal_status = current_goal_status

        # Update other visualization elements
        self._prev_in_hand_status = utils.update_in_hand_status(
            self.robot, self.vis_mats, self._prev_in_hand_status
        )

        self._prev_grasp_status = utils.update_grasp_status(
            self.robot, self.eef_cylinder_geoms, self._prev_grasp_status
        )

        self._prev_base_motion = utils.update_reachability_visualizers(
            self.reachability_visualizers, self._joint_cmd, self._prev_base_motion
        )

        utils.update_camera_blinking_visualizers(
            self.camera_blinking_visualizers,
            self.camera_paths[self.active_camera_id],
            self.obs,
            self._blink_frequency,
        )

        # Update checkpoint if needed
        self._frame_counter = utils.update_checkpoint(
            self.env, self._frame_counter, self._recording_path, self.event_queue
        )

    def get_action(self):
        """
        Generate action based on current joint commands

        Returns:
            torch.Tensor: Action for the robot
        """
        # Start an empty action
        action = th.zeros(self.robot.action_dim)

        # Apply arm action + extra dimension from base
        if self.is_bimanual_mobile:
            # Apply arm action
            left_act = (
                self._joint_cmd["left"]
                .clone()
                .clip(
                    self._arm_joint_limits["left"]["lower"],
                    self._arm_joint_limits["left"]["upper"],
                )
            )
            right_act = (
                self._joint_cmd["right"]
                .clone()
                .clip(
                    self._arm_joint_limits["right"]["lower"],
                    self._arm_joint_limits["right"]["upper"],
                )
            )

            # If we're in cooldown, clip values based on max delta value
            if self._in_cooldown:
                robot_pos = self.robot.get_joint_positions()
                robot_left_pos, robot_right_pos = [
                    robot_pos[self.robot.arm_control_idx[arm]]
                    for arm in ("left", "right")
                ]
                robot_left_delta = left_act - robot_left_pos
                robot_right_delta = right_act - robot_right_pos
                left_act = robot_left_pos + robot_left_delta.clip(
                    -self._reset_max_arm_delta, self._reset_max_arm_delta
                )
                right_act = robot_right_pos + robot_right_delta.clip(
                    -self._reset_max_arm_delta, self._reset_max_arm_delta
                )

            left_act[0] += (
                self._current_trunk_tilt * self._arm_shoulder_directions["left"]
            )
            right_act[0] += (
                self._current_trunk_tilt * self._arm_shoulder_directions["right"]
            )
            action[self.robot.arm_action_idx["left"]] = left_act
            action[self.robot.arm_action_idx["right"]] = right_act

            # Apply base action
            action[self.robot.base_action_idx] = self._joint_cmd["base"].clone()

            # Apply gripper action
            for arm in self.robot.arm_names:
                gripper_signal = self._joint_cmd[f"{arm}_gripper"].item()
                gripper_changed = self._gripper_action_signal_detectors[
                    arm
                ].process_sample(gripper_signal)
                if gripper_changed:
                    self._grasp_action[arm] = -self._grasp_action[arm]
                action[self.robot.gripper_action_idx[arm]] = self._grasp_action[arm]

            # Apply trunk action
            if SIMPLIFIED_TRUNK_CONTROL:
                # Update trunk translation (height)
                self._current_trunk_translate = float(
                    th.clamp(
                        th.tensor(self._current_trunk_translate, dtype=th.float)
                        - th.tensor(
                            self._joint_cmd["trunk"][0].item()
                            * og.sim.get_sim_step_dt(),
                            dtype=th.float,
                        ),
                        0.0,
                        2.0,
                    )
                )
                trunk_action = utils.infer_torso_qpos_from_trunk_translate(
                    self._current_trunk_translate, self._teleop_config
                )

                # Update trunk tilt offset
                self._current_trunk_tilt_offset = float(
                    th.clamp(
                        th.tensor(self._current_trunk_tilt_offset, dtype=th.float)
                        + th.tensor(
                            self._joint_cmd["trunk"][1].item()
                            * og.sim.get_sim_step_dt(),
                            dtype=th.float,
                        ),
                        self._trunk_tilt_limits["lower"] - trunk_action[2],
                        self._trunk_tilt_limits["upper"] - trunk_action[2],
                    )
                )
                trunk_action[2] = trunk_action[2] + self._current_trunk_tilt_offset

                action[self.robot.trunk_action_idx] = trunk_action

            # Update vertical visualizers
            if USE_VERTICAL_VISUALIZERS:
                for arm in ["left", "right"]:
                    arm_position = self.robot.eef_links[arm].get_position_orientation(
                        frame="world"
                    )[0]
                    self.vertical_visualizers[arm].set_position_orientation(
                        position=arm_position - th.tensor([0, 0, 1.0]),
                        orientation=th.tensor([0, 0, 0, 1.0]),
                        frame="world",
                    )
        else:
            action[self.robot.arm_action_idx[self.active_arm]] = self._joint_cmd[
                self.active_arm
            ].clone()

        # Optionally update ghost robot
        if self.ghosting and self._frame_counter % GHOST_UPDATE_FREQ == 0:
            self._ghost_appear_counter = utils.update_ghost_robot(
                self.ghost,
                self.robot,
                action,
                self._ghost_appear_counter,
                self.ghost_info,
            )

        return action

    def pause(self):
        self._waiting_to_resume = True
        for detector in self._gripper_action_signal_detectors.values():
            detector.reset()

    def reset(self, increment_instance=True):
        """
        Reset the environment and robot state

        Args:
            increment_instance (bool): If True and self.instance_id is not None, will increment the instance to reset to
                and reset to the updated instance id's initial state
        """
        if self._recording_path is not None:
            reset_text = "Resetting environment, episode recorded"
        else:
            reset_text = "Resetting environment"
        utils.add_status_event(self.event_queue, "reset", reset_text)
        # Reset internal variables
        self._ghost_appear_counter = {arm: 0 for arm in self.robot.arm_names}
        # Hide ghost robot on reset
        if self.ghosting:
            for link in self.ghost.links.values():
                link.visible = False
        self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
        self._in_cooldown = True
        self._current_trunk_translate = DEFAULT_TRUNK_TRANSLATE
        self._current_trunk_tilt_offset = 0.0
        self._current_trunk_tilt = 0.0
        self._waiting_to_resume = True
        self._joint_state = self.robot.reset_joint_pos
        self._joint_cmd = {
            arm: self._joint_state[self.robot.arm_control_idx[arm]]
            for arm in self.robot.arm_names
        }
        self._should_update_checkpoint = False
        self._grasp_action = {arm: 1 for arm in self.robot.arm_names}
        for detector in self._gripper_action_signal_detectors.values():
            detector.reset()
        for arm in self.robot.arm_names:
            self._joint_cmd[f"{arm}_gripper"] = th.ones(
                len(self.robot.gripper_action_idx[arm])
            )
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
        should_update_initial_file = self._needs_initial_file_update

        # Update the instance id / initial state if the instance ID is specified
        # We will manually update the task relevant objects (TRO) state
        if self.instance_id is not None and increment_instance:
            should_update_initial_file = True
            self.instance_id += 1
            scene_model = self.env.task.scene_name
            tro_filename = self.env.task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=self.env.task.activity_name,
                activity_definition_id=self.env.task.activity_definition_id,
                activity_instance_id=self.instance_id,
            )
            tro_file_path = os.path.join(get_task_instance_path(
                scene_model,
                f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
            ))
            # check if tro_file_path exists, if not, then presumbaly we are done
            if not os.path.exists(tro_file_path):
                print(
                    f"Task {self.env.task.activity_name} instance id: {self.instance_id} does not exist"
                )
                print("No more task instances to load, exiting...")
                self.stop()
            with open(tro_file_path, "r") as f:
                tro_state = recursively_convert_to_torch(json.load(f))
            self.env.scene.reset()
            for tro_key, tro_state in tro_state.items():
                if tro_key == "robot_poses":
                    presampled_robot_poses = tro_state
                    # make all robot name lower case
                    presampled_robot_poses = {
                        k.lower(): v for k, v in presampled_robot_poses.items()
                    }
                    if "robot" in presampled_robot_poses:
                        robot_pose = presampled_robot_poses["robot"][0]
                    elif self.robot.model in presampled_robot_poses:
                        print("No generic presampled robot pose found, using robot-specific pose.")
                        robot_pose = presampled_robot_poses[self.robot.model][0]
                    else:
                        raise KeyError(f"No generic or model-specific presampled robot pose found for {self.robot.model}!")
                    # Only set pose (we assume this is a holonomic robot, so ignore Rx / Ry and only take Rz component
                    # for orientation
                    robot_pos, robot_quat = robot_pose["position"], robot_pose["orientation"]
                    self.robot.set_position_orientation(robot_pos, robot_quat)
                    # Write robot poses to scene metadata
                    self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
                else:
                    self.env.task.object_scope[tro_key].load_state(
                        tro_state, serialized=False
                    )

            print(
                f"\nLoading task {self.env.task.activity_name} instance id: {self.instance_id}\n"
            )
            utils.update_instance_id_label(self.instance_id_label, self.instance_id)


        if should_update_initial_file:
            # Try to ensure that all task-relevant objects are stable before caching a reset baseline.
            # They should already be stable from the sampled instance, but loading the state can cause jitter.
            for _ in range(25):
                og.sim.step_physics()
                self.robot.keep_still()
                for entity in self.env.task.object_scope.values():
                    if entity is not None and not isinstance(entity, BaseSystem):
                        entity.keep_still()
            self.robot.keep_still()
            self.env.scene.update_initial_file()
            self._needs_initial_file_update = False

        # Reset env
        self.env.reset()
        self.robot.keep_still()

        # If we're recording, record the retroactively record the instance ID from the previous episode
        if (
            self._recording_path is not None
            and self.instance_id is not None
            and self.env.traj_count > 0
        ):
            instance_id = (
                self.instance_id - 1 if increment_instance else self.instance_id
            )
            group = self.env.hdf5_file[f"data/demo_{self.env.traj_count - 1}"]
            self.env.add_metadata(group=group, name="instance_id", data=instance_id)

    def stop(self) -> None:
        """Stop the server and clean up resources"""
        self._zmq_server_thread.terminate()
        self._zmq_server_thread.join()

        if self._recording_path is not None:
            # Sanity check if we are in the middle of an episode; always flush the current trajectory
            if len(self.env.current_traj_history) > 0:
                self.env.flush_current_traj()

            self.env.save_data()

        if VIEWING_MODE == ViewingMode.VR:
            self.vr_system.stop()

        og.shutdown()

    def __del__(self) -> None:
        """Clean up when object is deleted"""
        self.stop()
