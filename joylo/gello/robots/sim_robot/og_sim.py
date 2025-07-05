import os
import time
import torch as th
import numpy as np
from typing import Optional
import json

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.envs import DataCollectionWrapper
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.robots.r1 import R1
from omnigibson.robots.r1pro import R1Pro
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.tasks import BehaviorTask
from omnigibson.systems.system_base import BaseSystem
from omnigibson.systems.macro_particle_system import MacroVisualParticleSystem
from omnigibson.utils.teleop_utils import OVXRSystem
from omnigibson.object_states import Filled
from omnigibson.prims.xform_prim import XFormPrim
import omnigibson.utils.transform_utils as T
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.python_utils import recursively_convert_to_torch

from gello.devices import DEVICE_LIB

from gello.robots.sim_robot.og_teleop_cfg import *
import gello.robots.sim_robot.og_teleop_utils as utils


class OGRobotServer:
    def __init__(
        self,
        robot: str = ROBOT_TYPE,
        teleop: str = "JoyLo",
        config: str = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        recording_path: Optional[str] = None,
        task_name: Optional[str] = None,
        partial_load: bool = True,
        instance_id: Optional[int] = None,
        ghosting: bool = True,
    ):      
        if task_name is not None:
            available_tasks = utils.load_available_tasks()
            assert task_name in available_tasks, f"Task {task_name} not found in available tasks"
            self.task_name = task_name
            self.task_cfg = available_tasks[self.task_name][0] # Regardless of whether we have multiple instances, we always load the seed instance by default; we will handle randomization for different instances during reset
            # Case 1: Both task name and instance id are provided; this is for formal data collection with domain randomization
            if instance_id is not None:
                assert task_name in VALIDATED_TASKS, f"Task {task_name} is not in the list of validated tasks: {VALIDATED_TASKS}"
                # Initialize instance ID, decrementing by 1 to ensure proper increment during the first reset
                self.instance_id = (instance_id - 1)
            # Case 2: Only task name is provided; this is for task validation and testing with the seed instance
            else:
                self.instance_id = None
        else:
            # Case 3: No task or instance specified; this is for testing in an empty environment
            self.task_name = None
            self.task_cfg = None
            self.instance_id = None

        utils.apply_omnigibson_macros()
        
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        robot_cls = REGISTERED_ROBOTS.get(robot, None)
        assert robot_cls is not None, f"Got invalid OmniGibson robot class: {robot}"
        assert issubclass(robot_cls, ManipulationRobot), f"Robot class {robot} is not a manipulation robot! Cannot use GELLO"
        assert robot in SUPPORTED_ROBOTS, f"Robot {robot} is not supported by GELLO! Supported robots: {SUPPORTED_ROBOTS}"

        if config is None:
            cfg = utils.generate_basic_environment_config(self.task_name, self.task_cfg)
        else:
            # Load config from file
            cfg = parse_config(config)

        robot_config = utils.generate_robot_config(self.task_name, self.task_cfg)
        cfg["robots"] = [robot_config]

        if self.task_name is not None and partial_load:
            relevant_rooms = utils.get_task_relevant_room_types(activity_name=self.task_name)
            if self.task_cfg:
                relevant_rooms = utils.augment_rooms(relevant_rooms, self.task_cfg["scene_model"], self.task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        self.env = og.Environment(configs=cfg)
        self.robot = self.env.robots[0]
        
        self.ghosting = ghosting
        if self.ghosting:
            self.ghost = utils.setup_ghost_robot(self.env.scene, self.task_cfg)
            og.sim.step() # Initialize ghost robot
            self._ghost_appear_counter = {arm: 0 for arm in self.robot.arm_names}
            self.ghost_info = utils.setup_ghost_robot_info(self.ghost, self.robot)

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
        self.status_window, self.status_labels = utils.setup_status_display_ui(og.sim.viewer_camera._viewport)
        self.event_queue = []

        # Set variables that are set during reset call
        self._resume_cooldown_time = None
        self._in_cooldown = False

        self._waiting_to_resume = True
        self._should_update_checkpoint = False
        self._rollback_checkpoint_idx = None
        self._blink_frequency = 1.0 # Hz

        # Recording configuration
        self._recording_path = recording_path
        if self._recording_path is not None:
            self.env = DataCollectionWrapper(
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
        self._button_toggled_state = {
            "x": False,
            "y": False,
            "a": False,
            "b": False,
            "left": False,
            "right": False,
        }

        # Setup teleop controller
        self.teleop_device = DEVICE_LIB[teleop](robot=self.robot, host=host, port=port)

        with og.sim.stopped():
            # # Set lower position iteration count for faster sim speed
            # og.sim._physics_context._physx_scene_api.GetMaxPositionIterationCountAttr().Set(8)
            # og.sim._physics_context._physx_scene_api.GetMaxVelocityIterationCountAttr().Set(1)
            isregistry = lazy.carb.settings.acquire_settings_interface()
            isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_NUM_THREADS, 0)
            # isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, int(1 / og.sim.get_physics_dt()))
            # isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, 30)

            # Enable CCD for all task-relevant objects
            if isinstance(self.env.task, BehaviorTask):
                for bddl_obj in self.env.task.object_scope.values():
                    if not bddl_obj.is_system and bddl_obj.exists:
                        for link in bddl_obj.wrapped_obj.links.values():
                            link.ccd_enabled = True
            # Postprocessing robot and objects
            for obj in self.env.scene.objects:
                if obj != self.robot:
                    if obj.category in VISUAL_ONLY_CATEGORIES:
                        obj.visual_only = True
                else:
                    if isinstance(obj, (R1, R1Pro)):
                        obj.base_footprint_link.mass = 250.0

            # Update ghost robot's masses to be uniform to avoid orthonormal errors
            if self.ghosting:
                for link in self.ghost.links.values():
                    link.mass = 0.1

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
        self.reset()

        # Take a single step
        action = self.get_action()
        self.env.step(action)

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

        assert teleop in DEVICE_LIB, f"Got invalid teleop type: {teleop}. Supported types: {list(DEVICE_LIB.keys())}"

    def _setup_teleop_support(self):
        """Set up cameras, visualizations, UI elements"""
        # Setup cameras
        self.camera_paths, self.viewports = utils.setup_cameras(
            self.robot, 
            self.env.external_sensors, 
            RESOLUTION
        )
        self.active_camera_id = 0
        
        # Setup camera blinking visualizers
        self.camera_blinking_visualizers = utils.setup_camera_blinking_visualizers(
            self.camera_paths, 
            self.env.scene
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
            self.overlay_window, self.text_labels, self.instance_id_label, self.bddl_goal_conditions = utils.setup_task_instruction_ui(
                self.task_name, 
                self.env,
                self.instance_id
            )
            
            # Initialize goal status tracking
            self._prev_goal_status = {
                'satisfied': [],
                'unsatisfied': list(range(len(self.bddl_goal_conditions)))
            }
            
            # Get task-relevant objects
            task_objects = [bddl_obj.wrapped_obj for bddl_obj in self.env.task.object_scope.values() 
                            if bddl_obj.wrapped_obj is not None and bddl_obj.exists]
            
            self.task_relevant_objects = [obj for obj in task_objects 
                                          if not isinstance(obj, BaseSystem)
                                          and obj.category != "agent" 
                                          and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES]
            
            # Setup object beacons
            self.object_beacons = utils.setup_object_beacons(self.task_relevant_objects, self.env.scene)
            
            # Setup task-specific visualizers
            self.task_visualizers = utils.setup_task_visualizers(self.task_relevant_objects, self.env.scene)
            
            # Get task-irrelevant objects
            self.task_irrelevant_objects = [obj for obj in self.env.scene.objects
                                            if not isinstance(obj, BaseSystem)
                                            and obj not in task_objects
                                            and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES]
        else:
            self.overlay_window = None
            self.text_labels = None
            self.bddl_goal_conditions = None
            self.task_relevant_objects = []
            self.task_irrelevant_objects = []
            self.object_beacons = {}

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
        self.sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)

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
            
            self.vr_system = OVXRSystem(
                robot=self.robot,
                show_control_marker=False,
                system="SteamVR",
                eef_tracking_mode="disabled",
                align_anchor_to=self.camera_prims[0],
            )
            self.vr_system.start()

    def resume_control(self):
        """Resume control after waiting"""
        if self._waiting_to_resume:
            self._waiting_to_resume = False
            self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
            self._in_cooldown = True
            self._rollback_checkpoint_idx = None
            utils.add_status_event(self.event_queue, "waiting", "Control Resumed, cooling down...")

    def serve(self) -> None:
        """Main serving loop"""
        # start the teleop controller
        self.teleop_device.start()

        while self.teleop_device.is_running():
            self._update_observations()

            # Process button inputs
            self._process_button_inputs()
            
            # Update status display
            self.event_queue = utils.update_status_display(
                self.status_window,
                self.status_labels,
                self.event_queue,
                time.time()
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
                utils.print_color(f"\rPress X (keyboard or JoyCon) to resume sim!{' ' * 30}", end="", flush=True)
                utils.add_status_event(self.event_queue, "waiting", "Waiting to Resume... Press X to start", persistent=True)
            else:
                # Generate action and deploy
                action = self.get_action()
                _, _, _, _, info = self.env.step(action)
                
                # Update checkpoint if queued
                if self._should_update_checkpoint:
                    self.env.update_checkpoint()
                    print("Auto recorded checkpoint due to goal status change!")
                    if self.event_queue:
                        utils.add_status_event(self.event_queue, "checkpoint", "Checkpoint Recorded due to goal status change")
                    self._should_update_checkpoint = False

                # Update visualizations and status
                self._update_visualization_and_status(info)

    def _process_button_inputs(self):
        """Process button inputs from controller"""
        # If X is toggled from OFF -> ON, either:
        # (a) begin receiving commands, if currently paused, or
        # (b) record checkpoint, if actively running, or
        # (c) rollback to checkpoint, if at least a single "Y" was pressed beforehand

        button_cmd = self.teleop_device.get_button_input_cmd()

        button_x_state = button_cmd["button_x"] != 0.0
        if button_x_state and not self._button_toggled_state["x"]:
            if self._waiting_to_resume:
                self.resume_control()
            else:
                if self._recording_path is not None:
                    if self._rollback_checkpoint_idx is not None:
                        print("Rolling back to checkpoint...watch out, GELLO will move on its own!")
                        utils.add_status_event(self.event_queue, "rollback",
                                               "Rolling back to checkpoint...watch out, GELLO will move on its own!")
                        self.env.rollback_to_checkpoint(index=-self._rollback_checkpoint_idx)
                        utils.optimize_sim_settings(vr_mode=(VIEWING_MODE == ViewingMode.VR))

                        # Extract trunk position values and calculate offsets
                        trunk_qpos = self.robot.get_joint_positions()[self.robot.trunk_control_idx]
                        self.teleop_device.current_trunk_translate = utils.infer_trunk_translate_from_torso_qpos(trunk_qpos)
                        base_trunk_pos = utils.infer_torso_qpos_from_trunk_translate(self.teleop_device.current_trunk_translate)
                        self.teleop_device.current_trunk_tilt_offset = float(trunk_qpos[2] - base_trunk_pos[2])
                        
                        # Handle gripper actions
                        for arm in self.robot.arm_names:
                            gripper_goal = float(self.robot.controllers[f"gripper_{arm}"].goal["target"])
                            checkpoint_gripper_action = 1 if gripper_goal > 0 else -1
                            self.teleop_device.grasp_action[arm] = checkpoint_gripper_action

                        print("Finished rolling back!")
                        self._waiting_to_resume = True
                    else:
                        self.env.update_checkpoint()
                        print("Checkpoint Recorded manually")
                        utils.add_status_event(self.event_queue, "checkpoint", "Checkpoint Recorded manually")
        self._button_toggled_state["x"] = button_x_state

        # If Y is toggled from OFF -> ON, rollback to checkpoint
        button_y_state = button_cmd["button_y"] != 0.0
        if button_y_state and not self._button_toggled_state["y"]:
            if self._recording_path is not None and len(self.env.checkpoint_states) > 0:
                # Increment rollback counter -- this means that we will rollback next time "X" is pressed
                if self._rollback_checkpoint_idx is None:
                    self._rollback_checkpoint_idx = 0
                self._rollback_checkpoint_idx = (self._rollback_checkpoint_idx % len(self.env.checkpoint_states)) + 1
                print(f"Preparing to rollback to checkpoint idx -{self._rollback_checkpoint_idx}")
                utils.add_status_event(self.event_queue, "rollback", f"Preparing to rollback to checkpoint idx -{self._rollback_checkpoint_idx}")
        self._button_toggled_state["y"] = button_y_state

        # If B is toggled from OFF -> ON, toggle camera
        button_b_state = button_cmd["button_b"] != 0.0
        if button_b_state and not self._button_toggled_state["b"]:
            self.active_camera_id = 1 - self.active_camera_id
            og.sim.viewer_camera.active_camera_path = self.camera_paths[self.active_camera_id]
            if VIEWING_MODE == ViewingMode.VR:
                self.vr_system.set_anchor_with_prim(
                    self.camera_prims[self.active_camera_id]
                )
        self._button_toggled_state["b"] = button_b_state

        # If A is toggled from OFF -> ON, toggle task-irrelevant object visibility
        button_a_state = button_cmd["button_a"] != 0.0
        if button_a_state and not self._button_toggled_state["a"]:
            for obj in self.task_irrelevant_objects:
                obj.visible = not obj.visible
            task_objects = [bddl_obj.wrapped_obj for bddl_obj in self.env.task.object_scope.values() 
                            if bddl_obj.wrapped_obj is not None and bddl_obj.exists]
            current_task_relevant_objects = [obj for obj in task_objects 
                                        if not isinstance(obj, BaseSystem)
                                        and obj.category != "agent" 
                                        and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES]
            should_highlight = not any(self.object_beacons[key].visible for key in current_task_relevant_objects if key in self.object_beacons)
            for entity in self.env.task.object_scope.values():
                entity_obj = entity.wrapped_obj
                entity_unwrapped = entity.unwrapped
                
                # Handle objects
                if entity_obj in current_task_relevant_objects:
                    obj = entity_obj
                    obj.highlighted = not obj.highlighted
                    if obj in self.object_beacons:
                        beacon = self.object_beacons[obj]
                        beacon.set_position_orientation(
                            position=obj.aabb_center + th.tensor([0, 0, BEACON_LENGTH / 2.0]),
                            orientation=T.euler2quat(th.tensor([0, 0, 0])),
                            frame="world"
                        )
                        beacon.visible = not beacon.visible
                    if obj.fixed_base and obj.articulated:
                        for name, link in obj.links.items():
                            if not 'meta' in name and link != obj.root_link:
                                link.visible = not obj.highlighted
                    for vis_list in self.task_visualizers.values():
                        for vis in vis_list:
                            vis.visible = obj.highlighted
                
                # Handle visual particle systems - infer action from beacon visibility
                elif isinstance(entity_unwrapped, MacroVisualParticleSystem) and entity_unwrapped.initialized:
                    if should_highlight:
                        entity_unwrapped.particle_object.material.enable_highlight(highlight_color=[1.0, 0.1, 0.92], highlight_intensity=10000.0)
                    else:
                        entity_unwrapped.particle_object.material.disable_highlight()
        self._button_toggled_state["a"] = button_a_state

        # If capture is toggled from OFF -> ON, breakpoint
        if button_cmd["button_capture"] != 0.0:
            if not self._in_cooldown:
                breakpoint()

        # If home is toggled from OFF -> ON, reset env
        if button_cmd["button_home"] != 0.0:
            if not self._in_cooldown:
                self.reset()

        # If left arrow is toggled from OFF -> ON, toggle flashlight on left eef
        button_left_arrow_state = button_cmd["button_left"] != 0.0
        if button_left_arrow_state and not self._button_toggled_state["left"]:
            if self.flashlights["left"].GetVisibilityAttr().Get() == "invisible":
                self.flashlights["left"].MakeVisible()
            else:
                self.flashlights["left"].MakeInvisible()
        self._button_toggled_state["left"] = button_left_arrow_state
        
        # If right arrow is toggled from OFF -> ON, toggle flashlight on right eef
        button_right_arrow_state = button_cmd["button_right"] != 0.0
        if button_right_arrow_state and not self._button_toggled_state["right"]:
            if self.flashlights["right"].GetVisibilityAttr().Get() == "invisible":
                self.flashlights["right"].MakeVisible()
            else:
                self.flashlights["right"].MakeInvisible()
        self._button_toggled_state["right"] = button_right_arrow_state

    def _update_visualization_and_status(self, info):
        """Update visualization and status based on new information"""
        # Update task goal status if task is active
        if self.task_name is not None and 'done' in info:
            current_goal_status = utils.update_goal_status(
                self.text_labels,
                info['done']['goal_status'],
                self._prev_goal_status,
                self.env,
                self._recording_path,
                self.event_queue
            )
            
            # Update checkpoint if new goals are satisfied
            if AUTO_CHECKPOINTING and len(current_goal_status['satisfied']) > len(self._prev_goal_status['satisfied']):
                if self._recording_path is not None:
                    self._should_update_checkpoint = True
            
            self._prev_goal_status = current_goal_status
        
        # Update other visualization elements
        self._prev_in_hand_status = utils.update_in_hand_status(
            self.robot,
            self.vis_mats,
            self._prev_in_hand_status
        )
        
        self._prev_grasp_status = utils.update_grasp_status(
            self.robot,
            self.eef_cylinder_geoms,
            self._prev_grasp_status
        )
        
        self._prev_base_motion = utils.update_reachability_visualizers(
            self.reachability_visualizers,
            self.teleop_device.get_base_cmd(),
            self._prev_base_motion
        )
        
        utils.update_camera_blinking_visualizers(
            self.camera_blinking_visualizers,
            self.camera_paths[self.active_camera_id],
            self.teleop_device.obs,
            self._blink_frequency,
        )
        
        # Update checkpoint if needed
        self._frame_counter = utils.update_checkpoint(
            self.env,
            self._frame_counter,
            self._recording_path,
            self.event_queue
        )

    def get_action(self):
        """
        Generate action based on current joint commands
        
        Returns:
            torch.Tensor: Action for the robot
        """
        action = self.teleop_device.get_action(in_cooldown=self._in_cooldown)

        # Update vertical visualizers
        if isinstance(self.robot, R1) and USE_VERTICAL_VISUALIZERS:
            for arm in ["left", "right"]:
                arm_position = self.robot.eef_links[arm].get_position_orientation(frame="world")[0]
                self.vertical_visualizers[arm].set_position_orientation(
                    position=arm_position - th.tensor([0, 0, 1.0]), 
                    orientation=th.tensor([0, 0, 0, 1.0]), 
                    frame="world"
                )

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

    def _update_observations(self) -> None:
        self.teleop_device.update_observations()
        # update cooldown and wait info
        self.teleop_device.obs["in_cooldown"] = self._in_cooldown
        self.teleop_device.obs["waiting_to_resume"] = self._waiting_to_resume

    def pause(self):
        self._waiting_to_resume = True
        self.teleop_device.pause()


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
        self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
        self._in_cooldown = True
        self._waiting_to_resume = True

        self._should_update_checkpoint = False
        
        self.teleop_device.reset()

        # Update the instance id / initial state if the instance ID is specified
        # We will manually update the task relevant objects (TRO) state
        if self.instance_id is not None and increment_instance:
            self.instance_id += 1
            scene_model = self.env.task.scene_name
            tro_filename = self.env.task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=self.env.task.activity_name,
                activity_definition_id=self.env.task.activity_definition_id,
                activity_instance_id=self.instance_id,
            )
            tro_file_path = f"{gm.DATASET_PATH}/scenes/{scene_model}/json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json"
            # check if tro_file_path exists, if not, then presumbaly we are done
            if not os.path.exists(tro_file_path):
                print(f"Task {self.env.task.activity_name} instance id: {self.instance_id} does not exist")
                print("No more task instances to load, exiting...")
                self.stop()
            with open(tro_file_path, "r") as f:
                tro_state = recursively_convert_to_torch(json.load(f))
            self.env.scene.reset()
            for bddl_name, obj_state in tro_state.items():
                if "agent" in bddl_name:
                    # Only set pose (we assume this is a holonomic robot, so ignore Rx / Ry and only take Rz component
                    # for orientation
                    robot_pos = obj_state["joint_pos"][:3] + obj_state["root_link"]["pos"]
                    robot_quat = T.euler2quat(th.tensor([0, 0, obj_state["joint_pos"][5]]))
                    self.env.task.object_scope[bddl_name].set_position_orientation(robot_pos, robot_quat)
                else:
                    self.env.task.object_scope[bddl_name].load_state(obj_state, serialized=False)
                    
            # Try to ensure that all task-relevant objects are stable
            # They should already be stable from the sampled instance, but there is some issue where loading the state
            # causes some jitter (maybe for small mass / thin objects?)
            for _ in range(25):
                og.sim.step_physics()
                for entity in self.env.task.object_scope.values():
                    if not entity.is_system and entity.exists:
                        entity.keep_still()
            self.env.scene.update_initial_file()
            print(f"\nLoading task {self.env.task.activity_name} instance id: {self.instance_id}\n")
            utils.update_instance_id_label(self.instance_id_label, self.instance_id)

        # Reset env
        self.env.reset()

        # If we're recording, record the retroactively record the instance ID from the previous episode
        if self._recording_path is not None and self.instance_id is not None and self.env.traj_count > 0:
            instance_id = self.instance_id - 1 if increment_instance else self.instance_id
            group = self.env.hdf5_file[f"data/demo_{self.env.traj_count - 1}"]
            self.env.add_metadata(group=group, name="instance_id", data=instance_id)

    def stop(self) -> None:
        """Stop the controller and clean up resources"""
        
        if self._recording_path is not None:
            # Sanity check if we are in the middle of an episode; always flush the current trajectory
            if len(self.env.current_traj_history) > 0:
                self.env.flush_current_traj()

            self.env.save_data()
        
        if VIEWING_MODE == ViewingMode.VR:
            self.vr_system.stop()
        
        self.teleop_device.stop()
        
        og.shutdown()

    def __del__(self) -> None:
        """Clean up when object is deleted"""
        self.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OmniGibson Robot Server")
    parser.add_argument("--robot", type=str, default=ROBOT_TYPE, help="Robot type")
    parser.add_argument("--teleop", type=str, default="JoyLo", help="Teleop type")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5556, help="Server port")
    parser.add_argument("--recording-path", type=str, default=None, help="Path to save recordings")
    parser.add_argument("--task", type=str, default=None, help="Task name")
    parser.add_argument("--no-ghost", action="store_true", help="Disable ghost robot visualization")
    
    args = parser.parse_args()
    
    sim = OGRobotServer(
        robot=args.robot,
        teleop=args.teleop,
        host=args.host,
        port=args.port,
        recording_path=args.recording_path,
        task_name=args.task,
        ghosting=not args.no_ghost
    )
    
    sim.serve()
    print("Server stopped")