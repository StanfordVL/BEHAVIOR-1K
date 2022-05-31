import ctypes
import logging
import platform
import time
from time import sleep

import numpy as np


from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR, VrSettings
from igibson.render.viewer import ViewerVR
from igibson.robots.behavior_robot import BODY_ANGULAR_VELOCITY, BODY_LINEAR_VELOCITY, HAND_BASE_ROTS
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.simulator import Simulator
from igibson.utils.vr_utils import VR_CONTROLLERS, VR_DEVICES, VrData, calc_offset, calc_z_rot_from_right

ATTACHMENT_BUTTON_TIME_THRESHOLD = 1  # second


class SimulatorVR(Simulator):
    """
    Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(
        self,
        gravity=9.8,
        physics_timestep=1 / 120.0,
        render_timestep=1 / 30.0,
        solver_iterations=100,
        mode="vr",
        image_width=128,
        image_height=128,
        vertical_fov=90,
        device_idx=0,
        rendering_settings=MeshRendererSettings(),
        vr_settings=VrSettings(),
        use_pb_gui=False,
    ):
        """
        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param solver_iterations: number of solver iterations to feed into pybullet, can be reduced to increase speed.
            pybullet default value is 50.
        :param use_variable_step_num: whether to use a fixed (1) or variable physics step number
        :param mode: choose mode from headless, headless_tensor, gui_interactive, gui_non_interactive
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param rendering_settings: settings to use for mesh renderer
        :param vr_settings: settings to use for VR in simulator and MeshRendererVR
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        if platform.system() == "Windows":
            # By default, windows does not provide ms level timing accuracy
            winmm = ctypes.WinDLL("winmm")  # type: ignore
            winmm.timeBeginPeriod(1)

        # Blend highlight for VR overlay
        rendering_settings.blend_highlight = True

        # Whether the VR system is actively hooked up to the VR agent.
        self.vr_attached = False
        self._vr_attachment_button_previously_pressed = False

        # Starting position for the VR (default set to None if no starting position is specified by the user)
        self.vr_settings = vr_settings
        self.vr_overlay_initialized = False
        self.vr_start_pos = None
        self.max_haptic_duration = 4000

        # Duration of a vsync frame - assumes 90Hz refresh rate
        self.vsync_frame_dur = 11.11e-3
        # Timing variables for functions called outside of step() that also take up frame time
        self.frame_end_time = None

        # Variables for data saving and replay in VR
        self.last_physics_timestep = -1
        self.last_render_timestep = -1
        self.last_physics_step_num = -1
        self.last_frame_dur = -1

        super().__init__(
            gravity,
            physics_timestep,
            render_timestep,
            solver_iterations,
            mode,
            image_width,
            image_height,
            vertical_fov,
            device_idx,
            rendering_settings,
            use_pb_gui,
        )

        # Get expected number of vsync frames per iGibson frame Note: currently assumes a 90Hz VR system
        self.vsync_frame_num = int(round(self.render_timestep / self.vsync_frame_dur))

        # Total amount of time we want non-blocking actions to take each frame
        # Leave a small amount of time before the last vsync, just in case we overrun
        self.non_block_frame_time = (self.vsync_frame_num - 1) * self.vsync_frame_dur + (
            5e-3 if self.vr_settings.curr_device == "OCULUS" else 10e-3
        )

        self._most_recent_trigger_fraction = {}

    def initialize_renderer(self):
        self.visual_objects = {}
        self.renderer = MeshRendererVR(
            rendering_settings=self.rendering_settings, vr_settings=self.vr_settings, simulator=self
        )
        self.viewer = ViewerVR(
            self.vr_settings.use_companion_window,
            frame_save_path=self.vr_settings.frame_save_path,
            renderer=self.renderer,
        )

    def add_vr_overlay_text(
        self,
        text_data="PLACEHOLDER: PLEASE REPLACE!",
        font_name="OpenSans",
        font_style="Regular",
        font_size=48,
        color=[0, 0, 0],
        pos=[20, 80],
        size=[70, 80],
        scale=1.0,
        background_color=[1, 1, 1, 0.8],
    ):
        """
        Creates Text for use in a VR overlay. Returns the text object to the caller,
        so various settings can be changed - eg. text content, position, scale, etc.
        :param text_data: starting text to display (can be changed at a later time by set_text)
        :param font_name: name of font to render - same as font folder in iGibson assets
        :param font_style: style of font - one of [regular, italic, bold]
        :param font_size: size of font to render
        :param color: [r, g, b] color
        :param pos: [x, y] position of top-left corner of text box, in percentage across screen
        :param size: [w, h] size of text box in percentage across screen-space axes
        :param scale: scale factor for resizing text
        :param background_color: color of the background in form [r, g, b, a] - default is semi-transparent white so text is easy to read in VR
        """
        if not self.vr_overlay_initialized:
            # This function automatically creates a VR text overlay the first time text is added
            self.renderer.gen_vr_hud()
            self.vr_overlay_initialized = True

        # Note: For pos/size - (0,0) is bottom-left and (100, 100) is top-right
        # Calculate pixel positions for text
        pixel_pos = [int(pos[0] / 100.0 * self.renderer.width), int(pos[1] / 100.0 * self.renderer.height)]
        pixel_size = [int(size[0] / 100.0 * self.renderer.width), int(size[1] / 100.0 * self.renderer.height)]
        return self.renderer.add_text(
            text_data=text_data,
            font_name=font_name,
            font_style=font_style,
            font_size=font_size,
            color=color,
            pixel_pos=pixel_pos,
            pixel_size=pixel_size,
            scale=scale,
            background_color=background_color,
            render_to_tex=True,
        )

    def add_overlay_image(self, image_fpath, width=1, pos=[0, 0, -1]):
        """
        Add an image with a given file path to the VR overlay. This image will be displayed
        in addition to any text that the users wishes to display. This function returns a handle
        to the VrStaticImageOverlay, so the user can display/hide it at will.
        """
        return self.renderer.gen_static_overlay(image_fpath, width=width, pos=pos)

    def set_hud_show_state(self, show_state):
        """
        Shows/hides the main VR HUD.
        :param show_state: whether to show HUD or not
        """
        self.renderer.vr_hud.set_overlay_show_state(show_state)

    def get_hud_show_state(self):
        """
        Returns the show state of the main VR HUD.
        """
        return self.renderer.vr_hud.get_overlay_show_state()

    def step_vr_system(self):
        # Update VR compositor and VR data
        vr_system_start = time.perf_counter()
        # First sync VR compositor - this is where Oculus blocks (as opposed to Vive, which blocks in update_vr_data)
        self.sync_vr_compositor()
        # Note: this should only be called once per frame - use get_vr_events to read the event data list in
        # subsequent read operations
        self.poll_vr_events()
        # This is necessary to fix the eye tracking value for the current frame, since it is multi-threaded
        self.fix_eye_tracking_value()
        # Move user to their starting location
        self.perform_vr_start_pos_move()
        # Update VR data and wait until 3ms before the next vsync
        self.renderer.update_vr_data()
        # Update VR system data - eg. offsets, haptics, etc.
        self.vr_system_update()
        vr_system_dur = time.perf_counter() - vr_system_start
        return vr_system_dur

    def step(self, print_stats=False):
        """
        Step the simulation when using VR. Order of function calls:
        1) Simulate physics
        2) Render frame
        3) Submit rendered frame to VR compositor
        4) Update VR data for use in the next frame
        """
        assert (
            self.scene is not None
        ), "A scene must be imported before running the simulator. Use EmptyScene for an empty scene."

        # Calculate time outside of step
        outside_step_dur = 0
        if self.frame_end_time is not None:
            outside_step_dur = time.perf_counter() - self.frame_end_time
        # Simulate Physics in PyBullet
        physics_start_time = time.perf_counter()
        for _ in range(self.physics_timestep_num):
            p.stepSimulation()
        physics_dur = time.perf_counter() - physics_start_time

        non_physics_start_time = time.perf_counter()
        self._non_physics_step()
        non_physics_dur = time.perf_counter() - non_physics_start_time

        # Sync PyBullet bodies to renderer and then render to Viewer
        render_start_time = time.perf_counter()
        self.sync()
        render_dur = time.perf_counter() - render_start_time

        # Sleep until last possible Vsync
        pre_sleep_dur = outside_step_dur + physics_dur + non_physics_dur + render_dur
        sleep_start_time = time.perf_counter()
        if pre_sleep_dur < self.non_block_frame_time:
            sleep(self.non_block_frame_time - pre_sleep_dur)
        sleep_dur = time.perf_counter() - sleep_start_time

        vr_system_dur = self.step_vr_system()

        # Calculate final frame duration
        # Make sure it is non-zero for FPS calculation (set to max of 1000 if so)
        frame_dur = max(1e-3, pre_sleep_dur + sleep_dur + vr_system_dur)

        # Set variables for data saving and replay
        self.last_physics_timestep = physics_dur
        self.last_render_timestep = render_dur
        self.last_frame_dur = frame_dur

        if print_stats:
            print("Frame number {} statistics (ms)".format(self.frame_count))
            print("Total out-of-step duration: {}".format(outside_step_dur * 1000))
            print("Total physics duration: {}".format(physics_dur * 1000))
            print("Total non-physics duration: {}".format(non_physics_dur * 1000))
            print("Total render duration: {}".format(render_dur * 1000))
            print("Total sleep duration: {}".format(sleep_dur * 1000))
            print("Total VR system duration: {}".format(vr_system_dur * 1000))
            print("Total frame duration: {} and fps: {}".format(frame_dur * 1000, 1 / frame_dur))
            print(
                "Realtime factor: {}".format(round((self.physics_timestep_num * self.physics_timestep) / frame_dur, 3))
            )
            print("-------------------------")

        self.frame_count += 1
        self.frame_end_time = time.perf_counter()

    def vr_system_update(self):
        """
        Updates the VR system for a single frame. This includes moving the vr offset,
        adjusting the user's height based on button input, and triggering haptics.
        """
        # Update VR offset using appropriate controller
        if self.vr_settings.touchpad_movement:
            vr_offset_device = "{}_controller".format(self.vr_settings.movement_controller)
            is_valid, _, _ = self.get_data_for_vr_device(vr_offset_device)
            if is_valid:
                _, touch_x, touch_y, _ = self.get_button_data_for_controller(vr_offset_device)
                new_offset = calc_offset(
                    self, touch_x, touch_y, self.vr_settings.movement_speed, self.vr_settings.relative_movement_device
                )
                self.set_vr_offset(new_offset)

        # Adjust user height based on y-axis (vertical direction) touchpad input
        vr_height_device = "left_controller" if self.vr_settings.movement_controller == "right" else "right_controller"
        is_height_valid, _, _ = self.get_data_for_vr_device(vr_height_device)
        if is_height_valid:
            curr_offset = self.get_vr_offset()
            hmd_height = self.get_hmd_world_pos()[2]
            _, _, height_y, _ = self.get_button_data_for_controller(vr_height_device)
            if height_y < -0.7:
                vr_z_offset = -0.01
                if hmd_height + curr_offset[2] + vr_z_offset >= self.vr_settings.height_bounds[0]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])
            elif height_y > 0.7:
                vr_z_offset = 0.01
                if hmd_height + curr_offset[2] + vr_z_offset <= self.vr_settings.height_bounds[1]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])

        # Update haptics for body and hands
        if self.main_vr_robot:
            vr_body_id = self.main_vr_robot.base_link.body_id
            vr_hands = [
                ("left_controller", "left_hand"),
                ("right_controller", "right_hand"),
            ]

            # Check for body haptics
            wall_ids = [bid for x in self.scene.objects_by_category["walls"] for bid in x.get_body_ids()]
            for c_info in p.getContactPoints(vr_body_id):
                if wall_ids and (c_info[1] in wall_ids or c_info[2] in wall_ids):
                    for controller in ["left_controller", "right_controller"]:
                        is_valid, _, _ = self.get_data_for_vr_device(controller)
                        if is_valid:
                            # Use 90% strength for body to warn user of collision with wall
                            self.trigger_haptic_pulse(controller, 0.9)

            # Check for hand haptics
            for hand_device, hand_name in vr_hands:
                is_valid, _, _ = self.get_data_for_vr_device(hand_device)
                if is_valid:
                    if (
                        len(p.getContactPoints(self.main_vr_robot.eef_links[hand_name].body_id))  # TODO: Generalize
                        or self.main_vr_robot.is_grasping(hand_name) == IsGraspingState.TRUE
                    ):
                        # Only use 30% strength for normal collisions, to help add realism to the experience
                        self.trigger_haptic_pulse(hand_device, 0.3)

    def register_main_vr_robot(self, vr_robot):
        """
        Register the robot representing the VR user.
        """
        self.main_vr_robot = vr_robot

    def gen_vr_data(self):
        """
        Generates a VrData object containing all of the data required to describe the VR system in the current frame.
        This data is used to power the BehaviorRobot each frame.
        """

        v = dict()
        for device in VR_DEVICES:
            is_valid, trans, rot = self.get_data_for_vr_device(device)
            device_data = [is_valid, trans.tolist(), rot.tolist()]
            device_data.extend(self.get_device_coordinate_system(device))
            v[device] = device_data
            if device in VR_CONTROLLERS:
                v["{}_button".format(device)] = self.get_button_data_for_controller(device)

        # Store final rotations of hands, with model rotation applied
        for hand in ["right", "left"]:
            # Base rotation quaternion
            base_rot = HAND_BASE_ROTS[hand]
            # Raw rotation of controller
            controller_rot = v["{}_controller".format(hand)][2]
            # Use dummy translation to calculation final rotation
            final_rot = p.multiplyTransforms([0, 0, 0], controller_rot, [0, 0, 0], base_rot)[1]
            v["{}_controller".format(hand)].append(final_rot)

        is_valid, torso_trans, torso_rot = self.get_data_for_vr_tracker(self.vr_settings.torso_tracker_serial)
        v["torso_tracker"] = [is_valid, torso_trans, torso_rot]
        v["eye_data"] = self.get_eye_tracking_data()
        v["event_data"] = self.get_vr_events()
        reset_actions = []
        for controller in VR_CONTROLLERS:
            reset_actions.append(self.query_vr_event(controller, "reset_agent"))
        v["reset_actions"] = reset_actions
        v["vr_positions"] = [self.get_vr_pos().tolist(), list(self.get_vr_offset())]

        return VrData(v)

    def gen_vr_robot_action(self):
        """
        Generates an action for the BehaviorRobot to perform based on VrData collected this frame.

        Action space (all non-normalized values that will be clipped if they are too large)
        * See BehaviorRobot.py for details on the clipping thresholds for
        Body:
        - 6DOF pose delta - relative to body frame from previous frame
        Eye:
        - 6DOF pose delta - relative to body frame (where the body will be after applying this frame's action)
        Left hand, right hand (in that order):
        - 6DOF pose delta - relative to body frame (same as above)
        - Trigger fraction delta
        - Action reset value

        Total size: 28
        """
        # Actions are stored as 1D numpy array
        action = np.zeros((28,))

        # Get VrData for the current frame
        v = self.gen_vr_data()

        # If a double button press is recognized for ATTACHMENT_BUTTON_TIME_THRESHOLD seconds, attach/detach the VR
        # system as needed. Forward a zero action to the robot if deactivated or if a switch is recognized.
        attach_or_detach = self.get_action_button_state(
            "left_controller", "reset_agent", v
        ) and self.get_action_button_state("right_controller", "reset_agent", v)
        if attach_or_detach:
            # If the button just recently started being pressed, record the time.
            if self._vr_attachment_button_press_timestamp is None:
                logging.info("Double button press detected.")
                self._vr_attachment_button_press_timestamp = time.time()

            # If the button has been pressed for ATTACHMENT_BUTTON_TIME_THRESHOLD seconds, attach/detach.
            if time.time() - self._vr_attachment_button_press_timestamp > ATTACHMENT_BUTTON_TIME_THRESHOLD:
                # Replace timestamp with infinity so that the condition won't retrigger until button is released.
                self._vr_attachment_button_press_timestamp = float("inf")

                # Flip the attachment state.
                self.vr_attached = not self.vr_attached
                logging.info("VR kit {} BehaviorRobot.".format("attached to" if self.vr_attached else "detached from"))

                # We don't want to fill in an action in this case.
                return action
        else:
            # If the button is released, stop keeping track.
            self._vr_attachment_button_press_timestamp = None

        # If the VR system is not attached to the robot, return a zero action.
        if not self.vr_attached:
            return action

        # Update body action space
        hmd_is_valid, hmd_pos, hmd_orn, hmd_r = v.query("hmd")[:4]
        torso_is_valid, torso_pos, torso_orn = v.query("torso_tracker")
        vr_body = self.main_vr_robot.base_link
        prev_body_pos, prev_body_orn = vr_body.get_position_orientation()
        inv_prev_body_pos, inv_prev_body_orn = p.invertTransform(prev_body_pos, prev_body_orn)

        if self.vr_settings.using_tracked_body:
            if torso_is_valid:
                des_body_pos, des_body_orn = torso_pos, torso_orn
            else:
                des_body_pos, des_body_orn = prev_body_pos, prev_body_orn
        else:
            if hmd_is_valid:
                des_body_pos, des_body_orn = hmd_pos, p.getQuaternionFromEuler([0, 0, calc_z_rot_from_right(hmd_r)])
            else:
                des_body_pos, des_body_orn = prev_body_pos, prev_body_orn

        body_delta_pos, body_delta_orn = p.multiplyTransforms(
            inv_prev_body_pos, inv_prev_body_orn, des_body_pos, des_body_orn
        )
        body_delta_rot = p.getEulerFromQuaternion(body_delta_orn)
        action[self.main_vr_robot.controller_action_idx["base"]] = np.concatenate([body_delta_pos, body_delta_rot])

        # Get new body position so we can calculate correct relative transforms for other VR objects
        clipped_body_delta_pos = np.clip(body_delta_pos, -BODY_LINEAR_VELOCITY, BODY_LINEAR_VELOCITY).tolist()
        clipped_body_delta_orn = p.getQuaternionFromEuler(
            np.clip(body_delta_rot, -BODY_ANGULAR_VELOCITY, BODY_ANGULAR_VELOCITY).tolist()
        )
        new_body_pos, new_body_orn = p.multiplyTransforms(
            prev_body_pos, prev_body_orn, clipped_body_delta_pos, clipped_body_delta_orn
        )
        # Also calculate its inverse for further local transform calculations
        inv_new_body_pos, inv_new_body_orn = p.invertTransform(new_body_pos, new_body_orn)

        # Update action space for other VR objects
        body_relative_parts = [
            ("right_hand", self.main_vr_robot.eef_links["right_hand"]),
            ("left_hand", self.main_vr_robot.eef_links["left_hand"]),
            ("eye", self.main_vr_robot.links["eyes"]),
        ]
        for part_name, vr_part in body_relative_parts:
            # Process local transform adjustments
            prev_world_pos, prev_world_orn = vr_part.get_position_orientation()
            prev_local_pos, prev_local_orn = vr_part.get_local_position_orientation()
            _, inv_prev_local_orn = p.invertTransform(prev_local_pos, prev_local_orn)
            if part_name == "eye":
                valid, world_pos, world_orn = hmd_is_valid, hmd_pos, hmd_orn
            else:
                controller_name = "{}_controller".format(part_name.replace("_hand", ""))
                valid, world_pos, _ = v.query(controller_name)[:3]
                # Need rotation of the model so it will appear aligned with the physical controller in VR
                world_orn = v.query(controller_name)[6]

            # Keep in same world position as last frame if controller/tracker data is not valid
            if not valid:
                world_pos, world_orn = prev_world_pos, prev_world_orn

            # Get desired local position and orientation transforms
            des_local_pos, des_local_orn = p.multiplyTransforms(
                inv_new_body_pos, inv_new_body_orn, world_pos, world_orn
            )

            # Get the delta local orientation in the reference frame of the body
            _, delta_local_orn = p.multiplyTransforms(
                [0, 0, 0],
                des_local_orn,
                [0, 0, 0],
                inv_prev_local_orn,
            )
            delta_local_orn = p.getEulerFromQuaternion(delta_local_orn)

            # Get the delta local position in the reference frame of the body
            delta_local_pos = np.array(des_local_pos) - np.array(prev_local_pos)

            controller_name = "camera" if part_name == "eye" else "arm_" + part_name
            action[self.main_vr_robot.controller_action_idx[controller_name]] = np.concatenate(
                [delta_local_pos, delta_local_orn]
            )

            # Process trigger fraction and reset for controllers
            if part_name in ["right_hand", "left_hand"]:
                if valid:
                    button_name = "{}_controller_button".format(part_name.replace("_hand", ""))
                    trig_frac = v.query(button_name)[0]
                    self._most_recent_trigger_fraction[part_name] = trig_frac
                else:
                    # Use the last trigger fraction if no valid input was received from controller.
                    trig_frac = (
                        self._most_recent_trigger_fraction[part_name]
                        if part_name in self._most_recent_trigger_fraction
                        else 0.0
                    )

                grip_controller_name = "gripper_" + part_name
                scaled_trig_frac = -2 * trig_frac + 1  # Map from (0, 1) to (-1, 1) range, inverted.
                action[self.main_vr_robot.controller_action_idx[grip_controller_name]] = scaled_trig_frac

                # If we reset, action is 1, otherwise 0
                reset_action = v.query("reset_actions")[0] if part_name == "left" else v.query("reset_actions")[1]
                reset_action_val = 1.0 if reset_action else 0.0
                action[self.main_vr_robot.controller_action_idx["reset_%s" % part_name]] = reset_action_val

        return action

    def sync_vr_compositor(self):
        """
        Sync VR compositor.
        """
        self.renderer.vr_compositor_update()

    def perform_vr_start_pos_move(self):
        """
        Sets the VR position on the first step iteration where the hmd tracking is valid. Not to be confused
        with self.set_vr_start_pos, which simply records the desired start position before the simulator starts running.
        """
        # Update VR start position if it is not None and the hmd is valid
        # This will keep checking until we can successfully set the start position
        if self.vr_start_pos:
            hmd_is_valid, _, _, _ = self.renderer.vrsys.getDataForVRDevice("hmd")
            if hmd_is_valid:
                offset_to_start = np.array(self.vr_start_pos) - self.get_hmd_world_pos()
                if self.vr_height_offset is not None:
                    offset_to_start[2] = self.vr_height_offset
                self.set_vr_offset(offset_to_start)
                self.vr_start_pos = None

    def fix_eye_tracking_value(self):
        """
        Calculates and fixes eye tracking data to its value during step(). This is necessary, since multiple
        calls to get eye tracking data return different results, due to the SRAnipal multithreaded loop that
        runs in parallel to the iGibson main thread
        """
        self.eye_tracking_data = self.renderer.vrsys.getEyeTrackingData()

    def poll_vr_events(self):
        """
        Returns VR event data as list of lists.
        List is empty if all events are invalid. Components of a single event:
        controller: 0 (left_controller), 1 (right_controller)
        button_idx: any valid idx in EVRButtonId enum in openvr.h header file
        press: 0 (unpress), 1 (press)
        """

        self.vr_event_data = self.renderer.vrsys.pollVREvents()
        # Enforce store_first_button_press_per_frame option, if user has enabled it
        if self.vr_settings.store_only_first_event_per_button:
            temp_event_data = []
            # Make sure we only store the first (button, press) combo of each type
            event_set = set()
            for ev_data in self.vr_event_data:
                controller, button_idx, _ = ev_data
                key = (controller, button_idx)
                if key not in event_set:
                    temp_event_data.append(ev_data)
                    event_set.add(key)
            self.vr_event_data = temp_event_data[:]

        return self.vr_event_data

    def get_vr_events(self):
        """
        Returns the VR events processed by the simulator
        """
        return self.vr_event_data

    def query_vr_event(self, controller, action):
        """
        Queries system for a VR event, and returns true if that event happened this frame
        :param controller: device to query for - can be left_controller or right_controller
        :param action: an action name listed in "action_button_map" dictionary for the current device in the vr_config.yml
        """
        # Return false if any of input parameters are invalid
        if (
            controller not in ["left_controller", "right_controller"]
            or action not in self.vr_settings.action_button_map.keys()
        ):
            return False

        # Search through event list to try to find desired event
        controller_id = 0 if controller == "left_controller" else 1
        button_idx, press_id = self.vr_settings.action_button_map[action]
        for ev_data in self.vr_event_data:
            if controller_id == ev_data[0] and button_idx == ev_data[1] and press_id == ev_data[2]:
                return True

        # Return false if event was not found this frame
        return False

    def get_data_for_vr_device(self, device_name):
        """
        Call this after step - returns all VR device data for a specific device
        Returns is_valid (indicating validity of data), translation and rotation in Gibson world space
        :param device_name: can be hmd, left_controller or right_controller
        """

        # Use fourth variable in list to get actual hmd position in space
        is_valid, translation, rotation, _ = self.renderer.vrsys.getDataForVRDevice(device_name)
        if not is_valid:
            translation = np.array([0, 0, 0])
            rotation = np.array([0, 0, 0, 1])
        return [is_valid, translation, rotation]

    def get_data_for_vr_tracker(self, tracker_serial_number):
        """
        Returns the data for a tracker with a specific serial number. This number can be found
        by looking in the SteamVR device information.
        :param tracker_serial_number: the serial number of the tracker
        """

        if not tracker_serial_number:
            return [False, [0, 0, 0], [0, 0, 0, 0]]

        tracker_data = self.renderer.vrsys.getDataForVRTracker(tracker_serial_number)
        # Set is_valid to false, and assume the user will check for invalid data
        if not tracker_data:
            return [False, np.array([0, 0, 0]), np.array([0, 0, 0, 1])]

        is_valid, translation, rotation = tracker_data
        return [is_valid, translation, rotation]

    def get_hmd_world_pos(self):
        """
        Get world position of HMD without offset
        """

        _, _, _, hmd_world_pos = self.renderer.vrsys.getDataForVRDevice("hmd")
        return hmd_world_pos

    def get_button_data_for_controller(self, controller_name):
        """
        Call this after getDataForVRDevice - returns analog data for a specific controller
        Returns trigger_fraction, touchpad finger position x, touchpad finger position y
        Data is only valid if isValid is true from previous call to getDataForVRDevice
        Trigger data: 1 (closed) <------> 0 (open)
        Analog data: X: -1 (left) <-----> 1 (right) and Y: -1 (bottom) <------> 1 (top)
        :param controller_name: one of left_controller or right_controller
        """

        # Test for validity when acquiring button data
        if self.get_data_for_vr_device(controller_name)[0]:
            trigger_fraction, touch_x, touch_y, buttons_pressed = self.renderer.vrsys.getButtonDataForController(
                controller_name
            )
        else:
            trigger_fraction, touch_x, touch_y, buttons_pressed = 0.0, 0.0, 0.0, 0
        return [trigger_fraction, touch_x, touch_y, buttons_pressed]

    def get_action_button_state(self, controller, action, vr_data):
        """This function can be used to extract the _state_ of a button from the vr_data's buttons_pressed vector.

        If only key press/release events are required, use the event polling mechanism. This function is meant for
        providing access to the continuous pressed/released state of the button.
        """
        # Find the controller and find the button mapping for this action in the config.
        if (
            controller not in ["left_controller", "right_controller"]
            or action not in self.vr_settings.action_button_map.keys()
        ):
            return False

        # Find the button index for this action from the config.
        button_idx, _ = self.vr_settings.action_button_map[action]

        # Get the bitvector corresponding to the buttons currently pressed on the controller.
        buttons_pressed = int(vr_data.query("%s_button" % controller)[3])

        # Extract and return the value of the bit corresponding to the button.
        return bool(buttons_pressed & (1 << button_idx))

    def get_scroll_input(self):
        """
        Gets scroll input. This uses the non-movement-controller, and determines whether
        the user wants to scroll by testing if they have pressed the touchpad, while keeping
        their finger on the left/right of the pad. Return True for up and False for down (-1 for no scroll)
        """
        mov_controller = self.vr_settings.movement_controller
        other_controller = "right" if mov_controller == "left" else "left"
        other_controller = "{}_controller".format(other_controller)
        # Data indicating whether user has pressed top or bottom of the touchpad
        _, touch_x, _ = self.renderer.vrsys.getButtonDataForController(other_controller)
        # Detect no touch in extreme regions of x axis
        if touch_x > 0.7 and touch_x <= 1.0:
            return 1
        elif touch_x < -0.7 and touch_x >= -1.0:
            return 0
        else:
            return -1

    def get_eye_tracking_data(self):
        """
        Returns eye tracking data as list of lists. Order: is_valid, gaze origin, gaze direction, gaze point,
        left pupil diameter, right pupil diameter (both in millimeters)
        Call after getDataForVRDevice, to guarantee that latest HMD transform has been acquired
        """
        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = self.eye_tracking_data
        # Set other values to 0 to avoid very small/large floating point numbers
        if not is_valid:
            return [False, [0, 0, 0], [0, 0, 0], 0, 0]
        else:
            return [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter]

    def set_vr_start_pos(self, start_pos=None, vr_height_offset=None):
        """
        Sets the starting position of the VR system in iGibson space
        :param start_pos: position to start VR system at
        :param vr_height_offset: starting height offset. If None, uses absolute height from start_pos
        """

        # The VR headset will actually be set to this position during the first frame.
        # This is because we need to know where the headset is in space when it is first picked
        # up to set the initial offset correctly.
        self.vr_start_pos = start_pos
        # This value can be set to specify a height offset instead of an absolute height.
        # We might want to adjust the height of the camera based on the height of the person using VR,
        # but still offset this height. When this option is not None it offsets the height by the amount
        # specified instead of overwriting the VR system height output.
        self.vr_height_offset = vr_height_offset

    def set_vr_pos(self, pos=None, keep_height=False):
        """
        Sets the world position of the VR system in iGibson space
        :param pos: position to set VR system to
        :param keep_height: whether the current VR height should be kept
        """

        offset_to_pos = np.array(pos) - self.get_hmd_world_pos()
        if keep_height:
            curr_offset_z = self.get_vr_offset()[2]
            self.set_vr_offset([offset_to_pos[0], offset_to_pos[1], curr_offset_z])
        else:
            self.set_vr_offset(offset_to_pos)

    def get_vr_pos(self):
        """
        Gets the world position of the VR system in iGibson space.
        """
        return self.get_hmd_world_pos() + np.array(self.get_vr_offset())

    def set_vr_offset(self, pos=None):
        """
        Sets the translational offset of the VR system (HMD, left controller, right controller) from world space coordinates.
        Can be used for many things, including adjusting height and teleportation-based movement
        :param pos: must be a list of three floats, corresponding to x, y, z in Gibson coordinate space
        """

        self.renderer.vrsys.setVROffset(-pos[1], pos[2], -pos[0])

    def get_vr_offset(self):
        """
        Gets the current VR offset vector in list form: x, y, z (in iGibson coordinates)
        """

        x, y, z = self.renderer.vrsys.getVROffset()
        return [x, y, z]

    def get_device_coordinate_system(self, device):
        """
        Gets the direction vectors representing the device's coordinate system in list form: x, y, z (in Gibson coordinates)
        List contains "right", "up" and "forward" vectors in that order
        :param device: can be one of "hmd", "left_controller" or "right_controller"
        """

        vec_list = []

        coordinate_sys = self.renderer.vrsys.getDeviceCoordinateSystem(device)
        for dir_vec in coordinate_sys:
            vec_list.append(dir_vec)

        return vec_list

    def trigger_haptic_pulse(self, device, strength):
        """
        Triggers a haptic pulse of the specified strength (0 is weakest, 1 is strongest)
        :param device: device to trigger haptic for - can be any one of [left_controller, right_controller]
        :param strength: strength of haptic pulse (0 is weakest, 1 is strongest)
        """
        assert device in ["left_controller", "right_controller"]

        self.renderer.vrsys.triggerHapticPulseForDevice(device, int(self.max_haptic_duration * strength))
