from collections import defaultdict
import itertools
import contextlib
import os
from pathlib import Path

import numpy as np
import json
import omni
import carb
import omni.physics
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import open_stage, create_new_stage
from omni.physx.bindings._physx import ContactEventType, SimulationEvent
import omni.kit.loop._loop as omni_loop
from pxr import Usd, PhysicsSchemaTools, UsdUtils
from omni.physx import get_physx_interface, get_physx_simulation_interface, get_physx_scene_query_interface

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.utils.constants import LightingMode
from omnigibson.utils.config_utils import NumpyEncoder
from omnigibson.utils.python_utils import clear as clear_pu, create_object_from_init_info, Serializable
from omnigibson.utils.sim_utils import meets_minimum_isaac_version
from omnigibson.utils.usd_utils import clear as clear_uu, BoundingBoxAPI, FlatcacheAPI, RigidContactAPI
from omnigibson.utils.ui_utils import CameraMover, disclaimer, create_module_logger, suppress_omni_log
from omnigibson.scenes import Scene
from omnigibson.objects.object_base import BaseObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.object_states.contact_subscribed_state_mixin import ContactSubscribedStateMixin
from omnigibson.object_states.joint_break_subscribed_state_mixin import JointBreakSubscribedStateMixin
from omnigibson.object_states.factory import get_states_by_dependency_order
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.transition_rules import TransitionRuleAPI

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_VIEWER_CAMERA_POS = (-0.201028, -2.72566 ,  1.0654)
m.DEFAULT_VIEWER_CAMERA_QUAT = (0.68196617, -0.00155408, -0.00166678,  0.73138017)


class Simulator(SimulationContext, Serializable):
    """
    Simulator class for directly interfacing with the physx physics engine.

    NOTE: This is a monolithic class.
        All created Simulator() instances will reference the same underlying Simulator object

    Args:
        gravity (float): gravity on z direction.
        physics_dt (float): dt between physics steps. Defaults to 1.0 / 60.0.
        rendering_dt (float): dt between rendering steps. Note: rendering means rendering a frame of the current
            application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will
            be refreshed with this dt as well if running non-headless. Defaults to 1.0 / 60.0.
        stage_units_in_meters (float): The metric units of assets. This will affect gravity value..etc.
            Defaults to 0.01.
        viewer_width (int): width of the camera image, in pixels
        viewer_height (int): height of the camera image, in pixels
        device (None or str): specifies the device to be used if running on the gpu with torch backend
        """
    _world_initialized = False

    def __init__(
            self,
            gravity=9.81,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
            stage_units_in_meters=1.0,
            viewer_width=gm.DEFAULT_VIEWER_WIDTH,
            viewer_height=gm.DEFAULT_VIEWER_HEIGHT,
            device=None,
    ):
        # Store vars needed for initialization
        self.gravity = gravity
        self._viewer_camera = None
        self._camera_mover = None

        # Run super init
        super().__init__(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=stage_units_in_meters,
            device=device,
        )

        if self._world_initialized:
            return
        Simulator._world_initialized = True

        # Store other references to variables that will be initialized later
        self._scene = None
        self._physx_interface = None
        self._physx_simulation_interface = None
        self._physx_scene_query_interface = None
        self._contact_callback = None
        self._simulation_event_callback = None
        # List of objects that need to be initialized during whenever the next sim step occurs
        self._objects_to_initialize = []
        self._objects_require_contact_callback = False
        self._objects_require_joint_break_callback = False

        # Maps callback name to callback
        self._callbacks_on_play = dict()
        self._callbacks_on_stop = dict()
        self._callbacks_on_import_obj = dict()
        self._callbacks_on_remove_obj = dict()

        # Mapping from link IDs assigned from omni to the object that they reference
        self._link_id_to_objects = dict()

        # Set of categories that can be grasped by assisted grasping
        self.object_state_types = get_states_by_dependency_order()
        self.object_state_types_requiring_update = \
            [state for state in self.object_state_types if issubclass(state, UpdateStateMixin)]
        self.object_state_types_on_contact = \
            {state for state in self.object_state_types if issubclass(state, ContactSubscribedStateMixin)}
        self.object_state_types_on_joint_break = \
            {state for state in self.object_state_types if issubclass(state, JointBreakSubscribedStateMixin)}

        # Auto-load the dummy stage
        self.clear()

        # Set the viewer dimensions
        # TODO: Make this toggleable so we don't always have a viewer if we don't want to
        self.viewer_width = viewer_width
        self.viewer_height = viewer_height

        # Toggle simulator state once so that downstream omni features can be used without bugs
        # e.g.: particle sampling, which for some reason requires sim.play() to be called at least once
        self.play()
        self.stop()

        # Update the physics settings
        # This needs to be done now, after an initial step + stop for some reason if we want to use GPU
        # dynamics, otherwise we get very strange behavior, e.g., PhysX complains about invalid transforms
        # and crashes
        self._set_physics_engine_settings()

    def __new__(
        cls,
        gravity=9.81,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
        viewer_width=gm.DEFAULT_VIEWER_WIDTH,
        viewer_height=gm.DEFAULT_VIEWER_HEIGHT,
        device_idx=0,
    ):
        # Overwrite since we have different kwargs
        if Simulator._instance is None:
            Simulator._instance = object.__new__(cls)
        else:
            carb.log_info("Simulator is defined already, returning the previously defined one")
        return Simulator._instance

    def _set_viewer_camera(self, prim_path="/World/viewer_camera", viewport_name="Viewport"):
        """
        Creates a camera prim dedicated for this viewer at @prim_path if it doesn't exist,
        and sets this camera as the active camera for the viewer

        Args:
            prim_path (str): Path to check for / create the viewer camera
            viewport_name (str): Name of the viewport this camera should attach to. Default is "Viewport", which is
                the default viewport's name in Isaac Sim
        """
        self._viewer_camera = VisionSensor(
            prim_path=prim_path,
            name=prim_path.split("/")[-1],                  # Assume name is the lowest-level name in the prim_path
            modalities="rgb",
            image_height=self.viewer_height,
            image_width=self.viewer_width,
            viewport_name=viewport_name,
        )
        if not self._viewer_camera.loaded:
            self._viewer_camera.load()

        # We update its clipping range and focal length so we get a good FOV and so that it doesn't clip
        # nearby objects (default min is 1 m)
        self._viewer_camera.clipping_range = [0.001, 10000000.0]
        self._viewer_camera.focal_length = 17.0

        # Initialize the sensor
        self._viewer_camera.initialize()

        # Also need to potentially update our camera mover if it already exists
        if self._camera_mover is not None:
            self._camera_mover.set_cam(cam=self._viewer_camera)

    def _set_physics_engine_settings(self):
        """
        Set the physics engine with specified settings
        """
        assert self.is_stopped(), f"Cannot set simulator physics settings while simulation is playing!"
        self._physics_context.set_gravity(value=-self.gravity)
        # Also make sure we invert the collision group filter settings so that different collision groups cannot
        # collide with each other, and modify settings for speed optimization
        self._physics_context.set_invert_collision_group_filter(True)
        self._physics_context.enable_ccd(gm.ENABLE_CCD)

        if meets_minimum_isaac_version("2023.0.0"):
            self._physics_context.enable_fabric(gm.ENABLE_FLATCACHE)
        else:
            self._physics_context.enable_flatcache(gm.ENABLE_FLATCACHE)

        # Enable GPU dynamics based on whether we need omni particles feature
        if gm.USE_GPU_DYNAMICS:
            self._physics_context.enable_gpu_dynamics(True)
            self._physics_context.set_broadphase_type("GPU")
        else:
            self._physics_context.enable_gpu_dynamics(False)
            self._physics_context.set_broadphase_type("MBP")

        # Set GPU Pairs capacity and other GPU settings
        self._physics_context.set_gpu_found_lost_pairs_capacity(gm.GPU_PAIRS_CAPACITY)
        self._physics_context.set_gpu_found_lost_aggregate_pairs_capacity(gm.GPU_AGGR_PAIRS_CAPACITY)
        self._physics_context.set_gpu_total_aggregate_pairs_capacity(gm.GPU_AGGR_PAIRS_CAPACITY)
        self._physics_context.set_gpu_max_particle_contacts(gm.GPU_MAX_PARTICLE_CONTACTS)

    def _set_renderer_settings(self):
        # TODO: For now we are setting these to some reasonable high-performance values but these can be made configurable.
        carb.settings.get_settings().set_bool("/rtx/reflections/enabled", False)  # Can be True with a 10fps penalty
        carb.settings.get_settings().set_bool("/rtx/indirectDiffuse/enabled", True)  # Can be False with a 5fps gain
        carb.settings.get_settings().set_bool("/rtx/directLighting/sampledLighting/enabled", True)
        carb.settings.get_settings().set_int("/rtx/raytracing/showLights", 1)
        carb.settings.get_settings().set_float("/rtx/sceneDb/ambientLightIntensity", 0.1)
        # TODO: Think of better setting defaults. Below works well for indoor-only scenes, but if skybox is the only light source then this looks very bad
        # carb.settings.get_settings().set_int("/rtx/domeLight/upperLowerStrategy", 3)  # "Limited image-based"

    @property
    def viewer_visibility(self):
        """
        Returns:
            bool: Whether the viewer is visible or not
        """
        return self._viewer_camera.viewer_visibility

    @viewer_visibility.setter
    def viewer_visibility(self, visible):
        """
        Sets whether the viewer should be visible or not in the Omni UI

        Args:
            visible (bool): Whether the viewer should be visible or not
        """
        self._viewer_camera.viewer_visibility = visible

    @property
    def viewer_height(self):
        """
        Returns:
            int: viewer height of this sensor, in pixels
        """
        # If the viewer camera hasn't been created yet, utilize the default width
        return gm.DEFAULT_VIEWER_HEIGHT if self._viewer_camera is None else self._viewer_camera.image_height

    @viewer_height.setter
    def viewer_height(self, height):
        """
        Sets the viewer height @height for this sensor

        Args:
            height (int): viewer height, in pixels
        """
        self._viewer_camera.image_height = height

    @property
    def viewer_width(self):
        """
        Returns:
            int: viewer width of this sensor, in pixels
        """
        # If the viewer camera hasn't been created yet, utilize the default height
        return gm.DEFAULT_VIEWER_WIDTH if self._viewer_camera is None else self._viewer_camera.image_width

    @viewer_width.setter
    def viewer_width(self, width):
        """
        Sets the viewer width @width for this sensor

        Args:
            width (int): viewer width, in pixels
        """
        self._viewer_camera.image_width = width

    def set_lighting_mode(self, mode):
        """
        Sets the active lighting mode in the current simulator. Valid options are one of LightingMode

        Args:
            mode (LightingMode): Lighting mode to set
        """
        omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode=mode)

    def enable_viewer_camera_teleoperation(self):
        """
        Enables keyboard control of the active viewer camera for this simulation
        """
        self._camera_mover = CameraMover(cam=self._viewer_camera)
        self._camera_mover.print_info()
        return self._camera_mover

    def import_scene(self, scene):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        Args:
            scene (Scene): a scene object to load
        """
        assert self.is_stopped(), "Simulator must be stopped while importing a scene!"
        assert isinstance(scene, Scene), "import_scene can only be called with Scene"

        # Clear the existing scene if any
        self.clear()

        self._scene = scene
        self._scene.load()

        # Make sure simulator is not running, then start it so that we can initialize the scene
        assert self.is_stopped(), "Simulator must be stopped after importing a scene!"
        self.play()

        # Initialize the scene
        self._scene.initialize()

        # Need to one more step for particle systems to work
        self.step()
        self.stop()
        log.info("Imported scene.")

    def initialize_object_on_next_sim_step(self, obj):
        """
        Initializes the object upon the next simulation step

        Args:
            obj (BasePrim): Object to initialize as soon as a new sim step is called
        """
        self._objects_to_initialize.append(obj)

    def import_object(self, obj, register=True):
        """
        Import an object into the simulator.

        Args:
            obj (BaseObject): an object to load
            register (bool): whether to register this object internally in the scene registry
        """
        assert isinstance(obj, BaseObject), "import_object can only be called with BaseObject"

        # Make sure scene is loaded -- objects should not be loaded unless we have a reference to a scene
        assert self.scene is not None, "import_object needs to be called after import_scene"

        # Load the object in omniverse by adding it to the scene
        self.scene.add_object(obj, register=register, _is_call_from_simulator=True)

        # Run any callbacks
        for callback in self._callbacks_on_import_obj.values():
            callback(obj)

        # Cache the mapping from link IDs to object
        for link in obj.links.values():
            self._link_id_to_objects[PhysicsSchemaTools.sdfPathToInt(link.prim_path)] = obj

        # Lastly, additionally add this object automatically to be initialized as soon as another simulator step occurs
        self.initialize_object_on_next_sim_step(obj=obj)

    def remove_object(self, obj):
        """
        Remove a non-robot object from the simulator.

        Args:
            obj (BaseObject): a non-robot object to load
        """
        # Run any callbacks
        for callback in self._callbacks_on_remove_obj.values():
            callback(obj)

        # pop all link ids
        for link in obj.links.values():
            self._link_id_to_objects.pop(PhysicsSchemaTools.sdfPathToInt(link.prim_path))

        # If it was queued up to be initialized, remove it from the queue as well
        for i, initialize_obj in enumerate(self._objects_to_initialize):
            if obj.name == initialize_obj.name:
                self._objects_to_initialize.pop(i)
                break

        self._scene.remove_object(obj)
        self.app.update()

        # Update all handles that are now broken because objects have changed
        self._update_handles()

        # Refresh all current rules
        TransitionRuleAPI.prune_active_rules()

    def _reset_variables(self):
        """
        Reset internal variables when a new stage is loaded
        """

    def _update_handles(self):
        # First, refresh the physics sim view
        self._physics_sim_view = omni.physics.tensors.create_simulation_view(self.backend)
        self._physics_sim_view.set_subspace_roots("/")

        # Then update the handles for all objects
        if self.scene is not None and self.scene.initialized:
            for obj in self.scene.objects:
                # Only need to update if object is already initialized as well
                if obj.initialized:
                    obj.update_handles()

        # Finally update any unified views
        if self.is_playing():
            RigidContactAPI.initialize_view()

    def _non_physics_step(self):
        """
        Complete any non-physics steps such as state updates.
        """
        # If we don't have a valid scene, immediately return
        if self._scene is None:
            return

        # Update omni
        self._omni_update_step()

        # If we're playing we, also run additional logic
        if self.is_playing():
            # Check to see if any objects should be initialized (only done IF we're playing)
            n_objects_to_initialize = len(self._objects_to_initialize)
            if n_objects_to_initialize > 0 and self.is_playing():
                # We iterate through the objects to initialize
                # Note that we don't explicitly do for obj in self._objects_to_initialize because additional objects
                # may be added mid-iteration!!
                # For this same reason, after we finish the loop, we keep any objects that are yet to be initialized
                # First call zero-physics step update, so that handles are properly propagated
                og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
                for i in range(n_objects_to_initialize):
                    obj = self._objects_to_initialize[i]
                    obj.initialize()
                    if len(obj.states.keys() & self.object_state_types_on_contact) > 0:
                        self._objects_require_contact_callback = True
                    if len(obj.states.keys() & self.object_state_types_on_joint_break) > 0:
                        self._objects_require_joint_break_callback = True

                self._objects_to_initialize = self._objects_to_initialize[n_objects_to_initialize:]

                # Re-initialize the physics view because the number of objects has changed
                self._update_handles()

                # Also refresh the transition rules that are currently active
                TransitionRuleAPI.refresh_all_rules()

            # Update any system-related state
            for system in self.scene.systems:
                system.update()

            # Propagate states if the feature is enabled
            if gm.ENABLE_OBJECT_STATES:
                # Step the object states in global topological order (if the scene exists)
                for state_type in self.object_state_types_requiring_update:
                    for obj in self.scene.get_objects_with_state(state_type):
                        # Only update objects that have been initialized so far
                        if obj.initialized:
                            obj.states[state_type].update()

                for obj in self.scene.objects:
                    # Only update visuals for objects that have been initialized so far
                    if isinstance(obj, StatefulObject) and obj.initialized:
                        obj.update_visuals()

            # Possibly run transition rule step
            if gm.ENABLE_TRANSITION_RULES:
                TransitionRuleAPI.step()

    def _omni_update_step(self):
        """
        Step any omni-related things
        """
        # Clear the bounding box and contact caches so that they get updated during the next time they're called
        BoundingBoxAPI.clear()
        RigidContactAPI.clear()

    def play(self):
        if not self.is_playing():
            # Track whether we're starting the simulator fresh -- i.e.: whether we were stopped previously
            was_stopped = self.is_stopped()

            # Run super first
            # We suppress warnings from omni.usd because it complains about values set in the native USD
            # These warnings occur because the native USD file has some type mismatch in the `scale` property,
            # where the property expects a double but for whatever reason the USD interprets its values as floats
            # We supperss omni.physicsschema.plugin when kinematic_only objects are placed with scale ~1.0, to suppress
            # the following error:
            # [omni.physicsschema.plugin] ScaleOrientation is not supported for rigid bodies, prim path: [...] You may
            #   ignore this if the scale is close to uniform.
            # We also need to suppress the following error when flat cache is used:
            # [omni.physx.plugin] Transformation change on non-root links is not supported.
            channels = ["omni.usd", "omni.physicsschema.plugin"]
            if gm.ENABLE_FLATCACHE:
                channels.append("omni.physx.plugin")
            with suppress_omni_log(channels=channels):
                super().play()

            # If we're stopped, take a physics step and update the physics sim view. This must happen BEFORE the
            # handles are updated, since updating the physics view makes the per-object physics view invalid
            self.step_physics()

            # Take a render step -- this is needed so that certain (unknown, maybe omni internal state?) is populated
            # correctly.
            self.render()

            # Update all object handles
            self._update_handles()

            if was_stopped:
                # We need to update controller mode because kp and kd were set to the original (incorrect) values when
                # sim was stopped. We need to reset them to default_kp and default_kd defined in ControllableObject.
                # We also need to take an additional sim step to make sure simulator is functioning properly.
                # We need to do this because for some reason omniverse exhibits strange behavior if we do certain
                # operations immediately after playing; e.g.: syncing USD poses when flatcache is enabled
                if self.scene is not None and self.scene.initialized:
                    for robot in self.scene.robots:
                        if robot.initialized:
                            robot.update_controller_mode()

            # Additionally run non physics things
            self._non_physics_step()

        # Run all callbacks
        for callback in self._callbacks_on_play.values():
            callback()

    def pause(self):
        if not self.is_paused():
            super().pause()

    def stop(self):
        if not self.is_stopped():
            super().stop()

        # If we're using flatcache, we also need to reset its API
        if gm.ENABLE_FLATCACHE:
            FlatcacheAPI.reset()

        # Run all callbacks
        for callback in self._callbacks_on_stop.values():
            callback()

    @property
    def n_physics_timesteps_per_render(self):
        """
        Number of physics timesteps per rendering timestep. rendering_dt has to be a multiple of physics_dt.

        Returns:
            int: Discrete number of physics timesteps to take per step
        """
        n_physics_timesteps_per_render = self.get_rendering_dt() / self.get_physics_dt()
        assert n_physics_timesteps_per_render.is_integer(), "render_timestep must be a multiple of physics_timestep"
        return int(n_physics_timesteps_per_render)

    def step(self, render=True):
        """
        Step the simulation at self.render_timestep

        Args:
            render (bool): Whether rendering should occur or not
        """
        # If we have imported any objects within the last timestep, we render the app once, since otherwise calling
        # step() may not step physics
        if len(self._objects_to_initialize) > 0:
            self.render()

        if render:
            super().step(render=True)
        else:
            for i in range(self.n_physics_timesteps_per_render):
                super().step(render=False)

        # Additionally run non physics things
        self._non_physics_step()

        # TODO (eric): After stage changes (e.g. pose, texture change), it will take two super().step(render=True) for
        #  the result to propagate to the rendering. We could have called super().render() here but it will introduce
        #  a big performance regression.

    def step_physics(self):
        """
        Step the physics a single step.
        """
        self._physics_context._step(current_time=self.current_time)
        self._omni_update_step()

    def _on_contact(self, contact_headers, contact_data):
        """
        This callback will be invoked after every PHYSICS step if there is any contact.
        For each of the pair of objects in each contact, we invoke the on_contact function for each of its states
        that subclass ContactSubscribedStateMixin. These states update based on contact events.
        """
        if gm.ENABLE_OBJECT_STATES and self._objects_require_contact_callback:
            headers = defaultdict(list)
            for contact_header in contact_headers:
                actor0_obj = self._link_id_to_objects.get(contact_header.actor0, None)
                actor1_obj = self._link_id_to_objects.get(contact_header.actor1, None)
                # If any of the objects cannot be found, skip
                if actor0_obj is None or actor1_obj is None:
                    continue
                # If any of the objects is not initialized, skip
                if not actor0_obj.initialized or not actor1_obj.initialized:
                    continue
                # If any of the objects is not stateful, skip
                if not isinstance(actor0_obj, StatefulObject) or not isinstance(actor1_obj, StatefulObject):
                    continue
                # If any of the objects doesn't have states that require on_contact callbacks, skip
                if len(actor0_obj.states.keys() & self.object_state_types_on_contact) == 0 or len(actor1_obj.states.keys() & self.object_state_types_on_contact) == 0:
                    continue
                headers[tuple(sorted((actor0_obj, actor1_obj), key=lambda x: x.uuid))].append(contact_header)

            for (actor0_obj, actor1_obj) in headers:
                for obj0, obj1 in [(actor0_obj, actor1_obj), (actor1_obj, actor0_obj)]:
                    for state_type in self.object_state_types_on_contact:
                        if state_type in obj0.states:
                            obj0.states[state_type].on_contact(obj1, headers[(actor0_obj, actor1_obj)], contact_data)

    def _on_simulation_event(self, event):
        """
        This callback will be invoked if there is any simulation event. Currently it only processes JOINT_BREAK event.
        """
        if gm.ENABLE_OBJECT_STATES:
            if event.type == int(SimulationEvent.JOINT_BREAK) and self._objects_require_joint_break_callback:
                joint_path = str(PhysicsSchemaTools.decodeSdfPath(event.payload["jointPath"][0], event.payload["jointPath"][1]))
                obj = None
                # TODO: recursively try to find the parent object of this joint
                tokens = joint_path.split("/")
                for i in range(2, len(tokens) + 1):
                    obj = self._scene.object_registry("prim_path", "/".join(tokens[:i]))
                    if obj is not None:
                        break

                if obj is None or not obj.initialized or not isinstance(obj, StatefulObject):
                    return
                if len(obj.states.keys() & self.object_state_types_on_joint_break) == 0:
                    return
                for state_type in self.object_state_types_on_joint_break:
                    if state_type in obj.states:
                        obj.states[state_type].on_joint_break(joint_path)

    def is_paused(self):
        """
        Returns:
            bool: True if the simulator is paused, otherwise False
        """
        return not (self.is_stopped() or self.is_playing())

    @contextlib.contextmanager
    def stopped(self):
        """
        A context scope for making sure the simulator is stopped during execution within this scope.
        Upon leaving the scope, the prior simulator state is restored.
        """
        # Infer what state we're currently in, then stop, yield, and then restore the original state
        sim_is_playing, sim_is_paused = self.is_playing(), self.is_paused()
        if sim_is_playing or sim_is_paused:
            self.stop()
        yield
        if sim_is_playing: self.play()
        elif sim_is_paused: self.pause()

    @contextlib.contextmanager
    def playing(self):
        """
        A context scope for making sure the simulator is playing during execution within this scope.
        Upon leaving the scope, the prior simulator state is restored.
        """
        # Infer what state we're currently in, then stop, yield, and then restore the original state
        sim_is_stopped, sim_is_paused = self.is_stopped(), self.is_paused()
        if sim_is_stopped or sim_is_paused:
            self.play()
        yield
        if sim_is_stopped: self.stop()
        elif sim_is_paused: self.pause()

    @contextlib.contextmanager
    def paused(self):
        """
        A context scope for making sure the simulator is paused during execution within this scope.
        Upon leaving the scope, the prior simulator state is restored.
        """
        # Infer what state we're currently in, then stop, yield, and then restore the original state
        sim_is_stopped, sim_is_playing = self.is_stopped(), self.is_playing()
        if sim_is_stopped or sim_is_playing:
            self.pause()
        yield
        if sim_is_stopped: self.stop()
        elif sim_is_playing: self.play()

    @contextlib.contextmanager
    def slowed(self, dt):
        """
        A context scope for making the simulator simulation dt slowed, e.g.: for taking micro-steps for propagating
        instantaneous kinematics with minimal impact on physics propagation.

        NOTE: This will set both the physics dt and rendering dt to the same value during this scope.

        Upon leaving the scope, the prior simulator state is restored.
        """
        # Set dt, yield, then restore the original dt
        physics_dt, rendering_dt = self.get_physics_dt(), self.get_rendering_dt()
        self.set_simulation_dt(physics_dt=dt, rendering_dt=dt)
        yield
        self.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)

    def add_callback_on_play(self, name, callback):
        """
        Adds a function @callback, referenced by @name, to be executed every time sim.play() is called

        Args:
            name (str): Name of the callback
            callback (function): Callback function. Function signature is expected to be:

                def callback() --> None
        """
        self._callbacks_on_play[name] = callback

    def add_callback_on_stop(self, name, callback):
        """
        Adds a function @callback, referenced by @name, to be executed every time sim.stop() is called

        Args:
            name (str): Name of the callback
            callback (function): Callback function. Function signature is expected to be:

                def callback() --> None
        """
        self._callbacks_on_stop[name] = callback

    def add_callback_on_import_obj(self, name, callback):
        """
        Adds a function @callback, referenced by @name, to be executed every time sim.import_object() is called

        Args:
            name (str): Name of the callback
            callback (function): Callback function. Function signature is expected to be:

                def callback(obj: BaseObject) --> None
        """
        self._callbacks_on_import_obj[name] = callback

    def add_callback_on_remove_obj(self, name, callback):
        """
        Adds a function @callback, referenced by @name, to be executed every time sim.remove_object() is called

        Args:
            name (str): Name of the callback
            callback (function): Callback function. Function signature is expected to be:

                def callback(obj: BaseObject) --> None
        """
        self._callbacks_on_remove_obj[name] = callback

    def remove_callback_on_play(self, name):
        """
        Remove play callback whose reference is @name

        Args:
            name (str): Name of the callback
        """
        self._callbacks_on_play.pop(name)

    def remove_callback_on_stop(self, name):
        """
        Remove stop callback whose reference is @name

        Args:
            name (str): Name of the callback
        """
        self._callbacks_on_stop.pop(name)

    def remove_callback_on_import_obj(self, name):
        """
        Remove stop callback whose reference is @name

        Args:
            name (str): Name of the callback
        """
        self._callbacks_on_import_obj.pop(name)

    def remove_callback_on_remove_obj(self, name):
        """
        Remove stop callback whose reference is @name

        Args:
            name (str): Name of the callback
        """
        self._callbacks_on_remove_obj.pop(name)

    @classmethod
    def clear_instance(cls):
        SimulationContext.clear_instance()
        Simulator._world_initialized = None
        return

    def __del__(self):
        SimulationContext.__del__(self)
        Simulator._world_initialized = None
        return

    @property
    def pi(self):
        """
        Returns:
            PhysX: Physx Interface (pi) for controlling low-level physx engine
        """
        return self._physx_interface

    @property
    def psi(self):
        """
        Returns:
            IPhysxSimulation: Physx Simulation Interface (psi) for controlling low-level physx simulation
        """
        return self._physx_simulation_interface

    @property
    def psqi(self):
        """
        Returns:
            PhysXSceneQuery: Physx Scene Query Interface (psqi) for running low-level scene queries
        """
        return self._physx_scene_query_interface

    @property
    def scene(self):
        """
        Returns:
            None or Scene: Scene currently loaded in this simulator. If no scene is loaded, returns None
        """
        return self._scene

    @property
    def viewer_camera(self):
        """
        Returns:
            VisionSensor: Active camera sensor corresponding to the active viewport window instance shown in the omni UI
        """
        return self._viewer_camera

    @property
    def camera_mover(self):
        """
        Returns:
            None or CameraMover: If enabled, the teleoperation interface for controlling the active viewer camera
        """
        return self._camera_mover

    @property
    def world_prim(self):
        """
        Returns:
            Usd.Prim: Prim at /World
        """
        return get_prim_at_path(prim_path="/World")

    def clear(self) -> None:
        """
        Clears the stage leaving the PhysicsScene only if under /World.
        """
        # Stop the physics
        self.stop()

        # Clear any pre-existing scene if it exists
        if self._scene is not None:
            self.scene.clear()
        self._scene = None

        # Clear all vision sensors and remove viewer camera reference and camera mover reference
        VisionSensor.clear()
        self._viewer_camera = None
        if self._camera_mover is not None:
            self._camera_mover.clear()
            self._camera_mover = None

        # Clear all transition rules if being used
        if gm.ENABLE_TRANSITION_RULES:
            TransitionRuleAPI.clear()

        # Clear uniquely named items and other internal states
        clear_pu()
        clear_uu()
        self._objects_to_initialize = []
        self._objects_require_contact_callback = False
        self._objects_require_joint_break_callback = False
        self._link_id_to_objects = dict()

        self._callbacks_on_play = dict()
        self._callbacks_on_stop = dict()
        self._callbacks_on_import_obj = dict()
        self._callbacks_on_remove_obj = dict()

        # Load dummy stage, but don't clear sim to prevent circular loops
        self._open_new_stage()

    def write_metadata(self, key, data):
        """
        Writes metadata @data to the current global metadata dict using key @key

        Args:
            key (str): Keyword entry in the global metadata dictionary to use
            data (dict): Data to write to @key in the global metadata dictionary
        """
        self.world_prim.SetCustomDataByKey(key, data)

    def get_metadata(self, key):
        """
        Grabs metadata from the current global metadata dict using key @key

        Args:
            key (str): Keyword entry in the global metadata dictionary to use
        """
        return self.world_prim.GetCustomDataByKey(key)

    def restore(self, json_path):
        """
        Restore a simulation environment from @json_path.

        Args:
            json_path (str): Full path of JSON file to load, which contains information
                to recreate a scene.
        """
        if not json_path.endswith(".json"):
            log.error(f"You have to define the full json_path to load from. Got: {json_path}")
            return

        # Load the info from the json
        with open(json_path, "r") as f:
            scene_info = json.load(f)
        init_info = scene_info["init_info"]
        state = scene_info["state"]

        # Override the init info with our json path
        init_info["args"]["scene_file"] = json_path

        # Also make sure we have any additional modifications necessary from the specific scene
        og.REGISTERED_SCENES[init_info["class_name"]].modify_init_info_for_restoring(init_info=init_info)

        # Recreate and import the saved scene
        og.sim.stop()
        recreated_scene = create_object_from_init_info(init_info)
        self.import_scene(scene=recreated_scene)

        # Start the simulation and restore the dynamic state of the scene and then pause again
        self.play()
        self.load_state(state, serialized=False)

        log.info("The saved simulation environment loaded.")

        return

    def save(self, json_path):
        """
        Saves the current simulation environment to @json_path.

        Args:
            json_path (str): Full path of JSON file to save (should end with .json), which contains information
                to recreate the current scene.
        """
        # Make sure the sim is not stopped, since we need to grab joint states
        assert not self.is_stopped(), "Simulator cannot be stopped when saving to USD!"

        # Make sure there are no objects in the initialization queue, if not, terminate early and notify user
        # Also run other sanity checks before saving
        if len(self._objects_to_initialize) > 0:
            log.error("There are still objects to initialize! Please take one additional sim step and then save.")
            return
        if not self.scene:
            log.warning("Scene has not been loaded. Nothing to save.")
            return
        if not json_path.endswith(".json"):
            log.error(f"You have to define the full json_path to save the scene to. Got: {json_path}")
            return

        # Update scene info
        self.scene.update_objects_info()

        # Dump saved current state and also scene init info
        scene_info = {
            "metadata": self.world_prim.GetCustomData(),
            "state": self.scene.dump_state(serialized=False),
            "init_info": self.scene.get_init_info(),
            "objects_info": self.scene.get_objects_info(),
        }

        # Write this to the json file
        Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
        with open(json_path, "w+") as f:
            json.dump(scene_info, f, cls=NumpyEncoder, indent=4)

        log.info("The current simulation environment saved.")

    def _open_new_stage(self):
        """
        Opens a new stage
        """
        # Stop the physics if we're playing
        if not self.is_stopped():
            log.warning("Stopping simulation in order to open new stage.")
            self.stop()

        # Store physics dt and rendering dt to reuse later
        # Note that the stage may have been deleted previously; if so, we use the default values
        # of 1/60, 1/60
        try:
            physics_dt = self.get_physics_dt()
        except:
            print("WARNING: Invalid or non-existent physics scene found. Setting physics dt to 1/60.")
            physics_dt = 1 / 60.
        rendering_dt = self.get_rendering_dt()

        # Open new stage -- suppressing warning that we're opening a new stage
        with suppress_omni_log(None):
            create_new_stage()

        # Clear physics context
        self._physics_context = None
        if meets_minimum_isaac_version("2023.0.0"):
            self._physx_fabric_interface = None

        # Create world prim
        self.stage.DefinePrim("/World", "Xform")

        self._init_stage(physics_dt=physics_dt, rendering_dt=rendering_dt)

    def _load_stage(self, usd_path):
        """
        Open the stage specified by USD file at @usd_path

        Args:
            usd_path (str): Absolute filepath to USD stage that should be loaded
        """
        # Stop the physics if we're playing
        if not self.is_stopped():
            log.warning("Stopping simulation in order to load stage.")
            self.stop()

        # Store physics dt and rendering dt to reuse later
        # Note that the stage may have been deleted previously; if so, we use the default values
        # of 1/60, 1/60
        try:
            physics_dt = self.get_physics_dt()
        except:
            print("WARNING: Invalid or non-existent physics scene found. Setting physics dt to 1/60.")
            physics_dt = 1/60.
        rendering_dt = self.get_rendering_dt()

        # Open new stage -- suppressing warning that we're opening a new stage
        with suppress_omni_log(None):
            open_stage(usd_path=usd_path)

        self._init_stage(physics_dt=physics_dt, rendering_dt=rendering_dt)

    def _init_stage(
        self,
        physics_dt=None,
        rendering_dt=None,
        stage_units_in_meters=None,
        physics_prim_path="/physicsScene",
        sim_params=None,
        set_defaults=True,
        backend="numpy",
        device=None,
    ):
        # Run super first
        super()._init_stage(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=stage_units_in_meters,
            physics_prim_path=physics_prim_path,
            sim_params=sim_params,
            set_defaults=set_defaults,
            backend=backend,
            device=device,
        )

        # Update internal vars
        self._physx_interface = get_physx_interface()
        self._physx_simulation_interface = get_physx_simulation_interface()
        self._physx_scene_query_interface = get_physx_scene_query_interface()

        # Update internal settings
        self._set_physics_engine_settings()
        self._set_renderer_settings()

        # Update internal callbacks
        self._setup_default_callback_fns()
        self._stage_open_callback = (
            omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._stage_open_callback_fn)
        )
        self._contact_callback = self._physics_context._physx_sim_interface.subscribe_contact_report_events(self._on_contact)
        self._simulation_event_callback = self._physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop(self._on_simulation_event)

        # Set the lighting mode to be stage by default
        self.set_lighting_mode(mode=LightingMode.STAGE)

        # Set the viewer camera, and then set its default pose
        self._set_viewer_camera()
        self.viewer_camera.set_position_orientation(
            position=np.array(m.DEFAULT_VIEWER_CAMERA_POS),
            orientation=np.array(m.DEFAULT_VIEWER_CAMERA_QUAT),
        )

    def close(self):
        """
        Shuts down the OmniGibson application
        """
        self._app.shutdown()

    @property
    def stage_id(self):
        """
        Returns:
            int: ID of the current active stage
        """
        return UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()

    @property
    def device(self):
        """
        Returns:
            device (None or str): Device used in simulation backend
        """
        return self._device

    @device.setter
    def device(self, device):
        """
        Sets the device used for sim backend

        Args:
            device (None or str): Device to set for the simulation backend
        """
        self._device = device
        if self._device is not None and "cuda" in self._device:
            device_id = self._settings.get_as_int("/physics/cudaDevice")
            self._device = f"cuda:{device_id}"

    @property
    def state_size(self):
        # Total state size is the state size of our scene
        return self._scene.state_size

    def _dump_state(self):
        # Default state is from the scene
        return self._scene.dump_state(serialized=False)

    def _load_state(self, state):
        # Default state is from the scene
        self._scene.load_state(state=state, serialized=False)

    def load_state(self, state, serialized=False):
        # We need to make sure the simulator is playing since joint states only get updated when playing
        assert self.is_playing()

        # Run super
        super().load_state(state=state, serialized=serialized)

        # Highlight that at the current step, the non-kinematic states are potentially inaccurate because a sim
        # step is needed to propagate specific states in physics backend
        # TODO: This should be resolved in a future omniverse release!
        disclaimer("Attempting to load simulator state.\n"
                   "Currently, omniverse does not support exclusively stepping kinematics, so we cannot update some "
                   "of our object states relying on updated kinematics until a simulator step is taken!\n"
                   "Object states such as OnTop, Inside, etc. relying on relative spatial information will inaccurate"
                   "until a single sim step is taken.\n"
                   "This should be resolved by the next NVIDIA Isaac Sim release.")

    def _serialize(self, state):
        # Default state is from the scene
        return self._scene.serialize(state=state)

    def _deserialize(self, state):
        # Default state is from the scene
        return self._scene.deserialize(state=state), self._scene.state_size
