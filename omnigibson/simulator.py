# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import defaultdict
import itertools
import logging

import numpy as np
import json
import omni
import carb
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_ancestral, get_prim_type_name, is_prim_no_delete, get_prim_at_path, \
    is_prim_path_valid
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.dynamic_control import _dynamic_control
import omni.kit.loop._loop as omni_loop
import builtins
from pxr import Usd, Gf, UsdGeom, Sdf, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.loggers import DataLogger
from typing import Optional, List

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.config_utils import NumpyEncoder
from omnigibson.utils.python_utils import clear as clear_pu, create_object_from_init_info, Serializable
from omnigibson.utils.usd_utils import clear as clear_uu, BoundingBoxAPI, get_usd_metadata, update_usd_metadata
from omnigibson.utils.asset_utils import get_og_avg_category_specs
from omnigibson.utils.ui_utils import CameraMover
from omnigibson.scenes import Scene
from omnigibson.objects.object_base import BaseObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.object_states.factory import get_states_by_dependency_order
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.transition_rules import DEFAULT_RULES, TransitionResults
from omni.kit.viewport_legacy import acquire_viewport_interface
from omni.syntheticdata import SyntheticData


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_VIEWER_CAMERA_POS = (-0.201028, -2.72566 ,  1.0654)
m.DEFAULT_VIEWER_CAMERA_QUAT = (0.68196617, -0.00155408, -0.00166678,  0.73138017)


class Simulator(SimulationContext, Serializable):
    """ This class inherits from SimulationContext which provides the following.

        SimulationContext provide functions that take care of many time-related events such as
        perform a physics or a render step for instance. Adding/ removing callback functions that
        gets triggered with certain events such as a physics step, timeline event
        (pause or play..etc), stage open/ close..etc.

        It also includes an instance of PhysicsContext which takes care of many physics related
        settings such as setting physics dt, solver type..etc.

        In addition to what is provided from SimulationContext, this class allows the user to add a
        task to the world and it contains a scene object.

        To control the default reset state of different objects easily, the object could be added to
        a Scene. Besides this, the object is bound to a short keyword that facilitates objects retrievals,
        like in a dict.

        Checkout the required tutorials at
        https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html

        Args:
            :param gravity: gravity on z direction.
            physics_dt (Optional[float], optional): dt between physics steps. Defaults to 1.0 / 60.0.
            rendering_dt (Optional[float], optional): dt between rendering steps. Note: rendering means
                                                       rendering a frame of the current application and not
                                                       only rendering a frame to the viewports/ cameras. So UI
                                                       elements of Isaac Sim will be refereshed with this dt
                                                       as well if running non-headless.
                                                       Defaults to 1.0 / 60.0.
            stage_units_in_meters (float, optional): The metric units of assets. This will affect gravity value..etc.
                                                      Defaults to 0.01.
            :param viewer_width: width of the camera image
            :param viewer_height: height of the camera image
            :param vertical_fov: vertical field of view of the camera image in degrees
            :param device: None or str, specifies the device to be used if running on the gpu with torch backend
            apply_transitions (bool): True to apply the transition rules.
        """

    _world_initialized = False

    def __init__(
            self,
            gravity=9.81,
            physics_dt: float = 1.0 / 60.0,
            rendering_dt: float = 1.0 / 60.0,
            stage_units_in_meters: float = 1.0,
            viewer_width=gm.DEFAULT_VIEWER_WIDTH,
            viewer_height=gm.DEFAULT_VIEWER_HEIGHT,
            vertical_fov=90,
            device=None,
            apply_transitions=False,
    ) -> None:
        super().__init__(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=stage_units_in_meters,
            device=device,
        )
        if Simulator._world_initialized:
            return
        Simulator._world_initialized = True
        self._scene_finalized = False
        # self._current_tasks = dict()
        self._dc_interface = _dynamic_control.acquire_dynamic_control_interface()
        # if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
        #     self.start_simulation()
        set_camera_view()
        self._data_logger = DataLogger()

        # Store other internal vars
        n_physics_timesteps_per_render = rendering_dt / physics_dt
        assert n_physics_timesteps_per_render.is_integer(), "render_timestep must be a multiple of physics_timestep"
        self.n_physics_timesteps_per_render = int(n_physics_timesteps_per_render)
        self.gravity = gravity
        self.vertical_fov = vertical_fov        # TODO: This currently does nothing

        # Store other references to variables that will be initialized later
        self._viewer = None
        self._viewer_camera = None
        self._camera_mover = None
        self._scene = None

        # Initialize viewer
        # self._set_physics_engine_settings()
        # TODO: Make this toggleable so we don't always have a viewer if we don't want to
        self._set_viewer_settings()
        self.viewer_width = viewer_width
        self.viewer_height = viewer_height

        # List of objects that need to be initialized during whenever the next sim step occurs
        self._objects_to_initialize = []

        # TODO: Fix
        # Set of categories that can be grasped by assisted grasping
        self.object_state_types = get_states_by_dependency_order()

        # Set of all non-Omniverse transition rules to apply.
        self._apply_transitions = apply_transitions
        self._transition_rules = DEFAULT_RULES

        # Toggle simulator state once so that downstream omni features can be used without bugs
        # e.g.: particle sampling, which for some reason requires sim.play() to be called at least once
        self.play()
        self.stop()

        # Finally, update the physics settings
        # This needs to be done now, after an initial step + stop for some reason if we want to use GPU
        # dynamics, otherwise we get very strange behavior, e.g., PhysX complains about invalid transforms
        # and crashes
        self._set_physics_engine_settings()

    def __new__(
        cls,
        gravity=9.81,
        physics_dt: float = 1.0 / 60.0,
        rendering_dt: float = 1.0 / 60.0,
        stage_units_in_meters: float = 1.0,
        viewer_width=gm.DEFAULT_VIEWER_WIDTH,
        viewer_height=gm.DEFAULT_VIEWER_HEIGHT,
        vertical_fov=90,
        device_idx=0,
        apply_transitions=False,
    ) -> None:
        # Overwrite since we have different kwargs
        if Simulator._instance is None:
            Simulator._instance = object.__new__(cls)
        else:
            carb.log_info("Simulator is defined already, returning the previously defined one")
        return Simulator._instance

    def _reset_variables(self):
        """
        Reset state of internal variables
        """
        pass

    def _set_viewer_camera(self, prim_path="/World/viewer_camera"):
        """
        Creates a camera prim dedicated for this viewer at @prim_path if it doesn't exist,
        and sets this camera as the active camera for the viewer

        Args:
            prim_path (str): Path to check for / create the viewer camera
        """
        vp = acquire_viewport_interface()
        viewers_to_names = {vp.get_viewport_window(h): vp.get_viewport_window_name(h) for h in vp.get_instance_list()}
        self._viewer_camera = VisionSensor(
            prim_path=prim_path,
            name=prim_path.split("/")[-1],                  # Assume name is the lowest-level name in the prim_path
            modalities="rgb",
            image_height=self.viewer_height,
            image_width=self.viewer_width,
            viewport_name=viewers_to_names[self._viewer],
        )
        if not self._viewer_camera.loaded:
            self._viewer_camera.load(simulator=self)

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
        self._physics_context.enable_flatcache(gm.ENABLE_FLATCACHE)

        # Enable GPU dynamics based on whether we need omni particles feature
        if gm.ENABLE_OMNI_PARTICLES:
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

    def _set_viewer_settings(self):
        """
        Initializes a reference to the viewer in the App, and sets the frame size
        """
        # Store reference to viewer (see https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/reference_python_snippets.html#get-camera-parameters)
        viewport = acquire_viewport_interface()
        viewport_handle = viewport.get_instance("Viewport")
        self._viewer = viewport.get_viewport_window(viewport_handle)

        # Set viewer camera and frame size
        self._set_viewer_camera()

    def enable_viewer_camera_teleoperation(self):
        """
        Enables keyboard control of the active viewer camera for this simulation
        """
        self._camera_mover = CameraMover(cam=self._viewer_camera)
        self._camera_mover.print_info()

    def import_scene(self, scene):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: a scene object to load
        """
        assert self.is_stopped(), "Simulator must be stopped while importing a scene!"
        assert isinstance(scene, Scene), "import_scene can only be called with Scene"

        # Clear the existing scene if any
        self.clear()

        self._scene = scene
        self._scene.load(self)

        # Make sure simulator is not running, then start it so that we can initialize the scene
        assert self.is_stopped(), "Simulator must be stopped after importing a scene!"
        self.play()

        # Initialize the scene
        self._scene.initialize()

        # Need to one more step for particle systems to work
        self.step()
        self.stop()

    # # TODO
    # def import_particle_system(self, particle_system):
    #     """
    #     Import a particle system into the simulator. Called by objects owning a particle-system, via reference to the Simulator instance.
    #
    #     :param particle_system: a ParticleSystem object to load
    #     """
    #
    #     assert isinstance(
    #         particle_system, ParticleSystem
    #     ), "import_particle_system can only be called with ParticleSystem"
    #
    #     self.particle_systems.append(particle_system)
    #     particle_system.initialize(self)

    def initialize_object_on_next_sim_step(self, obj):
        """
        Initializes the object upon the next simulation step

        Args:
            obj (BasePrim): Object to initialize as soon as a new sim step is called
        """
        self._objects_to_initialize.append(obj)

    def import_object(self, obj, register=True, auto_initialize=True):
        """
        Import an object into the simulator.

        Args:
            obj (BaseObject): an object to load
            register (bool): whether to register this object internally in the scene registry
            auto_initialize (bool): If True, will auto-initialize the requested object on the next simulation step.
                Otherwise, we assume that the object will call initialize() on its own!
        """
        assert isinstance(obj, BaseObject), "import_object can only be called with BaseObject"

        # TODO
        # if False:#isinstance(obj, VisualMarker) or isinstance(obj, Particle):
        #     # Marker objects can be imported without a scene.
        #     obj.load(self)
        # else:

        # Make sure scene is loaded -- objects should not be loaded unless we have a reference to a scene
        assert self.scene is not None, "import_object needs to be called after import_scene"

        # Load the object in omniverse by adding it to the scene
        self.scene.add_object(obj, self, register=register, _is_call_from_simulator=True)

        # Lastly, additionally add this object automatically to be initialized as soon as another simulator step occurs
        # if requested
        if auto_initialize:
            self.initialize_object_on_next_sim_step(obj=obj)
    
    def remove_object(self, obj):
        """
        Remove a non-robot object from the simulator.

        Args:
            obj (BaseObject): a non-robot object to load
        """
        self._scene.remove_object(obj, simulator=self)
        self.app.update()

    def _non_physics_step(self):
        """
        Complete any non-physics steps such as state updates.
        """
        # Check to see if any objects should be initialized (only done IF we're playing)
        if len(self._objects_to_initialize) > 0 and self.is_playing():
            for obj in self._objects_to_initialize:
                obj.initialize()
            self._objects_to_initialize = []
            # Also update the scene registry
            # TODO: A better place to put this perhaps?
            self._scene.object_registry.update(keys="root_handle")

        # Propagate states if the feature is enabled
        if gm.ENABLE_OBJECT_STATES:

            # Cache values from all of the micro and macro particle systems.
            # This is used to store system-wide state which can be queried
            # by the object state system.
            for system in self.scene.systems:
                system.cache()

            # Step the object states in global topological order (if the scene exists).
            if self.scene is not None:
                for state_type in self.object_state_types:
                    for obj in self.scene.get_objects_with_state(state_type):
                        # Only update objects that have been initialized so far
                        if obj.initialized:
                            obj.states[state_type].update()

            # Perform system level updates to the micro and macro particle systems.
            # This allows for the states to handle changes in response to changes
            # induced by the object state system.
            for system in self.scene.systems:
                system.update()

            for obj in self.scene.objects:
                # Only update visuals for objects that have been initialized so far
                if isinstance(obj, StatefulObject) and obj.initialized:
                    obj.update_visuals()

        # Clear the bounding box cache so that it gets updated during the next time it's called
        BoundingBoxAPI.clear()

    def _transition_rule_step(self):
        """Applies all internal non-Omniverse transition rules."""
        # Create a dict from rule to filter to objects we care about.
        obj_dict = defaultdict(lambda: defaultdict(list))
        for obj in self.scene.objects:
            for rule in self._transition_rules:
                for f in rule.filters:
                    if f(obj):
                        obj_dict[rule][f].append(obj)

        # For each rule, create a subset of the dict and apply it if applicable.
        added_obj_attrs = []
        removed_objs = []
        for rule in self._transition_rules:
            if rule not in obj_dict:
                continue
            # Create lists of objects that this rule potentially cares about.
            # Skip the rule if any of the object lists is empty.
            obj_list_rule = list(obj_dict[rule][f] for f in rule.filters)
            if any(not obj_list_filter for obj_list_filter in obj_list_rule):
                continue
            # For each possible combination of objects, check if the rule is
            # applicable, and if so, apply the transition defined by the rule.
            # If objects are to be added / removed, the transition function is
            # expected to return an instance of TransitionResults containing
            # information about those objects.
            # TODO: Consider optimizing itertools.product.
            # TODO: Track what needs to be added / removed at the Scene object level.
            # Comments from a PR on possible changes:
            # - Make the transition function immediately apply the transition.
            # - Addition / removal tracking on the Scene object.
            # - Check if the objects are still in the scene in each step.
            for obj_tuple in itertools.product(*obj_list_rule):
                if rule.condition(self, *obj_tuple):
                    t_results = rule.transition(self, *obj_tuple)
                    if isinstance(t_results, TransitionResults):
                        added_obj_attrs.extend(t_results.add)
                        removed_objs.extend(t_results.remove)

        # Process all transition results.
        for added_obj_attr in added_obj_attrs:
            new_obj = added_obj_attr.obj
            self.import_object(added_obj_attr.obj)
            pos, orn = added_obj_attr.pos, added_obj_attr.orn
            new_obj.set_position_orientation(position=pos, orientation=orn)
        for removed_obj in removed_objs:
            self.remove_object(removed_obj)

    def reset_scene(self):
        """
        Resets ths scene (if it exists) and its corresponding objects
        """
        if self.scene is not None and self.scene.initialized:
            self.scene.reset()

    def play(self):
        super().play()

        # Update all object / robot handles
        if self.scene is not None and self.scene.initialized:
            for obj in self.scene.objects:
                # Only need to update handles if object is already initialized as well
                if obj.initialized:
                    obj.update_handles()

            for robot in self.scene.robots:
                # Only need to update handles if robot is already initialized as well
                if robot.initialized:
                    robot.update_handles()

        # Check to see if any objects should be initialized
        if len(self._objects_to_initialize) > 0:
            for obj in self._objects_to_initialize:
                obj.initialize()
            self._objects_to_initialize = []
            # Also update the scene registry
            # TODO: A better place to put this perhaps?
            self._scene.object_registry.update(keys="root_handle")

    def step(self, render=True, force_playing=False):
        """
        Step the simulation at self.render_timestep

        Args:
            render (bool): Whether rendering should occur or not
            force_playing (bool): If True, will force physics to propagate (i.e.: set simulation, if paused / stopped,
                to "play" mode)
        """
        # Possibly force playing
        if force_playing and not self.is_playing():
            self.play()

        # Note that we bypass super().step() because there seems to be some issues with app.update()
        # In theory, app.update() should be equivalent to step_physics() and then render().
        # However, emperically, app.update() causes a bug in gpu dynamics.
        if self.physics_sim_view is not None:
            self.physics_sim_view.flush()

        for i in range(self.n_physics_timesteps_per_render):
            self.step_physics()

        if render:
            self.render()

        # Additionally run non physics things if we have a valid scene
        if self._scene is not None:
            self._non_physics_step()
            if self._apply_transitions:
                self._transition_rule_step()

        # TODO (eric): After stage changes (e.g. pose, texture change), it will take two super().step(render=True) for
        #  the result to propagate to the rendering. We could have called super().render() here but it will introduce
        #  a big performance regression.

    def step_physics(self):
        """
        Step the physics a single step.

        """
        self._physics_context._step(current_time=self.current_time)

    # TODO: Do we need this?
    # def sync(self, force_sync=False):
    #     """
    #     Update positions in renderer without stepping the simulation. Usually used in the reset() function.
    #
    #     :param force_sync: whether to force sync the objects in renderer
    #     """
    #     self.body_links_awake = 0
    #     for instance in self.renderer.instances:
    #         if instance.dynamic:
    #             self.body_links_awake += self.update_position(instance, force_sync=force_sync or self.first_sync)
    #     if self.viewer is not None:
    #         self.viewer.update()
    #     if self.first_sync:
    #         self.first_sync = False

    def is_paused(self) -> bool:
        """Returns: True if the simulator is paused."""
        return not (self.is_stopped() or self.is_playing())

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
    def dc_interface(self) -> _dynamic_control.DynamicControl:
        """[summary]

        Returns:
            _dynamic_control.DynamicControl: [description]
        """
        return self._dc_interface

    @property
    def scene(self) -> Scene:
        """[summary]

        Returns:
            Scene: [description]
        """
        return self._scene

    @property
    def viewer(self):
        """
        Returns:
            ViewportWindow: Active viewport window instance shown in the omni UI
        """
        return self._viewer

    @property
    def viewer_camera(self):
        """
        Returns:
            VisionSensor: Active camera sensor corresponding to the active viewport window instance shown in the omni UI
        """
        return self._viewer_camera

    @property
    def world_prim(self):
        """
        Returns:
            Usd.Prim: Prim at /World
        """
        return get_prim_at_path(prim_path="/World")

    def _clear_state(self):
        """
        Clears the internal state of this simulation
        """
        # Clear uniquely named items and other internal states
        clear_pu()
        clear_uu()

    def clear(self) -> None:
        """Clears the stage leaving the PhysicsScene only if under /World.
        """
        # Stop the physics
        self.stop()

        # if self.scene is not None:
        #     self.scene.clear()
        # self._current_tasks = dict()
        self._scene_finalized = False
        self._scene = None
        self._data_logger = DataLogger()

        # TODO: Handle edge-case for when we clear sim without loading new scene in. self._scene should be None
        # but scene.load(sim) requires scene to be defined!

        # def check_deletable_prim(prim_path):
        #     print(f"checking prim path: {prim_path}")
        #     if is_prim_no_delete(prim_path):
        #         return False
        #     if is_prim_ancestral(prim_path):
        #         return False
        #     if get_prim_type_name(prim_path=prim_path) == "PhysicsScene":
        #         return False
        #     if prim_path == "/World":
        #         return False
        #     if prim_path == "/":
        #         return False
        #     return True
        #
        # clear_stage(predicate=check_deletable_prim)

        # Load dummy stage, but don't clear sim to prevent circular loops
        self.load_stage(usd_path=f"{og.assets_path}/models/misc/clear_stage.usd")

    def restore(self, usd_path):
        """
        Restore a simulation environment from @usd_path.

        Args:
            usd_path (str): Full path of USD file to load, which contains information
                to recreate a scene.
        """
        if not usd_path.endswith(".usd"):
            logging.error(f"You have to define the full usd_path to load from. Got: {usd_path}")
            return

        # Load saved stage to get saved_info.
        self.load_stage(usd_path)

        # TODO: Save / loading emptyscene fails because we're not loading the saved USD (no usd_path arg to pass into the scene constructor)
        # TODO: Really need to iron this out in a cleaner way -- how to reload scene robustly?

        # Load saved info
        scene_state = json.loads(self.world_prim.GetCustomDataByKey("scene_state"))
        scene_init_info = json.loads(self.world_prim.GetCustomDataByKey("scene_init_info"))
        # Overwrite the usd with our desired usd file
        scene_init_info["args"]["usd_path"] = usd_path
        # Clear the current environment and delete any currently loaded scene.
        self.clear()

        # Recreate and import the saved scene
        recreated_scene = create_object_from_init_info(scene_init_info)
        self.import_scene(scene=recreated_scene)

        # Start the simulation and restore the dynamic state of the scene and then pause again
        self.play()
        self._scene.load_state(scene_state, serialized=False)
        self.app.update()
        self.pause()

        logging.info("The saved simulation environment loaded.")

        return

    def save(self, usd_path):
        """
        Saves the current simulation environment to @usd_path.

        Args:
            usd_path (str): Full path of USD file to load, which contains information
                to recreate the current scene.
        """
        # TODO: Make sure all objects have been initialized

        if not self.scene:
            logging.warning("Scene has not been loaded. Nothing to save.")
            return
        if not usd_path.endswith(".usd"):
            logging.error(f"You have to define the full usd_path to save the scene to. Got: {usd_path}")
            return

        # Update scene info
        self.scene.update_scene_info()

        # Dump saved current state and also scene init info
        saved_state_str = json.dumps(self.scene.dump_state(serialized=False), cls=NumpyEncoder)
        self.world_prim.SetCustomDataByKey("scene_state", saved_state_str)
        scene_init_info = self.scene.get_init_info()
        scene_init_info_str = json.dumps(scene_init_info, cls=NumpyEncoder)
        self.world_prim.SetCustomDataByKey("scene_init_info", scene_init_info_str)

        # Update USD Metadata -- we need this info in order to, e.g., update material filepaths appropriately
        # in case this USD is opened on another machine
        update_usd_metadata()

        # Save stage. This needs to happen at the end since some objects may get reset after sim.stop().
        # We also need to reset the Synthetic Data Utilities, so that we can re-initialize it when we reload the USD
        # Otherwise when we try to reload the USD and init Synthetic Data again we will run into an error since the
        # Synthetic Data interface had already existed when we had saved the USD beforehand
        self.stop()
        SyntheticData.Reset()

        self.stage.Export(usd_path)

        # Re-initialize the synthetic data and re-initialize all sensors
        # This is needed because we destroyed and recreated the synthetic data interface, and so the specific sensor
        # modalities need to be initialized again within the Synthetic Data interface
        SyntheticData.Initialize()
        for sensor in VisionSensor.SENSORS.values():
            sensor.initialize_sensors(names=sensor.modalities)

        logging.info("The current simulation environment saved.")

        return

    def get_data_logger(self) -> DataLogger:
        """Returns the data logger of the world.

        Returns:
            DataLogger: [description]
        """
        return self._data_logger

    # TODO: Extend to update internal info
    def load_stage(self, usd_path):
        """
        Open the stage specified by USD file at @usd_path

        Args:
            usd_path (str): Absolute filepath to USD stage that should be loaded
        """
        # Stop the physics if we're playing
        if not self.is_stopped():
            logging.warning("Stopping simulation in order to load stage.")
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

        # Clear simulation state
        self._clear_state()

        open_stage(usd_path=usd_path)

        # Re-initialize necessary internal vars
        self._app = omni.kit.app.get_app_interface()
        self._framework = carb.get_framework()
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.set_auto_update(True)
        self._dynamic_control = _dynamic_control.acquire_dynamic_control_interface()
        self._cached_rate_limit_enabled = self._settings.get_as_bool("/app/runLoops/main/rateLimitEnabled")
        self._cached_rate_limit_frequency = self._settings.get_as_int("/app/runLoops/main/rateLimitFrequency")
        self._cached_min_frame_rate = self._settings.get_as_int("persistent/simulation/minFrameRate")
        self._loop_runner = omni_loop.acquire_loop_interface()

        self._init_stage(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=self._initial_stage_units_in_meters,
        )
        self._set_physics_engine_settings()
        self._setup_default_callback_fns()
        self._stage_open_callback = (
            omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._stage_open_callback_fn)
        )

        # Set the viewer camera, and then set its default pose
        self._set_viewer_camera()
        og.sim.viewer_camera.set_position_orientation(
            position=np.array(m.DEFAULT_VIEWER_CAMERA_POS),
            orientation=np.array(m.DEFAULT_VIEWER_CAMERA_QUAT),
        )

    def close(self):
        """
        Shuts down the OmniGibson application
        """
        self._app.shutdown()

    @property
    def device(self) -> str:
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
        # If we're using GPU, we have to do a super stupid workaround to avoid physx crashing
        # For some reason, trying to load large states after n >= 3 steps are taken after the simulator starts playing
        # results in a crash. So, since we are resetting the entire sim state anyways, we will stop and start the
        # simulator to reset the frame count
        assert self.is_playing()
        if gm.ENABLE_OMNI_PARTICLES:
            self.stop()
            self.play()
        # Run super
        super().load_state(state=state, serialized=serialized)

        # TODO: verify if this is still needed
        # # We also need to manually update the simulator app
        # self._simulator.app.update()

    def _serialize(self, state):
        # Default state is from the scene
        return self._scene.serialize(state=state)

    def _deserialize(self, state):
        # Default state is from the scene
        end_idx = self._scene.state_size
        return self._scene.deserialize(state=state[:end_idx]), end_idx
