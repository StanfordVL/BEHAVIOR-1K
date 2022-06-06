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

import json
import omni
import carb
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_ancestral, get_prim_type_name, is_prim_no_delete, get_prim_at_path, \
    is_prim_path_valid
from omni.isaac.core.utils.stage import clear_stage, save_stage, open_stage
from omni.isaac.dynamic_control import _dynamic_control
import omni.kit.loop._loop as omni_loop
import builtins
from pxr import Usd, UsdGeom, Sdf, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.loggers import DataLogger
from typing import Optional, List

from igibson import assets_path
import igibson.macros as m
from igibson.robots.robot_base import BaseRobot
from igibson.utils.utils import NumpyEncoder
from igibson.utils.python_utils import clear as clear_pu, create_object_from_init_info
from igibson.utils.usd_utils import clear as clear_uu, BoundingBoxAPI
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.scenes import Scene
from igibson.objects.object_base import BaseObject
from igibson.object_states.factory import get_states_by_dependency_order
from igibson.transition_rules import DEFAULT_RULES, TransitionResults

# Import viewport getter based on isaacsim version
if m.IS_PUBLIC_ISAACSIM:
    from omni.kit.viewport import get_viewport_interface as acquire_viewport_interface
else:
    from omni.kit.viewport_legacy import acquire_viewport_interface


class Simulator(SimulationContext):
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
            :param device_idx: GPU device index to run rendering on
        """

    _world_initialized = False

    def __init__(
            self,
            gravity=9.81,
            physics_dt: float = 1.0 / 60.0,
            rendering_dt: float = 1.0 / 60.0,
            stage_units_in_meters: float = 1.0,
            viewer_width=1280,
            viewer_height=720,
            vertical_fov=90,
            device_idx=0,
    ) -> None:
        super().__init__(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=stage_units_in_meters,
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
        self.viewer_width = viewer_width
        self.viewer_height = viewer_height
        self.vertical_fov = vertical_fov        # TODO: This currently does nothing
        self.device_idx = device_idx            # TODO: This currently does nothing

        # Store other references to variables that will be initialized later
        self._viewer = None
        self._viewer_camera = None
        self._scene = None
        self.particle_systems = []
        self.frame_count = 0
        self.body_links_awake = 0
        self.first_sync = True          # First sync always sync all objects (regardless of their sleeping states)

        # Initialize viewer
        # self._set_physics_engine_settings()
        # TODO: Make this toggleable so we don't always have a viewer if we don't want to
        self._set_viewer_settings()

        # List of objects that need to be initialized during whenever the next sim step occurs
        self._objects_to_initialize = []

        # TODO: Fix
        # Set of categories that can be grasped by assisted grasping
        self.object_state_types = get_states_by_dependency_order()

        # Set of all non-Omniverse transition rules to apply.
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
        stage_units_in_meters: float = 0.01,
        viewer_width=1280,
        viewer_height=720,
        vertical_fov=90,
        device_idx=0,
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
        self.particle_systems = []
        self.frame_count = 0
        self.body_links_awake = 0
        self.first_sync = True          # First sync always sync all objects (regardless of their sleeping states)

    def _set_viewer_camera(self, prim_path="/World/viewer_camera"):
        """
        Creates a camera prim dedicated for this viewer at @prim_path if it doesn't exist,
        and sets this camera as the active camera for the viewer

        Args:
            prim_path (str): Path to check for / create the viewer camera
        """
        self._viewer_camera = get_prim_at_path(prim_path=prim_path) if is_prim_path_valid(prim_path=prim_path) else \
            UsdGeom.Camera.Define(self.stage, "/World/viewer_camera").GetPrim()
        self._viewer.set_active_camera(str(self._viewer_camera.GetPrimPath()))

    def _set_physics_engine_settings(self):
        """
        Set the physics engine with specified settings
        """
        assert self.is_stopped(), f"Cannot set simulator physics settings while simulation is playing!"
        self._physics_context.set_gravity(value=-self.gravity)
        # Also make sure we invert the collision group filter settings so that different collision groups cannot
        # collide with each other, and modify settings for speed optimization

        if m.IS_PUBLIC_ISAACSIM:
            # Only have access to invert collision filter and CCD setting, no flatcache
            self._physics_context._physx_scene_api.GetInvertCollisionGroupFilterAttr().Set(True)
            self._physics_context._physx_scene_api.GetEnableCCDAttr().Set(m.ENABLE_CCD)
        else:
            self._physics_context.set_invert_collision_group_filter(True)
            self._physics_context.enable_ccd(m.ENABLE_CCD)
            self._physics_context.enable_flatcache(m.ENABLE_FLATCACHE)

        # Enable GPU dynamics based on whether we need omni particles feature
        if m.ENABLE_OMNI_PARTICLES:
            self._physics_context.enable_gpu_dynamics(True)
            self._physics_context.set_broadphase_type("GPU")
        else:
            self._physics_context.enable_gpu_dynamics(False)
            self._physics_context.set_broadphase_type("MBP")

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
        self._viewer.set_texture_resolution(self.viewer_width, self.viewer_height)

    def import_scene(self, scene):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: a scene object to load
        """
        assert self.is_stopped(), "Simulator must be stopped while importing a scene!"
        assert isinstance(scene, Scene), "import_scene can only be called with Scene"
        self._scene = scene
        scene.load(self)

        # Make sure simulator is not running, then start it, then pause it so we can initialize the scene
        assert self.is_stopped(), "Simulator must be stopped after importing a scene!"
        self.play()
        self.pause()
        # Initialize the scene
        self._scene.initialize()

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
        self._scene.remove_object(obj)

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

        # Step all of the particle systems.
        for particle_system in self.particle_systems:
            particle_system.update(self)

        # Step the object states in global topological order (if the scene exists).
        if self.scene is not None:
            for state_type in self.object_state_types:
                for obj in self.scene.get_objects_with_state(state_type):
                    # Only update objects that have been initialized so far
                    if obj.initialized:
                        obj.states[state_type].update()


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

    # TODO: Potentially make apply_transitions a ctor argument.
    def step(self, render=True, force_playing=False, apply_transitions=False):
        """
        Step the simulation at self.render_timestep

        Args:
            render (bool): Whether rendering should occur or not
            force_playing (bool): If True, will force physics to propagate (i.e.: set simulation, if paused / stopped,
                to "play" mode)
            apply_transitions (bool): If True, apply the internal transition
                logic before physics, render steps.
        """
        if apply_transitions:
            self._transition_rule_step()

        # Possibly force playing
        if force_playing and not self.is_playing():
            self.play()

        for i in range(self.n_physics_timesteps_per_render - 1):
            # No rendering for intermediate steps for efficiency
            super().step(render=False)

        # Render on final step unless input says otherwise
        super().step(render=render)

        self._non_physics_step()
        # self.sync()
        self.frame_count += 1

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
        return self._viewer

    @property
    def world_prim(self):
        """
        Returns:
            Usd.Prim: Prim at /World
        """
        return get_prim_at_path(prim_path="/World")

    # def get_current_tasks(self) -> List[BaseTask]:
    #     """[summary]
    #
    #     Returns:
    #         List[BaseTask]: [description]
    #     """
    #     return self._current_tasks
    #
    # def get_task(self, name: str) -> BaseTask:
    #     if name not in self._current_tasks:
    #         raise Exception("task name {} doesn't exist in the current world tasks.".format(name))
    #     return self._current_tasks[name]

    # def _finalize_scene(self) -> None:
    #     """[summary]
    #     """
    #     if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
    #         self.play()
    #     self._scene._finalize()
    #     return

    def clear_and_remove_scene(self) -> None:
        self._scene = None
        self.clear()

    def clear(self) -> None:
        """Clears the stage leaving the PhysicsScene only if under /World.
        """
        # Stop the physics
        self.stop()

        # TODO: Handle edge-case for when we clear sim without loading new scene in. self._scene should be None
        # but scene.load(sim) requires scene to be defined!

        # Clear uniquely named items and other internal states
        clear_pu()
        clear_uu()

        # if self.scene is not None:
        #     self.scene.clear()
        self._current_tasks = dict()
        self._scene_finalized = False
        self._data_logger = DataLogger()

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
        self.load_stage(usd_path=f"{assets_path}/models/misc/clear_stage.usd")

        return

    # def reset(self) -> None:
    #     """ Resets the stage to its initial state and each object included in the Scene to its default state
    #         as specified by .set_default_state and the __init__ funcs.
    #
    #         Note:
    #         - All tasks should be added before the first reset is called unless a .clear() was called.
    #         - All articulations should be added before the first reset is called unless a .clear() was called.
    #         - This method takes care of initializing articulation handles with the first reset called.
    #         - This will do one step internally regardless
    #         - calls post_reset on each object in the Scene
    #         - calls post_reset on each Task
    #
    #         things like setting pd gains for instance should happend at a Task reset or a Robot reset since
    #         the defaults are restored after .stop() is called.
    #     """
    #     if not self._scene_finalized:
    #         for task in self._current_tasks.values():
    #             task.set_up_scene(self.scene)
    #         self._finalize_scene()
    #         self._scene_finalized = True
    #     self.stop()
    #     for task in self._current_tasks.values():
    #         task.cleanup()
    #     self.play()
    #     self.scene.post_reset()
    #     for task in self._current_tasks.values():
    #         task.post_reset()
    #     return
    #
    # async def reset_async(self) -> None:
    #     """Resets the stage to its initial state and each object included in the Scene to its default state
    #         as specified by .set_default_state and the __init__ funcs.
    #
    #         Note:
    #         - All tasks should be added before the first reset is called unless a .clear() was called.
    #         - All articulations should be added before the first reset is called unless a .clear() was called.
    #         - This method takes care of initializing articulation handles with the first reset called.
    #         - This will do one step internally regardless
    #         - calls post_reset on each object in the Scene
    #         - calls post_reset on each Task
    #
    #         things like setting pd gains for instance should happend at a Task reset or a Robot reset since
    #         the defaults are restored after .stop() is called.
    #     """
    #     if not self._scene_finalized:
    #         for task in self._current_tasks.values():
    #             task.set_up_scene(self.scene)
    #         await self.play_async()
    #         self._finalize_scene()
    #         self._scene_finalized = True
    #     await self.stop_async()
    #     for task in self._current_tasks.values():
    #         task.cleanup()
    #     await self.play_async()
    #     self._scene.post_reset()
    #     for task in self._current_tasks.values():
    #         task.post_reset()
    #     return

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

        # Load saved info
        scene_state = json.loads(self.world_prim.GetCustomDataByKey("scene_state"))
        scene_init_info = json.loads(self.world_prim.GetCustomDataByKey("scene_init_info"))

        # Clear the current environment and delete any currently loaded scene.
        self.clear_and_remove_scene()

        # Recreate and import the saved scene.
        # Note that the imported scene only have the default objects loaded.
        recreated_scene = create_object_from_init_info(scene_init_info)
        self.import_scene(scene=recreated_scene)

        # Start the simulation and restore the dynamic state of the scene and then pause again
        self.play()
        self.scene.load_state(scene_state, serialized=False)
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
        # Overwrite the usd with our desired usd file
        scene_init_info["args"]["usd_path"] = usd_path
        scene_init_info_str = json.dumps(scene_init_info, cls=NumpyEncoder)
        self.world_prim.SetCustomDataByKey("scene_init_info", scene_init_info_str)

        # Save stage. This needs to happen at the end since some objects may get reset after sim.stop().
        self.stop()
        self.stage.Export(usd_path)

        logging.info("The current simulation environment saved.")

        return

    def add_task(self, task: BaseTask) -> None:
        """Tasks should have a unique name.


        Args:
            task (BaseTask): [description]
        """
        if task.name in self._current_tasks:
            raise Exception("Task name should be unique in the world")
        self._current_tasks[task.name] = task
        return

    def get_observations(self, task_name: Optional[str] = None) -> dict:
        """Gets observations from all the tasks that were added

        Args:
            task_name (Optional[str], optional): [description]. Defaults to None.

        Returns:
            dict: [description]
        """
        if task_name is not None:
            return self._current_tasks[task_name].get_observations()
        else:
            observations = dict()
            for task in self._current_tasks.values():
                observations.update(task.get_observations())
            return observations

    def calculate_metrics(self, task_name: Optional[str] = None) -> None:
        """Gets metrics from all the tasks that were added

        Args:
            task_name (Optional[str], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if task_name is not None:
            return self._current_tasks[task_name].calculate_metrics()
        else:
            metrics = dict()
            for task in self._current_tasks.values():
                metrics.update(task.calculate_metrics())
            return metrics

    def is_done(self, task_name: Optional[str] = None) -> None:
        """[summary]

        Args:
            task_name (Optional[str], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if task_name is not None:
            return self._current_tasks[task_name].is_done()
        else:
            result = [task.is_done() for task in self._current_tasks.values()]
            return all(result)

    # def step(self, render: bool = True) -> None:
    #     """Steps the physics simulation while rendering or without.
    #
    #        - Note: task pre_step is called here.
    #
    #     Args:
    #         render (bool, optional): Set to False to only do a physics simulation without rendering. Note:
    #                                  app UI will be frozen (since its not rendering) in this case.
    #                                  Defaults to True.
    #
    #     """
    #     if self._scene_finalized:
    #         for task in self._current_tasks.values():
    #             task.pre_step(self.current_time_step_index, self.current_time)
    #     if self.scene._enable_bounding_box_computations:
    #         self.scene._bbox_cache.SetTime(Usd.TimeCode(self._current_time))
    #     SimulationContext.step(self, render=render)
    #     if self._data_logger.is_started():
    #         if self._data_logger._data_frame_logging_func is None:
    #             raise Exception("You need to add data logging function before starting the data logger")
    #         data = self._data_logger._data_frame_logging_func(tasks=self.get_current_tasks(), scene=self.scene)
    #         self._data_logger.add_data(
    #             data=data, current_time_step=self.current_time_step_index, current_time=self.current_time
    #         )
    #     return

    def step_async(self, step_size: float) -> None:
        """Calls all functions that should be called pre stepping the physics

           - Note: task pre_step is called here.

        Args:
            step_size (float): [description]

        Raises:
            Exception: [description]
        """
        if self._scene_finalized:
            for task in self._current_tasks.values():
                task.pre_step(self.current_time_step_index, self.current_time)
        if self.scene._enable_bounding_box_computations:
            self.scene._bbox_cache.SetTime(Usd.TimeCode(self._current_time))
        if self._data_logger.is_started():
            if self._data_logger._data_frame_logging_func is None:
                raise Exception("You need to add data logging function before starting the data logger")
            data = self._data_logger._data_frame_logging_func(tasks=self.get_current_tasks(), scene=self.scene)
            self._data_logger.add_data(
                data=data, current_time_step=self.current_time_step_index, current_time=self.current_time
            )
        return

    def get_data_logger(self) -> DataLogger:
        """Returns the data logger of the world.

        Returns:
            DataLogger: [description]
        """
        return self._data_logger

    def save_stage(self, usd_path):
        """
        Save the current stage in this simulator to specified @usd_path

        Args:
            usd_path (str): Absolute filepath to where this stage should be saved
        """
        # Make sure simulator is stopped
        assert self.is_stopped(), "Simulator must be stopped before the stage can be saved!"
        save_stage(usd_path=usd_path)

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
            physics_dt=self._initial_physics_dt,
            rendering_dt=self._initial_rendering_dt,
            stage_units_in_meters=self._stage_units_in_meters if m.IS_PUBLIC_ISAACSIM else self._initial_stage_units_in_meters,
        )
        self._set_physics_engine_settings()
        self._setup_default_callback_fns()
        self._stage_open_callback = (
            omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._stage_open_callback_fn)
        )

        # Set the viewer camera
        self._set_viewer_camera()

    def close(self):
        """
        Shuts down the iGibson application
        """
        self._app.close()
