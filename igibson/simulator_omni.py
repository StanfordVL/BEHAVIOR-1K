# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_ancestral, get_prim_type_name, is_prim_no_delete
from omni.isaac.core.utils.stage import clear_stage
from omni.isaac.dynamic_control import _dynamic_control
import builtins
from pxr import Usd, Sdf, UsdPhysics, PhysxSchema
from omni.kit.viewport import get_viewport_interface
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.loggers import DataLogger
from typing import Optional, List

from igibson.scenes import Scene
from igibson.objects.object_base import BaseObject


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
            :param image_width: width of the camera image
            :param image_height: height of the camera image
            :param vertical_fov: vertical field of view of the camera image in degrees
            :param device_idx: GPU device index to run rendering on
        """

    _world_initialized = False

    def __init__(
            self,
            gravity=9.8,
            physics_dt: float = 1.0 / 60.0,
            rendering_dt: float = 1.0 / 60.0,
            stage_units_in_meters: float = 1.0,
            image_width=1280,
            image_height=720,
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
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_fov = vertical_fov
        self.device_idx = device_idx

        # Store other references to variables that will be initialized later
        self._viewer = None
        self._scene = None
        self.particle_systems = []
        self.frame_count = 0
        self.body_links_awake = 0
        self.first_sync = True          # First sync always sync all objects (regardless of their sleeping states)

        # Initialize viewer
        self._set_physics_engine_settings()
        self._set_viewer_settings()

        # TODO: Fix
        # Set of categories that can be grasped by assisted grasping
        self.object_state_types = {} #get_states_by_dependency_order()

        # TODO: Once objects are in place, uncomment and test this
        # self.assist_grasp_category_allow_list = self.gen_assisted_grasping_categories()
        # self.assist_grasp_mass_thresh = 10.0

    def _reset_variables(self):
        """
        Reset state of internal variables
        """
        self.particle_systems = []
        self.frame_count = 0
        self.body_links_awake = 0
        self.first_sync = True          # First sync always sync all objects (regardless of their sleeping states)

    def _set_physics_engine_settings(self):
        """
        Set the physics engine with specified settings
        """
        self._physics_context.set_gravity(value=-self.gravity)

    def _set_viewer_settings(self):
        """
        Initializes a reference to the viewer in the App, and sets the frame size
        """
        # Store reference to viewer (see https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/reference_python_snippets.html#get-camera-parameters)
        viewport = get_viewport_interface()
        viewport_handle = viewport.get_instance("Viewport")
        self._viewer = viewport.get_viewport_window(viewport_handle)

        # Set viewer frame size
        self._viewer.set_texture_resolution(self.image_width, self.image_height)

    def import_scene(self, scene):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: a scene object to load
        """
        assert isinstance(scene, Scene), "import_scene can only be called with Scene"
        scene.load(self)
        self._scene = scene

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

    def import_object(self, obj):
        """
        Import a non-robot object into the simulator.

        :param obj: BaseObject, a non-robot object to load
        """
        assert isinstance(obj, BaseObject), "import_object can only be called with BaseObject"

        # TODO
        if False:#isinstance(obj, VisualMarker) or isinstance(obj, Particle):
            # Marker objects can be imported without a scene.
            obj.load(self)
        else:
            # Non-marker objects require a Scene to be imported.
            # Load the object in pybullet. Returns a pybullet id that we can use to load it in the renderer
            assert self.scene is not None, "import_object needs to be called after import_scene"
            self.scene.add_object(obj, self, _is_call_from_simulator=True)
    #
    # # TODO
    # def import_robot(self, robot):
    #     """
    #     Import a robot into the simulator.
    #     :param robot: a robot object to load
    #     """
    #     # TODO: Remove this function in favor of unifying with import_object.
    #     assert isinstance(robot, (BaseRobot, BehaviorRobot)), "import_robot can only be called with Robots"
    #     assert self.scene is not None, "import_robot needs to be called after import_scene"
    #
    #     # TODO: remove this if statement after BehaviorRobot refactoring
    #     if isinstance(robot, BaseRobot):
    #         assert (
    #             robot.control_freq is None
    #         ), "control_freq should NOT be specified in robot config. Currently this value is automatically inferred from simulator.render_timestep!"
    #         control_freq = 1.0 / self.render_timestep
    #         robot.control_freq = control_freq
    #
    #     self.scene.add_object(robot, self, _is_call_from_simulator=True)

    def _non_physics_step(self):
        """
        Complete any non-physics steps such as state updates.
        """
        # Step all of the particle systems.
        for particle_system in self.particle_systems:
            particle_system.update(self)

        # Step the object states in global topological order.
        for state_type in self.object_state_types:
            for obj in self.scene.get_objects_with_state(state_type):
                obj.states[state_type].update()

        # TODO
        # # Step the object procedural materials based on the updated object states.
        # for obj in self.scene.get_objects():
        #     if hasattr(obj, "procedural_material") and obj.procedural_material is not None:
        #         obj.procedural_material.update()

    def step(self, render=None):
        """
        Step the simulation at self.render_timestep

        :param render: None or bool, if set, will override internal rendering such that at every timestep rendering
            either occurs or does not occur
        """
        for _ in range(self.n_physics_timesteps_per_render - 1):
            super().step(render=False if render is None else render)

        # Render on final step unless input says otherwise
        super().step(render=True if render is None else render)

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

    def gen_assisted_grasping_categories(self):
        """
        Generate a list of categories that can be grasped using assisted grasping,
        using labels provided in average category specs file.
        """
        assisted_grasp_category_allow_list = set()
        avg_category_spec = get_ig_avg_category_specs()
        for k, v in avg_category_spec.items():
            if v["enable_ag"]:
                assisted_grasp_category_allow_list.add(k)
        return assisted_grasp_category_allow_list

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

    def get_current_tasks(self) -> List[BaseTask]:
        """[summary]

        Returns:
            List[BaseTask]: [description]
        """
        return self._current_tasks

    def get_task(self, name: str) -> BaseTask:
        if name not in self._current_tasks:
            raise Exception("task name {} doesn't exist in the current world tasks.".format(name))
        return self._current_tasks[name]

    def _finalize_scene(self) -> None:
        """[summary]
        """
        if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            self.play()
        self._scene._finalize()
        return

    def clear(self) -> None:
        """Clears the stage leaving the PhysicsScene only if under /World.
        """
        self.scene.clear()
        self._current_tasks = dict()
        self._scene_finalized = False
        self._data_logger = DataLogger()

        def check_deletable_prim(prim_path):
            if is_prim_no_delete(prim_path):
                return False
            if is_prim_ancestral(prim_path):
                return False
            if get_prim_type_name(prim_path=prim_path) == "PhysicsScene":
                return False
            if prim_path == "/World":
                return False
            if prim_path == "/":
                return False
            return True

        clear_stage(predicate=check_deletable_prim)
        return

    def reset(self) -> None:
        """ Resets the stage to its initial state and each object included in the Scene to its default state
            as specified by .set_default_state and the __init__ funcs.

            Note:
            - All tasks should be added before the first reset is called unless a .clear() was called.
            - All articulations should be added before the first reset is called unless a .clear() was called.
            - This method takes care of initializing articulation handles with the first reset called.
            - This will do one step internally regardless
            - calls post_reset on each object in the Scene
            - calls post_reset on each Task

            things like setting pd gains for instance should happend at a Task reset or a Robot reset since
            the defaults are restored after .stop() is called.
        """
        if not self._scene_finalized:
            for task in self._current_tasks.values():
                task.set_up_scene(self.scene)
            self._finalize_scene()
            self._scene_finalized = True
        self.stop()
        for task in self._current_tasks.values():
            task.cleanup()
        self.play()
        self.scene.post_reset()
        for task in self._current_tasks.values():
            task.post_reset()
        return

    async def reset_async(self) -> None:
        """Resets the stage to its initial state and each object included in the Scene to its default state
            as specified by .set_default_state and the __init__ funcs.

            Note:
            - All tasks should be added before the first reset is called unless a .clear() was called.
            - All articulations should be added before the first reset is called unless a .clear() was called.
            - This method takes care of initializing articulation handles with the first reset called.
            - This will do one step internally regardless
            - calls post_reset on each object in the Scene
            - calls post_reset on each Task

            things like setting pd gains for instance should happend at a Task reset or a Robot reset since
            the defaults are restored after .stop() is called.
        """
        if not self._scene_finalized:
            for task in self._current_tasks.values():
                task.set_up_scene(self.scene)
            await self.play_async()
            self._finalize_scene()
            self._scene_finalized = True
        await self.stop_async()
        for task in self._current_tasks.values():
            task.cleanup()
        await self.play_async()
        self._scene.post_reset()
        for task in self._current_tasks.values():
            task.post_reset()
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

    def create_joint(self, prim_path, joint_type, body0=None, body1=None):
        """
        Creates a joint between @body0 and @body1 of specified type @joint_type

        :param prim_path: str, absolute path to where the joint will be created
        :param joint_type: str, type of joint to create. Valid options are:
            "FixedJoint", "Joint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"
        :param body0: str, absolute path to the first body's prim. At least @body0 or @body1 must be specified.
        :param body1: str, absolute path to the second body's prim. At least @body0 or @body1 must be specified.

        :return UsdPhysics.<JointType>: Created joint
        """
        # Make sure we have valid joint_type
        assert joint_type in {"Joint", "FixedJoint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"},\
            f"Invalid joint specified for creation: {joint_type}"

        # Make sure at least body0 or body1 is specified
        assert body0 is not None and body1 is not None, \
            f"At least either body0 or body1 must be specified when creating a joint!"

        # Create the joint
        joint = UsdPhysics.__dict__[joint_type].Define(self.stage, prim_path)

        # Possibly add body0, body1 targets
        if body0 is not None:
            joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
        if body1 is not None:
            joint.GetBody1Rel().SetTargets([Sdf.Path(body1)])

        # Apply this joint
        PhysxSchema.PhysxJointAPI.Apply()

        # Return this joint
        return joint

