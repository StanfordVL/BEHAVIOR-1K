import contextlib
import json
import os
import shutil
import tempfile
from abc import ABC
from pathlib import Path

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.asset_utils
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects.object_base import BaseObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.robots.robot_base import REGISTERED_ROBOTS, m as robot_macros
from omnigibson.systems import Cloth
from omnigibson.systems.micro_particle_system import FluidSystem
from omnigibson.systems.macro_particle_system import MacroParticleSystem
from omnigibson.systems.system_base import (
    BaseSystem,
    PhysicalParticleSystem,
    VisualParticleSystem,
    create_system_from_metadata,
    get_all_system_names,
)
from omnigibson.transition_rules import TransitionRuleAPI
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.constants import STRUCTURAL_DOOR_CATEGORIES
from omnigibson.utils.python_utils import (
    Recreatable,
    Registerable,
    Serializable,
    classproperty,
    create_object_from_init_info,
    get_uuid,
    recursively_convert_to_torch,
)
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import CollisionAPI, add_asset_to_stage

# Create module logger
log = create_module_logger(module_name=__name__)

# Global dicts that will contain mappings
REGISTERED_SCENES = dict()

# Prebuilt USDs that are cached per scene file to speed things up.
PREBUILT_USDS = dict()


class Scene(Serializable, Registerable, Recreatable, ABC):
    """
    Base class for all Scene objects.
    Contains the base functionalities for an arbitrary scene with an arbitrary set of added objects
    """

    def __init__(
        self,
        scene_file=None,
        use_floor_plane=True,
        floor_plane_visible=True,
        floor_plane_color=(1.0, 1.0, 1.0),
        use_skybox=True,
        include_robots=True,
    ):
        """
        Args:
            scene_file (None or str or dict): If specified, full path of JSON file to load (with .json) or the
                pre-loaded scene state from that json.
                None results in no additional objects being loaded into the scene
            use_floor_plane (bool): whether to load a flat floor plane into the simulator
            floor_plane_visible (bool): whether to render the additionally added floor plane
            floor_plane_color (3-array): if @floor_plane_visible is True, this determines the (R,G,B) color assigned
                to the generated floor plane
            use_skybox (bool): whether to load a skybox into the simulator
            include_robots (bool): whether to also include the robot(s) defined in the scene
        """
        # Store internal variables
        self.scene_file = scene_file
        self._loaded = False  # Whether this scene exists in the stage or not
        self._initialized = False  # Whether this scene has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._registry = None
        self._scene_prim = None
        self._initial_file = None
        self._objects_info = None  # Information associated with this scene
        self._idx = None
        self._use_floor_plane = use_floor_plane
        self._floor_plane_visible = floor_plane_visible
        self._floor_plane_color = floor_plane_color
        self._use_skybox = use_skybox
        self._transition_rule_api = None
        self._available_systems = None
        self._pose_info = None
        self._updated_state_objects = None
        self._include_robots = include_robots
        self._task_metadata = {}

        # Call super init
        super().__init__()

        # Prepare the initialization dicts
        self._init_objs = {}
        self._init_state = {}
        self._init_systems = []

        # If we have any scene file specified, use it to create the objects within it
        if self.scene_file is not None:
            # Grab objects info from the scene file
            if isinstance(self.scene_file, str):
                with open(self.scene_file, "r") as f:
                    scene_info = json.load(f)
            else:
                scene_info = self.scene_file
            init_info = scene_info["objects_info"]["init_info"]
            # TODO: Remove this backwards-compatibility once newer RC is released
            self._init_state = (
                scene_info["state"]["registry"]["object_registry"]
                if "registry" in scene_info["state"]
                else scene_info["state"]["object_registry"]
            )
            self._init_systems = (
                list(scene_info["state"]["registry"]["system_registry"].keys())
                if "registry" in scene_info["state"]
                else list(scene_info["state"]["system_registry"].keys())
            )
            task_metadata = (
                scene_info["metadata"]["task"] if "metadata" in scene_info and "task" in scene_info["metadata"] else {}
            )

            # Iterate over all scene info, and instantiate object classes linked to the objects found on the stage accordingly
            for obj_name, obj_info in init_info.items():
                # Check whether we should load the object or not
                if not self._should_load_object(obj_info=obj_info, task_metadata=task_metadata):
                    continue
                # Create object class instance
                obj = create_object_from_init_info(obj_info)
                self._init_objs[obj_name] = obj

            # Store presampled robot poses from metadata
            for key, data in task_metadata.items():
                self.write_task_metadata(key=key, data=data)

    @property
    def registry(self):
        """
        Returns:
            SerializableRegistry: Master registry containing sub-registries of objects, robots, systems, etc.
        """
        return self._registry

    @property
    def object_registry(self):
        """
        Returns:
            SerializableRegistry: Object registry containing all active standalone objects in the scene
        """
        return self._registry(key="name", value="object_registry")

    @property
    def system_registry(self):
        """
        Returns:
            SerializableRegistry: System registry containing all systems in the scene (e.g.: water, dust, etc.)
        """
        return self._registry(key="name", value="system_registry")

    @property
    def objects(self):
        """
        Get the objects in the scene.

        Returns:
            list of BaseObject: Standalone object(s) that are currently in this scene
        """
        return self.object_registry.objects

    @property
    def updated_state_objects(self):
        """
        Returns:
            set of StatefulObject: set of stateful objects in the scene that have had at least a single object state
                updated since the last simulator's non_physics_step()
        """
        return self._updated_state_objects

    @property
    def robots(self):
        """
        Robots in the scene

        Returns:
            list of BaseRobot: Robot(s) that are currently in this scene
        """
        return list(sorted(self.object_registry("category", robot_macros.ROBOT_CATEGORY, []), key=lambda x: x.name))

    @property
    def systems(self):
        """
        Active systems in the scene

        Returns:
            list of BaseSystem: Active system(s) in this scene
        """
        return self.system_registry.objects

    @property
    def available_systems(self):
        """
        Available systems in the scene

        Returns:
            dict: Maps all system names to corresponding systems that are available to use in this scene
        """
        return self._available_systems

    @property
    def object_registry_unique_keys(self):
        """
        Returns:
            list of str: Keys with which to index into the object registry. These should be valid public attributes of
                prims that we can use as unique IDs to reference prims, e.g., prim.prim_path, prim.name, etc.
        """
        return ["name", "prim_path", "uuid"]

    @property
    def object_registry_group_keys(self):
        """
        Returns:
            list of str: Keys with which to index into the object registry. These should be valid public attributes of
                prims that we can use as grouping IDs to reference prims, e.g., prim.in_rooms
        """
        return ["prim_type", "states", "category", "fixed_base", "in_rooms", "abilities"]

    @property
    def loaded(self):
        return self._loaded

    @property
    def idx(self):
        """Index of this scene in the simulator. Should not change."""
        assert self._idx is not None, "This scene is not loaded yet!"
        return self._idx

    @property
    def initialized(self):
        return self._initialized

    @property
    def use_floor_plane(self):
        return self._use_floor_plane

    @property
    def transition_rule_api(self):
        return self._transition_rule_api

    def clear_updated_objects(self):
        self._updated_state_objects = set()

    def prebuild(self):
        """
        Prebuild the scene USD before loading it into the simulator. This is useful for caching the scene USD for faster
        loading times.

        Returns:
            str: Path to the prebuilt USD file
        """
        # Prebuild and cache the scene USD using the objects
        if isinstance(self.scene_file, str):
            scene_file_path = self.scene_file
        else:
            # The scene file is a dict, so write it to disk directly
            scene_file_str = json.dumps(self.scene_file, cls=TorchEncoder, indent=4)
            scene_file_hash = get_uuid(scene_file_str, deterministic=True)
            scene_file_path = os.path.join(og.tempdir, f"scene_file_{scene_file_hash}.json")
            with open(scene_file_path, "w+") as f:
                json.dump(self.scene_file, f, cls=TorchEncoder, indent=4)

        if scene_file_path not in PREBUILT_USDS:
            # Prebuild the scene USD
            log.info(f"Prebuilding scene file {scene_file_path}...")

            # Create a new stage inside the tempdir, named after this scene's file.
            decrypted_fd, usd_path = tempfile.mkstemp(os.path.basename(scene_file_path) + ".usd", dir=og.tempdir)
            os.close(decrypted_fd)
            stage = lazy.pxr.Usd.Stage.CreateNew(usd_path)

            # Create the world prim and make it the default
            world_prim = stage.DefinePrim("/World", "Xform")
            stage.SetDefaultPrim(world_prim)

            # Iterate through all objects and add them to the stage
            for obj_name, obj in self._init_objs.items():
                obj.prebuild(stage)

            stage.Save()
            del stage

            PREBUILT_USDS[scene_file_path] = usd_path

        # Copy the prebuilt USD to a new path
        decrypted_fd, instance_usd_path = tempfile.mkstemp(os.path.basename(scene_file_path) + ".usd", dir=og.tempdir)
        os.close(decrypted_fd)
        shutil.copyfile(PREBUILT_USDS[scene_file_path], instance_usd_path)
        return instance_usd_path

    def _load(self):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.
        """
        # There's nothing to load for the base scene. Subclasses can implement this method.
        # Do nothing here
        pass

    def _load_systems(self):
        system_dir = os.path.join(get_dataset_path("behavior-1k-assets"), "systems")

        available_systems = (
            {
                system_name: create_system_from_metadata(system_name=system_name)
                for system_name in get_all_system_names()
            }
            if os.path.exists(system_dir)
            else dict()
        )

        # Manually add cloth system since it is a special system that doesn't have any corresponding directory in
        # the B1K database
        cloth_system = Cloth(name="cloth")
        available_systems["cloth"] = cloth_system
        self._available_systems = available_systems

    def _load_scene_prim_with_objects(self, last_scene_edge, initial_scene_prim_z_offset, scene_margin):
        """
        Loads scene objects based on metadata information found in the current USD stage's scene info
        (information stored in the world prim's CustomData)
        """
        # Add the prebuilt scene USD to the stage
        scene_relative_path = f"/scene_{self.idx}"
        scene_absolute_path = f"/World{scene_relative_path}"

        # If there is already a prim at the absolute path, the scene has been loaded. If not, load the prebuilt scene USD now.
        if self.scene_file is not None:
            scene_prim_obj = og.sim.stage.GetPrimAtPath(scene_absolute_path)
            if not scene_prim_obj:
                scene_prim_obj = add_asset_to_stage(asset_path=self.prebuild(), prim_path=scene_absolute_path)

        # Store world prim and load the scene into the simulator
        self._scene_prim = XFormPrim(
            relative_prim_path=scene_relative_path,
            name=f"scene_{self.idx}",
        )
        self._scene_prim.load(None)
        if self.scene_file is not None:
            assert self._scene_prim.prim_path == scene_prim_obj.GetPath().pathString, "Scene prim path mismatch!"

        # Go through and load all systems.
        self._load_systems()

        # Create desired systems
        for system_name in self._init_systems:
            system = self.get_system(system_name, force_init=False)
            if gm.USE_GPU_DYNAMICS or isinstance(system, MacroParticleSystem):
                self.get_system(system_name, force_init=True)
            else:
                log.warning(f"System {system_name} is not supported without GPU dynamics! Skipping...")

        # Position the scene prim initially at a z offset to avoid collision
        self._scene_prim.set_position_orientation(
            position=th.tensor([0, 0, initial_scene_prim_z_offset if self.idx != 0 else 0])
        )

        # Now load the objects with their own logic
        with og.sim.adding_objects(objs=self._init_objs.values()):
            for obj_name, obj in self._init_objs.items():
                # Import into the simulator
                self.add_object(obj, _batched_call=True)
                # Set the init pose accordingly
                obj.set_position_orientation(
                    position=self._init_state[obj_name]["root_link"]["pos"],
                    orientation=self._init_state[obj_name]["root_link"]["ori"],
                )

        # Position the scene prim based on the last scene's right edge
        if self.idx != 0:
            aabb_min, aabb_max = lazy.omni.usd.get_context().compute_path_world_bounding_box(scene_absolute_path)
            left_edge_to_center = -aabb_min[0]
            self._scene_prim.set_position_orientation(
                position=[last_scene_edge + scene_margin + left_edge_to_center, 0, 0]
            )
            new_scene_edge = last_scene_edge + scene_margin + (aabb_max[0] - aabb_min[0])
        else:
            aabb_min, aabb_max = lazy.omni.usd.get_context().compute_path_world_bounding_box(scene_absolute_path)
            new_scene_edge = aabb_max[0]

        return new_scene_edge

    def _load_metadata_from_scene_file(self):
        """
        Loads metadata from self.scene_file and stores it within the world prim's CustomData
        """
        if isinstance(self.scene_file, str):
            with open(self.scene_file, "r") as f:
                scene_info = json.load(f)
        else:
            scene_info = self.scene_file

        # Write the metadata
        for key, data in scene_info.get("metadata", dict()).items():
            self.write_task_metadata(key=key, data=data)

    def _should_load_object(self, obj_info, task_metadata):
        """
        Helper function to check whether we should load an object given its init_info. Useful for potentially filtering
        objects based on, e.g., their category, size, etc.

        By default, this checks whether robot should be loaded.
        Subclasses should call super and implement additional logic.

        Args:
            obj_info (dict): Dictionary of object kwargs that will be used to load the object

        Returns:
            bool: Whether this object should be loaded or not
        """
        # Check whether this is an agent and we allow agents
        return self._include_robots or obj_info["class_name"] not in REGISTERED_ROBOTS

    def load(self, idx, **kwargs):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.
        Do not override this function. Override _load instead.
        """
        # Make sure simulator is stopped
        assert og.sim.is_stopped(), "Simulator should be stopped when loading this scene!"

        # Check if scene is already loaded
        if self._loaded:
            raise ValueError("This scene is already loaded.")

        self._idx = idx
        self.clear_updated_objects()

        # Create the registry for tracking all objects in the scene
        self._registry = self._create_registry()

        # Load floor plane and skybox
        if self.use_floor_plane:
            og.sim.add_ground_plane(
                floor_plane_visible=self._floor_plane_visible, floor_plane_color=self._floor_plane_color
            )
        if self._use_skybox:
            og.sim.add_skybox()

        # Go through whatever else loading the scene needs to do.
        self._load()

        # We're now loaded
        self._loaded = True

        # If we have any scene file specified, use it to load the objects within it and also update the initial state
        # and metadata
        new_scene_edge = self._load_scene_prim_with_objects(**kwargs)
        if self.scene_file is not None:
            self._load_metadata_from_scene_file()

        # Cache this scene's pose
        pos_ori = self._scene_prim.get_position_orientation()
        pose = T.pose2mat(pos_ori)
        self._pose_info = {
            "pos_ori": pos_ori,
            "pose": T.pose2mat(pos_ori),
            "pose_inv": th.linalg.inv_ex(pose).inverse,
        }

        if gm.ENABLE_TRANSITION_RULES:
            assert gm.ENABLE_OBJECT_STATES, "Transition rules require object states to be enabled!"
            self._transition_rule_api = TransitionRuleAPI(scene=self)

        # Always stop the sim if we started it internally
        if not og.sim.is_stopped():
            og.sim.stop()

        return new_scene_edge

    def clear(self):
        """
        Clears any internal state before the scene is destroyed
        """
        # Clears systems so they can be re-initialized.
        for system in self.active_systems.values():
            self.clear_system(system_name=system.name)

        # Remove all of the scene's objects.
        og.sim.batch_remove_objects(list(self.objects))

        # Remove the scene prim.
        self._scene_prim.remove()

        if gm.ENABLE_TRANSITION_RULES:
            # Clear the transition rule API
            self._transition_rule_api.clear()

    def _initialize(self):
        """
        Initializes state of this scene and sets up any references necessary post-loading. Should be implemented by
        sub-class for extended utility
        """
        pass

    def initialize(self):
        """
        Initializes state of this scene and sets up any references necessary post-loading. Subclasses should
        implement / extend the _initialize() method.
        """
        assert not self._initialized, "Scene can only be initialized once! (It is already initialized)"
        assert og.sim.is_playing(), "Simulator must be playing in order to initialize the scene!"
        self._initialize()

        # Grab relevant objects info
        self.update_objects_info()
        self.wake_scene_objects()

        self._initialized = True

        # Store initial state, which may be loaded from a scene file if specified
        if self.scene_file is not None:
            if isinstance(self.scene_file, str):
                with open(self.scene_file, "r") as f:
                    scene_info = json.load(f)
            else:
                scene_info = self.scene_file
            init_state = scene_info["state"]
            init_state = recursively_convert_to_torch(init_state)
            # In VectorEnvironment, the scene pose loaded from the file should be updated
            init_state["pos"], init_state["ori"] = self._pose_info["pos_ori"]
            for obj_name, obj_info in init_state["registry"]["object_registry"].items():
                # Convert the pose to be in the scene's coordinate frame
                pos, ori = obj_info["root_link"]["pos"], obj_info["root_link"]["ori"]
                # apply scene pose to all objects in this scene
                obj_info["root_link"]["pos"], obj_info["root_link"]["ori"] = T.pose_transform(
                    *T.mat2pose(self.pose), pos, ori
                )
            # TODO: also handle system registry here
            self.load_state(init_state, serialized=False)
        self._initial_file = self.save(as_dict=True)

    def _create_registry(self):
        """
        Creates the internal registry used for tracking all objects

        Returns:
            SerializableRegistry: registry for tracking all objects
        """

        # Create meta registry and populate with internal registries for robots, objects, and systems
        registry = SerializableRegistry(
            name=f"scene_registry_{self.idx}",
            class_types=SerializableRegistry,
        )

        # Add registry for systems -- this is already created externally, so we just update it and pull it directly
        registry.add(
            obj=SerializableRegistry(
                name="system_registry",
                class_types=BaseSystem,
                default_key="name",
                hash_key="uuid",
                unique_keys=["name", "prim_path", "uuid"],
            )
        )

        # Add registry for objects
        registry.add(
            obj=SerializableRegistry(
                name="object_registry",
                class_types=BaseObject,
                default_key="name",
                hash_key="uuid",
                unique_keys=self.object_registry_unique_keys,
                group_keys=self.object_registry_group_keys,
            )
        )

        return registry

    def wake_scene_objects(self):
        """
        Force wakeup sleeping objects
        """
        for obj in self.objects:
            obj.wake()

    def get_objects_with_state(self, state):
        """
        Get the objects with a given state in the scene.

        Args:
            state (BaseObjectState): state of the objects to get

        Returns:
            set: all objects with the given state
        """
        return self.object_registry("states", state, set())

    def get_objects_with_state_recursive(self, state):
        """
        Get the objects with a given state and its subclasses in the scene.

        Args:
            state (BaseObjectState): state of the objects to get

        Returns:
            set: all objects with the given state and its subclasses
        """
        objs = set()
        states = {state}
        while states:
            next_states = set()
            for state in states:
                objs |= self.object_registry("states", state, set())
                next_states |= set(state.__subclasses__())
            states = next_states
        return objs

    def _add_object(self, obj):
        """
        Add an object to the scene's internal object tracking mechanisms.

        Note that if the scene is not loaded, it should load this added object alongside its other objects when
        scene.load() is called. The object should also be accessible through scene.objects.

        Args:
            obj (BaseObject): the object to load into the simulator
        """
        pass

    def add_object(self, obj, register=True, _batched_call=False):
        """
        Add an object to the scene. The scene should already be loaded.

        Args:
            obj (BaseObject): the object to load
            register (bool): Whether to register @obj internally in the scene object registry or not, as well as run
                additional scene-specific logic in addition to the obj being loaded
            _batched_call (bool): Whether this is from a batched call or not. If True, will avoid running
                a context externally. In general, this should NOT be explicitly set by the user
        """
        cxt = contextlib.nullcontext() if _batched_call else og.sim.adding_objects(objs=[obj])
        with cxt:
            # Make sure all objects in this scene are uniquely named
            assert (
                obj.name not in self.object_registry.object_names
            ), f"Object with name {obj.name} already exists in scene!"

            # Load the object.
            obj.load(self)

            if register:
                # If this object is fixed and is NOT an agent, disable collisions between the fixed links of the fixed objects
                # This is to account for cases such as Tiago, which has a fixed base which is needed for its global base joints
                # We do this by adding the object to our tracked collision groups
                if obj.fixed_base and obj.category != robot_macros.ROBOT_CATEGORY and not obj.visual_only:
                    obj_fixed_links = obj.get_fixed_link_names_in_subtree()
                    for link_name, link in obj.links.items():
                        if link_name in obj_fixed_links:
                            CollisionAPI.add_to_collision_group(
                                col_group=("fixed_base_fixed_links"),
                                prim_path=link.prim_path,
                            )
                        elif obj.category in STRUCTURAL_DOOR_CATEGORIES:
                            CollisionAPI.add_to_collision_group(
                                col_group=("structural_doors"),
                                prim_path=link.prim_path,
                            )

                # Add this object to our registry based on its type, if we want to register it
                self.object_registry.add(obj)

                # Run any additional scene-specific logic with the created object
                self._add_object(obj)

    def remove_object(self, obj, _batched_call=False):
        """
        Method to remove an object from the simulator

        Args:
            obj (BaseObject): Object to remove
            _batched_call (bool): Whether this is from a batched call or not. If True, will avoid running
                a context externally. In general, this should NOT be explicitly set by the user
        """
        cxt = contextlib.nullcontext() if _batched_call else og.sim.removing_objects(objs=[obj])
        with cxt:
            # Remove from the appropriate registry if registered.
            # Sometimes we don't register objects to the object registry during add_object (e.g. particle templates)
            if self.object_registry.object_is_registered(obj):
                self.object_registry.remove(obj)

            # Remove from omni stage
            obj.remove()

    def reset(self, hard=True):
        """
        Resets this scene

        Args:
            hard (bool): If set, will force the set of active objects currently in the sim to match
                the specified objects stored in self._initial_file. Otherwise, will only load the kinematic and semantic
                state for any objects that are currently in the sim, ignoring any additional / missing objects
        """
        # Make sure the simulator is playing
        assert og.sim.is_playing(), "Simulator must be playing in order to reset the scene!"

        # Reset the states of all objects (including robots), including (non-)kinematic states and internal variables.
        # This also forces the scene to align with the correct set of initial objects / systems, in case any
        # were removed / added during runtime
        assert self._initial_file is not None
        if hard:
            self.restore(scene_file=self._initial_file)
        else:
            self.load_state(self._initial_file["state"], serialized=False)

        og.sim.step_physics()

    def save(self, json_path=None, as_dict=False):
        """
        Saves the current scene environment to @json_path.

        Args:
            json_path (None or str): Full path of JSON file to save (should end with .json), which will
                contain information to recreate the current scene, if specified. If None, will return the raw JSON
                string instead.
            as_dict (bool): If set and @json_path is None, will return the saved environment as a dictionary instead
                of encoded json strings

        Returns:
            None or str or dict: If @json_path is None, returns dumped json string (or dict if @as_dict is set).
                Else, None
        """
        # Make sure the sim is not stopped, since we need to grab joint states
        assert not og.sim.is_stopped(), "Simulator cannot be stopped when saving scene!"

        if json_path is not None:
            if not json_path.endswith(".json"):
                log.error(f"You have to define the full json_path to save the scene to. Got: {json_path}")
                return

        # Update scene info
        self.update_objects_info()

        # Dump saved current state and also scene init info
        scene_info = {
            # TODO: Use these to verify compatibility at load time.
            "versions": {
                "omnigibson": {
                    "version": omnigibson.utils.asset_utils.get_omnigibson_version(),
                    "git_hash": omnigibson.utils.asset_utils.get_omnigibson_git_hash(),
                },
                "bddl": {
                    "version": omnigibson.utils.asset_utils.get_bddl_version(),
                    "git_hash": omnigibson.utils.asset_utils.get_bddl_git_hash(),
                },
                "behavior-1k-assets": {
                    "version": omnigibson.utils.asset_utils.get_behavior_1k_assets_version(),
                },
                "omnigibson-robot-assets": {
                    "version": omnigibson.utils.asset_utils.get_omnigibson_robot_asset_version(),
                    "git_hash": omnigibson.utils.asset_utils.get_omnigibson_robot_asset_git_hash(),
                },
            },
            "metadata": self._task_metadata,
            "state": self.dump_state(serialized=False),
            "init_info": self.get_init_info(),
            "objects_info": self.get_objects_info(),
        }

        # Write this to the json file
        if json_path is None:
            return scene_info if as_dict else json.dumps(scene_info, cls=TorchEncoder, indent=4)

        else:
            Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
            with open(json_path, "w+") as f:
                json.dump(scene_info, f, cls=TorchEncoder, indent=4)

            log.info(f"Scene {self.idx} saved to {json_path}.")

    def restore(self, scene_file, update_initial_file=False):
        """
        Restores this scene given @scene_file

        Args:
            scene_file (str or dict): Full path of either JSON file or loaded scene file to load, which contains
                information to recreate a scene. Should be the output of self.save()
            update_initial_file (bool): Whether to update this scene's initial file or not
        """
        # Make sure the sim is not stopped, since we need to grab joint states
        assert not og.sim.is_stopped(), "Simulator cannot be stopped when restoring scene!"

        if isinstance(scene_file, str):
            if not scene_file.endswith(".json"):
                log.error(f"You have to define the full json_path to load from. Got: {scene_file}")
                return

            # Load the info from the json
            with open(scene_file, "r") as f:
                scene_info = json.load(f)
        else:
            scene_info = scene_file
        init_info = scene_info["init_info"]
        # The saved state are lists, convert them to torch tensors
        state = recursively_convert_to_torch(scene_info["state"])

        # Recover metadata
        for key, data in scene_info.get("metadata", dict()).items():
            self.write_task_metadata(key=key, data=data)

        # Make sure the class type is the same
        if self.__class__.__name__ != init_info["class_name"]:
            log.error(
                f"Got mismatch in scene type: current is type {self.__class__.__name__}, trying to load type {init_info['class_name']}"
            )

        # Synchronize systems -- we need to check for pruning currently-existing systems,
        # as well as creating any non-existing systems
        current_systems = set(self.active_systems.keys())
        load_systems = set(scene_info["state"]["registry"]["system_registry"].keys())
        systems_to_remove = current_systems - load_systems
        systems_to_add = load_systems - current_systems
        for name in systems_to_remove:
            self.clear_system(name)
        for name in systems_to_add:
            self.get_system(name, force_init=True)

        current_obj_names = set(self.object_registry.get_dict("name").keys())
        load_obj_names = set(scene_info["objects_info"]["init_info"].keys())

        objs_to_remove = current_obj_names - load_obj_names
        objs_to_add = load_obj_names - current_obj_names

        # Delete any extra objects that currently exist in the scene stage
        objects_to_remove = [self.object_registry("name", obj_to_remove) for obj_to_remove in objs_to_remove]
        og.sim.batch_remove_objects(objects_to_remove)

        # Add any extra objects that do not currently exist in the scene stage
        objects_to_add = [
            create_object_from_init_info(scene_info["objects_info"]["init_info"][obj_to_add])
            for obj_to_add in objs_to_add
        ]
        og.sim.batch_add_objects(objects_to_add, scenes=[self] * len(objects_to_add))

        # Load state
        self.load_state(state, serialized=False)

        if update_initial_file:
            self.update_initial_file(scene_file=scene_file)

    def get_task_metadata(self, key):
        return self._task_metadata.get(key, None)

    def write_task_metadata(self, key, data):
        self._task_metadata[key] = data

    def get_position_orientation(self):
        """
        Get the position and orientation of the scene

        Returns:
            2-tuple:
                - th.Tensor: (3,) position of the scene
                - th.Tensor: (4,) orientation of the scene
        """
        return self._pose_info["pos_ori"]

    def set_position_orientation(self, position=None, orientation=None):
        """
        Set the position and orientation of the scene

        Args:
            position (th.Tensor): (3,) position of the scene
            orientation (th.Tensor): (4,) orientation of the scene
        """
        self._scene_prim.set_position_orientation(position=position, orientation=orientation)
        # Need to update sim here -- this is because downstream setters called immediately may not be respected,
        # e.g. during load_state() call when specific objects have just been added to the simulator in this scene
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        # Update the cached pose and inverse pose
        pos_ori = self._scene_prim.get_position_orientation()
        pose = T.pose2mat(pos_ori)
        self._pose_info = {
            "pos_ori": pos_ori,
            "pose": T.pose2mat(pos_ori),
            "pose_inv": th.linalg.inv_ex(pose).inverse,
        }

    @property
    def prim_path(self):
        """
        Returns:
            str: the prim path of the scene
        """
        assert self._scene_prim is not None, "Scene prim is not loaded yet!"
        return self._scene_prim.prim_path

    @property
    def n_floors(self):
        """
        Returns:
            int: Number of floors in this scene
        """
        # Default is a single floor
        return 1

    @property
    def n_objects(self):
        """
        Returns:
            int: number of objects
        """
        return len(self.objects)

    @property
    def fixed_objects(self):
        """
        Returns:
            dict: Keyword-mapped objects that are fixed in the scene, IGNORING any robots.
                Maps object name to their object class instances (DatasetObject)
        """
        return {
            obj.name: obj
            for obj in self.object_registry("fixed_base", True, default_val=[])
            if obj.category != robot_macros.ROBOT_CATEGORY
        }

    @property
    def pose(self):
        """
        Returns:
            th.Tensor: (4,4) homogeneous transformation matrix representing this scene's global pose
        """
        return self._pose_info["pose"]

    @property
    def pose_inv(self):
        """
        Returns:
            th.Tensor: (4,4) homogeneous transformation matrix representing this scene's global inverse pose
        """
        return self._pose_info["pose_inv"]

    def convert_world_pose_to_scene_relative(self, position, orientation):
        """
        Convert a world pose to a scene-relative pose.

        Args:
            position (th.Tensor): (3,) position in world frame
            orientation (th.Tensor): (4,) orientation in world frame

        Returns:
            2-tuple:
                - th.Tensor: (3,) position in scene frame
                - th.Tensor: (4,) orientation in scene frame
        """
        return T.mat2pose(self.pose_inv @ T.pose2mat((position, orientation)))

    def convert_scene_relative_pose_to_world(self, position, orientation):
        """
        Convert a scene-relative pose to a world pose.

        Args:
            position (th.Tensor): (3,) position in scene frame
            orientation (th.Tensor): (4,) orientation in scene frame

        Returns:
            2-tuple:
                - th.Tensor: (3,) position in world frame
                - th.Tensor: (4,) orientation in world frame
        """
        return T.mat2pose(self.pose @ T.pose2mat((position, orientation)))

    def is_system_active(self, system_name):
        return self.get_system(system_name, force_init=False).initialized

    def is_visual_particle_system(self, system_name):
        return isinstance(self.get_system(system_name, force_init=False), VisualParticleSystem)

    def is_physical_particle_system(self, system_name):
        return isinstance(self.get_system(system_name, force_init=False), PhysicalParticleSystem)

    def is_fluid_system(self, system_name):
        return isinstance(self.get_system(system_name, force_init=False), FluidSystem)

    def get_system(self, system_name, force_init=True):
        """
        Grab the system @system_name, and optionally initialize it if @force_init is set

        Args:
            system_name (str): Name of the system to grab
            force_init (bool): Whether to force the system to be initialized and added to set of active_systems
                if not already

        Returns:
            BaseSystem: Requested system
        """
        # Make sure scene exists
        assert self.loaded, "Cannot get systems until scene is imported!"
        assert system_name in self._available_systems, f"System {system_name} is not a valid system name"
        # If system is not initialized, initialize and add it to our registry
        system = self._available_systems[system_name]
        if not system.initialized and force_init:
            system.initialize(scene=self)
            self.system_registry.add(system)
        return system

    def clear_system(self, system_name):
        """
        Clear the system @system_name and remove it from our set of active systems

        Args:
            system_name (str): Name of the system to remove
        """
        system = self.system_registry("name", system_name)
        if system is not None:
            # Remove from system registry and clear
            self.system_registry.remove(system)
            system.clear()

    @property
    def active_systems(self):
        return {system.name: system for system in self.systems if not isinstance(system, Cloth)}

    def get_random_floor(self):
        """
        Sample a random floor among all existing floor_heights in the scene.
        Most scenes in OmniGibson only have a single floor.

        Returns:
            int: an integer between 0 and self.n_floors-1
        """
        return th.randint(0, self.n_floors)

    def get_random_point(self, floor=None, reference_point=None, robot=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.
        If @reference_point is given, sample a point in the same connected component as the previous point.

        Args:
            floor (None or int): floor number. None means the floor is randomly sampled
                                 Warning: if @reference_point is given, @floor must be given;
                                          otherwise, this would lead to undefined behavior
            reference_point (3-array): (x,y,z) if given, sample a point in the same connected component as this point

        Returns:
            2-tuple:
                - int: floor number. This is the sampled floor number if @floor is None
                - 3-array: (x,y,z) randomly sampled point
        """
        raise NotImplementedError()

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None):
        """
        Get the shortest path from one point to another point.

        Args:
            floor (int): floor number
            source_world (2-array): (x,y) 2D source location in world reference frame (metric)
            target_world (2-array): (x,y) 2D target location in world reference frame (metric)
            entire_path (bool): whether to return the entire path
            robot (None or BaseRobot): if given, erode the traversability map to account for the robot's size

        Returns:
            2-tuple:
                - (N, 2) array: array of path waypoints, where N is the number of generated waypoints
                - float: geodesic distance of the path
        """
        raise NotImplementedError()

    def get_floor_height(self, floor=0):
        """
        Get the height of the given floor. Default is 0.0, since we only have a single floor

        Args:
            floor (int): an integer identifying the floor

        Returns:
            int: height of the given floor
        """
        return 0.0

    def update_initial_file(self, scene_file=None):
        """
        Updates the initial scene file for this scene (which the scene will get reset to upon calling reset())

        Args:
            scene_file (None or str or dict): If specified, the state to set internally. Can be a full path to the scene
                file or the already-loaded dictionary equivalent. This should be the output of self.save(). Otherwise,
                will set the initial file to be the current scene file that will be cached immediately
        """
        if scene_file is None:
            scene_file = self.save(as_dict=True)
        else:
            if isinstance(scene_file, str):
                with open(scene_file, "r") as f:
                    scene_file = json.load(f)
            else:
                assert isinstance(
                    scene_file, dict
                ), f"Expected scene_file to be a dictionary, but got: {type(scene_file)}"

        scene_file["state"] = recursively_convert_to_torch(scene_file["state"])
        self._initial_file = scene_file

    def update_objects_info(self):
        """
        Updates the scene-relevant information and saves it to the active USD. Useful for reloading a scene directly
        from a saved USD in this format.
        """
        # Save relevant information

        # Iterate over all objects and save their init info
        init_info = {obj.name: obj.get_init_info() for obj in self.object_registry.objects}

        # Compose as single dictionary and store internally
        self._objects_info = dict(init_info=init_info)

    def get_objects_info(self):
        """
        Stored information, if any, for this scene. Structure is:

            "init_info":
                "<obj0>": <obj0> init kw/args
                ...
                "<robot0>": <robot0> init kw/args
                ...

        Returns:
            None or dict: If it exists, nested dictionary of relevant objects' information
        """
        return self._objects_info

    def _dump_state(self):
        # Default state for the scene is from the registry alone
        pos, ori = self.get_position_orientation()
        return {
            "pos": pos,
            "ori": ori,
            "registry": self._registry.dump_state(serialized=False),
        }

    def _load_state(self, state):
        # Load scene state, then registry
        # TODO: Remove backwards compatible check once new scene RC is updated
        if "pos" in state:
            self.set_position_orientation(position=state["pos"], orientation=state["ori"])
            self._registry.load_state(state=state["registry"], serialized=False)
        else:
            self._registry.load_state(state=state, serialized=False)

    def serialize(self, state):
        return th.cat([state["pos"], state["ori"], self._registry.serialize(state=state["registry"])])

    def deserialize(self, state):
        state_dict = {
            "pos": state[:3],
            "ori": state[3:7],
        }
        idx = 7
        registry_dict, registry_idx = self._registry.deserialize(state=state[idx:])
        state_dict["registry"] = registry_dict
        return state_dict, idx + registry_idx

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_SCENES
        return REGISTERED_SCENES

    @classmethod
    def modify_init_info_for_restoring(cls, init_info):
        """
        Helper function to modify a given init info for restoring a scene from corresponding scene info.
        Note that this function modifies IN-PLACE!

        Args:
            init_info (dict): Information for this scene from @self.get_init_info()
        """
        # Default is pass
        pass
