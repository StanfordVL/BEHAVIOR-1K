import json
import os
import shutil
import tempfile
from abc import ABC
from itertools import combinations

import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.light_object import LightObject
from omnigibson.objects.object_base import BaseObject
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.robots.robot_base import m as robot_macros
from omnigibson.systems.micro_particle_system import FluidSystem
from omnigibson.systems.system_base import (
    BaseSystem,
    PhysicalParticleSystem,
    VisualParticleSystem,
    create_system_from_metadata,
)
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
from omnigibson.utils.python_utils import (
    Recreatable,
    Registerable,
    Serializable,
    classproperty,
    create_object_from_init_info,
)
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import CollisionAPI, add_asset_to_stage

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Margin between the AABBs of two different scenes.
m.SCENE_MARGIN = 10.0

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
    ):
        """
        Args:
            scene_file (None or str): If specified, full path of JSON file to load (with .json).
                None results in no additional objects being loaded into the scene
        """
        # Store internal variables
        self.scene_file = scene_file
        self._loaded = False  # Whether this scene exists in the stage or not
        self._initialized = False  # Whether this scene has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._registry = None
        self._scene_prim = None
        self._initial_state = None
        self._objects_info = None  # Information associated with this scene

        # Call super init
        super().__init__()

        # Prepare the initialization dicts
        self._init_info = {}
        self._init_state = {}
        self._init_systems = []
        self._task_metadata = None
        self._init_objs = {}

        # If we have any scene file specified, use it to create the objects within it
        if self.scene_file is not None:
            # Grab objects info from the scene file
            with open(self.scene_file, "r") as f:
                scene_info = json.load(f)
            self._init_info = scene_info["objects_info"]["init_info"]
            self._init_state = scene_info["state"]["object_registry"]
            self._init_systems = list(scene_info["state"]["system_registry"].keys())
            self._task_metadata = {}
            try:
                self._task_metadata = scene_info["metadata"]["task"]
            except:
                pass

            # Iterate over all scene info, and instantiate object classes linked to the objects found on the stage
            # accordingly
            self._init_objs = {}
            for obj_name, obj_info in self._init_info.items():
                # Check whether we should load the object or not
                if not self._should_load_object(obj_info=obj_info, task_metadata=self._task_metadata):
                    continue
                # Create object class instance
                obj = create_object_from_init_info(obj_info)
                self._init_objs[obj_name] = obj

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
        return self._registry(key="name", value=f"object_registry")

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
    def robots(self):
        """
        Robots in the scene

        Returns:
            list of BaseRobot: Robot(s) that are currently in this scene
        """
        return list(self.object_registry("category", robot_macros.ROBOT_CATEGORY, []))

    @property
    def systems(self):
        """
        Systems in the scene

        Returns:
            list of BaseSystem: System(s) that are available to use in this scene
        """
        return self.system_registry.objects

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
        return og.sim.scenes.index(self)

    @property
    def initialized(self):
        return self._initialized

    def prebuild(self):
        """
        Prebuild the scene before loading it into the simulator. This is useful for caching the scene USD for faster
        loading times.

        Returns:
            str: Path to the prebuilt USD file
        """
        # Prebuild and cache the scene USD using the objects
        if self.scene_file not in PREBUILT_USDS:
            # Prebuild the scene USD
            log.info(f"Prebuilding scene file {self.scene_file}...")

            # Create a new stage inside the tempdir, named after this scene's file.
            decrypted_fd, usd_path = tempfile.mkstemp(os.path.basename(self.scene_file) + ".usd", dir=og.tempdir)
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

            PREBUILT_USDS[self.scene_file] = usd_path

        # Copy the prebuilt USD to a new path
        decrypted_fd, instance_usd_path = tempfile.mkstemp(os.path.basename(self.scene_file) + ".usd", dir=og.tempdir)
        os.close(decrypted_fd)
        shutil.copyfile(PREBUILT_USDS[self.scene_file], instance_usd_path)
        return instance_usd_path

    def _load(self):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.
        """
        # There's nothing to load for the base scene. Subclasses can implement this method.
        # We simply check that the simulator has a floor plane.
        assert og.sim.floor_plane, "Simulator must have a floor plane if using an empty Scene!"

    def _load_systems(self):
        system_dir = os.path.join(gm.DATASET_PATH, "systems")

        if os.path.exists(system_dir):
            system_names = os.listdir(system_dir)
            for system_name in system_names:
                self.system_registry.add(create_system_from_metadata(system_name=system_name))

    def _load_objects_from_scene_file(self):
        """
        Loads scene objects based on metadata information found in the current USD stage's scene info
        (information stored in the world prim's CustomData)
        """
        # Create desired systems
        for system_name in self._init_systems:
            if gm.USE_GPU_DYNAMICS:
                self.get_system(system_name)
            else:
                log.warning(f"System {system_name} is not supported without GPU dynamics! Skipping...")

        # Add the prebuilt scene USD to the stage
        scene_relative_path = f"/scene_{self.idx}"
        scene_absolute_path = f"/World{scene_relative_path}"

        # If there is already a prim at the absolute path, the scene has been loaded. If not, load the prebuilt scene USD now.
        scene_prim_obj = og.sim.stage.GetPrimAtPath(scene_absolute_path)
        if not scene_prim_obj:
            scene_prim_obj = add_asset_to_stage(asset_path=self.prebuild(), prim_path=scene_absolute_path)

        # Store world prim and load the scene into the simulator
        self._scene_prim = XFormPrim(
            relative_prim_path=scene_relative_path,
            name=f"scene_{self.idx}",
            load_config={"created_manually": True},
        )
        self._scene_prim.load(None)
        assert self._scene_prim.prim_path == scene_prim_obj.GetPath().pathString, "Scene prim path mismatch!"

        # Go through and load all systems.
        self._load_systems()

        # Position the scene prim based on its index in the simulator.
        x, y = T.integer_spiral_coordinates(self.idx)
        self._scene_prim.set_position([x * m.SCENE_MARGIN, y * m.SCENE_MARGIN, 0])

        # Now load the objects with their own logic
        for obj_name, obj in self._init_objs.items():
            # Import into the simulator
            self.add_object(obj)
            # Set the init pose accordingly
            obj.set_local_pose(
                position=self._init_state[obj_name]["root_link"]["pos"],
                orientation=self._init_state[obj_name]["root_link"]["ori"],
            )

        # TODO(parallel-hang): Compute the scene AABB use it for scene placement.
        # aabb_min, aabb_max = lazy.omni.usd.get_context().compute_path_world_bounding_box(scene_absolute_path)
        # place with a z offset initially; after load, find aabb and then reposition (note: origin is not necessarily at the center)

    def _load_metadata_from_scene_file(self):
        """
        Loads metadata from self.scene_file and stores it within the world prim's CustomData
        """
        with open(self.scene_file, "r") as f:
            scene_info = json.load(f)

        # Write the metadata
        for key, data in scene_info.get("metadata", dict()).items():
            og.sim.write_metadata(key=key, data=data)

    def _should_load_object(self, obj_info, task_metadata):
        """
        Helper function to check whether we should load an object given its init_info. Useful for potentially filtering
        objects based on, e.g., their category, size, etc.

        Subclasses can implement additional logic. By default, this returns True

        Args:
            obj_info (dict): Dictionary of object kwargs that will be used to load the object

        Returns:
            bool: Whether this object should be loaded or not
        """
        return True

    def load(self):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.
        """
        # Do not override this function. Override _load instead.

        # Make sure simulator is stopped
        assert og.sim.is_stopped(), "Simulator should be stopped when loading this scene!"

        # Check if scene is already loaded
        if self._loaded:
            raise ValueError("This scene is already loaded.")

        # Create the registry for tracking all objects in the scene
        self._registry = self._create_registry()

        # Go through whatever else loading the scene needs to do.
        self._load()

        # If we have any scene file specified, use it to load the objects within it and also update the initial state
        # and metadata
        if self.scene_file is not None:
            self._load_objects_from_scene_file()
            self._load_metadata_from_scene_file()

        # We're now loaded
        self._loaded = True

        # Always stop the sim if we started it internally
        if not og.sim.is_stopped():
            og.sim.stop()

    def clear(self):
        """
        Clears any internal state before the scene is destroyed
        """
        # Clears systems so they can be re-initialized.
        clear_all_systems()

        # Remove all of the scene's objects.
        og.sim.remove_object(list(self.objects))

        # Remove the scene prim.
        self._scene_prim.remove()

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
        if self.scene_file is None:
            init_state = self.dump_state(serialized=False)
        else:
            with open(self.scene_file, "r") as f:
                scene_info = json.load(f)
            init_state = scene_info["state"]
            self.load_state(init_state, serialized=False)

        self._initial_state = init_state

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
                unique_keys=["name", "prim_path", "uuid"],
            )
        )

        # Add registry for objects
        registry.add(
            obj=SerializableRegistry(
                name=f"object_registry",
                class_types=BaseObject,
                default_key="name",
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

    def add_object(self, obj):
        """
        Add an object to the scene. The scene should already be loaded.

        Args:
            obj (BaseObject): the object to load

        Returns:
            Usd.Prim: the prim of the loaded object if the scene was already loaded, or None if the scene is not loaded
                (in that case, the object is stored to be loaded together with the scene)
        """
        # Load the object.
        prim = obj.load(self)

        # If this object is fixed and is NOT an agent, disable collisions between the fixed links of the fixed objects
        # This is to account for cases such as Tiago, which has a fixed base which is needed for its global base joints
        # We do this by adding the object to our tracked collision groups
        if obj.fixed_base and obj.category != robot_macros.ROBOT_CATEGORY and not obj.visual_only:
            # TODO: Remove structure hotfix once asset collision meshes are fixed!!
            if obj.category in STRUCTURE_CATEGORIES:
                CollisionAPI.add_to_collision_group(col_group="structures", prim_path=obj.prim_path)
            else:
                for link in obj.links.values():
                    CollisionAPI.add_to_collision_group(
                        col_group="fixed_base_root_links" if link == obj.root_link else "fixed_base_nonroot_links",
                        prim_path=link.prim_path,
                    )

        # Add this object to our registry based on its type, if we want to register it
        self.object_registry.add(obj)

        # Run any additional scene-specific logic with the created object
        self._add_object(obj)

        # Inform the simulator that there is a new object here.
        og.sim.post_import_object(obj)
        return prim

    def remove_object(self, obj):
        """
        Method to remove an object from the simulator

        Args:
            obj (BaseObject): Object to remove
        """
        # Remove from the appropriate registry if registered.
        # Sometimes we don't register objects to the object registry during add_object (e.g. particle templates)
        if self.object_registry.object_is_registered(obj):
            self.object_registry.remove(obj)

        # Remove from omni stage
        obj.remove()

    def reset(self):
        """
        Resets this scene
        """
        # Make sure the simulator is playing
        assert og.sim.is_playing(), "Simulator must be playing in order to reset the scene!"

        # Reset the states of all objects (including robots), including (non-)kinematic states and internal variables.
        assert self._initial_state is not None
        self.load_state(self._initial_state)
        og.sim.step_physics()

    @property
    def prim(self):
        """
        Returns:
            XFormPrim: the prim of the scene
        """
        assert self._scene_prim is not None, "Scene prim is not loaded yet!"
        return self._scene_prim

    @property
    def prim_path(self):
        """
        Returns:
            str: the prim path of the scene
        """
        assert self._scene_prim is not None, "Scene prim is not loaded yet!"
        return self.prim.prim_path

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

    def is_system_active(self, system_name):
        system = self.system_registry("name", system_name)
        if system is None:
            return False
        return system.initialized

    def is_visual_particle_system(self, system_name):
        system = self.system_registry("name", system_name)
        assert system is not None, f"System {system_name} not in system registry."
        return issubclass(system, VisualParticleSystem)

    def is_physical_particle_system(self, system_name):
        system = self.system_registry("name", system_name)
        assert system is not None, f"System {system_name} not in system registry."
        return issubclass(system, PhysicalParticleSystem)

    def is_fluid_system(self, system_name):
        system = self.system_registry("name", system_name)
        assert system is not None, f"System {system_name} not in system registry."
        return issubclass(system, FluidSystem)

    def get_system(self, system_name, force_active=True):
        # Make sure scene exists
        assert self.loaded, "Cannot get systems until scene is imported!"
        # If system_name is not in REGISTERED_SYSTEMS, create from metadata
        system = self.system_registry("name", system_name)
        assert system is not None, f"System {system_name} not in system registry."
        if not system.initialized and force_active:
            system.initialize()
        return system

    def get_active_systems(self):
        return [system for system in self.systems if system.initialized]

    def get_random_floor(self):
        """
        Sample a random floor among all existing floor_heights in the scene.
        Most scenes in OmniGibson only have a single floor.

        Returns:
            int: an integer between 0 and self.n_floors-1
        """
        return np.random.randint(0, self.n_floors)

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
            floor: an integer identifying the floor

        Returns:
            int: height of the given floor
        """
        return 0.0

    def update_initial_state(self, state=None):
        """
        Updates the initial state for this scene (which the scene will get reset to upon calling reset())

        Args:
            state (None or dict): If specified, the state to set internally. Otherwise, will set the initial state to
                be the current state
        """
        self._initial_state = self.dump_state(serialized=False) if state is None else state

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
        return self._registry.dump_state(serialized=False)

    def _load_state(self, state):
        # Default state for the scene is from the registry alone
        self._registry.load_state(state=state, serialized=False)

    def _serialize(self, state):
        # Default state for the scene is from the registry alone
        return self._registry.serialize(state=state)

    def deserialize(self, state):
        # Default state for the scene is from the registry alone
        return self._registry._deserialize(state=state)

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
