import json
from abc import ABC
from collections import OrderedDict
from itertools import combinations
from omni.isaac.core.objects.ground_plane import GroundPlane
import numpy as np
import omnigibson as og
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.python_utils import classproperty, Serializable, Registerable, Recreatable, \
    create_object_from_init_info
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.objects.object_base import BaseObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.systems import SYSTEMS_REGISTRY
from omnigibson.robots.robot_base import m as robot_macros

# Global dicts that will contain mappings
REGISTERED_SCENES = OrderedDict()


class Scene(Serializable, Registerable, Recreatable, ABC):
    """
    Base class for all Scene objects.
    Contains the base functionalities for an arbitary scene with an arbitrary set of added objects
    """
    def __init__(
            self,
            scene_file=None,
            use_floor_plane=True,
            floor_plane_visible=True,
            floor_plane_color=(1.0, 1.0, 1.0),
    ):
        """
        Args:
            scene_file (None or str): If specified, full path of JSON file to load (with .json).
                None results in no additional objects being loaded into the scene
            use_floor_plane (bool): whether to load a flat floor plane into the simulator
            floor_plane_visible (bool): whether to render the additionally added floor plane
            floor_plane_color (3-array): if @floor_plane_visible is True, this determines the (R,G,B) color assigned
                to the generated floor plane
        """
        # Store internal variables
        self.scene_file = scene_file
        self._loaded = False                    # Whether this scene exists in the stage or not
        self._initialized = False               # Whether this scene has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._registry = None
        self._world_prim = None
        self._initial_state = None
        self._objects_info = None                       # Information associated with this scene
        self._use_floor_plane = use_floor_plane
        self._floor_plane_visible = floor_plane_visible
        self._floor_plane_color = floor_plane_color
        self._floor_plane = None

        # Call super init
        super().__init__()

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
            SerializableRegistry: System registry containing all physical systems in the scene (e.g.: WaterSystem,
                DustSystem, etc.)
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
                prims that we can use as unique IDs to reference prims, e.g., prim.prim_path, prim.name, prim.handle, etc.
        """
        return ["name", "prim_path", "root_handle", "uuid"]

    @property
    def object_registry_group_keys(self):
        """
        Returns:
            list of str: Keys with which to index into the object registry. These should be valid public attributes of
                prims that we can use as grouping IDs to reference prims, e.g., prim.in_rooms
        """
        return ["prim_type", "states", "category", "fixed_base", "in_rooms", "states"]

    @property
    def loaded(self):
        return self._loaded

    @property
    def initialized(self):
        return self._initialized

    def _load(self, simulator):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.

        Args:
            simulator (Simulator): the simulator to load the scene into
        """
        # We just add a ground plane if requested
        if self._use_floor_plane:
            self.add_ground_plane(color=self._floor_plane_color, visible=self._floor_plane_visible)

    def _load_objects_from_scene_file(self, simulator):
        """
        Loads scene objects based on metadata information found in the current USD stage's scene info
        (information stored in the world prim's CustomData)
        """
        # Grab objects info from the scene file
        with open(self.scene_file, "r") as f:
            scene_info = json.load(f)
        init_info = scene_info["objects_info"]["init_info"]
        init_state = scene_info["state"]["object_registry"]

        # Iterate over all scene info, and instantiate object classes linked to the objects found on the stage
        # accordingly
        for obj_name, obj_info in init_info.items():
            # Check whether we should load the object or not
            if not self._should_load_object(obj_info=obj_info):
                continue
            # Create object class instance
            obj = create_object_from_init_info(obj_info)
            # Import into the simulator
            simulator.import_object(obj)
            # Set the init pose accordingly
            obj.set_position_orientation(
                position=init_state[obj_name]["root_link"]["pos"],
                orientation=init_state[obj_name]["root_link"]["ori"],
            )

        # disable collision between the fixed links of the fixed objects
        fixed_objs = self.object_registry("fixed_base", True, default_val=[])
        if len(fixed_objs) > 1:
            # We iterate over all pairwise combinations of fixed objects
            for obj_a, obj_b in combinations(fixed_objs, 2):
                obj_a.root_link.add_filtered_collision_pair(obj_b.root_link)

    def _should_load_object(self, obj_info):
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

    def load(self, simulator):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.

        Args:
            simulator (Simulator): the simulator to load the scene into
        """
        # Make sure simulator is stopped
        assert simulator.is_stopped(), "Simulator should be stopped when loading this scene!"

        # Do not override this function. Override _load instead.
        if self._loaded:
            raise ValueError("This scene is already loaded.")

        # Create the registry for tracking all objects in the scene
        self._registry = self._create_registry()

        # Store world prim
        self._world_prim = simulator.world_prim

        # Clear the systems
        self.clear_systems()

        self._load(simulator)

        # Initialize systems
        self.initialize_systems(simulator)

        # If we have any scene file specified, use it to load the objects within it
        if self.scene_file is not None:
            self._load_objects_from_scene_file(simulator)

        # We're now loaded
        self._loaded = True

        # Always stop the sim if we started it internally
        if not simulator.is_stopped():
            simulator.stop()

    def initialize_systems(self, simulator):
        # Initialize registries
        for system in self.systems:
            print(f"Initializing system: {system.name}")
            system.initialize(simulator=simulator)

    def clear_systems(self):
        # Clears systems so they can be re-initialized
        for system in self.systems:
            print(f"Clearing system: {system.name}")
            system.clear()

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
        self._initialize()

        # Grab relevant objects info and re-initialize object registry by handle since now handles are populated
        self.update_objects_info()
        self.object_registry.update(keys="root_handle")
        self.wake_scene_objects()

        self._initialized = True

        # Store initial state, which may be loaded from a scene file if specified
        if self.scene_file is None:
            init_state = self.dump_state(serialized=False)
        else:
            with open(self.scene_file, "r") as f:
                scene_info = json.load(f)
            init_state = scene_info["state"]
            og.sim.load_state(init_state, serialized=False)

        self._initial_state = init_state

    def _create_registry(self):
        """
        Creates the internal registry used for tracking all objects

        Returns:
            SerializableRegistry: registry for tracking all objects
        """

        # Create meta registry and populate with internal registries for robots, objects, and systems
        registry = SerializableRegistry(
            name="master_registry",
            class_types=SerializableRegistry,
        )

        # Add registry for systems -- this is already created externally, so we just pull it directly
        registry.add(obj=SYSTEMS_REGISTRY)

        # Add registry for objects
        registry.add(obj=SerializableRegistry(
            name="object_registry",
            class_types=BaseObject,
            default_key="name",
            unique_keys=self.object_registry_unique_keys,
            group_keys=self.object_registry_group_keys,
        ))

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

    def _add_object(self, obj):
        """
        Add an object to the scene's internal object tracking mechanisms.

        Note that if the scene is not loaded, it should load this added object alongside its other objects when
        scene.load() is called. The object should also be accessible through scene.objects.

        Args:
            obj (BaseObject): the object to load into the simulator
        """
        pass

    def add_object(self, obj, simulator, register=True, _is_call_from_simulator=False):
        """
        Add an object to the scene, loading it if the scene is already loaded.

        Note that calling add_object to an already loaded scene should only be done by the simulator's import_object()
        function.

        Args:
            obj (BaseObject): the object to load
            simulator (Simulator): the simulator to add the object to
            register (bool): whether to track this object internally in the scene registry
            _is_call_from_simulator (bool): whether the caller is the simulator. This should
            **not** be set by any callers that are not the Simulator class

        Returns:
            Usd.Prim: the prim of the loaded object if the scene was already loaded, or None if the scene is not loaded
                (in that case, the object is stored to be loaded together with the scene)
        """
        # Make sure the simulator is the one calling this function
        assert _is_call_from_simulator, "Use import_object() for adding objects to a simulator and scene!"

        # If the scene is already loaded, we need to load this object separately. Otherwise, don't do anything now,
        # let scene._load() load the object when called later on.
        prim = obj.load(simulator)

        # Add this object to our registry based on its type, if we want to register it
        if register:
            self.object_registry.add(obj)

            # Run any additional scene-specific logic with the created object
            self._add_object(obj)

        return prim

    def remove_object(self, obj, simulator):
        """
        Method to remove an object from the simulator

        Args:
            obj (BaseObject): Object to remove
            simulator (Simulator): current simulation context
        """
        # Remove from the appropriate registry
        self.object_registry.remove(obj)

        # Remove from omni stage
        obj.remove(simulator=simulator)

    def reset(self):
        """
        Resets this scene
        """
        # Reset all systems
        for system in self.systems:
            system.reset()

        # Reset all object and robot states
        for obj in self.objects:
            if isinstance(obj, StatefulObject):
                obj.reset_states()

        # Reset the pose and joint configuration of all scene objects.
        if self._initial_state is not None:
            self.load_state(self._initial_state)
            og.app.update()

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
            dict: Keyword-mapped objects that are fixed in the scene. Maps object name to their object class instances
                (DatasetObject)
        """
        return {obj.name: obj for obj in self.object_registry("fixed_base", True)}

    def get_random_floor(self):
        """
        Sample a random floor among all existing floor_heights in the scene.
        Most scenes in OmniGibson only have a single floor.

        Returns:
            int: an integer between 0 and self.n_floors-1
        """
        return np.random.randint(0, self.n_floors)

    def get_random_point(self, floor=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.
        Should be implemented by subclass.

        Args:
            floor (None or int): floor number. None means the floor is randomly sampled

        Returns:
            2-tuple:
                - int: floor number. This is the sampled floor number if @floor is None
                - 3-array: (x,y,z) randomly sampled point
        """
        raise NotImplementedError()

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get the shortest path from one point to another point.

        Args:
            floor (int): floor number
            source_world (2-array): (x,y) 2D source location in world reference frame (metric)
            target_world (2-array): (x,y) 2D target location in world reference frame (metric)
            entire_path (bool): whether to return the entire path

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

    def add_ground_plane(
        self,
        size=None,
        z_position: float = 0,
        name="ground_plane",
        prim_path: str = "/World/groundPlane",
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.8,
        color=None,
        visible=True,
    ):
        """
        Generate a ground plane into the simulator

        Args:
            size (None or float): If specified, sets the (x,y) size of the generated plane
            z_position (float): Z position of the generated plane
            name (str): Name to assign to the generated plane
            prim_path (str): Prim path for the generated plane
            static_friction (float): Static friction of the generated plane
            dynamic_friction (float): Dynamics friction of the generated plane
            restitution (float): Restitution of the generated plane
            color (None or 3-array): If specified, sets the (R,G,B) color of the generated plane
            visible (bool): Whether the plane should be visible or not
        """
        plane = GroundPlane(
            prim_path=prim_path,
            name=name,
            z_position=z_position,
            size=size,
            color=None if color is None else np.array(color),
            visible=visible,

            # TODO: update with new PhysicsMaterial API
            # static_friction=static_friction,
            # dynamic_friction=dynamic_friction,
            # restitution=restitution,
        )

        self._floor_plane = XFormPrim(
            prim_path=plane.prim_path,
            name=plane.name,
        )

    def update_initial_state(self):
        """
        Updates the initial state for this scene (which the scene will get reset to upon calling reset())
        """
        self._initial_state = self.dump_state(serialized=False)

    def update_objects_info(self):
        """
        Updates the scene-relevant information and saves it to the active USD. Useful for reloading a scene directly
        from a saved USD in this format.
        """
        # Save relevant information

        # Iterate over all objects and save their init info
        init_info = {obj.name: obj.get_init_info() for obj in self.object_registry.objects}

        # Compose as single dictionary and store internally
        self._objects_info = OrderedDict(init_info=init_info)

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

    @property
    def state_size(self):
        # Total state size is the state size of our registry
        return self._registry.state_size

    def _dump_state(self):
        # Default state for the scene is from the registry alone
        return self._registry.dump_state(serialized=False)

    def _load_state(self, state):
        # Default state for the scene is from the registry alone
        self._registry.load_state(state=state, serialized=False)

    def _serialize(self, state):
        # Default state for the scene is from the registry alone
        return self._registry.serialize(state=state)

    def _deserialize(self, state):
        # Default state for the scene is from the registry alone
        # We split this into two explicit steps, because the actual registry state size might dynamically change
        # as we're deserializing
        state_dict = self._registry.deserialize(state=state)
        return state_dict, self._registry.state_size

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
