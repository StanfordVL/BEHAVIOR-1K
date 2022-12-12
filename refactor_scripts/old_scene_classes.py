import json
import logging
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.dom import minidom
from itertools import combinations
from collections import OrderedDict

import numpy as np

from pxr.Sdf import ValueTypeNames as VT
from omni.isaac.core.utils.rotations import gf_quat_to_np_array

from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.maps.segmentation_map import SegmentationMap
from omnigibson.utils.asset_utils import (
    get_3dfront_scene_path,
    get_cubicasa_scene_path,
    get_og_category_ids,
    get_og_category_path,
    get_og_model_path,
    get_og_scene_path,
)
from omnigibson.utils.python_utils import create_object_from_init_info
from omnigibson.utils.constants import JointType
from omnigibson.utils.sim_utils import check_collision

SCENE_SOURCE_PATHS = {
    "IG": get_og_scene_path,
    "CUBICASA": get_cubicasa_scene_path,
    "THREEDFRONT": get_3dfront_scene_path,
}


import logging
from abc import ABC

from omnigibson.scenes.scene_base import Scene
from omnigibson.maps.traversable_map import TraversableMap
from omnigibson.utils.python_utils import classproperty


import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from omni.isaac.core.objects.ground_plane import GroundPlane
from pxr import Usd, UsdGeom
import numpy as np
from omni.isaac.core.utils.stage import get_current_stage
from omnigibson import app
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.python_utils import classproperty, Serializable, Registerable, Recreatable
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.utils.config_utils import NumpyEncoder
from omnigibson.objects.object_base import BaseObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.systems import SYSTEMS_REGISTRY
from omnigibson.robots.robot_base import m as robot_macros


class SceneOld(Scene):
    """
    Base class for all Scene objects.
    Contains the base functionalities and the functions that all derived classes need to implement.
    """

    def __init__(self):
        # Store internal variables
        self._loaded = False                    # Whether this scene exists in the stage or not
        self._initialized = False               # Whether this scene has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._registry = None
        self._world_prim = None
        self._initial_state = None
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
        return ["prim_type", "states", "category"]

    @property
    def loaded(self):
        return self._loaded

    @property
    def initialized(self):
        return self._initialized

    @abstractmethod
    def _load(self, simulator):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.

        :param simulator: the simulator to load the scene into
        :return: a list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        raise NotImplementedError()

    def load(self, simulator):
        """
        Load the scene into simulator
        The elements to load may include: floor, building, objects, etc.

        :param simulator: the simulator to load the scene into
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
        self._loaded = True

        # TODO (eric): make scene have _post_load() function that calls obj._post_load() for each object in the scene.
        # Then we should be able to sandwich self.initialize_systems() between self._load() and self._post_load()
        # The systems should be initialized internally within self._load
        for system in self.systems:
            assert system.initialized, f"System not initialized: {system.name}"

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
        self._initialized = True

        # Store object states
        scene_info = self.get_scene_info()
        self._initial_state = self.dump_state(serialized=False) if scene_info is None else \
            scene_info["init_state"]

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

    # TODO: Refactor
    def object_exists(self, name: str) -> bool:
        """[summary]

        Args:
            name (str): [description]

        Returns:
            XFormPrim: [description]
        """
        if self._scene_registry.name_exists(name):
            return True
        else:
            return False

    def get_objects_with_state(self, state):
        """
        Get the objects with a given state in the scene.

        :param state: state of the objects to get
        :return: a set of objects with the given state
        """
        return self.object_registry("states", state, set())

    def _add_object(self, obj):
        """
        Add an object to the scene's internal object tracking mechanisms.

        Note that if the scene is not loaded, it should load this added object alongside its other objects when
        scene.load() is called. The object should also be accessible through scene.objects.

        :param obj: the object to load
        """
        pass

    def add_object(self, obj, simulator, register=True, _is_call_from_simulator=False):
        """
        Add an object to the scene, loading it if the scene is already loaded.

        Note that calling add_object to an already loaded scene should only be done by the simulator's import_object()
        function.

        :param obj: the object to load
        :param simulator: the simulator to add the object to
        :param register: whether to track this object internally in the scene registry
        :param _is_call_from_simulator: whether the caller is the simulator. This should
            **not** be set by any callers that are not the Simulator class
        :return: the prim of the loaded object if the scene was already loaded, or None if the scene is not loaded
            (in that case, the object is stored to be loaded together with the scene)
        """
        # Make sure the simulator is the one calling this function
        assert _is_call_from_simulator, "Use import_object() for adding objects to a simulator and scene!"

        # TODO
        # if isinstance(obj, VisualMarker) or isinstance(obj, Particle):
        #     raise ValueError("VisualMarker and Particle objects and subclasses should be added directly to simulator.")

        # If the scene is already loaded, we need to load this object separately. Otherwise, don't do anything now,
        # let scene._load() load the object when called later on.
        prim = obj.load(simulator)

        # Add this object to our registry based on its type, if we want to register it
        if register:
            self.object_registry.add(obj)

            # Run any additional scene-specific logic with the created object
            self._add_object(obj)

        return prim

    # def clear(self):
    #     """
    #     Clears all scene data from this scene
    #     """
    #     # Remove all object, robot, system info

    def remove_object(self, obj, simulator):
        # Remove from the appropriate registry
        self.object_registry.remove(obj)

        # Remove from omni stage
        obj.remove(simulator=simulator)

    # TODO: Integrate good features of this
    #
    # def add(self, obj: XFormPrim) -> XFormPrim:
    #     """[summary]
    #
    #     Args:
    #         obj (XFormPrim): [description]
    #
    #     Raises:
    #         Exception: [description]
    #         Exception: [description]
    #
    #     Returns:
    #         XFormPrim: [description]
    #     """
    #     if self._scene_registry.name_exists(obj.name):
    #         raise Exception("Cannot add the object {} to the scene since its name is not unique".format(obj.name))
    #     if isinstance(obj, RigidPrim):
    #         self._scene_registry.add_rigid_object(name=obj.name, rigid_object=obj)
    #     elif isinstance(obj, GeometryPrim):
    #         self._scene_registry.add_geometry_object(name=obj.name, geometry_object=obj)
    #     elif isinstance(obj, Robot):
    #         self._scene_registry.add_robot(name=obj.name, robot=obj)
    #     elif isinstance(obj, Articulation):
    #         self._scene_registry.add_articulated_system(name=obj.name, articulated_system=obj)
    #     elif isinstance(obj, XFormPrim):
    #         self._scene_registry.add_xform(name=obj.name, xform=obj)
    #     else:
    #         raise Exception("object type is not supported yet")
    #     return obj

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

        for robot in self.robots:
            robot.reset_states()

        # Reset the pose and joint configuration of all scene objects.
        if self._initial_state is not None:
            self.load_state(self._initial_state)
            app.update()

    @property
    def has_connectivity_graph(self):
        """
        Returns:
            bool: Whether this scene has a connectivity graph
        """
        # Default is no connectivity graph
        return False

    @property
    def num_floors(self):
        """
        Returns:
            int: Number of floors in this scene
        """
        # Default is a single floor
        return 1

    def get_random_floor(self):
        """
        Sample a random floor among all existing floor_heights in the scene.
        While Gibson v1 scenes can have several floor_heights, the Scene, StadiumScene and scenes from OmniGibson
        have only a single floor.

        :return: an integer between 0 and NumberOfFloors-1
        """
        return np.random.randint(0, self.num_floors)

    def get_random_point(self, floor=None):
        """
        Sample a random valid location in the given floor.

        :param floor: integer indicating the floor, or None if randomly sampled
        :return: a tuple of random floor and random valid point (3D) in that floor
        """
        raise NotImplementedError()

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Query the shortest path between two points in the given floor.

        :param floor: floor to compute shortest path in
        :param source_world: initial location in world reference frame
        :param target_world: target location in world reference frame
        :param entire_path: flag indicating if the function should return the entire shortest path or not
        :return: a tuple of path (if indicated) as a list of points, and geodesic distance (lenght of the path)
        """
        raise NotImplementedError()

    def get_floor_height(self, floor=0):
        """
        Get the height of the given floor.

        :param floor: an integer identifying the floor
        :return: height of the given floor
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
        """[summary]

        Args:
            size (Optional[float], optional): [description]. Defaults to None.
            z_position (float, optional): [description]. Defaults to 0.
            name (str, optional): [description]. Defaults to "ground_plane".
            prim_path (str, optional): [description]. Defaults to "/World/groundPlane".
            static_friction (float, optional): [description]. Defaults to 0.5.
            dynamic_friction (float, optional): [description]. Defaults to 0.5.
            restitution (float, optional): [description]. Defaults to 0.8.
            color (Optional[np.ndarray], optional): [description]. Defaults to None.
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

    def update_scene_info(self):
        """
        Updates the scene-relevant information and saves it to the active USD. Useful for reloading a scene directly
        from a saved USD in this format.
        """
        # Save relevant information

        # Iterate over all objects and save their init info
        init_info = {obj.name: obj.get_init_info() for obj in self.object_registry.objects}

        # Save initial object state info
        init_state_info = self._initial_state

        # Compose as single dictionary and dump into custom data field in world prim
        scene_info = {"init_info": init_info, "init_state": init_state_info}
        scene_info_str = json.dumps(scene_info, cls=NumpyEncoder)
        self._world_prim.SetCustomDataByKey("scene_info", scene_info_str)

    def get_scene_info(self):
        """
        Stored information, if any, for this scene. Structure is:

            "init_info":
                "<obj0>": <obj0> init kw/args
                ...
                "<robot0>": <robot0> init kw/args
                ...
            "init_state":
                dict: State of the scene upon episode initialization; output from self.dump_state(serialized=False)

        Returns:
            None or dict: If it exists, nested dictionary of relevant scene information
        """
        scene_info_str = self._world_prim.GetCustomDataByKey("scene_info")
        return None if scene_info_str is None else json.loads(scene_info_str)

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
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("SceneOld")
        return classes


class TraversableSceneOld(SceneOld):
    """
    Traversable scene class.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        scene_model,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Load a traversable scene and compute traversability

        :param scene_model: Scene model
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_with_objects: whether to use objects or not when constructing graph
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        """
        super().__init__()
        logging.info("TraversableScene model: {}".format(scene_model))
        self.scene_model = scene_model

        # Create traversable map
        self._trav_map = TraversableMap(
            map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_with_objects=trav_map_with_objects,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )

    @property
    def trav_map(self):
        """
        Returns:
            TraversableMap: Map for computing connectivity between nodes for this scene
        """
        return self._trav_map

    @property
    def has_connectivity_graph(self):
        # Connectivity graph is determined by travserable map
        return self._trav_map.build_graph

    def get_random_point(self, floor=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.

        :param floor: floor number
        :return floor: floor number
        :return point: randomly sampled point in [x, y, z]
        """
        return self._trav_map.get_random_point(floor=floor)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        :param floor: floor number
        :param source_world: 2D source location in world reference frame (metric)
        :param target_world: 2D target location in world reference frame (metric)
        :param entire_path: whether to return the entire path
        """
        assert self._trav_map.build_graph, "cannot get shortest path without building the graph"

        return self._trav_map.get_shortest_path(
            floor=floor,
            source_world=source_world,
            target_world=target_world,
            entire_path=entire_path,
        )

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("TraversableSceneOld")
        return classes




class InteractiveTraversableSceneOld(TraversableSceneOld):
    """
    Create an interactive scene defined with OmniGibson Scene Description Format (iGSDF).
    iGSDF is an extension of URDF that we use to define an interactive scene.
    It has support for URDF scaling, URDF nesting and randomization.
    InteractiveIndoorScene inherits from TraversableScene the functionalities to compute shortest path and other
    navigation functionalities.
    """
    def __init__(
        self,
        scene_model,
        usd_file=None,
        usd_path=None,
        # pybullet_filename=None,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_with_objects=True,
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        texture_randomization=False,
        object_randomization=False,
        link_collision_tolerance=0.03,
        predefined_object_randomization_idx=None,
        should_open_all_doors=False,
        load_object_categories=None,
        not_load_object_categories=None,
        load_room_types=None,
        load_room_instances=None,
        seg_map_resolution=0.1,
        scene_source="IG",
        # merge_fixed_links=True,
        rendering_params=None,
        include_robots=True,
    ):
        """
        # TODO: Update
        :param scene_model: Scene model, e.g.: Rs_int
        # TODO: Update doc -- usd_file / usd_path naming convention is too ambiguous / similar
        :param usd_file: name of usd file to load (without .urdf), default to og_dataset/scenes/<scene_model>/urdf/<urdf_file>.urdf
        :param usd_path: full path of URDF file to load (with .urdf)
        # :param pybullet_filename: optional specification of which pybullet file to restore after initialization
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_with_objects: whether to use objects or not when constructing graph
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param texture_randomization: whether to randomize material/texture
        :param object_randomization: whether to randomize object
        :param link_collision_tolerance: tolerance of the percentage of links that cannot be fully extended after object randomization
        :param predefined_object_randomization_idx: None or int, index of a pre-computed object randomization model that guarantees good scene quality.
            If None, a fully random sampling from object categories will be used.
        :param should_open_all_doors: whether to open all doors after episode reset (usually required for navigation tasks)
        :param load_object_categories: only load these object categories into the scene (a list of str)
        :param not_load_object_categories: do not load these object categories into the scene (a list of str)
        :param load_room_types: only load objects in these room types into the scene (a list of str)
        :param load_room_instances: only load objects in these room instances into the scene (a list of str)
        :param seg_map_resolution: room segmentation map resolution
        :param scene_source: source of scene data; among IG, CUBICASA, THREEDFRONT
        :param merge_fixed_links: whether to merge fixed links in pybullet
        :param rendering_params: additional rendering params to be passed into object initializers (e.g. texture scale)
        :param include_robots: whether to also include the robot(s) defined in the scene
        """
        # Run super init first
        super().__init__(
            scene_model=scene_model,
            trav_map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_with_objects=trav_map_with_objects,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
        )

        # Store attributes from inputs
        self.texture_randomization = texture_randomization
        self.object_randomization = object_randomization
        self.predefined_object_randomization_idx = predefined_object_randomization_idx
        self.rendering_params = rendering_params
        self.should_open_all_doors = should_open_all_doors
        self.scene_source = scene_source

        # Other values that will be loaded at runtime
        self.scene_file = None
        self.scene_dir = None
        self.load_object_categories = None
        self.not_load_object_categories = None
        self.load_room_instances = None

        # self.scene_tree = ET.parse(self.scene_file)
        # self.pybullet_filename = pybullet_filename
        self.random_groups = {}
        self.category_ids = get_og_category_ids()
        # self.merge_fixed_links = merge_fixed_links
        self.include_robots = include_robots

        # Get scene information
        self.get_scene_loading_info(usd_file=usd_file, usd_path=usd_path)

        # Load room semantic and instance segmentation map (must occur AFTER inferring scene directory)
        self._seg_map = SegmentationMap(scene_dir=self.scene_dir, map_resolution=seg_map_resolution)

        # Decide which room(s) and object categories to load
        self.filter_rooms_and_object_categories(
            load_object_categories, not_load_object_categories, load_room_types, load_room_instances
        )

        # load overlapping bboxes in scene annotation
        self.overlapped_bboxes = self.load_overlapped_bboxes()

        # percentage of objects allowed that CANNOT extend their joints by >66%
        self.link_collision_tolerance = link_collision_tolerance

        # Agent placeholder
        self.agent = {}

        # ObjectMultiplexer
        self.object_multiplexers = defaultdict(dict)

        # ObjectGrouper
        self.object_groupers = defaultdict(dict)

    def get_scene_loading_info(self, usd_file=None, usd_path=None):
        """
        Gets scene loading info to know what single USD file to load, either specified indirectly via @usd_file or
        directly by the fpath from @usd_path. Note that if both are specified, @usd_path takes precidence.
        If neither are specified, then a file will automatically be chosen based on self.scene_model and
        self.object_randomization

        Args:
            usd_file (None or str): If specified, should be name of usd file to load. (without .usd), default to
                og_dataset/scenes/<scene_model>/usd/<usd_file>.usd
            usd_path (None or str): If specified, should be absolute filepath to the USD file to load (with .usd)
        """
        # Grab scene source path
        assert self.scene_source in SCENE_SOURCE_PATHS, f"Unsupported scene source: {self.scene_source}"
        self.scene_dir = SCENE_SOURCE_PATHS[self.scene_source](self.scene_model)

        scene_file = usd_path
        # Possibly grab the USD directly from a specified fpath
        if usd_path is None:
            if usd_file is not None:
                fname = usd_file
            else:
                if not self.object_randomization:
                    fname = "{}_best_template".format(self.scene_model)
                else:
                    fname = self.scene_model if self.predefined_object_randomization_idx is None else \
                        "{}_random_{}".format(self.scene_model, self.predefined_object_randomization_idx)
            scene_file = os.path.join(self.scene_dir, "usd", "{}.usd".format(fname))

        # Store values internally
        self.scene_file = scene_file

    def filter_rooms_and_object_categories(
        self, load_object_categories, not_load_object_categories, load_room_types, load_room_instances
    ):
        """
        Handle partial scene loading based on object categories, room types or room instances

        :param load_object_categories: only load these object categories into the scene (a list of str)
        :param not_load_object_categories: do not load these object categories into the scene (a list of str)
        :param load_room_types: only load objects in these room types into the scene (a list of str)
        :param load_room_instances: only load objects in these room instances into the scene (a list of str)
        """
        self.load_object_categories = [load_object_categories] if \
            isinstance(load_object_categories, str) else load_object_categories

        self.not_load_object_categories = [not_load_object_categories] if \
            isinstance(not_load_object_categories, str) else not_load_object_categories

        if load_room_instances is not None:
            if isinstance(load_room_instances, str):
                load_room_instances = [load_room_instances]
            load_room_instances_filtered = []
            for room_instance in load_room_instances:
                if room_instance in self._seg_map.room_ins_name_to_ins_id:
                    load_room_instances_filtered.append(room_instance)
                else:
                    logging.warning("room_instance [{}] does not exist.".format(room_instance))
            self.load_room_instances = load_room_instances_filtered
        elif load_room_types is not None:
            if isinstance(load_room_types, str):
                load_room_types = [load_room_types]
            load_room_instances_filtered = []
            for room_type in load_room_types:
                if room_type in self._seg_map.room_sem_name_to_ins_name:
                    load_room_instances_filtered.extend(self._seg_map.room_sem_name_to_ins_name[room_type])
                else:
                    logging.warning("room_type [{}] does not exist.".format(room_type))
            self.load_room_instances = load_room_instances_filtered
        else:
            self.load_room_instances = None

    def load_overlapped_bboxes(self):
        """
        Load overlapped bounding boxes in scene definition.
        E.g. a dining table usually has overlaps with the surrounding dining chairs
        """
        bbox_overlap_file = os.path.join(self.scene_dir, "misc", "bbox_overlap.json")
        if os.path.isfile(bbox_overlap_file):
            with open(bbox_overlap_file) as f:
                return json.load(f)
        else:
            return []

    def randomize_texture(self):
        """
        Randomize texture/material for all objects in the scene
        """
        if not self.texture_randomization:
            logging.warning("calling randomize_texture while texture_randomization is False during initialization.")
            return
        for obj in self.objects:
            obj.randomize_texture()

    # TODO
    def check_scene_quality(self, simulator):
        """
        Helper function to check for scene quality.
        1) Objects should have no collision with each other.
        2) Fixed, articulated objects that cannot fully extend their joints should be less than self.link_collision_tolerance

        Args:
            simulator (Simulator): Active simulator object

        :return: whether scene passes quality check
        """
        quality_check = True

        body_body_collision = []
        body_link_collision = []

        # # build mapping from body_id to object name for debugging
        # body_id_to_name = {}
        # for name in self.objects_by_name:
        #     for body_id in self.objects_by_name[name].get_body_ids():
        #         body_id_to_name[body_id] = name
        # self.body_id_to_name = body_id_to_name

        # collect body ids for overlapped bboxes (e.g. tables and chairs,
        # sofas and coffee tables)
        overlapped_objs = []
        for obj1_name, obj2_name in self.overlapped_bboxes:
            if obj1_name not in self.object_registry or obj2_name not in self.object_registry:
                # This could happen if only part of the scene is loaded (e.g. only a subset of rooms)
                continue
            overlapped_objs.append((self.object_registry("name", obj1_name), self.object_registry("name", obj2_name)))

        # TODO -- current method is probably insufficient
        # cache pybullet initial state
        state = self.save_state()

        # check if these overlapping bboxes have collision
        simulator.step()
        for obj_a, obj_b in overlapped_objs:
            has_collision = obj_a.in_contact(prims=obj_b)
            quality_check = quality_check and (not has_collision)
            if has_collision:
                body_body_collision.append((obj_a, obj_b))

        # check if fixed, articulated objects can extend their joints
        # without collision with other fixed objects
        joint_collision_allowed = int(self.get_num_objects() * self.link_collision_tolerance)
        joint_collision_so_far = 0
        for fixed_obj in self.fixed_objects.values():
            # This only applies if the object is articulated
            if fixed_obj.articulated:
                joint_quality = True
                for joint in fixed_obj.joints.values():
                    j_low, j_high = joint.lower_limit, joint.upper_limit
                    if joint.joint_type not in [JointType.JOINT_REVOLUTE, JointType.JOINT_PRISMATIC]:
                        continue
                    # this is the continuous joint (e.g. wheels for office chairs)
                    if j_low >= j_high:
                        continue

                    # usually j_low and j_high includes j_default = 0.0
                    # if not, set j_default to be j_low
                    j_default = 0.0
                    if not (j_low <= j_default <= j_high):
                        j_default = j_low

                    # check three joint positions, 0%, 33% and 66%
                    j_range = j_high - j_low
                    j_low_perc = j_range * 0.33 + j_low
                    j_high_perc = j_range * 0.66 + j_low

                    # check if j_default, j_low_per, or j_high_perc has collision
                    for j_pos in (j_default, j_low_perc, j_high_perc):
                        self.restore_state(state)
                        joint.set_pos(pos=j_pos)
                        has_collision = check_collision(prims=fixed_obj.links[joint.child_name], prims_check=self.fixed_objects)
                        joint_quality = joint_quality and (not has_collision)

                if not joint_quality:
                    joint_collision_so_far += 1
                    body_link_collision.append(fixed_obj)

        quality_check = quality_check and (joint_collision_so_far <= joint_collision_allowed)

        # restore state to the initial state before testing collision
        self.restore_state(state)

        body_collision_set = set()
        for obj_a, obj_b in body_body_collision:
            logging.warning(f"scene quality check: {obj_a.name} and {obj_b.name} has collision.")
            body_collision_set.add(obj_a.name)
            body_collision_set.add(obj_b.name)

        link_collision_set = set()
        for obj in body_link_collision:
            logging.warning(f"scene quality check: {obj.name} has joint that cannot extend for >66%.")
            link_collision_set.add(obj.name)

        return quality_check

    # TODO
    def open_one_obj(self, body_id, mode="random"):
        """
        Attempt to open one object without collision

        :param body_id: body id of the object
        :param mode: opening mode (zero, max, or random)
        """
        body_joint_pairs = []
        for joint_id in range(p.getNumJoints(body_id)):
            # cache current physics state
            state_id = p.saveState()

            j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
            j_type = p.getJointInfo(body_id, joint_id)[2]
            parent_idx = p.getJointInfo(body_id, joint_id)[-1]
            if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                p.removeState(state_id)
                continue
            # this is the continuous joint
            if j_low >= j_high:
                p.removeState(state_id)
                continue
            # this is the 2nd degree joint, ignore for now
            if parent_idx != -1:
                p.removeState(state_id)
                continue

            if mode == "max":
                # try to set the joint to the maxr value until no collision
                # step_size is 5cm for prismatic joint and 5 degrees for revolute joint
                step_size = np.pi / 36.0 if j_type == p.JOINT_REVOLUTE else 0.05
                for j_pos in np.arange(0.0, j_high + step_size, step=step_size):
                    p.resetJointState(body_id, joint_id, j_high - j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(body_a=body_id, link_a=joint_id)
                    restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_high - j_pos)
                        break

            elif mode == "random":
                # try to set the joint to a random value until no collision
                reset_success = False
                # make 10 attemps
                for _ in range(10):
                    j_pos = np.random.uniform(j_low, j_high)
                    p.resetJointState(body_id, joint_id, j_pos)
                    p.stepSimulation()
                    has_collision = self.check_collision(body_a=body_id, link_a=joint_id)
                    restoreState(state_id)
                    if not has_collision:
                        p.resetJointState(body_id, joint_id, j_pos)
                        reset_success = True
                        break

                # if none of the random values work, set it to 0.0 by default
                if not reset_success:
                    p.resetJointState(body_id, joint_id, 0.0)
            elif mode == "zero":
                p.resetJointState(body_id, joint_id, 0.0)
            else:
                assert False

            body_joint_pairs.append((body_id, joint_id))
            # Remove cached state to avoid memory leak.
            p.removeState(state_id)

        return body_joint_pairs

    # TODO! Signature is correct tho
    def open_all_objs_by_category(self, category, mode="random", p=1.0):
        """
        Attempt to open all objects of a certain category without collision

        Args:
            category (str): Object category to check for opening joints
            mode (str): Opening mode, one of {zero, max, random}
            p (float): Probability in range [0, 1] for opening a given object

        Returns:
            list of DatasetObject: Object(s) whose joints were "opened"
        """
        return []
        # body_joint_pairs = []
        # if category not in self.object_registry.get_ids("category"):
        #     return body_joint_pairs
        # for obj in self.object_registry("category", category):
        #     # open probability
        #     if np.random.random() > prob:
        #         continue
        #     for body_id in obj.get_body_ids():
        #         body_joint_pairs += self.open_one_obj(body_id, mode=mode)
        # return body_joint_pairs

    # TODO
    def open_all_objs_by_categories(self, categories, mode="random", prob=1.0):
        """
        Attempt to open all objects of a number of categories without collision

        :param categories: object categories (a list of str)
        :param mode: opening mode (zero, max, or random)
        :param prob: opening probability
        """
        return []
        # body_joint_pairs = []
        # for category in categories:
        #     body_joint_pairs += self.open_all_objs_by_category(category, mode=mode, prob=prob)
        # return body_joint_pairs

    def open_all_doors(self):
        """
        Attempt to open all doors to maximum values without collision
        """
        return self.open_all_objs_by_category("door", mode="max")

    def _create_obj_from_template_xform(self, simulator, prim):
        """
        Creates the object specified from a template xform @prim, presumed to be in a template USD file,
        and simultaneously replaces the template prim with the actual object prim


        Args:
            simulator (Simulator): Active simulation object
            prim: Usd.Prim: Object template Xform prim

        Returns:
            None or DatasetObject: Created OmniGibson object if a valid objet is found at @prim
        """
        obj = None
        info = {}

        # Extract relevant info from template
        name, prim_path = prim.GetName(), prim.GetPrimPath().__str__()
        category, model, bbox, bbox_center_pos, bbox_center_ori, fixed, in_rooms, random_group, scale, bddl_obj_scope = self._extract_obj_info_from_template_xform(prim=prim)
        info["bbox_center_pos"] = bbox_center_pos
        info["bbox_center_ori"] = bbox_center_ori

        # Delete the template prim
        simulator.stage.RemovePrim(prim_path)

        # Skip multiplexer and grouper because they are not real objects
        if category == "multiplexer":
            raise NotImplementedError()
            # self.object_multiplexers[object_name]["current_index"] = link.attrib["current_index"]
            # continue

        elif category == "grouper":
            raise NotImplementedError()
            # self.object_groupers[object_name]["pose_offsets"] = json.loads(link.attrib["pose_offsets"])
            # self.object_groupers[object_name]["multiplexer"] = link.attrib["multiplexer"]
            # self.object_multiplexers[link.attrib["multiplexer"]]["grouper"] = object_name
            # continue

        # TODO! Handle
        elif category == "agent" and not self.include_robots:
            raise NotImplementedError()
            # continue

        # Robot object
        # TODO! Handle
        elif category == "agent":
            pass
            # raise NotImplementedError()
            # robot_config = json.loads(link.attrib["robot_config"]) if "robot_config" in link.attrib else {}
            # assert model in REGISTERED_ROBOTS, "Got invalid robot to instantiate: {}".format(model)
            # obj = REGISTERED_ROBOTS[model](name=object_name, **robot_config)

        # Non-robot object
        else:
            usd_path = None

            # Need to sanity check to make sure we want to load them

            # Do not load these object categories (can blacklist building structures as well)
            not_blacklisted = self.not_load_object_categories is None or category not in self.not_load_object_categories

            # Only load these object categories (no need to white list building structures)
            whitelisted = self.load_object_categories is None or category in self.load_object_categories

            # This object is not located in one of the selected rooms, skip
            valid_room = self.load_room_instances is None or len(set(self.load_room_instances) & set(in_rooms)) >= 0

            # We only load this model if all the above conditions are met
            if not_blacklisted and whitelisted and valid_room:

                # Make sure objects exist in the actual requested category
                category_path = get_og_category_path(category)
                assert len(os.listdir(category_path)) != 0, "No models in category folder {}".format(category_path)

                # Potentially grab random object
                if model == "random":
                    if random_group is None:
                        model = random.choice(os.listdir(category_path))
                    else:
                        # Using random group to assign the same model to a group of objects
                        # E.g. we want to use the same model for a group of chairs around the same dining table
                        # random_group is a unique integer within the category
                        random_group_key = (category, random_group)

                        if random_group_key in self.random_groups:
                            model = self.random_groups[random_group_key]
                        else:
                            # We create a new unique entry for this random group if it doesn't already exist
                            model = random.choice(os.listdir(category_path))
                            self.random_groups[random_group_key] = model

                model_path = get_og_model_path(category, model)
                # TODO: Remove "usd" in the middle when we simply have the model directory directly contain the USD
                usd_path = os.path.join(model_path, "usd", model + ".usd")

            # Only create the object if a valid usd_path is specified
            if usd_path is not None:

                # Make sure only a bounding box OR scale is specified
                assert bbox is None or scale is None, f"Both scale and bounding box size was defined for a USDObject in the template scene!"

                # Make sure scale is, at the minimum, specified
                if bbox is None and scale is None:
                    scale = np.array([1.0, 1.0, 1.0])

                # Create the object (finally!)
                obj = DatasetObject(
                    prim_path=prim_path,
                    name=name,
                    category=category,
                    model=model,
                    scale=scale,
                    # rendering_params=self.rendering_params,
                    fixed_base=fixed,
                    bounding_box=bbox,
                    in_rooms=in_rooms,
                    # texture_randomization=self.texture_randomization,
                    bddl_object_scope=bddl_obj_scope,
                )

                # TODO: Are all of these necessary now that we can directly save USD snapshots?
                # bbox_center_pos = np.array([float(val) for val in connecting_joint.find("origin").attrib["xyz"].split(" ")])
                # if "rpy" in connecting_joint.find("origin").attrib:
                #     bbx_center_orn = np.array(
                #         [float(val) for val in connecting_joint.find("origin").attrib["rpy"].split(" ")]
                #     )
                # else:
                #     bbx_center_orn = np.array([0.0, 0.0, 0.0])
                # bbx_center_orn = p.getQuaternionFromEuler(bbx_center_orn)
                #
                # base_com_pose = json.loads(link.attrib["base_com_pose"]) if "base_com_pose" in link.attrib else None
                # base_velocities = json.loads(link.attrib["base_velocities"]) if "base_velocities" in link.attrib else None
                # if "joint_states" in link.keys():
                #     joint_states = json.loads(link.attrib["joint_states"])
                # elif "joint_positions" in link.keys():
                #     # Backward compatibility, assuming multi-sub URDF object don't have any joints
                #     joint_states = {
                #         key: (position, 0.0) for key, position in json.loads(link.attrib["joint_positions"])[0].items()
                #     }
                # else:
                #     joint_states = None
                #
                # if "states" in link.keys():
                #     non_kinematic_states = json.loads(link.attrib["states"])
                # else:
                #     non_kinematic_states = None
                # print(f"obj: {name}, bbox center pos: {bbox_center_pos}, bbox center ori: {bbox_center_ori}")

            # self.object_states.add_object(
            #     obj_name=name,
            #     bbox_center_pose=(bbox_center_pos, bbox_center_ori),
            #     base_com_pose=None,#(np.zeros(3), np.array([0, 0, 0, 1.0])),
            #     base_velocities=None,
            #     joint_states=None,
            #     non_kinematic_states=None,
            # )

            # TODO: Handle multiplexing / groupers
            # if "multiplexer" in link.keys() or "grouper" in link.keys():
            #     if "multiplexer" in link.keys():
            #         self.object_multiplexers[link.attrib["multiplexer"]]["whole_object"] = obj
            #     else:
            #         grouper = self.object_groupers[link.attrib["grouper"]]
            #         if "object_parts" not in grouper:
            #             grouper["object_parts"] = []
            #         grouper["object_parts"].append(obj)
            #
            #         # Once the two halves are added, this multiplexer is ready to be added to the scene
            #         if len(grouper["object_parts"]) == 2:
            #             multiplexer = grouper["multiplexer"]
            #             current_index = int(self.object_multiplexers[multiplexer]["current_index"])
            #             whole_object = self.object_multiplexers[multiplexer]["whole_object"]
            #             object_parts = grouper["object_parts"]
            #             pose_offsets = grouper["pose_offsets"]
            #             grouped_obj_parts = ObjectGrouper(list(zip(object_parts, pose_offsets)))
            #             obj = ObjectMultiplexer(multiplexer, [whole_object, grouped_obj_parts], current_index)
            #             self.add_object(obj, simulator=None)
            # else:
            #     self.add_object(obj, simulator=None)

        return obj, info

    def _extract_obj_info_from_template_xform(self, prim):
        """
        Extracts relevant OmniGibson object information from a template xform, presumed to be in a template USD file

        Args:
            prim: Usd.Prim: Object template Xform prim

        Returns:
            6-tuple:
                - category (str): object category
                - model (str): object model ("random" implies should be randomly selected)
                - bbox (None or 3-array): (x,y,z) bounding box for the object in the scene, if specified
                - pos (3-array): (x,y,z) global coordinate for this object
                - ori (4-array): (x,y,z,w) global quaternion orientation for this object
                - fixed (bool): whether to fix this object in the scene via a fixed joint
                - rooms (list of str): room(s) allowed by this object
                - random_group (None or str): if specified, group from which to grab a randomized instance
                    if @model is "random"
                - scale (None or 3-array): (x,y,z) scale for the object in the scene, if specified
                - obj_scope (None or str): If specified, the scope of this object
        """
        category = prim.GetAttribute("ig:category").Get()
        model = prim.GetAttribute("ig:model").Get()
        bbox = prim.GetAttribute("ig:boundingBox").Get()
        bbox = None if bbox is None else np.array(bbox)
        pos = np.array(prim.GetAttribute("ig:position").Get())
        ori = gf_quat_to_np_array(prim.GetAttribute("ig:orientation").Get())[[1, 2, 3, 0]]
        fixed = prim.GetAttribute("ig:fixedJoint").Get()
        rooms = prim.GetAttribute("ig:rooms").Get().split(",")
        random_group = prim.GetAttribute("ig:randomGroup").Get()
        scale = prim.GetAttribute("ig:scale").Get()
        scale = None if scale is None else np.array(scale)
        obj_scope = prim.GetAttribute("ig:objectScope").Get()

        return category, model, bbox, pos, ori, fixed, rooms, random_group, scale, obj_scope

    def _load(self, simulator):
        """
        Load all scene objects into the simulator.
        """
        # Notify user we're loading the scene
        logging.info("Clearing stage and loading scene USD: {}".format(self.scene_file))

        # We first load the initial USD file
        simulator.load_stage(usd_path=self.scene_file)

        self.initialize_systems(simulator)

        # Store stage reference and refresh world prim reference
        self._world_prim = simulator.world_prim

        # Check if current stage is a template based on ig:isTemplate value, and set the value to False if it does not exist
        if "ig:isTemplate" in self._world_prim.GetPropertyNames():
            is_template = self._world_prim.GetAttribute("ig:isTemplate").Get()
        else:
            # Create the property
            self._world_prim.CreateAttribute("ig:isTemplate", VT.Bool)
            is_template = True

        # Load objects using logic based on whether the current USD is a template or not
        if is_template:
            self._load_objects_from_template(simulator=simulator)
        else:
            self._load_objects_from_scene_info(simulator=simulator)

        # Set this to be False -- we are no longer a template after we load
        self._world_prim.GetAttribute("ig:isTemplate").Set(False)

        # disable collision between the fixed links of the fixed objects
        fixed_objs = self.object_registry("fixed_base", True, default_val=[])
        if len(fixed_objs) > 1:
            # We iterate over all pairwise combinations of fixed objects
            for obj_a, obj_b in combinations(fixed_objs, 2):
                obj_a.root_link.add_filtered_collision_pair(obj_b.root_link)

        # Load the traversability map
        maps_path = os.path.join(self.scene_dir, "layout")
        if self.has_connectivity_graph:
            self._trav_map.load_map(maps_path)

        return list(self.objects)

    def _load_objects_from_template(self, simulator):
        """
        Loads scene objects based on metadata information found in the current USD stage, assumed to be a template
        (property ig:isTemplate is True)
        """
        # Iterate over all the children in the stage world
        for prim in self._world_prim.GetChildren():
            # Only process prims that are an Xform
            if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
                name = prim.GetName()
                category = prim.GetAttribute("ig:category").Get()

                # Create the object and load it into the simulator
                obj, info = self._create_obj_from_template_xform(simulator=simulator, prim=prim)

                # Only import the object if we received a valid object
                if obj is not None:
                    # Note that we don't auto-initialize because of some very specific state-setting logic that
                    # has to occur a certain way at the start of scene creation (sigh, Omniverse ): )
                    simulator.import_object(obj, auto_initialize=False)
                    # If we have additional info specified, we also directly set it's bounding box position since this is a known quantity
                    # This is also the only time we'll be able to set fixed object poses
                    pos = info["bbox_center_pos"]
                    ori = info["bbox_center_ori"]
                    obj.set_bbox_center_position_orientation(pos, ori)

    def _load_objects_from_scene_info(self, simulator):
        """
        Loads scene objects based on metadata information found in the current USD stage's scene info
        (information stored in the world prim's CustomData)
        """
        # Grab scene info
        scene_info = self.get_scene_info()

        # Iterate over all scene info, and instantiate object classes linked to the objects found on the stage
        # accordingly
        for obj_info in scene_info["init_info"].values():
            obj = create_object_from_init_info(obj_info)
            # Make sure this object is already loaded -- i.e.: it was linked to a pre-existing prim found on the stage
            assert obj.loaded, f"Object {obj.name} should have already been loaded when creating a BaseObject class" \
                               f"instance, but did not find any on the current stage! " \
                               f"Searched at prim_path: {obj.prim_path}"
            # Import into the simulator
            # Note that we don't auto-initialize because of some very specific state-setting logic that
            # has to occur a certain way at the start of scene creation (sigh, Omniverse ): )
            simulator.import_object(obj, auto_initialize=False)

    def _initialize(self):
        # First, we initialize all of our objects and add the object
        for obj in self.objects:
            obj.initialize()
            obj.keep_still()
        for robot in self.robots:
            robot.initialize()

        # Re-initialize our scene object registry by handle since now handles are populated
        self.object_registry.update(keys="root_handle")

        # # TODO: Additional restoring from USD state? Is this even necessary?
        # if self.pybullet_filename is not None:
        #     restoreState(fileName=self.pybullet_filename)

        # TODO: Need to check scene quality
        # self.check_scene_quality(simulator=simulator)

        # TODO: Necessary? Currently does nothing since sim is paused at this point
        # force wake up each body once
        self.wake_scene_objects()

    def wake_scene_objects(self):
        """
        Force wakeup sleeping objects
        """
        for obj in self.objects:
            obj.wake()

    def reset(self):
        # Run super first
        super().reset()

        # Also open all doors if self.should_open_all_doors is True
        if self.should_open_all_doors:
            self.wake_scene_objects()
            self.open_all_doors()

    def get_num_objects(self):
        """
        Get the number of objects

        :return: number of objects
        """
        return len(self.objects)

    def get_object_handles(self):
        """
        Return the object handles of all scene objects

        :return: object handles
        """
        return [obj.handle for obj in self.objects]

    @property
    def seg_map(self):
        """
        Returns:
            SegmentationMap: Map for segmenting this scene
        """
        return self._seg_map

    @property
    def object_registry_unique_keys(self):
        # Grab from super and add handle for objects
        return super().object_registry_unique_keys

    @property
    def object_registry_group_keys(self):
        # Grab from super and add additional keys for the dataset scene objects
        return super().object_registry_group_keys + ["category", "fixed_base", "in_rooms", "states"]

    @property
    def fixed_objects(self):
        """
        Returns:
            dict: Keyword-mapped objects that are are fixed in the scene. Maps object name to their object class instances
                (DatasetObject)
        """
        return {obj.name: obj for obj in self.object_registry("fixed_base", True)}
