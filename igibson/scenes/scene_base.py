from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from future.utils import with_metaclass

import carb
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene_registry import SceneRegistry
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_parent, get_prim_path, is_prim_root_path, is_prim_ancestral
import omni.usd.commands
from pxr import Usd, UsdGeom
import numpy as np
import builtins
from omni.isaac.core.utils.stage import get_current_stage, update_stage
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.isaac.core.utils.stage import add_reference_to_stage
from typing import Optional, Tuple
import gc
from igibson.utils.python_utils import classproperty, Serializable, Registerable
from igibson.utils.registry_utils import SerializableRegistry
from igibson.objects.object_base import BaseObject
from igibson.objects.dataset_object import DatasetObject
from igibson.systems import SYSTEMS_REGISTRY

# from igibson.objects.particles import Particle
# from igibson.objects.visual_marker import VisualMarker
from igibson.robots.robot_base import BaseRobot

# Global dicts that will contain mappings
REGISTERED_SCENES = OrderedDict()


class Scene(Serializable, Registerable, metaclass=ABCMeta):
    """
    Base class for all Scene objects.
    Contains the base functionalities and the functions that all derived classes need to implement.
    """

    def __init__(self):
        # Store internal variables
        self._loaded = False                    # Whether this scene exists in the stage or not
        self._initialized = False               # Whether this scene has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._registry = None
        self.floor_body_ids = []  # List of ids of the floor_heights

    @property
    def stage(self) -> Usd.Stage:
        """[summary]

        Returns:
            Usd.Stage: [description]
        """
        return get_current_stage()

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
    def robot_registry(self):
        """
        Returns:
            SerializableRegistry: Robot registry containing all active robots in the scene
        """
        return self._registry(key="name", value="robot_registry")

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
        return self.robot_registry.objects

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
        # Only use name and handle by default, the most general
        return ["name", "prim_path", "handle"]

    @property
    def object_registry_group_keys(self):
        """
        Returns:
            list of str: Keys with which to index into the object registry. These should be valid public attributes of
                prims that we can use as grouping IDs to reference prims, e.g., prim.in_rooms
        """
        # None by default
        return []

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
        :return: a list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        # Make sure simulator is stopped
        assert simulator.is_stopped(), "Simulator should be stopped when loading this scene!"

        # Do not override this function. Override _load instead.
        if self._loaded:
            raise ValueError("This scene is already loaded.")

        # Creat the registry for tracking all objects in the scene
        self._registry = self._create_registry()

        prims = self._load(simulator)
        self._loaded = True

        # Always stop the sim if we started it internally
        if not simulator.is_stopped():
            simulator.stop()

        return prims

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

        # Add registry for objects
        registry.add(obj=SerializableRegistry(
            name="object_registry",
            class_types=BaseObject,
            default_key="name",
            unique_keys=self.object_registry_unique_keys,
            group_keys=self.object_registry_group_keys,
        ))

        # Add registry for robots
        registry.add(obj=SerializableRegistry(
            name="robot_registry",
            class_types=BaseRobot,
            default_key="name",
            unique_keys=None,
            group_keys=["model_name"],
        ))

        # Add registry for systems -- this is already created externally, so we just pull it directly
        registry.add(obj=SYSTEMS_REGISTRY)

        return registry

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
        :return: a list of objects with the given state
        """
        return [item for item in self.objects if hasattr(item, "states") and state in item.states]

    def _add_object(self, obj):
        """
        Add an object to the scene's internal object tracking mechanisms.

        Note that if the scene is not loaded, it should load this added object alongside its other objects when
        scene.load() is called. The object should also be accessible through scene.objects.

        :param obj: the object to load
        """
        pass

    def add_object(self, obj, simulator, _is_call_from_simulator=False):
        """
        Add an object to the scene, loading it if the scene is already loaded.

        Note that calling add_object to an already loaded scene should only be done by the simulator's import_object()
        function.

        :param obj: the object to load
        :param simulator: the simulator to add the object to
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

        # Add this object to our registry based on its type
        if isinstance(obj, BaseRobot):
            self.robot_registry.add(obj)
        else:
            self.object_registry.add(obj)

        # Run any additional scene-specific logic with the created object
        self._add_object(obj)

        return prim

    # def clear(self):
    #     """
    #     Clears all scene data from this scene
    #     """
    #     # Remove all object, robot, system info


    def remove_object(self, obj):
        # Remove from this registry
        self.object_registry.remove(obj)

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
        Resets this scene. Default is no-op
        """
        pass

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
        While Gibson v1 scenes can have several floor_heights, the EmptyScene, StadiumScene and scenes from iGibson
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
    ) -> None:
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

        Returns:
            [type]: [description]
        """
        plane = GroundPlane(
            prim_path=prim_path,
            name=name,
            z_position=z_position,
            size=size,
            color=np.array(color),
            visible=visible,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )

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
        end_idx = self._registry.state_size
        return self._registry.deserialize(state=state[:end_idx]), end_idx

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("Scene")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_SCENES
        return REGISTERED_SCENES
