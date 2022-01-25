from abc import ABCMeta, abstractmethod

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

# from igibson.objects.particles import Particle
# from igibson.objects.visual_marker import VisualMarker
# from igibson.robots.robot_base import BaseRobot


class Scene(with_metaclass(ABCMeta)):
    """
    Base class for all Scene objects.
    Contains the base functionalities and the functions that all derived classes need to implement.
    """

    def __init__(self):
        self._scene_registry = SceneRegistry()


        self.loaded = False
        self.build_graph = False  # Indicates if a graph for shortest path has been built
        self.floor_body_ids = []  # List of ids of the floor_heights
        self.robots = []

    @property
    def stage(self) -> Usd.Stage:
        """[summary]

        Returns:
            Usd.Stage: [description]
        """
        return get_current_stage()

    @abstractmethod
    def _load(self, simulator):
        """
        Load the scene into simulator (pybullet and renderer).
        The elements to load may include: floor, building, objects, etc.

        :param simulator: the simulator to load the scene into
        :return: a list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        raise NotImplementedError()

    def load(self, simulator):
        """
        Load the scene into simulator (pybullet and renderer).
        The elements to load may include: floor, building, objects, etc.

        :param simulator: the simulator to load the scene into
        :return: a list of pybullet ids of elements composing the scene, including floors, buildings and objects
        """
        # Do not override this function. Override _load instead.
        if self.loaded:
            raise ValueError("This scene is already loaded.")

        self.loaded = True
        return self._load(simulator)

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

    @abstractmethod
    def get_objects(self):
        """
        Get the objects in the scene.

        :return: a list of objects
        """
        raise NotImplementedError()

    def get_objects_with_state(self, state):
        """
        Get the objects with a given state in the scene.

        :param state: state of the objects to get
        :return: a list of objects with the given state
        """
        return [item for item in self.get_objects() if hasattr(item, "states") and state in item.states]

    @abstractmethod
    def _add_object(self, obj):
        """
        Add an object to the scene's internal object tracking mechanisms.

        Note that if the scene is not loaded, it should load this added object alongside its other objects when
        scene.load() is called. The object should also be accessible through scene.get_objects().

        :param obj: the object to load
        """
        raise NotImplementedError()

    def add_object(self, obj, simulator, _is_call_from_simulator=False):
        """
        Add an object to the scene, loading it if the scene is already loaded.

        Note that calling add_object to an already loaded scene should only be done by the simulator's import_object()
        function.

        :param obj: the object to load
        :param simulator: the simulator to add the object to
        :param _is_call_from_simulator: whether the caller is the simulator. This should
            **not** be set by any callers that are not the Simulator class
        :return: the body ID(s) of the loaded object if the scene was already loaded, or None if the scene is not loaded
            (in that case, the object is stored to be loaded together with the scene)
        """
        if self.loaded and not _is_call_from_simulator:
            raise ValueError("To add an object to an already-loaded scene, use the Simulator's import_object function.")

        if isinstance(obj, VisualMarker) or isinstance(obj, Particle):
            raise ValueError("VisualMarker and Particle objects and subclasses should be added directly to simulator.")

        # If the scene is already loaded, we need to load this object separately. Otherwise, don't do anything now,
        # let scene._load() load the object when called later on.
        body_ids = None
        if self.loaded:
            body_ids = obj.load(simulator)

        self._add_object(obj)

        # Keeps track of all the robots separately
        if isinstance(obj, BaseRobot):
            self.robots.append(obj)

        return body_ids

    def add(self, obj: XFormPrim) -> XFormPrim:
        """[summary]

        Args:
            obj (XFormPrim): [description]

        Raises:
            Exception: [description]
            Exception: [description]

        Returns:
            XFormPrim: [description]
        """
        if self._scene_registry.name_exists(obj.name):
            raise Exception("Cannot add the object {} to the scene since its name is not unique".format(obj.name))
        if isinstance(obj, RigidPrim):
            self._scene_registry.add_rigid_object(name=obj.name, rigid_object=obj)
        elif isinstance(obj, GeometryPrim):
            self._scene_registry.add_geometry_object(name=obj.name, geometry_object=obj)
        elif isinstance(obj, Robot):
            self._scene_registry.add_robot(name=obj.name, robot=obj)
        elif isinstance(obj, Articulation):
            self._scene_registry.add_articulated_system(name=obj.name, articulated_system=obj)
        elif isinstance(obj, XFormPrim):
            self._scene_registry.add_xform(name=obj.name, xform=obj)
        else:
            raise Exception("object type is not supported yet")
        return obj

    def get_random_floor(self):
        """
        Sample a random floor among all existing floor_heights in the scene.
        While Gibson v1 scenes can have several floor_heights, the EmptyScene, StadiumScene and scenes from iGibson
        have only a single floor.

        :return: an integer between 0 and NumberOfFloors-1
        """
        return 0

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
        if Scene.object_exists(self, name=name):
            carb.log_info("ground floor already created with name {}.".format(name))
            return Scene.get_object(self, name=name)
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
        Scene.add(self, plane)
