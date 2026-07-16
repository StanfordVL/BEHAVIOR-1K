"""Newton-native scene API.

The public ``Scene`` path remains ``omnigibson.scenes.scene_base.Scene``, but
runtime identity is no longer a USD prim tree. Scene objects are declarative
asset specs before build and runtime Newton entities after the simulator binds
the scene to its ``EntityRegistry``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch as th


REGISTERED_SCENES = {}


class UnsupportedSceneFeature(NotImplementedError):
    """Raised for legacy scene APIs that depended on Isaac/PhysX prims."""


@dataclass(frozen=True)
class SceneObjectSpec:
    """An object instance to import into a Newton scene."""

    name: str
    object_type: str
    asset: object | None = None
    position: tuple[float, float, float] = (1.0, 0.0, 0.5)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: float | tuple[float, float, float] = 1.0
    usd_path: Path | None = None
    fixed_base: bool = False
    visual_only: bool = False

    @property
    def category(self):
        return self.asset.category if self.asset is not None else "usd_object"

    @property
    def kind(self):
        return "object"

    @property
    def prim_path(self):
        return f"/{self.name}"


@dataclass(frozen=True)
class SceneRobotSpec:
    """A robot instance to import into a Newton scene."""

    name: str
    asset: object
    position: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    asset_path: Path | None = None
    fixed_base: bool = False
    action_normalize: bool = True
    obs_modalities: tuple[str, ...] = ("rgb",)

    @property
    def category(self):
        return "robot"

    @property
    def kind(self):
        return "robot"

    @property
    def prim_path(self):
        return f"/{self.name}"


@dataclass(frozen=True)
class SceneLightSpec:
    """A viewer light declaration in a Newton scene."""

    name: str
    light_type: str = "Sphere"
    position: tuple[float, float, float] = (0.0, 0.0, 2.0)
    intensity: float = 1.0e5
    radius: float = 0.01


class Scene:
    """Newton-native scene with the original OmniGibson scene entry point.

    The class keeps high-level scene concepts such as object lists, robot lists,
    registries, metadata, reset, and save hooks. APIs whose behavior was
    fundamentally USD-prim/Isaac-specific are intentionally explicit
    ``NotImplementedError`` stubs until they are redesigned around Newton
    entities.
    """

    def __init__(
        self,
        name="scene",
        use_ground_plane=True,
        objects=(),
        robots=(),
        lights=(),
        scene_file=None,
        use_floor_plane=None,
        floor_plane_visible=True,
        floor_plane_color=(1.0, 1.0, 1.0),
        use_skybox=True,
        include_robots=True,
    ):
        self.name = name
        self.scene_file = scene_file
        self.use_ground_plane = use_ground_plane if use_floor_plane is None else use_floor_plane
        self.floor_plane_visible = floor_plane_visible
        self.floor_plane_color = floor_plane_color
        self.use_skybox = use_skybox
        self.include_robots = include_robots
        self.lights = tuple(lights)

        self._object_specs = tuple(objects)
        self._robot_specs = tuple(robots) if include_robots else tuple()
        self._simulator = None
        self._loaded = False
        self._initialized = False
        self._idx = None
        self._task_metadata = {}
        self._initial_entity_poses = None
        self._object_registry = _SceneObjectRegistry(self)

    @property
    def entities(self):
        return (*self.robots, *self.objects)

    @property
    def registry(self):
        return self._object_registry

    @property
    def object_registry(self):
        return self._object_registry

    @property
    def system_registry(self):
        raise UnsupportedSceneFeature("Scene systems are not implemented in the Newton runtime yet.")

    @property
    def objects(self):
        return self._object_specs

    @property
    def robots(self):
        if self._simulator is not None and self._simulator.model is not None:
            return tuple(self._simulator.robots)
        return self._robot_specs

    @property
    def object_specs(self):
        return self._object_specs

    @property
    def robot_specs(self):
        return self._robot_specs

    @property
    def systems(self):
        return tuple()

    @property
    def available_systems(self):
        return {}

    @property
    def active_systems(self):
        return {}

    @property
    def updated_state_objects(self):
        return set()

    @property
    def loaded(self):
        return self._loaded

    @property
    def initialized(self):
        return self._initialized

    @property
    def idx(self):
        if self._idx is None:
            raise AssertionError("This scene is not loaded yet.")
        return self._idx

    @property
    def use_floor_plane(self):
        return self.use_ground_plane

    @property
    def n_objects(self):
        return len(self.objects)

    @property
    def n_floors(self):
        return 1

    @property
    def fixed_objects(self):
        return {obj.name: obj for obj in self.objects if getattr(obj, "fixed_base", False)}

    @property
    def prim_path(self):
        raise UnsupportedSceneFeature("Newton scenes do not have a backing USD prim path.")

    @property
    def pose(self):
        return th.eye(4)

    @property
    def pose_inv(self):
        return th.eye(4)

    def bind_simulator(self, simulator):
        """Bind this scene to the simulator entity registry after Newton build."""
        self._simulator = simulator
        entities_by_name = {entity.name: entity for entity in simulator.objects}
        for obj in self._object_specs:
            entity = entities_by_name.get(obj.name)
            if entity is not None and hasattr(obj, "bind_entity"):
                obj.bind_entity(entity, scene=self)
        self._loaded = True
        self._initialized = True
        self._idx = 0 if self._idx is None else self._idx
        self._capture_initial_entity_poses()

    def unbind_simulator(self, simulator):
        if self._simulator is simulator:
            for obj in self._object_specs:
                if hasattr(obj, "unbind_entity"):
                    obj.unbind_entity()
            self._simulator = None
            self._loaded = False
            self._initialized = False

    def load(self, idx=0, **kwargs):
        self._idx = idx
        self._loaded = True
        return 0.0

    def initialize(self):
        self._initialized = True
        self._capture_initial_entity_poses()

    def clear(self):
        self._loaded = False
        self._initialized = False
        self._simulator = None
        self._initial_entity_poses = None

    def reset(self, hard=True):
        if self._initial_entity_poses is None:
            return
        for entity in self.entities:
            pose = self._initial_entity_poses.get(entity.name)
            if pose is None or not hasattr(entity, "set_position_orientation"):
                continue
            position, orientation = pose
            entity.set_position_orientation(position=position, orientation=orientation)
            if hasattr(entity, "keep_still"):
                entity.keep_still()

    def update_initial_file(self, scene_file=None):
        self._capture_initial_entity_poses()

    def save(self, json_path=None, as_dict=False):
        scene_info = {
            "metadata": {"task": self._task_metadata},
            "objects_info": {"init_info": {obj.name: obj for obj in self._object_specs}},
            "robots_info": {"init_info": {robot.name: robot for robot in self._robot_specs}},
        }
        if json_path is not None:
            raise UnsupportedSceneFeature("Saving Newton scenes to JSON is not implemented yet.")
        return scene_info if as_dict else str(scene_info)

    def restore(self, scene_file, update_initial_file=False):
        raise UnsupportedSceneFeature("Restoring Newton scenes from saved state is not implemented yet.")

    def get_task_metadata(self, key):
        return self._task_metadata.get(key, None)

    def write_task_metadata(self, key, data):
        self._task_metadata[key] = data

    def get_position_orientation(self):
        return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

    def set_position_orientation(self, position=None, orientation=None):
        if position is not None and any(float(v) != 0.0 for v in position):
            raise UnsupportedSceneFeature("Moving a Newton scene root is not implemented yet.")
        if orientation is not None and tuple(float(v) for v in orientation) != (0.0, 0.0, 0.0, 1.0):
            raise UnsupportedSceneFeature("Rotating a Newton scene root is not implemented yet.")

    def convert_world_pose_to_scene_relative(self, position, orientation):
        return position, orientation

    def convert_scene_relative_pose_to_world(self, position, orientation):
        return position, orientation

    def clear_updated_objects(self):
        return None

    def wake_scene_objects(self):
        for obj in self.objects:
            if hasattr(obj, "keep_still"):
                obj.keep_still()

    def add_object(self, obj, register=True, _batched_call=False):
        raise UnsupportedSceneFeature("Dynamic object insertion is not implemented in Newton scenes yet.")

    def remove_object(self, obj, _batched_call=False):
        raise UnsupportedSceneFeature("Dynamic object removal is not implemented in Newton scenes yet.")

    def get_system(self, system_name, force_init=True):
        raise UnsupportedSceneFeature("Scene systems are not implemented in the Newton runtime yet.")

    def clear_system(self, system_name):
        raise UnsupportedSceneFeature("Scene systems are not implemented in the Newton runtime yet.")

    def is_system_active(self, system_name):
        return False

    def is_visual_particle_system(self, system_name):
        return False

    def is_physical_particle_system(self, system_name):
        return False

    def is_fluid_system(self, system_name):
        return False

    def get_random_floor(self):
        return 0

    def get_random_point(self, floor=None, reference_point=None, robot=None):
        raise UnsupportedSceneFeature("Traversability sampling is not implemented in Newton scenes yet.")

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None):
        raise UnsupportedSceneFeature("Traversability planning is not implemented in Newton scenes yet.")

    def get_floor_height(self, floor=0):
        return 0.0

    def _capture_initial_entity_poses(self):
        poses = {}
        for entity in self.entities:
            if hasattr(entity, "get_position_orientation"):
                poses[entity.name] = entity.get_position_orientation()
        self._initial_entity_poses = poses


class _SceneObjectRegistry:
    def __init__(self, scene):
        self._scene = scene

    @property
    def objects(self):
        return tuple(self._scene.objects) + tuple(self._scene.robots)

    @property
    def object_names(self):
        return {obj.name for obj in self.objects}

    def __call__(self, key, value, default_val=None):
        matches = []
        for obj in self.objects:
            if key == "name" and obj.name == value:
                return obj
            if key == "prim_path" and getattr(obj, "prim_path", None) == value:
                return obj
            if key == "category" and getattr(obj, "category", None) == value:
                matches.append(obj)
            elif key == "kind" and getattr(obj, "kind", None) == value:
                matches.append(obj)
            elif key == "fixed_base" and getattr(obj, "fixed_base", None) == value:
                matches.append(obj)
        if key in {"category", "kind", "fixed_base"}:
            return tuple(matches) if matches else default_val
        if key in {"states", "abilities", "in_rooms", "prim_type", "uuid"}:
            return default_val
        raise KeyError(f"Newton scene object registry does not support key {key!r} yet.")

    def get_dict(self, key):
        if key == "name":
            return {obj.name: obj for obj in self.objects}
        raise KeyError(f"Newton scene object registry does not support key {key!r} yet.")

    def add(self, obj):
        raise UnsupportedSceneFeature("Dynamic object registration is not implemented in Newton scenes yet.")

    def remove(self, obj):
        raise UnsupportedSceneFeature("Dynamic object registration is not implemented in Newton scenes yet.")

    def object_is_registered(self, obj):
        return obj.name in self.object_names


class TraversableScene(Scene):
    """Compatibility alias for legacy traversable-scene imports."""


class StaticTraversableScene(TraversableScene):
    """Compatibility alias for legacy static traversable-scene imports."""


class InteractiveTraversableScene(TraversableScene):
    """Compatibility alias for legacy interactive traversable-scene imports."""


REGISTERED_SCENES.update(
    {
        "Scene": Scene,
        "TraversableScene": TraversableScene,
        "StaticTraversableScene": StaticTraversableScene,
        "InteractiveTraversableScene": InteractiveTraversableScene,
    }
)


NewtonObjectSpec = SceneObjectSpec
NewtonDatasetObjectSpec = SceneObjectSpec
NewtonRobotSpec = SceneRobotSpec
NewtonLightSpec = SceneLightSpec
NewtonSceneSpec = Scene


__all__ = [
    "InteractiveTraversableScene",
    "NewtonDatasetObjectSpec",
    "NewtonLightSpec",
    "NewtonObjectSpec",
    "NewtonRobotSpec",
    "NewtonSceneSpec",
    "REGISTERED_SCENES",
    "Scene",
    "SceneLightSpec",
    "SceneObjectSpec",
    "SceneRobotSpec",
    "StaticTraversableScene",
    "TraversableScene",
    "UnsupportedSceneFeature",
]
