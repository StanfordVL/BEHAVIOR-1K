"""Newton-native object descriptors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch as th


REGISTERED_OBJECTS = {}


class UnsupportedObjectFeature(NotImplementedError):
    """Raised for object APIs that still require a Newton-native redesign."""


class USDObject:
    """User-facing object descriptor backed by a Newton runtime entity after build."""

    def __init__(
        self,
        name,
        usd_path,
        encrypted=False,
        relative_prim_path=None,
        category="object",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=None,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        expected_file_hash=None,
        position=None,
        orientation=None,
        **kwargs,
    ):
        self.name = name
        self._usd_path = Path(usd_path).expanduser().resolve() if usd_path is not None else None
        self.encrypted = encrypted
        self.relative_prim_path = relative_prim_path or f"/{name}"
        self.category = category
        self.visible = visible
        self.fixed_base = bool(fixed_base)
        self.visual_only = bool(visual_only)
        self.kinematic_only = kinematic_only
        self.self_collisions = bool(self_collisions)
        self.prim_type = prim_type
        self.link_physics_materials = {} if link_physics_materials is None else link_physics_materials
        self.load_config = {} if load_config is None else dict(load_config)
        self.abilities = {} if abilities is None else dict(abilities)
        self.include_default_states = include_default_states
        self.expected_file_hash = expected_file_hash
        self.position = _to_float_tuple(position, 3, default=(1.0, 0.0, 0.5))
        self.orientation = _to_float_tuple(orientation, 4, default=(0.0, 0.0, 0.0, 1.0))
        self._scale = _normalize_scale(scale if scale is not None else self.load_config.get("scale", 1.0))
        self._entity = None
        self.scene = None
        self.uuid = _stable_uuid(name)
        self.object_type = self.__class__.__name__
        self._extra_kwargs = kwargs

    @property
    def usd_path(self):
        return self._usd_path

    @property
    def asset(self):
        return None

    @property
    def kind(self):
        return "object"

    @property
    def prim_path(self):
        return self.relative_prim_path

    @property
    def entity(self):
        return self._entity

    @property
    def scale(self):
        if self._entity is not None and hasattr(self._entity, "scale"):
            return self._entity.scale
        return self._scale.clone() if isinstance(self._scale, th.Tensor) else self._scale

    @scale.setter
    def scale(self, value):
        self._scale = _normalize_scale(value)
        if self._entity is not None and hasattr(self._entity, "scale"):
            self._entity.scale = self._scale

    @property
    def aabb_extent(self):
        return self._require_entity().aabb_extent

    @property
    def aabb_center(self):
        return self._require_entity().aabb_center

    @property
    def n_dof(self):
        return getattr(self._require_entity(), "n_dof", 0)

    @property
    def n_joints(self):
        return self.n_dof

    @property
    def joint_names(self):
        return getattr(self._require_entity(), "joint_names", [])

    @property
    def control_limits(self):
        return self._require_entity().control_limits

    @property
    def states(self):
        raise UnsupportedObjectFeature("Object states are not implemented in the Newton runtime yet.")

    @property
    def joints(self):
        return self._require_entity().joints

    @property
    def links(self):
        return self._require_entity().links

    @property
    def root_link(self):
        return self._require_entity().root_link

    def bind_entity(self, entity, scene=None):
        self._entity = entity
        self.scene = scene
        if self._scale is not None and hasattr(entity, "scale"):
            entity.scale = self._scale
        return self

    def unbind_entity(self):
        self._entity = None
        self.scene = None

    def get_position_orientation(self, *args, **kwargs):
        if self._entity is None:
            return th.tensor(self.position, dtype=th.float32), th.tensor(self.orientation, dtype=th.float32)
        return self._entity.get_position_orientation(*args, **kwargs)

    def set_position_orientation(self, position=None, orientation=None, *args, **kwargs):
        if position is not None:
            self.position = _to_float_tuple(position, 3)
        if orientation is not None:
            self.orientation = _to_float_tuple(orientation, 4)
        if self._entity is not None:
            return self._entity.set_position_orientation(position=position, orientation=orientation, *args, **kwargs)
        return None

    def set_joint_positions(self, *args, **kwargs):
        return self._require_entity().set_joint_positions(*args, **kwargs)

    def get_joint_positions(self, *args, **kwargs):
        return self._require_entity().get_joint_positions(*args, **kwargs)

    def set_joint_velocities(self, *args, **kwargs):
        return self._require_entity().set_joint_velocities(*args, **kwargs)

    def get_joint_velocities(self, *args, **kwargs):
        return self._require_entity().get_joint_velocities(*args, **kwargs)

    def keep_still(self):
        if self._entity is not None:
            self._entity.keep_still()

    def wake(self):
        self.keep_still()

    def load(self, scene):
        self.scene = scene
        return self

    def remove(self):
        self.unbind_entity()

    def get_init_info(self):
        return {
            "class_name": self.__class__.__name__,
            "args": {
                "name": self.name,
                "usd_path": str(self.usd_path) if self.usd_path is not None else None,
                "category": self.category,
                "scale": _scale_to_python(self._scale),
                "fixed_base": self.fixed_base,
                "visual_only": self.visual_only,
                "position": self.position,
                "orientation": self.orientation,
            },
        }

    def _require_entity(self):
        if self._entity is None:
            raise RuntimeError(f"Object {self.name!r} is not bound to a Newton entity yet.")
        return self._entity


class StatefulObject(USDObject):
    """Compatibility alias for object-state-enabled legacy imports."""


REGISTERED_OBJECTS.update({"USDObject": USDObject, "StatefulObject": StatefulObject})


def _to_float_tuple(value, length, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"Expected {length} values, got None.")
        return tuple(float(v) for v in default)
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().tolist()
    result = tuple(float(v) for v in value)
    if len(result) != length:
        raise ValueError(f"Expected {length} values, got {result}.")
    return result


def _normalize_scale(scale):
    if isinstance(scale, th.Tensor):
        scale = scale.detach().cpu()
        if scale.numel() == 1:
            return th.ones(3, dtype=th.float32) * float(scale.item())
        if scale.numel() == 3:
            return scale.to(dtype=th.float32).reshape(3)
    if isinstance(scale, (int, float)):
        return th.ones(3, dtype=th.float32) * float(scale)
    values = tuple(float(v) for v in scale)
    if len(values) != 3:
        raise ValueError(f"Expected scalar or 3-vector scale, got {scale!r}.")
    return th.tensor(values, dtype=th.float32)


def _scale_to_python(scale):
    if isinstance(scale, th.Tensor):
        values = scale.detach().cpu().tolist()
        return values[0] if len(values) == 3 and values[0] == values[1] == values[2] else values
    return scale


def _stable_uuid(name):
    return int(hashlib.md5(str(name).encode("utf-8")).hexdigest()[:8], 16)
