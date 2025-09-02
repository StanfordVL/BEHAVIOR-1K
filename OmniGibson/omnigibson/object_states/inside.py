import torch as th

import omnigibson as og
from omnigibson.macros import macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.object_state_utils import sample_kinematics


class Inside(RelativeObjectState, KinematicsMixin, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB})
        return deps

    def _set_value(self, other, new_value, reset_before_sampling=False, use_trav_map=False):
        if not new_value:
            raise NotImplementedError("Inside does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object inside a cloth object.")

        state = og.sim.dump_state(serialized=False)

        # Possibly reset this object if requested
        if reset_before_sampling:
            self.obj.reset()

        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            if sample_kinematics("inside", self.obj, other, use_trav_map=use_trav_map) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect if an object is inside a cloth object.")

        # First check that the inner object's position is inside the outer's AABB.
        # Since we usually check for a small set of outer objects, this is cheap
        aabb_lower, aabb_upper = self.obj.states[AABB].get_value()
        inner_object_pos = (aabb_lower + aabb_upper) / 2.0
        outer_object_aabb_lo, outer_object_aabb_hi = other.states[AABB].get_value()

        if not (
            th.le(outer_object_aabb_lo, inner_object_pos).all() and th.le(inner_object_pos, outer_object_aabb_hi).all()
        ):
            return False

        # TODO: Consider using the collision boundary points.
        # points = self.obj.collision_boundary_points_world
        points = inner_object_pos.reshape(1, 3)
        in_volume = th.zeros(points.shape[0], dtype=th.bool)
        for link in other.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                in_volume |= link.check_points_in_volume(points)

        return th.any(in_volume).item()
