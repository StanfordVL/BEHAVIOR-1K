import torch as th

import omnigibson as og
from omnigibson.macros import macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.usd_utils import RigidContactAPI


class Inside(RelativeObjectState, KinematicsMixin, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB})
        return deps

    def _set_value(self, other, new_value, reset_before_sampling=False):
        if not new_value:
            raise NotImplementedError("Inside does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object inside a cloth object.")

        container_link = None
        for link in other.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                container_link = link
                break

        assert container_link is not None, f"Container object {other.name} must have a fillable meta link"

        state = og.sim.dump_state(serialized=False)

        if reset_before_sampling:
            self.obj.reset()

        aabb_low, aabb_high = container_link.visual_aabb

        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            pos = None
            orientation = (
                self.obj.sample_orientation()
                if (hasattr(self.obj, "orientations") and self.obj.orientations is not None)
                else th.tensor([0, 0, 0, 1.0])
            )

            self.obj.set_position_orientation(position=th.tensor([100, 100, 10]), orientation=orientation)
            self.obj.keep_still()
            og.sim.step_physics()

            for _ in range(os_m.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS):
                pos = aabb_low + th.rand(3) * (aabb_high - aabb_low)

                if not container_link.check_points_in_volume(pos.unsqueeze(0)).item():
                    continue

                pos[2] += 0.05  # Add a small offset to ensure the object is above the bottom of the container
                self.obj.set_position_orientation(position=pos, orientation=orientation)
                self.obj.keep_still()

                n_steps_max = int(0.5 / og.sim.get_physics_dt())
                i = 0
                while (
                    not RigidContactAPI.is_in_contact(
                        scene_idx=self.obj.scene.idx,
                        query_set=[self.obj],
                        with_set=None,
                        ignore_set=None,
                        current_only=False,
                    )
                    and i < n_steps_max
                ):
                    og.sim.step_physics()
                    i += 1
                self.obj.keep_still()
                other.keep_still()

                for i in range(5):
                    og.sim.step_physics()
                i = 0
                while th.norm(self.obj.get_linear_velocity()) > 1e-3 and i < n_steps_max:
                    og.sim.step_physics()
                    i += 1

                og.sim.render()

                if self.get_value(other):
                    return True

                break

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
