import torch as th

import omnigibson as og
from omnigibson.macros import macros, create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.usd_utils import RigidContactAPI
import omnigibson.utils.transform_utils as T


m = create_module_macros(module_path=__file__)

m.CONTAINER_POSITION_CHANGE_THRESHOLD = 0.05  # 5cm
m.CONTAINER_ORIENTATION_CHANGE_THRESHOLD = th.deg2rad(th.tensor([10.0])).item()  # 10 degrees


class Inside(RelativeObjectState, KinematicsMixin, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB})
        return deps

    def _set_value(self, other, new_value, reset_before_sampling=False):
        """
        Set the Inside state for this object with respect to another object (container).

        This samples a random position inside the container's fillable volume, places the object,
        lets it settle via physics, and verifies it's still inside.

        The sampling strategy uses a two-phase approach:
        1. First half of attempts: Sample from inset AABB (container bounds minus object extent).
           This ensures the sampled position is at the object's centroid, not near the edges,
           which makes it more likely to fit inside.
        2. Second half of attempts: Sample from full container AABB. This is a fallback for
           cases where the object is too large to fit entirely within the inset bounds.

        Args:
            other: The container object to place this object inside.
            new_value: True to set Inside state (only True is supported).
            reset_before_sampling: If True, reset this object before sampling.

        Returns:
            True if successfully placed inside, False otherwise.
        """
        if not new_value:
            raise NotImplementedError("Inside does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object inside a cloth object.")

        # Save the initial position and orientation of the container
        container_pos_initial, container_orn_initial = other.get_position_orientation()

        # Find the container's fillable meta link (fillable or openfillable)
        container_link = None
        for link in other.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                container_link = link
                break

        assert container_link is not None, f"Container object {other.name} must have a fillable meta link"

        # Save simulator state for restoration on failed attempts
        state = og.sim.dump_state(serialized=False)

        if reset_before_sampling:
            self.obj.reset()

        # Get container's fillable volume bounds in world frame
        aabb_low, aabb_high = container_link.visual_aabb
        # Get the object extent to compute inset bounds
        obj_extent = self.obj.aabb_extent

        # Inset the container AABB by half the object extent in each dimension.
        # This ensures the sampled position (used as object centroid) won't place
        # any part of the object outside the container bounds.
        inset_aabb_low = aabb_low + obj_extent / 2.0
        inset_aabb_high = aabb_high - obj_extent / 2.0

        # Calculate the total attempt count. Here we don't have a sense of high/low-level attempts,
        # so to make the same numbr of attempts as the original implementation, we just multiply
        # the two sampling parameters.
        total_attempts = os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS * os_m.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS

        for attempt_idx in range(total_attempts):
            # Sample orientation if the object supports random orientations, otherwise use default
            orientation = (
                self.obj.sample_orientation()
                if (hasattr(self.obj, "orientations") and self.obj.orientations is not None)
                else th.tensor([0, 0, 0, 1.0])
            )

            # Also add a random world Z-axis offset to the orientation
            random_z_orientation = T.axisangle2quat(th.as_tensor([0, 0, th.rand(1) * 2 * th.pi]))
            orientation = T.quat_multiply(orientation, random_z_orientation)

            # First half: use inset bounds (smarter sampling)
            # Second half: use full bounds (fallback for large objects)
            # Also fallback if inset bounds are invalid (object too large for container)
            if attempt_idx < total_attempts // 2 and th.all(inset_aabb_low < inset_aabb_high):
                pos = inset_aabb_low + th.rand(3) * (inset_aabb_high - inset_aabb_low)
            else:
                pos = aabb_low + th.rand(3) * (aabb_high - aabb_low)

            # Rejection sampling #1: Verify the sampled point is actually inside the container volume
            if not container_link.check_points_in_volume(pos.unsqueeze(0)).item():
                og.sim.load_state(state, serialized=False)
                continue

            # Add small z-offset to avoid spawning inside the container floor
            pos[2] += 0.01
            self.obj.set_position_orientation(position=pos, orientation=orientation)
            self.obj.keep_still()

            # Rejection sampling #2: Check for collision after placement
            # Step until contact is made or max steps reached (0.5 seconds of sim time)
            n_steps_max = int(0.5 / og.sim.get_physics_dt())
            step_idx = 0
            while (
                not RigidContactAPI.is_in_contact(
                    scene_idx=self.obj.scene.idx,
                    query_set=[self.obj],
                    with_set=None,
                    ignore_set=None,
                    current_only=True,
                )
                and step_idx < n_steps_max
            ):
                og.sim.step_physics()
                step_idx += 1
            self.obj.keep_still()
            other.keep_still()

            # Step a few more times to let velocity stabilize
            for _ in range(5):
                og.sim.step_physics()
            settle_step_idx = 0
            while th.norm(self.obj.get_linear_velocity()) > 1e-3 and settle_step_idx < n_steps_max:
                og.sim.step_physics()
                settle_step_idx += 1

            # Check that the container object has not moved more than the thresholds
            container_pos, container_orn = other.get_position_orientation()
            position_difference = th.norm(container_pos - container_pos_initial)
            orientation_difference = T.get_orientation_diff_in_radian(container_orn, container_orn_initial)
            if (
                position_difference > m.CONTAINER_POSITION_CHANGE_THRESHOLD
                or orientation_difference > m.CONTAINER_ORIENTATION_CHANGE_THRESHOLD
            ):
                og.sim.load_state(state, serialized=False)
                continue

            # Rejection sampling #3: Verify object is still inside after settling
            if self.get_value(other):
                return True

        # Reset the simulator state to the initial state
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
