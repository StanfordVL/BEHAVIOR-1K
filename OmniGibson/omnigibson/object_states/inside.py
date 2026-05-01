import math

import torch as th

import omnigibson as og
from omnigibson.macros import macros, create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import (
    m as os_m,
    get_reachability_sampling_context,
    is_pose_reachable_for_predicate,
)
from omnigibson.utils.usd_utils import RigidContactAPI
import omnigibson.utils.transform_utils as T


m = create_module_macros(module_path=__file__)

m.CONTAINER_POSITION_CHANGE_THRESHOLD = 0.05  # 5cm
m.CONTAINER_ORIENTATION_CHANGE_THRESHOLD = th.deg2rad(th.tensor([10.0])).item()  # 10 degrees
m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_TRANSLATION = 1e-2  # 1cm
m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_ROTATION = math.radians(1)  # 1 degree


class Inside(RelativeObjectState, KinematicsMixin, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB})
        return deps

    def _set_value(self, other, new_value, reset_before_sampling=False, use_trav_map=False):
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

        Each candidate pose passes through several rejection-sampling stages:
        1. The sampled point must lie inside the container's fillable volume.
        2. After placement and a single physics step, the object must not already be
           intersecting anything (catches interpenetration with container walls).
        3. The object must come into contact with something after placement and half a second of
           physics steps.
        4. After settling, the container's root pose must not have moved
           beyond CONTAINER_POSITION_CHANGE_THRESHOLD / CONTAINER_ORIENTATION_CHANGE_THRESHOLD.
        5. The container's articulated joints must not have moved beyond the per-DOF-type
           CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_{TRANSLATION,ROTATION} thresholds
           (catches cases where the placed object swings a door/lid or pushes a drawer).
        6. The object must still register as Inside the container after settling, and
           reachable via the traversability map if use_trav_map is enabled.

        Args:
            other: The container object to place this object inside.
            new_value: True to set Inside state (only True is supported).
            reset_before_sampling: If True, reset this object before sampling.
            use_trav_map: Whether to use traversability-based reachability checks.
        Returns:
            True if successfully placed inside, False otherwise.
        """
        if not new_value:
            raise NotImplementedError("Inside does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object inside a cloth object.")

        # Save the initial position and orientation of the container
        container_pos_initial, container_orn_initial = other.get_position_orientation()
        container_joint_positions_initial = other.get_joint_positions() if other.n_joints > 0 else None

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

        if use_trav_map:
            reachability_context = get_reachability_sampling_context(
                objB=other,
                predicate="inside",
                use_trav_map=use_trav_map,
                warn_on_scene_mismatch=False,
            )
            use_trav_map = reachability_context is not None

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
            # Step physics once so the contact buffer gets populated for the newly-placed object
            og.sim.step_physics()

            # Rejection sampling #2: Reject if the object is already intersecting anything
            # immediately after placement (e.g. interpenetration with a container wall).
            if RigidContactAPI.is_in_contact(
                scene_idx=self.obj.scene.idx,
                query_set=[self.obj],
                with_set=None,
                ignore_set=None,
                current_only=True,
            ):
                og.sim.load_state(state, serialized=False)
                continue

            # Rejection sampling #3: step until contact is made or max steps reached
            # (0.5 seconds of sim time) to let the object settle onto a resting surface.
            # If it can't get into contact by then, we reject the placement.
            n_steps_max = int(0.5 / og.sim.get_physics_dt())
            for _ in range(n_steps_max):
                og.sim.step_physics()
                if RigidContactAPI.is_in_contact(
                    scene_idx=self.obj.scene.idx,
                    query_set=[self.obj],
                    with_set=None,
                    ignore_set=None,
                    current_only=True,
                ):
                    break
            else:
                og.sim.load_state(state, serialized=False)
                continue
            self.obj.keep_still()
            other.keep_still()

            # Step a few more times to let velocity stabilize
            for _ in range(5):
                og.sim.step_physics()
            settle_step_idx = 0
            while th.norm(self.obj.get_linear_velocity()) > 1e-3 and settle_step_idx < n_steps_max:
                og.sim.step_physics()
                settle_step_idx += 1

            # Rejection sampling #4: Reject if the container's root pose drifted past the
            # position/orientation thresholds (i.e. the placed object pushed the container).
            container_pos, container_orn = other.get_position_orientation()
            position_difference = th.norm(container_pos - container_pos_initial)
            orientation_difference = T.get_orientation_diff_in_radian(container_orn, container_orn_initial)
            if (
                position_difference > m.CONTAINER_POSITION_CHANGE_THRESHOLD
                or orientation_difference > m.CONTAINER_ORIENTATION_CHANGE_THRESHOLD
            ):
                og.sim.load_state(state, serialized=False)
                continue

            # Rejection sampling #5: Reject if any of the container's articulated joints moved
            # past the per-DOF-type delta thresholds (e.g. placed object swung a lid or pushed
            # a drawer). Thresholds are applied separately for rotational and translational DOFs.
            if container_joint_positions_initial is not None:
                container_joint_positions_final = other.get_joint_positions()
                joint_thresholds = th.where(
                    other.get_joint_dof_types(),
                    m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_ROTATION,
                    m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_TRANSLATION,
                )
                container_joint_positions_delta = th.abs(
                    container_joint_positions_final - container_joint_positions_initial
                )
                if th.any(container_joint_positions_delta > joint_thresholds):
                    og.sim.load_state(state, serialized=False)
                    continue

            # Rejection sampling #6: Verify object is still inside after settling and within reach if using trav map
            if self.get_value(other):
                if use_trav_map:
                    settled_pos, _ = self.obj.get_position_orientation()
                    if not is_pose_reachable_for_predicate(
                        pos=settled_pos,
                        objB=other,
                        predicate="inside",
                        reachability_context=reachability_context,
                    ):
                        og.sim.load_state(state, serialized=False)
                        continue
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
