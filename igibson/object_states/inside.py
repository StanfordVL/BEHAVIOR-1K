
from IPython import embed

import igibson
from igibson.object_states.aabb import AABB
from igibson.object_states.adjacency import HorizontalAdjacency, VerticalAdjacency, flatten_planes
from igibson.object_states.kinematics import KinematicsMixin
from igibson.object_states.memoization import PositionalValidationMemoizedObjectStateMixin
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState
from igibson.object_states.pose import Pose
from igibson.object_states.utils import sample_kinematics
from igibson.utils.usd_utils import BoundingBoxAPI
from igibson.utils.utils import restoreState


class Inside(PositionalValidationMemoizedObjectStateMixin, KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [AABB, Pose, HorizontalAdjacency, VerticalAdjacency]

    def _set_value(self, other, new_value, use_ray_casting_method=False):
        state = self.simulator.dump_state(serialized=True)

        for _ in range(10):
            sampling_success = sample_kinematics(
                "inside", self.obj, other, new_value, use_ray_casting_method=use_ray_casting_method
            )
            if sampling_success:
                self.obj.clear_cached_states()
                other.clear_cached_states()
                if self.get_value(other) != new_value:
                    sampling_success = False
                if igibson.debug_sampling:
                    print("Inside checking", sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                self.simulator.load_state(state, serialized=True)

        return sampling_success

    def _get_value(self, other, use_ray_casting_method=False):
        del use_ray_casting_method

        # First check that the inner object's position is inside the outer's AABB.
        # Since we usually check for a small set of outer objects, this is cheap.
        # Also note that this produces garbage values for fixed objects - but we are
        # assuming none of our inside-checking objects are fixed.
        inner_object_pos, _ = self.obj.states[Pose].get_value()
        outer_object_AABB = other.states[AABB].get_value()

        if not BoundingBoxAPI.aabb_contains_point(inner_object_pos, outer_object_AABB):
            return False

        # Our definition of inside: an object A is inside an object B if there
        # exists a 3-D coordinate space in which object B can be found on both
        # sides of object A in at least 2 out of 3 of the coordinate axes. To
        # check this, we sample a bunch of coordinate systems (for the sake of
        # simplicity, all have their 3rd axes aligned with the Z axis but the
        # 1st and 2nd axes are free.
        vertical_adjacency = self.obj.states[VerticalAdjacency].get_value()
        horizontal_adjacency = self.obj.states[HorizontalAdjacency].get_value()

        # First, check if the body can be found on both sides in Z
        body_ids = set(other.get_body_ids())
        on_both_sides_Z = not body_ids.isdisjoint(vertical_adjacency.negative_neighbors) and not body_ids.isdisjoint(
            vertical_adjacency.positive_neighbors
        )
        if on_both_sides_Z:
            # If the object is on both sides of Z, we already found 1 axis, so just
            # find another axis where the object is on both sides.
            on_both_sides_in_any_axis = any(
                (
                    not body_ids.isdisjoint(adjacency_list.positive_neighbors)
                    and not body_ids.isdisjoint(adjacency_list.negative_neighbors)
                )
                for adjacency_list in flatten_planes(horizontal_adjacency)
            )
            return on_both_sides_in_any_axis

        # If the object was not on both sides of Z, then we need to look at each
        # plane and try to find one where the object is on both sides of both
        # axes in that plane.
        on_both_sides_of_both_axes_in_any_plane = any(
            not body_ids.isdisjoint(adjacency_list_by_axis[0].positive_neighbors)
            and not body_ids.isdisjoint(adjacency_list_by_axis[0].negative_neighbors)
            and not body_ids.isdisjoint(adjacency_list_by_axis[1].positive_neighbors)
            and not body_ids.isdisjoint(adjacency_list_by_axis[1].negative_neighbors)
            for adjacency_list_by_axis in horizontal_adjacency
        )
        return on_both_sides_of_both_axes_in_any_plane
