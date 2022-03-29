import numpy as np

from igibson.object_states.aabb import AABB
from igibson.object_states.adjacency import HorizontalAdjacency, flatten_planes
from igibson.object_states.kinematics import KinematicsMixin
from igibson.object_states.memoization import PositionalValidationMemoizedObjectStateMixin
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState


class NextTo(PositionalValidationMemoizedObjectStateMixin, KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [HorizontalAdjacency]

    def _set_value(self, other, new_value):
        raise NotImplementedError()

    def _get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert AABB in objA_states
        assert AABB in objB_states

        objA_aabb = objA_states[AABB].get_value()
        objB_aabb = objB_states[AABB].get_value()

        objA_lower, objA_upper = objA_aabb
        objB_lower, objB_upper = objB_aabb
        distance_vec = []
        for dim in range(3):
            glb = max(objA_lower[dim], objB_lower[dim])
            lub = min(objA_upper[dim], objB_upper[dim])
            distance_vec.append(max(0, glb - lub))
        distance = np.linalg.norm(np.array(distance_vec))
        objA_dims = objA_upper - objA_lower
        objB_dims = objB_upper - objB_lower
        avg_aabb_length = np.mean(objA_dims + objB_dims)

        # If the distance is longer than acceptable, return False.
        if distance > avg_aabb_length * (1.0 / 6.0):
            return False

        # Otherwise, check if the other object shows up in the adjacency list.
        adjacency_this = self.obj.states[HorizontalAdjacency].get_value()
        other_body_ids = set(other.get_body_ids())
        in_any_horizontal_adjacency_of_this = any(
            (
                not other_body_ids.isdisjoint(adjacency_list.positive_neighbors)
                or not other_body_ids.isdisjoint(adjacency_list.negative_neighbors)
            )
            for adjacency_list in flatten_planes(adjacency_this)
        )
        if in_any_horizontal_adjacency_of_this:
            return True

        # If not, check in the adjacency lists of `other`. Maybe it's shorter than us etc.
        adjacency_other = other.states[HorizontalAdjacency].get_value()
        this_body_ids = set(self.obj.get_body_ids())
        in_any_horizontal_adjacency_of_other = any(
            (
                not this_body_ids.isdisjoint(adjacency_list.positive_neighbors)
                or not this_body_ids.isdisjoint(adjacency_list.negative_neighbors)
            )
            for adjacency_list in flatten_planes(adjacency_other)
        )

        return in_any_horizontal_adjacency_of_other
