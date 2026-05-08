import torch as th

from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.adjacency import Adjacency
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState


class NextTo(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Adjacency)
        return deps

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
        distance = th.norm(th.tensor(distance_vec, dtype=th.float32))
        objA_dims = objA_upper - objA_lower
        objB_dims = objB_upper - objB_lower
        avg_aabb_length = th.mean(objA_dims + objB_dims)

        # If the distance is longer than acceptable, return False.
        if distance > avg_aabb_length * (1.0 / 6.0):
            return False

        # Check if `other` shows up on any of self's horizontal axes (k=2..11 in Adjacency).
        adj_self = self.obj.states[Adjacency].get_value(other)
        if bool(adj_self[2:].any()):
            return True

        # Otherwise check the symmetric direction from `other` (it may be shorter than us etc.).
        adj_other = other.states[Adjacency].get_value(self.obj)
        return bool(adj_other[2:].any())
