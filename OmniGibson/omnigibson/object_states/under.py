import omnigibson as og
from omnigibson.object_states.adjacency import Adjacency
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.object_state_utils import sample_kinematics


class Under(RelativeObjectState, KinematicsMixin, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Adjacency)
        return deps

    def _set_value(self, other, new_value, reset_before_sampling=False, use_trav_map=False):
        if not new_value:
            raise NotImplementedError("Under does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object under a cloth object.")

        state = og.sim.dump_state(serialized=False)

        # Possibly reset this object if requested
        if reset_before_sampling:
            self.obj.reset()

        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            if sample_kinematics("under", self.obj, other, use_trav_map=use_trav_map) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect if an object is under a cloth object.")

        # Adjacency axis layout: k=0 is +Z (other above self), k=1 is -Z (other below self).
        # Under(self, other) is true when `other` is above self (and not below), and self is
        # not above `other` either (otherwise both could see each other on +Z, ambiguous).
        adj = self.obj.states[Adjacency].get_value(other)
        adj_other = other.states[Adjacency].get_value(self.obj)
        return bool(adj[0]) and not bool(adj[1]) and not bool(adj_other[0])
