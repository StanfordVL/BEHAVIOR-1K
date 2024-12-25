import omnigibson as og
from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.object_state_utils import sample_kinematics


class Under(RelativeObjectState, KinematicsMixin, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(VerticalAdjacency)
        return deps

    def _set_value(self, other, new_value, reset_before_sampling=False):
        if not new_value:
            raise NotImplementedError("Under does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object under a cloth object.")

        state = og.sim.dump_state(serialized=False)

        # Possibly reset this object if requested
        if reset_before_sampling:
            self.obj.reset()

        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            if sample_kinematics("under", self.obj, other) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect if an object is under a cloth object.")

        adjacency = self.obj.states[VerticalAdjacency].get_value()
        other_adjacency = other.states[VerticalAdjacency].get_value()
        return (
            other not in adjacency.negative_neighbors
            and other in adjacency.positive_neighbors
            and self.obj not in other_adjacency.positive_neighbors
        )
