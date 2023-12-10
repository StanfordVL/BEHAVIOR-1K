import omnigibson as og
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.object_state_utils import m as os_m


class OnTop(KinematicsMixin, RelativeObjectState, BooleanStateMixin):

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({Touching, VerticalAdjacency})
        return deps

    def _set_value(self, other, new_value):
        if not new_value:
            raise NotImplementedError("OnTop does not support set_value(False)")

        state = og.sim.dump_state(serialized=False)

        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            if sample_kinematics("onTop", self.obj, other) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        touching = self.obj.states[Touching].get_value(other)
        if not touching:
            return False

        adjacency = self.obj.states[VerticalAdjacency].get_value()
        other_adjacency = other.states[VerticalAdjacency].get_value()
        return other in adjacency.negative_neighbors and other not in adjacency.positive_neighbors and self.obj not in other_adjacency.negative_neighbors
