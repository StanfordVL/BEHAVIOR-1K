from IPython import embed

import omnigibson as og
from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.kinematics import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanState, RelativeObjectState
from omnigibson.utils.object_state_utils import sample_kinematics


class Under(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + RelativeObjectState.get_dependencies() + [VerticalAdjacency]

    def _set_value(self, other, new_value):
        if not new_value:
            raise NotImplementedError("Under does not support set_value(False)")

        state = og.sim.dump_state(serialized=False)

        if sample_kinematics("under", self.obj, other) and self.get_value(other):
            return True
        else:
            og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        adjacency = self.obj.states[VerticalAdjacency].get_value()
        return other not in adjacency.negative_neighbors and other in adjacency.positive_neighbors
