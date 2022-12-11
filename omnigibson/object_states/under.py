
from IPython import embed

import omnigibson
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

        state = self._simulator.dump_state(serialized=False)

        for _ in range(10):
            sampling_success = sample_kinematics("under", self.obj, other)
            if sampling_success:
                if self.get_value(other) != new_value:
                    sampling_success = False
                if omnigibson.debug_sampling:
                    print("Under checking", sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                self._simulator.load_state(state, serialized=False)

        return sampling_success

    def _get_value(self, other):
        adjacency = self.obj.states[VerticalAdjacency].get_value()
        return other not in adjacency.negative_neighbors and other in adjacency.positive_neighbors
