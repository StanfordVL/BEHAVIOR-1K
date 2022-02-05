
from IPython import embed

import igibson
from igibson.object_states.adjacency import VerticalAdjacency
from igibson.object_states.memoization import PositionalValidationMemoizedObjectStateMixin
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState
from igibson.object_states.utils import sample_kinematics
from igibson.utils.utils import restoreState


class Under(PositionalValidationMemoizedObjectStateMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies() + [VerticalAdjacency]

    def _set_value(self, other, new_value):
        state_id = p.saveState()

        for _ in range(10):
            sampling_success = sample_kinematics("under", self.obj, other, new_value)
            if sampling_success:
                self.obj.clear_cached_states()
                other.clear_cached_states()
                if self.get_value(other) != new_value:
                    sampling_success = False
                if igibson.debug_sampling:
                    print("Under checking", sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def _get_value(self, other):
        adjacency = self.obj.states[VerticalAdjacency].get_value()
        other_ids = set(other.get_body_ids())
        return not other_ids.isdisjoint(adjacency.positive_neighbors) and other_ids.isdisjoint(
            adjacency.negative_neighbors
        )
