
from IPython import embed

import igibson
from igibson.object_states.adjacency import VerticalAdjacency
from igibson.object_states.memoization import PositionalValidationMemoizedObjectStateMixin
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState
from igibson.object_states.touching import Touching
from igibson.object_states.utils import clear_cached_states, sample_kinematics
from igibson.utils.utils import restoreState


class OnTop(PositionalValidationMemoizedObjectStateMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies() + [Touching, VerticalAdjacency]

    def _set_value(self, other, new_value, use_ray_casting_method=False):
        state_id = p.saveState()

        for _ in range(10):
            sampling_success = sample_kinematics(
                "onTop", self.obj, other, new_value, use_ray_casting_method=use_ray_casting_method
            )
            if sampling_success:
                clear_cached_states(self.obj)
                clear_cached_states(other)
                if self.get_value(other) != new_value:
                    sampling_success = False
                if igibson.debug_sampling:
                    print("OnTop checking", sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                restoreState(state_id)

        p.removeState(state_id)

        return sampling_success

    def _get_value(self, other, use_ray_casting_method=False):
        del use_ray_casting_method

        # Touching is the less costly of our conditions.
        # Check it first.
        if not self.obj.states[Touching].get_value(other):
            return False

        # Then check vertical adjacency - it's the second least
        # costly.
        other_bids = set(other.get_body_ids())
        adjacency = self.obj.states[VerticalAdjacency].get_value()
        return not other_bids.isdisjoint(adjacency.negative_neighbors) and other_bids.isdisjoint(
            adjacency.positive_neighbors
        )
