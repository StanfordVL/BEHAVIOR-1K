
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
        state = self._simulator.dump_state(serialized=False)

        for _ in range(10):
            sampling_success = sample_kinematics("under", self.obj, other, new_value)
            if sampling_success:
                self.obj.clear_cached_states()
                other.clear_cached_states()
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
        # Call kinematics super call first to make sure poses are cached
        _ = super()._get_value(other)

        other_prim_paths = set(other.link_prim_paths)
        adjacency = self.obj.states[VerticalAdjacency].get_value()
        return not other_prim_paths.isdisjoint(adjacency.positive_neighbors) and other_prim_paths.isdisjoint(
            adjacency.negative_neighbors
        )
