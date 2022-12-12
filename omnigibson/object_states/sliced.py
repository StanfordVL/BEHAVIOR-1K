import numpy as np
from collections import OrderedDict
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanState


class Sliced(AbsoluteObjectState, BooleanState):
    def __init__(self, obj):
        super().__init__(obj)
        self.value = False

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        if self.value == new_value:
            return True

        if not new_value:
            raise ValueError("Cannot set sliced from True to False")

        # Transition rules handle removing / loading objects into the simulator,
        # so all we have to do is update our internal state
        self.value = new_value

        return True

    @property
    def state_size(self):
        return 1

    def _dump_state(self):
        return OrderedDict(sliced=self.value)

    def _load_state(self, state):
        self.value = state["sliced"]

    def _serialize(self, state):
        return np.array([state["sliced"]], dtype=float)

    def _deserialize(self, state):
        return OrderedDict(sliced=bool(state[0])), 1
