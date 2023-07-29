from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.temperature import Temperature
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
import numpy as np


class MaxTemperature(AbsoluteObjectState, UpdateStateMixin):
    """
    This state remembers the highest temperature reached by an object.
    """

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Temperature)
        return deps

    def __init__(self, obj):
        super(MaxTemperature, self).__init__(obj)

        self.value = float("-inf")

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        self.value = max(self.obj.states[Temperature].get_value(), self.value)

    @property
    def state_size(self):
        return 1

    def _dump_state(self):
        return dict(max_temperature=self.value)

    def _load_state(self, state):
        self.value = state["max_temperature"]

    def _serialize(self, state):
        return np.array([state["max_temperature"]], dtype=float)

    def _deserialize(self, state):
        return dict(max_temperature=state[0]), 1
