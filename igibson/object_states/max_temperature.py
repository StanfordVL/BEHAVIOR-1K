from igibson.object_states.object_state_base import AbsoluteObjectState, NONE
from igibson.object_states.temperature import Temperature
import numpy as np
from collections import OrderedDict


class MaxTemperature(AbsoluteObjectState):
    """
    This state remembers the highest temperature reached by an object.
    """

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [Temperature]

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
    def settable(self):
        return True

    def _dump_state(self):
        return OrderedDict(temperature=self.value)

    def _load_state(self, state):
        self.value = state["temperature"]

    def _serialize(self, state):
        return np.array([state["temperature"]])

    def _deserialize(self, state):
        return OrderedDict(temperature=state[0]), 1
