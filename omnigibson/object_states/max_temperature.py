from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.temperature import Temperature
from omnigibson.utils.python_utils import classproperty
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
    def state_size(self):
        return 1

    def _dump_state(self):
        return OrderedDict(temperature=self.value)

    def _load_state(self, state):
        self.value = state["temperature"]

    def _serialize(self, state):
        return np.array([state["temperature"]], dtype=float)

    def _deserialize(self, state):
        return OrderedDict(temperature=state[0]), 1
