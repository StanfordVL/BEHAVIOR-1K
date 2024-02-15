from omnigibson.object_states.temperature import Temperature
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
import numpy as np
from omnigibson.utils.python_utils import classproperty


class MaxTemperature(TensorizedValueState):
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

        # Set value to be default
        self._set_value(-np.inf)

    @classmethod
    def _update_values(cls, values):
        # Value is max between stored values and current temperature values
        return np.maximum(values, Temperature.VALUES)

    @classproperty
    def value_name(cls):
        return "max_temperature"
