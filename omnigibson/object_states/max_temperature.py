from omnigibson.object_states.temperature import Temperature
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
import numpy as np
from omnigibson.utils.python_utils import classproperty


class MaxTemperature(TensorizedValueState):
    """
    This state remembers the highest temperature reached by an object.
    """

    # list: Array of Temperature.VALUE indices that correspond to the internally tracked MaxTemperature objects
    TEMPERATURE_IDXS = None

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Temperature)
        return deps

    @classmethod
    def global_initialize(cls):
        # Call super first
        super().global_initialize()

        # Initialize other global variables
        cls.TEMPERATURE_IDXS = []

    @classmethod
    def global_clear(cls):
        # Call super first
        super().global_clear()

        # Clear other internal state
        cls.TEMPERATURE_IDXS = None

    @classmethod
    def _add_obj(cls, obj):
        # Call super first
        super()._add_obj(obj=obj)

        # Add to temperature index
        cls.TEMPERATURE_IDXS.append(Temperature.OBJ_IDXS[obj.name])

    @classmethod
    def _remove_obj(cls, obj):
        # Grab idx we'll delete before the object is deleted
        deleted_idx = cls.OBJ_IDXS[obj.name]

        # Remove from temperature index
        del cls.TEMPERATURE_IDXS[deleted_idx]

        # Call super
        super()._remove_obj(obj=obj)

    @classmethod
    def _update_values(cls, values):
        # Value is max between stored values and current temperature values
        return np.maximum(values, Temperature.VALUES[cls.TEMPERATURE_IDXS])

    @classproperty
    def value_name(cls):
        return "max_temperature"

    def __init__(self, obj):
        super(MaxTemperature, self).__init__(obj)

        # Set value to be default
        self._set_value(-np.inf)
