import torch as th

from omnigibson.object_states.temperature import Temperature
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty, torch_delete


class MaxTemperature(TensorizedValueState):
    """
    This state remembers the highest temperature reached by an object.
    """

    # th.tensor: Array of Temperature.VALUE indices that correspond to the internally tracked MaxTemperature objects
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
        cls.TEMPERATURE_IDXS = th.empty(0, dtype=int)

        # Add global callback to Temperature state so that temperature idxs will be updated
        def _update_temperature_idxs(obj):
            # Decrement all remaining temperature idxs -- they're strictly increasing so we can simply
            # subtract 1 from all downstream indices
            deleted_idx = Temperature.OBJ_IDXS[obj]
            cls.TEMPERATURE_IDXS = th.where(
                cls.TEMPERATURE_IDXS >= deleted_idx, cls.TEMPERATURE_IDXS - 1, cls.TEMPERATURE_IDXS
            )

        Temperature.add_callback_on_remove(
            name="MaxTemperature_temperature_idx_update", callback=_update_temperature_idxs
        )

    @classmethod
    def _add_obj(cls, obj):
        # Call super first
        super()._add_obj(obj=obj)

        # Add to temperature index
        cls.TEMPERATURE_IDXS = th.cat([cls.TEMPERATURE_IDXS, th.tensor([Temperature.OBJ_IDXS[obj]])])

    @classmethod
    def _remove_obj(cls, obj):
        # Grab idx we'll delete before the object is deleted
        deleted_idx = cls.OBJ_IDXS[obj]

        # Remove from temperature index
        cls.TEMPERATURE_IDXS = torch_delete(cls.TEMPERATURE_IDXS, [deleted_idx])

        # Decrement all remaining temperature idxs -- they're strictly increasing so we can simply
        # subtract 1 from all downstream indices
        if deleted_idx < len(cls.TEMPERATURE_IDXS):
            cls.TEMPERATURE_IDXS[deleted_idx:] -= 1

        # Call super
        super()._remove_obj(obj=obj)

    @classmethod
    def _update_values(cls, values):
        # Value is max between stored values and current temperature values
        return th.maximum(values, Temperature.VALUES[cls.TEMPERATURE_IDXS])

    @classproperty
    def value_name(cls):
        return "max_temperature"

    def __init__(self, obj):
        super(MaxTemperature, self).__init__(obj)

        # Set value to be default
        self._set_value(-float("inf"))
