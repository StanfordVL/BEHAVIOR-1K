import numpy as np

from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, NONE
from igibson.object_states.temperature import Temperature
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin

_DEFAULT_FREEZE_TEMPERATURE = 0.0

# When an object is set as frozen, we will sample it between
# the freeze temperature and these offsets.
_FROZEN_SAMPLING_RANGE_MAX = -10.0
_FROZEN_SAMPLING_RANGE_MIN = -50.0


class Frozen(AbsoluteObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj, freeze_temperature=_DEFAULT_FREEZE_TEMPERATURE):
        super(Frozen, self).__init__(obj)
        self.freeze_temperature = freeze_temperature
        self.value = False

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [Temperature]

    def _set_value(self, new_value):
        self.value = new_value
        if new_value:
            temperature = np.random.uniform(
                self.freeze_temperature + _FROZEN_SAMPLING_RANGE_MIN,
                self.freeze_temperature + _FROZEN_SAMPLING_RANGE_MAX,
            )
            return self.obj.states[Temperature].set_value(temperature)
        else:
            # We'll set the temperature just one degree above freezing. Hopefully the object
            # isn't in a fridge.
            return self.obj.states[Temperature].set_value(self.freeze_temperature + 1.0)

    def _get_value(self):
        return self.value 

    # Nothing needs to be done to save/load Frozen since it will happen due to temperature caching.

    def _update(self):
        value = self.obj.states[Temperature].get_value() <= self.freeze_temperature
        if self.value != value:
            self.update_texture()
        self.value = value
