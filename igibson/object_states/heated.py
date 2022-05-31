import numpy as np

from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, NONE
from igibson.object_states.temperature import Temperature
from igibson.object_states.steam_state_mixin import SteamStateMixin

_DEFAULT_HEAT_TEMPERATURE = 40

# When an object is set as heated, we will sample it between
# the heat temperature and these offsets.
_HEATED_SAMPLING_RANGE_MIN = 10.0
_HEATED_SAMPLING_RANGE_MAX = 20.0


class Heated(AbsoluteObjectState, BooleanState, SteamStateMixin):
    def __init__(self, obj, heat_temperature=_DEFAULT_HEAT_TEMPERATURE):
        super(Heated, self).__init__(obj)
        self.heat_temperature = heat_temperature

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [Temperature]

    def _set_value(self, new_value):
        if new_value:
            temperature = np.random.uniform(
                self.heat_temperature + _HEATED_SAMPLING_RANGE_MIN,
                self.heat_temperature + _HEATED_SAMPLING_RANGE_MAX,
            )
            return self.obj.states[Temperature].set_value(temperature)
        else:
            # We'll set the temperature just one degree below heating.
            return self.obj.states[Temperature].set_value(self.heat_temperature - 1.0)

    def _get_value(self):
        return self.obj.states[Temperature].get_value() >= self.heat_temperature

    # Nothing needs to be done to save/load Heated since it will happen due to temperature caching.

    def _update(self):
        self.update_steam()
