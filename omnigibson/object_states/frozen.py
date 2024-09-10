import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin
from omnigibson.object_states.temperature import Temperature

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_FREEZE_TEMPERATURE = 0.0

# When an object is set as frozen, we will sample it between
# the freeze temperature and these offsets.
m.FROZEN_SAMPLING_RANGE_MAX = -10.0
m.FROZEN_SAMPLING_RANGE_MIN = -50.0


class Frozen(AbsoluteObjectState, BooleanStateMixin):
    def __init__(self, obj, freeze_temperature=None):
        super(Frozen, self).__init__(obj)
        self.freeze_temperature = freeze_temperature if freeze_temperature is not None else m.DEFAULT_FREEZE_TEMPERATURE

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Temperature)
        return deps

    def _set_value(self, new_value):
        if new_value:
            temp_lo, temp_hi = (
                self.freeze_temperature + m.FROZEN_SAMPLING_RANGE_MIN,
                self.freeze_temperature + m.FROZEN_SAMPLING_RANGE_MAX,
            )
            temperature = (th.rand(1) * (temp_hi - temp_lo) + temp_lo).item()
            return self.obj.states[Temperature].set_value(temperature)
        else:
            # We'll set the temperature just one degree above freezing. Hopefully the object
            # isn't in a fridge.
            return self.obj.states[Temperature].set_value(self.freeze_temperature + 1.0)

    def _get_value(self):
        return self.obj.states[Temperature].get_value() <= self.freeze_temperature

    @staticmethod
    def get_texture_change_params():
        # Increase all channels by 0.3 (to make it white)
        albedo_add = 0.3
        # No final scaling
        diffuse_tint = th.tensor([1.0, 1.0, 1.0])
        return albedo_add, diffuse_tint

    # Nothing needs to be done to save/load Frozen since it will happen due to temperature caching.
