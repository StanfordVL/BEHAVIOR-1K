import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.max_temperature import MaxTemperature
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_BURN_TEMPERATURE = 200


class Burnt(AbsoluteObjectState, BooleanStateMixin):
    def __init__(self, obj, burn_temperature=None):
        super(Burnt, self).__init__(obj)
        self.burn_temperature = burn_temperature if burn_temperature is not None else m.DEFAULT_BURN_TEMPERATURE

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(MaxTemperature)
        return deps

    def _set_value(self, new_value):
        current_max_temp = self.obj.states[MaxTemperature].get_value()
        if new_value:
            # Set at exactly the burnt temperature (or higher if we have it in history)
            desired_max_temp = max(current_max_temp, self.burn_temperature)
        else:
            # Set at exactly one below burnt temperature (or lower if in history).
            desired_max_temp = min(current_max_temp, self.burn_temperature - 1.0)

        return self.obj.states[MaxTemperature].set_value(desired_max_temp)

    def _get_value(self):
        return self.obj.states[MaxTemperature].get_value() >= self.burn_temperature

    @staticmethod
    def get_texture_change_params():
        # Decrease all channels by 0.3 (to make it black)
        albedo_add = -0.3
        # No final scaling
        diffuse_tint = th.tensor([1.0, 1.0, 1.0])
        return albedo_add, diffuse_tint

    # Nothing needs to be done to save/load Burnt since it will happen due to
    # MaxTemperature caching.
