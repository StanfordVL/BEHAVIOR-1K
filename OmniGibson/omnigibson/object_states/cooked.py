import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.max_temperature import MaxTemperature
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_COOK_TEMPERATURE = 70


class Cooked(AbsoluteObjectState, BooleanStateMixin):
    def __init__(self, obj, cook_temperature=None):
        super(Cooked, self).__init__(obj)
        self.cook_temperature = cook_temperature if cook_temperature is not None else m.DEFAULT_COOK_TEMPERATURE

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(MaxTemperature)
        return deps

    def _set_value(self, new_value):
        current_max_temp = self.obj.states[MaxTemperature].get_value()
        if new_value:
            # Set at exactly the cook temperature (or higher if we have it in history)
            desired_max_temp = max(current_max_temp, self.cook_temperature)
        else:
            # Set at exactly one below cook temperature (or lower if in history).
            desired_max_temp = min(current_max_temp, self.cook_temperature - 1.0)

        return self.obj.states[MaxTemperature].set_value(desired_max_temp)

    def _get_value(self):
        return self.obj.states[MaxTemperature].get_value() >= self.cook_temperature

    @staticmethod
    def get_texture_change_params():
        # Increase all channels by 0.1
        albedo_add = 0.1
        # Then scale up "brown" color and scale down others
        diffuse_tint = th.tensor([1.5, 0.75, 0.25])
        return albedo_add, diffuse_tint

    # Nothing needs to be done to save/load Burnt since it will happen due to
    # MaxTemperature caching.
