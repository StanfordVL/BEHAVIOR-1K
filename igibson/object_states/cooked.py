from igibson.object_states.max_temperature import MaxTemperature
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, NONE
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin

_DEFAULT_COOK_TEMPERATURE = 70


class Cooked(AbsoluteObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj, cook_temperature=_DEFAULT_COOK_TEMPERATURE):
        super(Cooked, self).__init__(obj)
        self.cook_temperature = cook_temperature
        self.value = False

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [MaxTemperature]

    def _set_value(self, new_value):
        current_max_temp = self.obj.states[MaxTemperature].get_value()
        if new_value:
            # Set at exactly the cook temperature (or higher if we have it in history)
            desired_max_temp = max(current_max_temp, self.cook_temperature)
        else:
            # Set at exactly one below cook temperature (or lower if in history).
            desired_max_temp = min(current_max_temp, self.cook_temperature - 1.0)

        self.value = new_value

        return self.obj.states[MaxTemperature].set_value(desired_max_temp)

    def _get_value(self):
        return self.value

    # Nothing needs to be done to save/load Burnt since it will happen due to
    # MaxTemperature caching.

    def _update(self):
        value = self.obj.states[MaxTemperature].get_value() >= self.cook_temperature
        if value != self.value:
            self.update_texture()
        self.value = value
