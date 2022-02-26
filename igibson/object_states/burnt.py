from igibson.object_states.max_temperature import MaxTemperature
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, NONE
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin
from igibson.utils.utils import transform_texture
import numpy as np

_DEFAULT_BURN_TEMPERATURE = 200


class Burnt(AbsoluteObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj, burn_temperature=_DEFAULT_BURN_TEMPERATURE):
        super(Burnt, self).__init__(obj)
        self.burn_temperature = burn_temperature

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [MaxTemperature]

    @staticmethod
    def create_transformed_texture(diffuse_tex_filename, diffuse_tex_filename_transformed):
        # 0.8 mixture with black
        transform_texture(diffuse_tex_filename, diffuse_tex_filename_transformed, 0.8, (0, 0, 0))

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

    def _update(self):
        self.update_texture()
