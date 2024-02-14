import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
import omnigibson as og


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(AbsoluteObjectState, UpdateStateMixin):

    def __init__(self, obj):
        # Run super first
        super(Temperature, self).__init__(obj)

        self.value = m.DEFAULT_TEMPERATURE

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        deps.add(HeatSourceOrSink)
        return deps

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        # Apply temperature decay
        self.value += (m.DEFAULT_TEMPERATURE - self.value) * m.TEMPERATURE_DECAY_SPEED * og.sim.get_rendering_dt()

    def update_temperature_from_heatsource_or_sink(self, temperature, rate):
        """
        Updates this object's internal temperature based on @temperature and @rate

        Args:
            temperature (float): Heat source / sink temperature
            rate (float): Heating rate of the source / sink
        """
        self.value += (temperature - self.value) * rate * og.sim.get_rendering_dt()

    @property
    def state_size(self):
        return 1

    # For this state, we simply store its value.
    def _dump_state(self):
        return dict(temperature=self.value)

    def _load_state(self, state):
        self.value = state["temperature"]

    def _serialize(self, state):
        return np.array([state["temperature"]], dtype=float)

    def _deserialize(self, state):
        return dict(temperature=state[0]), 1
