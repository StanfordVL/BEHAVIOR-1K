import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
import omnigibson.utils.transform_utils as T
import omnigibson as og


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(AbsoluteObjectState, UpdateStateMixin):
    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @staticmethod
    def get_optional_dependencies():
        # Note that we don't include OnFire as the optional dependency because OnFire also depends on Temperature.
        # As a result, the temperature effect of objects that are on fire will have one step delay.
        return AbsoluteObjectState.get_optional_dependencies() + [HeatSourceOrSink]

    def __init__(self, obj):
        super(Temperature, self).__init__(obj)

        self.value = m.DEFAULT_TEMPERATURE

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        # Avoid circular import
        from omnigibson.object_states.on_fire import OnFire

        # Start at the current temperature.
        new_temperature = self.value

        # Find all heat source objects.
        affected_by_heat_source = False
        heat_source_objs = og.sim.scene.get_objects_with_state_recursive(HeatSourceOrSink)
        for obj2 in heat_source_objs:
            # Only external heat sources will affect the temperature.
            if obj2 == self.obj:
                continue

            heat_source = obj2.states.get(OnFire, obj2.states.get(HeatSourceOrSink, None))
            assert heat_source is not None, "Unknown HeatSourceOrSink subclass"

            # Compute delta to apply
            delta = heat_source.compute_temperature_delta(obj=self.obj)
            affected_by_heat_source = delta != 0
            new_temperature += delta * self._simulator.get_rendering_dt()

        # Apply temperature decay if not affected by any heat source.
        if not affected_by_heat_source:
            new_temperature += (
                (m.DEFAULT_TEMPERATURE - self.value) * m.TEMPERATURE_DECAY_SPEED * self._simulator.get_rendering_dt()
            )

        self.value = new_temperature

        # Also update cache to force syncing of temperature so get_value() is consistent with self.value
        self.update_cache(get_value_args=())

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
