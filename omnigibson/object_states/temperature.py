import numpy as np
from collections import OrderedDict
from omnigibson.macros import create_module_macros
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.aabb import AABB
import omnigibson.utils.transform_utils as T
import omnigibson as og


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(AbsoluteObjectState):
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
        heat_sources = og.sim.scene.get_objects_with_state(HeatSourceOrSink) | \
                       og.sim.scene.get_objects_with_state(OnFire)
        for obj2 in heat_sources:
            # Only external heat sources will affect the temperature.
            if obj2 == self.obj:
                continue

            heat_source = obj2.states[OnFire] if OnFire in obj2.states else obj2.states[HeatSourceOrSink]
            heat_source_state = heat_source.get_value()
            if heat_source_state:
                heat_source_position = heat_source.get_link_position()
                # If the object is on fire and there is no heat source position annotation, we use the AABB center
                if OnFire in obj2.states and heat_source_position is None:
                    aabb_lower, aabb_upper = obj2.states[AABB].get_value()
                    heat_source_position = (aabb_lower + aabb_upper) / 2.0
                # The heat source is on and there is a heat source position, we check distance.
                # If not, we check whether we are inside it or not.
                if heat_source_position is not None:
                    aabb_lower, aabb_upper = self.obj.states[AABB].get_value()
                    position = (aabb_lower + aabb_upper) / 2.0
                    # Compute distance to heat source from our position.
                    dist = T.l2_distance(heat_source_position, position)
                    if dist > heat_source.distance_threshold:
                        continue
                else:
                    if not self.obj.states[Inside].get_value(obj2):
                        continue

                new_temperature += (
                    (heat_source.temperature - self.value) * heat_source.heating_rate * self._simulator.get_rendering_dt()
                )
                affected_by_heat_source = True

        # Apply temperature decay if not affected by any heat source.
        if not affected_by_heat_source:
            new_temperature += (
                (m.DEFAULT_TEMPERATURE - self.value) * m.TEMPERATURE_DECAY_SPEED * self._simulator.get_rendering_dt()
            )

        self.value = new_temperature

    @property
    def state_size(self):
        return 1

    # For this state, we simply store its value.
    def _dump_state(self):
        return OrderedDict(temperature=self.value)

    def _load_state(self, state):
        self.value = state["temperature"]

    def _serialize(self, state):
        return np.array([state["temperature"]], dtype=float)

    def _deserialize(self, state):
        return OrderedDict(temperature=state[0]), 1
