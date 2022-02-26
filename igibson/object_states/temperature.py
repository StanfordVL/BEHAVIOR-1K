import numpy as np
from collections import OrderedDict
from igibson.object_states.heat_source_or_sink import HeatSourceOrSink
from igibson.object_states.inside import Inside
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.pose import Pose
from igibson.utils.utils import l2_distance

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(AbsoluteObjectState):
    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [Pose]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [HeatSourceOrSink]

    def __init__(self, obj):
        super(Temperature, self).__init__(obj)

        self.value = DEFAULT_TEMPERATURE

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        # Start at the current temperature.
        new_temperature = self.value

        # Find all heat source objects.
        affected_by_heat_source = False
        for obj2 in self.simulator.scene.get_objects_with_state(HeatSourceOrSink):
            # Obtain heat source position.
            heat_source = obj2.states[HeatSourceOrSink]
            heat_source_state, heat_source_position = heat_source.get_value()
            if heat_source_state:
                # The heat source is toggled on. If it has a position, we check distance.
                # If not, we check whether we are inside it or not.
                if heat_source_position is not None:
                    # Load our Pose. Note that this is cached already by the state.
                    # Also note that this produces garbage values for fixed objects - but we are
                    # assuming none of our temperature-enabled objects are fixed.
                    position, _ = self.obj.states[Pose].get_value()

                    # Compute distance to heat source from our position.
                    dist = l2_distance(heat_source_position, position)
                    if dist > heat_source.distance_threshold:
                        continue
                else:
                    if not self.obj.states[Inside].get_value(obj2):
                        continue

                new_temperature += (
                    (heat_source.temperature - self.value) * heat_source.heating_rate * self.simulator.render_timestep
                )
                affected_by_heat_source = True

        # Apply temperature decay if not affected by any heat source.
        if not affected_by_heat_source:
            new_temperature += (
                (DEFAULT_TEMPERATURE - self.value) * TEMPERATURE_DECAY_SPEED * self.simulator.render_timestep
            )

        self.value = new_temperature

    @property
    def settable(self):
        return True

    # For this state, we simply store its value.
    def _dump_state(self):
        return OrderedDict(temperature=self.value)

    def _load_state(self, state):
        self.value = state["temperature"]

    def _serialize(self, state):
        return np.array([state["temperature"]])

    def _deserialize(self, state):
        return OrderedDict(temperature=state[0]), 1
