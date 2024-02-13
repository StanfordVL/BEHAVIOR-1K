import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.update_state_mixin import GlobalUpdateStateMixin
import omnigibson as og


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(AbsoluteObjectState, GlobalUpdateStateMixin):

    # Numpy array of raw temperature values
    VALUES = np.array([])

    # Maps object name to index in VALUES
    OBJ_IDXS = dict()

    def __init__(self, obj):
        # Run super first
        super(Temperature, self).__init__(obj)

        # Add this object to the tracked global state
        self.VALUES = np.concatenate([self.VALUES, np.array([m.DEFAULT_TEMPERATURE])])
        self.OBJ_IDXS[self.obj.name] = len(self.VALUES) - 1

    @classmethod
    def remove_object(cls, obj):
        """
        Removes a tracked object from the array

        Args:
            obj (BaseObject): Object to potentially remove from internal global tracked state
        """
        if obj in cls.OBJ_IDXS:
            deleted_idx = cls.OBJ_IDXS.pop(obj.name)
            new_idxs = dict()
            for name, old_idx in cls.OBJ_IDXS.items():
                new_idxs[name] = old_idx - (0 if old_idx < new_idxs else 1)
            cls.OBJ_IDXS = new_idxs
            cls.VALUES = np.delete(cls.VALUES, [deleted_idx])

    @classmethod
    def global_update(cls):
        # This will globally decay all tracked temperatures
        cls.VALUES += (m.DEFAULT_TEMPERATURE - cls.VALUES) * m.TEMPERATURE_DECAY_SPEED * og.sim.get_rendering_dt()

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
        return self.VALUES[self.OBJ_IDXS[self.obj.name]]

    def _set_value(self, new_value):
        self.VALUES[self.OBJ_IDXS[self.obj.name]] = new_value
        return True

    def update_temperature_from_heatsource_or_sink(self, temperature, rate):
        """
        Updates this object's internal temperature based on @temperature and @rate

        Args:
            temperature (float): Heat source / sink temperature
            rate (float): Heating rate of the source / sink
        """
        self.VALUES[self.OBJ_IDXS[self.obj.name]] += (temperature - self.VALUES[self.OBJ_IDXS[self.obj.name]]) * rate * og.sim.get_rendering_dt()

    @property
    def state_size(self):
        return 1

    # For this state, we simply store its value.
    def _dump_state(self):
        return dict(temperature=self.VALUES[self.OBJ_IDXS[self.obj.name]])

    def _load_state(self, state):
        self.VALUES[self.OBJ_IDXS[self.obj.name]] = state["temperature"]

    def _serialize(self, state):
        return np.array([state["temperature"]], dtype=float)

    def _deserialize(self, state):
        return dict(temperature=state[0]), 1
