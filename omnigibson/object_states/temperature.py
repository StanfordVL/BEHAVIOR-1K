import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(TensorizedValueState):
    def __init__(self, obj):
        # Run super first
        super(Temperature, self).__init__(obj)

        # Set value to be default
        self._set_value(m.DEFAULT_TEMPERATURE)

    @classmethod
    def update_temperature_from_heatsource_or_sink(cls, objs, temperature, rate):
        """
        Updates @objs' internal temperatures based on @temperature and @rate

        Args:
            objs (Iterable of StatefulObject): Objects whose temperatures should be updated
            temperature (float): Heat source / sink temperature
            rate (float): Heating rate of the source / sink
        """
        # Get idxs for objs
        idxs = [cls.OBJ_IDXS[obj] for obj in objs]
        cls.VALUES[idxs] += (temperature - cls.VALUES[idxs]) * rate * og.sim.get_sim_step_dt()

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

    @classmethod
    def _update_values(cls, values):
        # Apply temperature decay
        return values + (m.DEFAULT_TEMPERATURE - values) * m.TEMPERATURE_DECAY_SPEED * og.sim.get_sim_step_dt()

    @classproperty
    def value_name(cls):
        return "temperature"
