from omnigibson.macros import create_module_macros
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.temperature import Temperature

# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Delete default values for this and make them required.
m.DEFAULT_IGNITION_TEMPERATURE = 250
m.DEFAULT_FIRE_TEMPERATURE = 1000
m.DEFAULT_HEATING_RATE = 0.04
m.DEFAULT_DISTANCE_THRESHOLD = 0.2


class OnFire(HeatSourceOrSink):
    """
    This state indicates the heat source is currently on fire.

    Once the temperature is above ignition_temperature, OnFire will become True and stay True.
    Its temperature will further raise to fire_temperature, and start heating other objects around it.
    It may include a heatsource_link annotation (e.g. candle wick), in which case the fire visualization will be placed
    under that meta link. Otherwise (e.g. charcoal), the fire visualization will be placed under the root link.
    """

    def __init__(
        self,
        obj,
        ignition_temperature=None,
        fire_temperature=None,
        heating_rate=None,
        distance_threshold=None,
    ):
        """
        Args:
            obj (StatefulObject): The object with the heat source ability.
            ignition_temperature (float): The temperature threshold above which on fire will become true.
            fire_temperature (float): The temperature of the fire (heat source) once on fire is true.
            heating_rate (float): Fraction in [0, 1] of the temperature difference with the
                heat source temperature should be received every step, per second.
            distance_threshold (float): The distance threshold which an object needs
                to be closer than in order to receive heat from this heat source.
        """
        ignition_temperature = (
            ignition_temperature if ignition_temperature is not None else m.DEFAULT_IGNITION_TEMPERATURE
        )
        fire_temperature = fire_temperature if fire_temperature is not None else m.DEFAULT_FIRE_TEMPERATURE
        heating_rate = heating_rate if heating_rate is not None else m.DEFAULT_HEATING_RATE
        distance_threshold = distance_threshold if distance_threshold is not None else m.DEFAULT_DISTANCE_THRESHOLD
        assert fire_temperature > ignition_temperature, "fire temperature should be higher than ignition temperature."

        super().__init__(
            obj,
            temperature=fire_temperature,
            heating_rate=heating_rate,
            distance_threshold=distance_threshold,
            requires_toggled_on=False,
            requires_closed=False,
            requires_inside=False,
        )
        self.ignition_temperature = ignition_temperature

    @classmethod
    def requires_meta_link(cls, **kwargs):
        # Does not require meta link to be specified
        return False

    @property
    def _default_link(self):
        # Fallback to root link
        return self.obj.root_link

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Temperature)
        return deps

    def _update(self):
        # Call super first
        super()._update()

        # If it's on fire, maintain the fire temperature
        if self.get_value():
            self.obj.states[Temperature].set_value(self.temperature)

    def _get_value(self):
        return self.obj.states[Temperature].get_value() >= self.ignition_temperature

    def _set_value(self, new_value):
        if new_value:
            return self.obj.states[Temperature].set_value(self.temperature)
        else:
            # We'll set the temperature just one degree below ignition.
            return self.obj.states[Temperature].set_value(self.ignition_temperature - 1)

    # Nothing needs to be done to save/load OnFire
