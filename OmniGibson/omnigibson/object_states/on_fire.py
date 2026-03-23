import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.temperature import Temperature
from omnigibson.utils.python_utils import torch_delete

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

    TEMPERATURE_IDXS = None  # (N,) int64   — index into Temperature.VALUES for each OnFire object
    IGNITION_TEMPERATURES = None  # (N,) float32  — ignition threshold per object

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls.TEMPERATURE_IDXS = th.empty(0, dtype=th.int64)
        cls.IGNITION_TEMPERATURES = th.empty(0, dtype=th.float32)

        # TEMPERATURE_IDXS[i] is an integer slot index into Temperature.VALUES.
        # When a Temperature object is removed, TensorizedValueState._remove_obj deletes its
        # row from Temperature.VALUES and all subsequent rows shift down by one.  Any stored
        # index that pointed past the deleted slot now refers to the wrong object.
        # This callback fires before the deletion and decrements every stored index that is
        # >= the deleted slot, keeping TEMPERATURE_IDXS in sync with Temperature.VALUES.
        # The same pattern is used for HeatSourceOrSink.TOGGLED_ON_IDXS (registered on
        # ToggledOn removal) and MaxTemperature.TEMPERATURE_IDXS (registered on Temperature removal).
        def _update_temperature_idxs(obj):
            if obj not in Temperature.OBJ_IDXS:
                return
            deleted_idx = Temperature.OBJ_IDXS[obj]
            valid = cls.TEMPERATURE_IDXS >= 0
            cls.TEMPERATURE_IDXS = th.where(
                valid & (cls.TEMPERATURE_IDXS >= deleted_idx),
                cls.TEMPERATURE_IDXS - 1,
                cls.TEMPERATURE_IDXS,
            )

        Temperature.add_callback_on_remove(name="OnFire_temperature_idx_update", callback=_update_temperature_idxs)

    @classmethod
    def _add_obj(cls, obj):
        super()._add_obj(obj)
        cls.TEMPERATURE_IDXS = th.cat([cls.TEMPERATURE_IDXS, th.full((1,), -1, dtype=th.int64)])
        cls.IGNITION_TEMPERATURES = th.cat([cls.IGNITION_TEMPERATURES, th.zeros(1, dtype=th.float32)])

    @classmethod
    def _remove_obj(cls, obj):
        deleted_idx = cls.OBJ_IDXS[obj]
        cls.TEMPERATURE_IDXS = torch_delete(cls.TEMPERATURE_IDXS, [deleted_idx])
        cls.IGNITION_TEMPERATURES = torch_delete(cls.IGNITION_TEMPERATURES, [deleted_idx])
        super()._remove_obj(obj)

    @classmethod
    def _update_values(cls, values):
        # OnFire is active when the object's current temperature >= its ignition threshold.
        # Pure tensor op: no loop needed.
        temps = Temperature.VALUES[cls.TEMPERATURE_IDXS]  # (N,) current temp of each fire object
        return temps >= cls.IGNITION_TEMPERATURES  # (N,) bool

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

        # Write OnFire-specific config into class tensors at our index.
        idx = type(self).OBJ_IDXS[obj]
        type(self).TEMPERATURE_IDXS[idx] = Temperature.OBJ_IDXS[obj]
        type(self).IGNITION_TEMPERATURES[idx] = ignition_temperature

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

    def _get_value(self):
        return self.obj.states[Temperature].get_value() >= self.ignition_temperature

    def _set_value(self, new_value):
        if new_value:
            return self.obj.states[Temperature].set_value(self.temperature)
        else:
            # We'll set the temperature just one degree below ignition.
            return self.obj.states[Temperature].set_value(self.ignition_temperature - 1)

    # Nothing needs to be done to save/load OnFire
