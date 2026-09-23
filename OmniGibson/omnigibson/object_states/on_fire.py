import torch as th
import warp as wp

import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.object_states.temperature import Temperature
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Delete default values for this and make them required.
m.DEFAULT_IGNITION_TEMPERATURE = 250


@wp.kernel
def _on_fire_kernel(
    self_temp_idx: wp.array(dtype=wp.int32),  # (N_of,) into Temperature N
    ignition_temperatures: wp.array(dtype=wp.float32),  # (N_of,)
    temperature_values: wp.array2d(dtype=wp.float32),  # (S, N_temp)
    out_values: wp.array2d(dtype=wp.uint8),  # (S, N_of) — uint8 view of bool
):
    """
    Per (scene, obj) thread: VALUES[s, h] = Temperature[s, self_temp_idx[h]] >= ignition[h].
    """
    s, h = wp.tid()
    n = self_temp_idx[h]
    if n < wp.int32(0):
        out_values[s, h] = wp.uint8(0)
        return
    if temperature_values[s, n] >= ignition_temperatures[h]:
        out_values[s, h] = wp.uint8(1)
    else:
        out_values[s, h] = wp.uint8(0)


class OnFire(TensorizedAbsoluteState):
    """
    Boolean state: True while the object's temperature is at or above its ignition temperature —
    a pure threshold detector over Temperature (its dependency), analogous to MaxTemperature.

    Everything the fire *does* lives elsewhere: the flammable ability pairs this state with a
    HeatSourceOrSink(requires_on_fire=True) on the same object. That heat source's activation
    gate reads this state's previous-step values (a deliberate lag-1 read that breaks the
    OnFire → Temperature → HeatSourceOrSink cycle), and Temperature then uses the active source
    to heat nearby objects and to hold the burning object itself at its fire temperature. That
    self-sustaining clamp is what makes this state sticky: once ignited, temperature stays at
    fire temperature (≥ ignition) until the object is deliberately cooled below ignition or
    set not-on-fire.
    """

    # Per-object config (N_of,) — uploaded once in initialize_view (single source of truth).
    _ignition_temperatures = None  # wp.array (N_of,) float32

    # Index map into Temperature's N dimension. Built directly in initialize_view — safe
    # because Temperature is a dependency, so its view is rebuilt before this one.
    _self_temp_idx = None  # wp.array (N_of,) int32

    def __init__(self, obj, ignition_temperature=None):
        """
        Args:
            obj (StatefulObject): The object with the flammable ability.
            ignition_temperature (float): The temperature threshold at / above which the object
                is on fire.
        """
        super().__init__(obj)
        self.ignition_temperature = (
            ignition_temperature if ignition_temperature is not None else m.DEFAULT_IGNITION_TEMPERATURE
        )

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Temperature)
        return deps

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "on_fire"

    @property
    def _companion_heat_source(self):
        """
        Returns:
            None or HeatSourceOrSink: The requires_on_fire heat source the flammable ability
                pairs with this state, if present.
        """
        heat_source = self.obj.states.get(HeatSourceOrSink)
        return heat_source if heat_source is not None and heat_source.requires_on_fire else None

    @property
    def temperature(self):
        """
        Returns:
            float: The fire temperature the object is held at while on fire.
        """
        companion = self._companion_heat_source
        return companion.temperature if companion is not None else self.ignition_temperature

    @property
    def heating_rate(self):
        """
        Returns:
            float: Heating rate of the fire towards nearby objects.
        """
        companion = self._companion_heat_source
        return companion.heating_rate if companion is not None else 0.0

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls._ignition_temperatures = None
        cls._self_temp_idx = None

    @classmethod
    def initialize_view(cls):
        super().initialize_view()

        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls._ignition_temperatures = None
            cls._self_temp_idx = None
            return

        ignition_temperatures = th.zeros(N, dtype=th.float32)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            for scene_row in cls.IDX_OBJS:
                if scene_row[obj_idx] is not None:
                    ignition_temperatures[obj_idx] = float(scene_row[obj_idx].states[cls].ignition_temperature)
                    break

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list
        cls._ignition_temperatures = create_tensor_from_list(ignition_temperatures, "float32", device="cuda")

        # Temperature is a hard dependency, so every OnFire object has a Temperature entry.
        idxs = [Temperature.OBJ_IDXS.get(rel_path, -1) for rel_path in cls.OBJ_IDXS]
        cls._self_temp_idx = wp.array(idxs, dtype=wp.int32, device="cuda")

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None or cls._self_temp_idx is None or Temperature.VALUES_WP is None:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return
        wp.launch(
            kernel=_on_fire_kernel,
            dim=(S, N),
            inputs=[cls._self_temp_idx, cls._ignition_temperatures, Temperature.VALUES_WP, cls.VALUES_WP],
            device="cuda",
        )

    def _get_value(self):
        if self.OBJ_IDXS is None or self.obj.relative_prim_path not in self.OBJ_IDXS:
            return False
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return bool(self.VALUES_CPU[s, obj_idx].item())

    def _set_value(self, new_value):
        """
        Direct setter: write Temperature so the threshold holds (and so the companion heat
        source's lag-1 gate picks the change up on the next pass), AND write VALUES immediately
        so get_value() reflects the new state without waiting for the next graph pass.
        """
        if self.OBJ_IDXS is None or self.obj.relative_prim_path not in self.OBJ_IDXS:
            return False
        s = self.obj.scene.idx
        h = self.OBJ_IDXS[self.obj.relative_prim_path]

        if new_value:
            # Push the temperature to the fire temperature (falls back to the ignition
            # threshold if no companion heat source exists).
            self.obj.states[Temperature].set_value(self.temperature)
        else:
            # Set temperature just below ignition.
            self.obj.states[Temperature].set_value(self.ignition_temperature - 1)
        self.VALUES[s, h] = bool(new_value)
        self.VALUES_CPU[s, h] = bool(new_value)
        return True

    # OnFire is fully derived from Temperature (a pure threshold detector), so loading must be a
    # no-op — matching main, where OnFire is not stateful at all. Overriding the inherited
    # TensorizedAbsoluteState._load_state is REQUIRED: the generic implementation routes through
    # _set_value(stored_value), and _set_value(False) writes Temperature = ignition_temperature - 1
    # (~249 C) — i.e. restoring a saved `on_fire: False` would HEAT a cold flammable object on
    # every scene load / reset.
    def _load_state(self, state):
        return
