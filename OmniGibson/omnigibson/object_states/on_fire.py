import torch as th
import warp as wp

from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.object_states.tensorized_state import TensorizedState
from omnigibson.object_states.temperature import Temperature
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.ONFIRE_META_LINK_TYPE = "heatsource"
# TODO: Delete default values for this and make them required.
m.DEFAULT_IGNITION_TEMPERATURE = 250
m.DEFAULT_FIRE_TEMPERATURE = 1000
m.DEFAULT_HEATING_RATE = 0.04
m.DEFAULT_DISTANCE_THRESHOLD = 0.2


@wp.kernel
def _on_fire_gate_kernel(
    self_temp_idx: wp.array(dtype=wp.int32),  # (N_of,)
    ignition_temperatures: wp.array(dtype=wp.float32),  # (N_of,)
    temperature_values: wp.array2d(dtype=wp.float32),  # (S, N_temp)
    out_values: wp.array2d(dtype=wp.uint8),  # (S, N_of) — uint8 view of bool
):
    """
    Per (scene, of) thread: VALUES[s, h] = Temperature[s, self_temp_idx[h]] >= ignition[h].
    """
    s, h = wp.tid()
    ti = self_temp_idx[h]
    if ti < wp.int32(0):
        out_values[s, h] = wp.uint8(0)
        return
    if temperature_values[s, ti] >= ignition_temperatures[h]:
        out_values[s, h] = wp.uint8(1)
    else:
        out_values[s, h] = wp.uint8(0)


@wp.kernel
def _on_fire_propagate_kernel(
    of_values: wp.array2d(dtype=wp.uint8),  # (S, N_of)
    fire_temperatures: wp.array(dtype=wp.float32),  # (N_of,)
    heating_rates: wp.array(dtype=wp.float32),  # (N_of,)
    distance_thresholds: wp.array(dtype=wp.float32),  # (N_of,)
    self_temp_idx: wp.array(dtype=wp.int32),  # (N_of,)
    link_flat_idx: wp.array(dtype=wp.int32),  # (N_of,)
    link_local_offset: wp.array(dtype=wp.vec3),  # (N_of,)
    temp_to_aabb_idx: wp.array(dtype=wp.int32),  # (N_temp,)
    pose_matrices: wp.array(dtype=wp.mat44),
    aabb_values: wp.array3d(dtype=wp.float32),  # AABB (S, N_aabb, 6)
    temperature_values: wp.array2d(dtype=wp.float32),  # (S, N_temp)
    incoming_heat_rate: wp.array2d(dtype=wp.float32),  # (S, N_temp) — out
):
    """
    Per (scene, of, target) thread: if on fire and target in range, atomic_add
    (T_fire - T_n) * rate into INCOMING_HEAT_RATE[s, n]. Consumed next step (lag-1, matches
    pre-tensorized behavior).
    """
    s, h, n = wp.tid()
    if of_values[s, h] == wp.uint8(0):
        return
    if self_temp_idx[h] == n:
        return
    a_n = temp_to_aabb_idx[n]
    if a_n < wp.int32(0):
        return
    li = link_flat_idx[h]
    if li < wp.int32(0):
        return
    link_pose = pose_matrices[li]
    off = link_local_offset[h]
    heat_world = wp.mul(link_pose, wp.vec4(off[0], off[1], off[2], wp.float32(1.0)))
    cx = (aabb_values[s, a_n, 0] + aabb_values[s, a_n, 3]) * wp.float32(0.5)
    cy = (aabb_values[s, a_n, 1] + aabb_values[s, a_n, 4]) * wp.float32(0.5)
    cz = (aabb_values[s, a_n, 2] + aabb_values[s, a_n, 5]) * wp.float32(0.5)
    dx = heat_world[0] - cx
    dy = heat_world[1] - cy
    dz = heat_world[2] - cz
    d2 = dx * dx + dy * dy + dz * dz
    thr = distance_thresholds[h]
    if d2 > thr * thr:
        return
    delta = (fire_temperatures[h] - temperature_values[s, n]) * heating_rates[h]
    wp.atomic_add(incoming_heat_rate, s, n, delta)


@wp.kernel
def _on_fire_clamp_kernel(
    of_values: wp.array2d(dtype=wp.uint8),  # (S, N_of)
    self_temp_idx: wp.array(dtype=wp.int32),  # (N_of,)
    fire_temperatures: wp.array(dtype=wp.float32),  # (N_of,)
    temperature_values: wp.array2d(dtype=wp.float32),  # (S, N_temp)
):
    """
    For each ignited OnFire entry, set Temperature[s, n_self] = max(current, fire_temp).
    This is the *one* legitimate cross-state Temperature write outside of Temperature:
    OnFire owns its own temperature when ignited so that subsequent decay/heatsource
    contributions cannot pull it back below the fire temperature.
    """
    s, h = wp.tid()
    if of_values[s, h] == wp.uint8(0):
        return
    ti = self_temp_idx[h]
    if ti < wp.int32(0):
        return
    if fire_temperatures[h] > temperature_values[s, ti]:
        temperature_values[s, ti] = fire_temperatures[h]


class OnFire(TensorizedAbsoluteState, LinkBasedStateMixin):
    """
    This state indicates the heat source is currently on fire.

    Once the temperature is above ignition_temperature, OnFire becomes True and stays True.
    Its temperature is further clamped to fire_temperature each step, heating nearby objects.

    Implementation runs inside the captured Warp graph in this order (after Temperature):
      1. Gate kernel: VALUES = Temperature >= ignition_temperature.
      2. Propagate kernel: atomic_add (T_fire - T_n) * rate into Temperature.INCOMING_HEAT_RATE
         for nearby Temperature-tracked targets (consumed NEXT step — lag-1).
      3. Clamp kernel: write Temperature.VALUES[n_self] = fire_temperature where ignited.

    After (3), Temperature.VALUES → Temperature.VALUES_CPU is re-mirrored so the clamp is
    visible to CPU readers without one step of staleness.
    """

    # Per-OnFire config (N_of,) — uploaded once in initialize_view.
    IGNITION_TEMPERATURES = None
    IGNITION_TEMPERATURES_WP = None
    FIRE_TEMPERATURES = None
    FIRE_TEMPERATURES_WP = None
    HEATING_RATES = None
    HEATING_RATES_WP = None
    DISTANCE_THRESHOLDS = None
    DISTANCE_THRESHOLDS_WP = None
    LINK_FLAT_IDX = None
    LINK_FLAT_IDX_WP = None
    LINK_LOCAL_OFFSET = None
    LINK_LOCAL_OFFSET_WP = None

    # Cross-state index maps — rebuilt in pre_update (Temperature/AABB initialize after OnFire).
    SELF_TEMP_IDX = None
    SELF_TEMP_IDX_WP = None
    TEMP_TO_AABB_IDX = None
    TEMP_TO_AABB_IDX_WP = None

    _CACHED_N_TEMP = 0

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
        super().__init__(obj)
        self.ignition_temperature = (
            ignition_temperature if ignition_temperature is not None else m.DEFAULT_IGNITION_TEMPERATURE
        )
        self._fire_temperature = fire_temperature if fire_temperature is not None else m.DEFAULT_FIRE_TEMPERATURE
        self._heating_rate = heating_rate if heating_rate is not None else m.DEFAULT_HEATING_RATE
        self.distance_threshold = distance_threshold if distance_threshold is not None else m.DEFAULT_DISTANCE_THRESHOLD
        assert (
            self._fire_temperature > self.ignition_temperature
        ), "fire temperature should be higher than ignition temperature."

    @classmethod
    def requires_meta_link(cls, **kwargs):
        # Does not require meta link to be specified
        return False

    @property
    def _default_link(self):
        # Fallback to root link
        return self.obj.root_link

    @classproperty
    def meta_link_types(cls):
        return [m.ONFIRE_META_LINK_TYPE]

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        # OnFire reads Temperature.VALUES (pre-step value, before this step's decay) for the
        # ignition gate, and writes back to Temperature via the clamp. It also needs AABB.
        deps.update({AABB, Temperature})
        return deps

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "on_fire"

    @property
    def temperature(self):
        """
        Returns:
            float: The fire temperature (clamped temperature when ignited).
        """
        return self._fire_temperature

    @property
    def heating_rate(self):
        return self._heating_rate

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls.IGNITION_TEMPERATURES = None
        cls.IGNITION_TEMPERATURES_WP = None
        cls.FIRE_TEMPERATURES = None
        cls.FIRE_TEMPERATURES_WP = None
        cls.HEATING_RATES = None
        cls.HEATING_RATES_WP = None
        cls.DISTANCE_THRESHOLDS = None
        cls.DISTANCE_THRESHOLDS_WP = None
        cls.LINK_FLAT_IDX = None
        cls.LINK_FLAT_IDX_WP = None
        cls.LINK_LOCAL_OFFSET = None
        cls.LINK_LOCAL_OFFSET_WP = None
        cls.SELF_TEMP_IDX = None
        cls.SELF_TEMP_IDX_WP = None
        cls.TEMP_TO_AABB_IDX = None
        cls.TEMP_TO_AABB_IDX_WP = None
        cls._CACHED_N_TEMP = 0

    @classmethod
    def initialize_view(cls):
        super().initialize_view()

        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls.IGNITION_TEMPERATURES_WP = None
            cls.FIRE_TEMPERATURES_WP = None
            cls.HEATING_RATES_WP = None
            cls.DISTANCE_THRESHOLDS_WP = None
            cls.LINK_FLAT_IDX_WP = None
            cls.LINK_LOCAL_OFFSET_WP = None
            cls.SELF_TEMP_IDX_WP = None
            cls.TEMP_TO_AABB_IDX_WP = None
            cls._CACHED_N_TEMP = 0
            return

        ignition_temperatures = th.zeros(N, dtype=th.float32)
        fire_temperatures = th.zeros(N, dtype=th.float32)
        heating_rates = th.zeros(N, dtype=th.float32)
        distance_thresholds = th.zeros(N, dtype=th.float32)
        link_flat_idx = th.full((N,), -1, dtype=th.int32)
        link_local_offset = th.zeros((N, 3), dtype=th.float32)

        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            inst = None
            for scene_row in cls.IDX_OBJS:
                if scene_row[obj_idx] is not None:
                    inst = scene_row[obj_idx].states[cls]
                    break
            if inst is None:
                continue
            ignition_temperatures[obj_idx] = float(inst.ignition_temperature)
            fire_temperatures[obj_idx] = float(inst._fire_temperature)
            heating_rates[obj_idx] = float(inst._heating_rate)
            distance_thresholds[obj_idx] = float(inst.distance_threshold)
            # OnFire always uses root_link (or meta link if present) as the heat element.
            # Offset is zero — the link's world position IS the heat-source position.
            if inst._links:
                link = inst.link
            else:
                link = inst.obj.root_link
            flat = RigidBodyViewAPI.get_flat_idx(link.prim_path)
            if flat is not None:
                link_flat_idx[obj_idx] = int(flat)

        cls.IGNITION_TEMPERATURES = ignition_temperatures.cuda()
        cls.FIRE_TEMPERATURES = fire_temperatures.cuda()
        cls.HEATING_RATES = heating_rates.cuda()
        cls.DISTANCE_THRESHOLDS = distance_thresholds.cuda()
        cls.LINK_FLAT_IDX = link_flat_idx.cuda()
        cls.LINK_LOCAL_OFFSET = link_local_offset.cuda()

        cls.IGNITION_TEMPERATURES_WP = wp.from_torch(cls.IGNITION_TEMPERATURES)
        cls.FIRE_TEMPERATURES_WP = wp.from_torch(cls.FIRE_TEMPERATURES)
        cls.HEATING_RATES_WP = wp.from_torch(cls.HEATING_RATES)
        cls.DISTANCE_THRESHOLDS_WP = wp.from_torch(cls.DISTANCE_THRESHOLDS)
        cls.LINK_FLAT_IDX_WP = wp.from_torch(cls.LINK_FLAT_IDX)
        cls.LINK_LOCAL_OFFSET_WP = wp.from_torch(cls.LINK_LOCAL_OFFSET, dtype=wp.vec3)

        # Cross-state maps deferred (Temperature N may not be sized yet).
        cls.SELF_TEMP_IDX_WP = None
        cls.TEMP_TO_AABB_IDX_WP = None
        cls._CACHED_N_TEMP = 0

    @classmethod
    def _rebuild_cross_state_maps(cls):
        N = len(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else 0
        N_temp = len(Temperature.OBJ_IDXS) if Temperature.OBJ_IDXS is not None else 0
        if N == 0:
            return

        temp_map = Temperature.OBJ_IDXS or {}
        aabb_map = AABB.OBJ_IDXS or {}

        self_temp_idx = th.full((N,), -1, dtype=th.int32)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            self_temp_idx[obj_idx] = temp_map.get(rel_path, -1)
        cls.SELF_TEMP_IDX = self_temp_idx.cuda()
        cls.SELF_TEMP_IDX_WP = wp.from_torch(cls.SELF_TEMP_IDX)

        temp_to_aabb = th.full((N_temp,), -1, dtype=th.int32)
        for rel_path, t_idx in temp_map.items():
            temp_to_aabb[t_idx] = aabb_map.get(rel_path, -1)
        if N_temp > 0:
            cls.TEMP_TO_AABB_IDX = temp_to_aabb.cuda()
            cls.TEMP_TO_AABB_IDX_WP = wp.from_torch(cls.TEMP_TO_AABB_IDX)
        else:
            cls.TEMP_TO_AABB_IDX_WP = None
        cls._CACHED_N_TEMP = N_temp

    @classmethod
    def pre_update(cls):
        super().pre_update()
        N_temp_now = len(Temperature.OBJ_IDXS) if Temperature.OBJ_IDXS is not None else 0
        if TensorizedState.graph_dirty or cls._CACHED_N_TEMP != N_temp_now or cls.SELF_TEMP_IDX_WP is None:
            cls._rebuild_cross_state_maps()

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None or cls.SELF_TEMP_IDX_WP is None or Temperature.VALUES_WP is None:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return

        # 1) Gate — VALUES[s, h] = (Temperature[s, n_self] >= ignition_temp)
        wp.launch(
            kernel=_on_fire_gate_kernel,
            dim=(S, N),
            inputs=[
                cls.SELF_TEMP_IDX_WP,
                cls.IGNITION_TEMPERATURES_WP,
                Temperature.VALUES_WP,
                cls.VALUES_WP,
            ],
            device="cuda",
        )

        # 2) Propagate — atomic_add (T_fire - T_n) * rate into INCOMING_HEAT_RATE
        if (
            Temperature.INCOMING_HEAT_RATE_WP is not None
            and cls.TEMP_TO_AABB_IDX_WP is not None
            and AABB.VALUES_WP is not None
        ):
            N_temp = Temperature.VALUES.shape[1]
            if N_temp > 0:
                wp.launch(
                    kernel=_on_fire_propagate_kernel,
                    dim=(S, N, N_temp),
                    inputs=[
                        cls.VALUES_WP,
                        cls.FIRE_TEMPERATURES_WP,
                        cls.HEATING_RATES_WP,
                        cls.DISTANCE_THRESHOLDS_WP,
                        cls.SELF_TEMP_IDX_WP,
                        cls.LINK_FLAT_IDX_WP,
                        cls.LINK_LOCAL_OFFSET_WP,
                        cls.TEMP_TO_AABB_IDX_WP,
                        RigidBodyViewAPI.POSE_MATRICES,
                        AABB.VALUES_WP,
                        Temperature.VALUES_WP,
                        Temperature.INCOMING_HEAT_RATE_WP,
                    ],
                    device="cuda",
                )

        # 3) Clamp — Temperature[s, n_self] = max(current, fire_temp) for ignited entries.
        wp.launch(
            kernel=_on_fire_clamp_kernel,
            dim=(S, N),
            inputs=[
                cls.VALUES_WP,
                cls.SELF_TEMP_IDX_WP,
                cls.FIRE_TEMPERATURES_WP,
                Temperature.VALUES_WP,
            ],
            device="cuda",
        )

        # Re-mirror Temperature.VALUES → Temperature.VALUES_CPU so the clamp is visible
        # to CPU readers this step (otherwise it lags by one step — the base global_update
        # already mirrored Temperature.VALUES_CPU before the clamp executed).
        if Temperature.VALUES_CPU_WP is not None and Temperature.VALUES_WP is not None:
            wp.copy(Temperature.VALUES_CPU_WP, Temperature.VALUES_WP)

    def _get_value(self):
        TensorizedState.maybe_refresh_caches()
        if self.OBJ_IDXS is None or self.obj.relative_prim_path not in self.OBJ_IDXS:
            return False
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return bool(self.VALUES_CPU[s, obj_idx].item())

    def _set_value(self, new_value):
        """
        Direct setter: write Temperature so the ignition gate fires (or doesn't) on the next
        step, AND write OnFire.VALUES immediately so `get_value()` reflects the new state
        without needing to wait for the next graph pass.
        """
        s = self.obj.scene.idx
        if self.obj.relative_prim_path not in self.OBJ_IDXS:
            return False
        h = self.OBJ_IDXS[self.obj.relative_prim_path]

        if new_value:
            # Push the temperature to fire_temperature and flip VALUES True.
            self.obj.states[Temperature].set_value(self._fire_temperature)
            self.VALUES[s, h] = True
            self.VALUES_CPU[s, h] = True
        else:
            # Set temperature just below ignition; flip VALUES False.
            self.obj.states[Temperature].set_value(self.ignition_temperature - 1)
            self.VALUES[s, h] = False
            self.VALUES_CPU[s, h] = False
        return True

    # Nothing needs to be done to save/load OnFire
