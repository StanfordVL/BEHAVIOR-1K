import torch as th
import warp as wp

import omnigibson.lazy as lazy
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
def _on_fire_is_active_kernel(
    self_temp_idx: wp.array(dtype=wp.int32),  # (N_of,)
    ignition_temperatures: wp.array(dtype=wp.float32),  # (N_of,)
    temperature_values: wp.array2d(dtype=wp.float32),  # (S, N_temp)
    out_values: wp.array2d(dtype=wp.uint8),  # (S, N_of) — uint8 view of bool
):
    """
    Per (scene, of) thread: VALUES[s, h] = Temperature[s, self_temp_idx[h]] >= ignition[h].
    """
    s, h = wp.tid()
    n_self = self_temp_idx[h]
    if n_self < wp.int32(0):
        out_values[s, h] = wp.uint8(0)
        return
    if temperature_values[s, n_self] >= ignition_temperatures[h]:
        out_values[s, h] = wp.uint8(1)
    else:
        out_values[s, h] = wp.uint8(0)


@wp.kernel
def _on_fire_can_influence_kernel(
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
    target_aabb_idx = temp_to_aabb_idx[n]
    if target_aabb_idx < wp.int32(0):
        return
    li = link_flat_idx[h]
    if li < wp.int32(0):
        return
    link_pose = pose_matrices[li]
    off = link_local_offset[h]
    heat_world = wp.mul(link_pose, wp.vec4(off[0], off[1], off[2], wp.float32(1.0)))
    cx = (aabb_values[s, target_aabb_idx, 0] + aabb_values[s, target_aabb_idx, 3]) * wp.float32(0.5)
    cy = (aabb_values[s, target_aabb_idx, 1] + aabb_values[s, target_aabb_idx, 4]) * wp.float32(0.5)
    cz = (aabb_values[s, target_aabb_idx, 2] + aabb_values[s, target_aabb_idx, 5]) * wp.float32(0.5)
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
    onfire_values: wp.array2d(dtype=wp.uint8),  # (S, N_of)
    self_temp_idx: wp.array(dtype=wp.int32),  # (N_of,)
    fire_temperatures: wp.array(dtype=wp.float32),  # (N_of,)
    temperature_values: wp.array2d(dtype=wp.float32),  # (S, N_temp)
):
    """
    For each ignited OnFire entry, set Temperature[s, n_self] = max(current, fire_temp).
    """
    s, h = wp.tid()
    if onfire_values[s, h] == wp.uint8(0):
        return
    self_idx_in_temp = self_temp_idx[h]
    if self_idx_in_temp < wp.int32(0):
        return
    if fire_temperatures[h] > temperature_values[s, self_idx_in_temp]:
        temperature_values[s, self_idx_in_temp] = fire_temperatures[h]


class OnFire(TensorizedAbsoluteState, LinkBasedStateMixin):
    """
    This state indicates the heat source is currently on fire.

    Once the temperature is above ignition_temperature, OnFire becomes True and stays True.
    Its temperature is further clamped to fire_temperature each step, heating nearby objects.

    Implementation runs inside the captured Warp graph in this order (after Temperature):
      1. set VALUES = Temperature >= ignition_temperature.
      2. update temperature change rate: atomic_add (T_fire - T_n) * rate into Temperature.INCOMING_HEAT_RATE
         for nearby Temperature-tracked targets (consumed NEXT step).
      3. clamp: write Temperature.VALUES[self] = fire_temperature where ignited.
    """

    # Per-OnFire config (N_of,) — uploaded once in initialize_view as wp.arrays (single source of truth).
    _ignition_temperatures = None  # wp.array (N_of,) float32
    _fire_temperatures = None  # wp.array (N_of,) float32
    _heating_rates = None  # wp.array (N_of,) float32
    _distance_thresholds = None  # wp.array (N_of,) float32
    _link_flat_idx = None  # wp.array (N_of,) int32 into RigidBodyViewAPI.POSE_MATRICES — -1 if N/A
    _link_local_offset = None  # wp.array (N_of,) vec3 — offset of heat element in link frame

    # Cross-state index maps — rebuilt in pre_update (Temperature/AABB initialize after OnFire).
    _self_temp_idx = None  # wp.array (N_of,) int32 into Temperature.OBJ_IDXS
    _temp_to_aabb_idx = None  # wp.array (N_temp,) int32 Temperature N → AABB N

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
        cls._ignition_temperatures = None
        cls._fire_temperatures = None
        cls._heating_rates = None
        cls._distance_thresholds = None
        cls._link_flat_idx = None
        cls._link_local_offset = None
        cls._self_temp_idx = None
        cls._temp_to_aabb_idx = None

    @classmethod
    def initialize_view(cls):
        super().initialize_view()

        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls._ignition_temperatures = None
            cls._fire_temperatures = None
            cls._heating_rates = None
            cls._distance_thresholds = None
            cls._link_flat_idx = None
            cls._link_local_offset = None
            cls._self_temp_idx = None
            cls._temp_to_aabb_idx = None
            return

        ignition_temperatures = th.zeros(N, dtype=th.float32)
        fire_temperatures = th.zeros(N, dtype=th.float32)
        heating_rates = th.zeros(N, dtype=th.float32)
        distance_thresholds = th.zeros(N, dtype=th.float32)
        link_flat_idx = th.full((N,), -1, dtype=th.int32)
        link_local_offset = th.zeros((N, 3), dtype=th.float32)

        for _, obj_idx in cls.OBJ_IDXS.items():
            onfire_state_instance = None
            for scene_row in cls.IDX_OBJS:
                if scene_row[obj_idx] is not None:
                    onfire_state_instance = scene_row[obj_idx].states[cls]
                    break
            if onfire_state_instance is None:
                continue
            ignition_temperatures[obj_idx] = float(onfire_state_instance.ignition_temperature)
            fire_temperatures[obj_idx] = float(onfire_state_instance._fire_temperature)
            heating_rates[obj_idx] = float(onfire_state_instance._heating_rate)
            distance_thresholds[obj_idx] = float(onfire_state_instance.distance_threshold)
            # OnFire always uses root_link (or meta link if present) as the heat element.
            # Offset is zero — the link's world position IS the heat-source position.
            if onfire_state_instance._links:
                link = onfire_state_instance.link
            else:
                link = onfire_state_instance.obj.root_link
            flat = RigidBodyViewAPI.get_flat_idx(link.prim_path)
            if flat is not None:
                link_flat_idx[obj_idx] = int(flat)

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list
        cls._ignition_temperatures = create_tensor_from_list(ignition_temperatures, "float32", device="cuda")
        cls._fire_temperatures = create_tensor_from_list(fire_temperatures, "float32", device="cuda")
        cls._heating_rates = create_tensor_from_list(heating_rates, "float32", device="cuda")
        cls._distance_thresholds = create_tensor_from_list(distance_thresholds, "float32", device="cuda")
        cls._link_flat_idx = create_tensor_from_list(link_flat_idx, "int32", device="cuda")
        # vec3 has no scalar-only helper — wp.array reinterprets the (N, 3) float32 CPU buffer as (N,) vec3.
        cls._link_local_offset = wp.array(link_local_offset, dtype=wp.vec3, device="cuda")

        # Cross-state maps deferred (Temperature N may not be sized yet).
        cls._self_temp_idx = None
        cls._temp_to_aabb_idx = None

    @classmethod
    def _rebuild_cross_state_maps(cls):
        N = len(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else 0
        N_temp = len(Temperature.OBJ_IDXS) if Temperature.OBJ_IDXS is not None else 0
        if N == 0:
            return

        temp_map = Temperature.OBJ_IDXS or {}
        aabb_map = AABB.OBJ_IDXS or {}

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list
        self_temp_idx = th.full((N,), -1, dtype=th.int32)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            self_temp_idx[obj_idx] = temp_map.get(rel_path, -1)
        cls._self_temp_idx = create_tensor_from_list(self_temp_idx, "int32", device="cuda")

        temp_to_aabb = th.full((N_temp,), -1, dtype=th.int32)
        for rel_path, temp_idx in temp_map.items():
            temp_to_aabb[temp_idx] = aabb_map.get(rel_path, -1)
        if N_temp > 0:
            cls._temp_to_aabb_idx = create_tensor_from_list(temp_to_aabb, "int32", device="cuda")
        else:
            cls._temp_to_aabb_idx = None

    @classmethod
    def pre_update(cls):
        super().pre_update()
        # Rebuild cross-state maps only when the captured graph is being rebuilt. graph_dirty is
        # the single trigger: every event that changes N_temp or invalidates the index tables goes
        # through some state's initialize_view, which sets graph_dirty. Rebuild reallocates GPU
        # buffers that must be baked into the freshly-captured graph, so rebuild and recapture must
        # stay coupled — gating on anything else risks reallocating without a recapture.
        if TensorizedState.graph_dirty:
            cls._rebuild_cross_state_maps()

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None or cls._self_temp_idx is None or Temperature.VALUES_WP is None:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return

        # 1) VALUES[s, h] = (Temperature[s, n_self] >= ignition_temp)
        wp.launch(
            kernel=_on_fire_is_active_kernel,
            dim=(S, N),
            inputs=[
                cls._self_temp_idx,
                cls._ignition_temperatures,
                Temperature.VALUES_WP,
                cls.VALUES_WP,
            ],
            device="cuda",
        )

        # 2) atomic_add (T_fire - T_n) * rate into INCOMING_HEAT_RATE
        if (
            Temperature.INCOMING_HEAT_RATE_WP is not None
            and cls._temp_to_aabb_idx is not None
            and AABB.VALUES_WP is not None
        ):
            N_temp = Temperature.VALUES.shape[1]
            if N_temp > 0:
                wp.launch(
                    kernel=_on_fire_can_influence_kernel,
                    dim=(S, N, N_temp),
                    inputs=[
                        cls.VALUES_WP,
                        cls._fire_temperatures,
                        cls._heating_rates,
                        cls._distance_thresholds,
                        cls._self_temp_idx,
                        cls._link_flat_idx,
                        cls._link_local_offset,
                        cls._temp_to_aabb_idx,
                        RigidBodyViewAPI.POSE_MATRICES,
                        AABB.VALUES_WP,
                        Temperature.VALUES_WP,
                        Temperature.INCOMING_HEAT_RATE_WP,
                    ],
                    device="cuda",
                )

        # 3) Temperature[s, n_self] = max(current, fire_temp) for ignited entries.
        wp.launch(
            kernel=_on_fire_clamp_kernel,
            dim=(S, N),
            inputs=[
                cls.VALUES_WP,
                cls._self_temp_idx,
                cls._fire_temperatures,
                Temperature.VALUES_WP,
            ],
            device="cuda",
        )

        # Re-mirror Temperature.VALUES to Temperature.VALUES_CPU so the clamp is visible
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
