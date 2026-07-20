import torch as th
import warp as wp

import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.object_states.tensorized_state import TensorizedState
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.HEATSOURCE_META_LINK_TYPE = "heatsource"
m.HEATING_ELEMENT_MARKER_SCALE = [1.0] * 3

# TODO: Delete default values for this and make them required.
m.DEFAULT_TEMPERATURE = 200
m.DEFAULT_FIRE_TEMPERATURE = 1000
m.DEFAULT_HEATING_RATE = 0.04
m.DEFAULT_DISTANCE_THRESHOLD = 0.2
m.DEFAULT_IGNITION_TEMPERATURE = 250


@wp.kernel
def _heatsource_is_active_kernel(
    requires_toggled_on: wp.array(dtype=wp.uint8),  # (N_hss,)
    requires_closed: wp.array(dtype=wp.uint8),  # (N_hss,)
    requires_on_fire: wp.array(dtype=wp.uint8),  # (N_hss,)
    toggle_idx: wp.array(dtype=wp.int32),  # (N_hss,) into ToggledOn.OBJ_IDXS, -1 if missing
    open_idx: wp.array(dtype=wp.int32),  # (N_hss,) into Open.OBJ_IDXS, -1 if missing
    onfire_idx: wp.array(dtype=wp.int32),  # (N_hss,) into OnFire.OBJ_IDXS, -1 if missing
    toggle_values: wp.array2d(dtype=wp.uint8),  # (S_toggle, N_toggle)
    open_values: wp.array2d(dtype=wp.uint8),  # (S_open, N_open)
    onfire_values: wp.array2d(dtype=wp.uint8),  # (S_onfire, N_onfire) — PREVIOUS step's values
    n_toggle_scenes: wp.int32,  # scene rows in toggle_values (0 if the state tracks nothing)
    n_open_scenes: wp.int32,
    n_onfire_scenes: wp.int32,
    out_values: wp.array2d(dtype=wp.uint8),  # (S, N_hss) — set to 1 if gate passes
):
    """
    hss = heat_source_or_sink
    Per (scene, hss) thread: compute whether this hss is active. 0 not active. 1 active.

    Active iff:
      - !requires_toggled_on OR ToggledOn[scene, toggle_idx[h]] is True
      - !requires_closed OR Open[scene, open_idx[h]] is False
      - !requires_on_fire OR OnFire[scene, onfire_idx[h]] is True (previous step's value —
        OnFire runs after this state in the captured graph; see class docstring)
    """
    s, h = wp.tid()
    active = wp.uint8(1)
    if requires_toggled_on[h] != wp.uint8(0):
        ti = toggle_idx[h]
        if ti < wp.int32(0) or s >= n_toggle_scenes:
            active = wp.uint8(0)
        elif toggle_values[s, ti] == wp.uint8(0):
            active = wp.uint8(0)
    if active == wp.uint8(1) and requires_closed[h] != wp.uint8(0):
        oi = open_idx[h]
        if oi < wp.int32(0) or s >= n_open_scenes:
            active = wp.uint8(0)
        elif open_values[s, oi] != wp.uint8(0):
            active = wp.uint8(0)
    if active == wp.uint8(1) and requires_on_fire[h] != wp.uint8(0):
        fi = onfire_idx[h]
        if fi < wp.int32(0) or s >= n_onfire_scenes:
            active = wp.uint8(0)
        elif onfire_values[s, fi] == wp.uint8(0):
            active = wp.uint8(0)
    out_values[s, h] = active


class HeatSourceOrSink(TensorizedAbsoluteState, LinkBasedStateMixin):
    """
    Boolean state: whether this heat source / sink is currently active, i.e. whether all of its
    activation gates are satisfied:
      - requires_toggled_on → ToggledOn must be True
      - requires_closed     → Open must be False
      - requires_on_fire    → OnFire must be True. OnFire depends on Temperature, which depends
        on this state, so inside the captured graph this gate reads the PREVIOUS step's OnFire
        values — the deliberate one-step lag that breaks the fire feedback cycle.

    This state only computes its own gate. The actual heat propagation — which targets an
    active source influences, by how much, and the self-sustaining fire clamp — is computed by
    Temperature (which depends on this state) from the per-source config arrays published here.
    `affects_obj()` queries Temperature's influence mask.

    A flammable object is modeled by the `flammable` ability as an OnFire threshold detector
    plus an instance of this state with requires_on_fire=True and temperature=fire_temperature.
    """

    # Per-HSS config (N_hss,) — uploaded once in initialize_view as wp.arrays (single source of
    # truth). Read by Temperature's kernels (Temperature depends on this state).
    _temperatures = None  # wp.array (N_hss,) float32
    _heating_rates = None  # wp.array (N_hss,) float32
    _distance_thresholds = None  # wp.array (N_hss,) float32
    _requires_toggled_on = None  # wp.array (N_hss,) uint8
    _requires_closed = None  # wp.array (N_hss,) uint8
    _requires_inside = None  # wp.array (N_hss,) uint8
    _requires_on_fire = None  # wp.array (N_hss,) uint8
    _ignition_temperatures = None  # wp.array (N_hss,) float32 — self-clamp threshold (requires_on_fire only)
    _link_flat_idx = None  # wp.array (N_hss,) int32 into RigidBodyViewAPI.POSE_MATRICES — -1 if requires_inside
    _link_local_offset = None  # wp.array (N_hss,) vec3 — offset of heat element in link frame

    # Index maps into the gate states' OBJ_IDXS — rebuilt in pre_update because OnFire's view
    # initializes AFTER this state's (OnFire depends on Temperature, which depends on this state).
    _self_toggle_idx = None  # wp.array (N_hss,) int32 into ToggledOn.OBJ_IDXS
    _self_open_idx = None  # wp.array (N_hss,) int32 into Open.OBJ_IDXS
    _self_onfire_idx = None  # wp.array (N_hss,) int32 into OnFire.OBJ_IDXS

    # Placeholder wp.array to fill into kernel arguments when an optional gate state
    # (ToggledOn, Open, OnFire) tracks no objects. Warp requires a valid wp.array of the
    # correct dtype for the signature.
    _placeholder_values = None  # wp.array (1, 1) uint8

    def __init__(
        self,
        obj,
        temperature=None,
        heating_rate=None,
        distance_threshold=None,
        requires_toggled_on=False,
        requires_closed=False,
        requires_inside=False,
        requires_on_fire=False,
        ignition_temperature=None,
    ):
        """
        Args:
            obj (StatefulObject): The object with the heat source ability.
            temperature (float): The temperature of the heat source. Defaults to
                DEFAULT_FIRE_TEMPERATURE when @requires_on_fire, otherwise DEFAULT_TEMPERATURE.
            heating_rate (float): Fraction in [0, 1] of the temperature difference with the
                heat source temperature should be received every step, per second.
            distance_threshold (float): The distance threshold which an object needs
                to be closer than in order to receive heat from this heat source.
            requires_toggled_on (bool): Whether the heat source object needs to be
                toggled on to emit heat. Requires toggleable ability if set to True.
            requires_closed (bool): Whether the heat source object needs to be
                closed (e.g. in terms of the joints) to emit heat. Requires openable
                ability if set to True.
            requires_inside (bool): Whether an object needs to be `inside` the
                heat source to receive heat. See the Inside state for details. This
                will mean that the "heating element" link for the object will be
                ignored.
            requires_on_fire (bool): Whether this heat source is the object's own fire, i.e.
                only active while the object's OnFire state is True. Set (together with
                @ignition_temperature) by the flammable ability, which pairs this state with
                OnFire. Requires the OnFire state if set to True.
            ignition_temperature (float): Only relevant if @requires_on_fire. Threshold at /
                above which the fire sustains itself: while active, Temperature holds this
                object at @temperature unless it has been cooled below this threshold
                (e.g. extinguished).
        """
        super().__init__(obj)
        default_temperature = m.DEFAULT_FIRE_TEMPERATURE if requires_on_fire else m.DEFAULT_TEMPERATURE
        self._temperature = temperature if temperature is not None else default_temperature
        self._heating_rate = heating_rate if heating_rate is not None else m.DEFAULT_HEATING_RATE
        self.distance_threshold = distance_threshold if distance_threshold is not None else m.DEFAULT_DISTANCE_THRESHOLD

        if requires_toggled_on:
            assert ToggledOn in self.obj.states
        self.requires_toggled_on = requires_toggled_on

        if requires_closed:
            assert Open in self.obj.states
        self.requires_closed = requires_closed

        self.requires_inside = requires_inside

        # OnFire is constructed after this state (it depends on Temperature, which depends on
        # this state), so its presence is validated in _initialize instead of here.
        self.requires_on_fire = requires_on_fire
        self.ignition_temperature = (
            ignition_temperature if ignition_temperature is not None else m.DEFAULT_IGNITION_TEMPERATURE
        )
        if requires_on_fire:
            assert (
                self._temperature > self.ignition_temperature
            ), "fire temperature should be higher than ignition temperature."

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason
        for kwarg, state_type in zip(("requires_toggled_on", "requires_closed"), (ToggledOn, Open)):
            if kwargs.get(kwarg, False) and state_type not in obj.states:
                return False, f"{cls.__name__} has {kwarg} but obj has no {state_type.__name__} state!"
        return True, None

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        compatible, reason = super().is_compatible_asset(prim, **kwargs)
        if not compatible:
            return compatible, reason
        for kwarg, state_type in zip(("requires_toggled_on", "requires_closed"), (ToggledOn, Open)):
            if kwargs.get(kwarg, False) and not state_type.is_compatible_asset(prim=prim, **kwargs)[0]:
                return False, f"{cls.__name__} has {kwarg} but obj has no {state_type.__name__} state!"
        return True, None

    @classproperty
    def meta_link_types(cls):
        return [m.HEATSOURCE_META_LINK_TYPE]

    @classmethod
    def requires_meta_link(cls, **kwargs):
        # No meta link required for containment-based sources or fire sources (which fall back
        # to the root link).
        return not (kwargs.get("requires_inside", False) or kwargs.get("requires_on_fire", False))

    @property
    def _default_link(self):
        # Fall back to the root link when no meta link is required
        if self.requires_inside or self.requires_on_fire:
            return self.obj.root_link
        return super()._default_link

    @property
    def heating_rate(self):
        """
        Returns:
            float: Temperature changing rate of this heat source / sink
        """
        return self._heating_rate

    @property
    def temperature(self):
        """
        Returns:
            float: Temperature of this heat source / sink
        """
        return self._temperature

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        # AABB / Inside are consumed by Temperature's heat-gather kernel, but the source object
        # itself must be registered with them (e.g. a target is heated by an oven only if it is
        # Inside the oven), so they are required here.
        deps.update({AABB, Inside})
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        # Gate states. NOTE: OnFire is deliberately NOT listed — it depends on Temperature,
        # which depends on this state, so declaring it would create a cycle. Its values are
        # read lag-1 instead (see class docstring).
        deps.update({ToggledOn, Open})
        return deps

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "heat_source_or_sink"

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()
        if self.requires_on_fire:
            # Local import to avoid a module-level cycle (on_fire imports temperature, which
            # imports this module).
            from omnigibson.object_states.on_fire import OnFire

            assert (
                OnFire in self.obj.states
            ), f"{type(self).__name__} on {self.obj.name} has requires_on_fire but obj has no OnFire state!"

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls._temperatures = None
        cls._heating_rates = None
        cls._distance_thresholds = None
        cls._requires_toggled_on = None
        cls._requires_closed = None
        cls._requires_inside = None
        cls._requires_on_fire = None
        cls._ignition_temperatures = None
        cls._link_flat_idx = None
        cls._link_local_offset = None
        cls._self_toggle_idx = None
        cls._self_open_idx = None
        cls._self_onfire_idx = None

        # Placeholder wp.array to fill into kernel arguments when an optional gate state is empty.
        cls._placeholder_values = wp.zeros((1, 1), dtype=wp.uint8, device="cuda")

    @classmethod
    def initialize_view(cls):
        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES
        super().initialize_view()

        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls._temperatures = None
            cls._heating_rates = None
            cls._distance_thresholds = None
            cls._requires_toggled_on = None
            cls._requires_closed = None
            cls._requires_inside = None
            cls._requires_on_fire = None
            cls._ignition_temperatures = None
            cls._link_flat_idx = None
            cls._link_local_offset = None
            cls._self_toggle_idx = None
            cls._self_open_idx = None
            cls._self_onfire_idx = None
            return

        temperatures = th.zeros(N, dtype=th.float32)
        heating_rates = th.zeros(N, dtype=th.float32)
        distance_thresholds = th.zeros(N, dtype=th.float32)
        requires_toggled_on = th.zeros(N, dtype=th.uint8)
        requires_closed = th.zeros(N, dtype=th.uint8)
        requires_inside = th.zeros(N, dtype=th.uint8)
        requires_on_fire = th.zeros(N, dtype=th.uint8)
        ignition_temperatures = th.zeros(N, dtype=th.float32)
        link_flat_idx = th.full((N,), -1, dtype=th.int32)
        link_local_offset = th.zeros((N, 3), dtype=th.float32)

        for _, hss_obj_idx in cls.OBJ_IDXS.items():
            hss_state_instance = None
            for scene_row in cls.IDX_OBJS:
                if scene_row[hss_obj_idx] is not None:
                    hss_state_instance = scene_row[hss_obj_idx].states[cls]
                    break
            if hss_state_instance is None:
                continue
            temperatures[hss_obj_idx] = float(hss_state_instance._temperature)
            heating_rates[hss_obj_idx] = float(hss_state_instance._heating_rate)
            distance_thresholds[hss_obj_idx] = float(hss_state_instance.distance_threshold)
            requires_toggled_on[hss_obj_idx] = 1 if hss_state_instance.requires_toggled_on else 0
            requires_closed[hss_obj_idx] = 1 if hss_state_instance.requires_closed else 0
            requires_inside[hss_obj_idx] = 1 if hss_state_instance.requires_inside else 0
            requires_on_fire[hss_obj_idx] = 1 if hss_state_instance.requires_on_fire else 0
            ignition_temperatures[hss_obj_idx] = float(hss_state_instance.ignition_temperature)
            if not hss_state_instance.requires_inside:
                # Store this hss' heat element link so Temperature can compute source positions:
                # the meta link when annotated (e.g. a candle wick), the root link for fire sources
                # without one.
                if hss_state_instance._links:
                    link = hss_state_instance.link
                elif hss_state_instance.requires_on_fire:
                    link = hss_state_instance.obj.root_link
                else:
                    link = None
                if link is not None:
                    flat = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                    if flat is not None:
                        link_flat_idx[hss_obj_idx] = int(flat)

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list
        cls._temperatures = create_tensor_from_list(temperatures, "float32", device="cuda")
        cls._heating_rates = create_tensor_from_list(heating_rates, "float32", device="cuda")
        cls._distance_thresholds = create_tensor_from_list(distance_thresholds, "float32", device="cuda")
        cls._requires_toggled_on = create_tensor_from_list(requires_toggled_on, "uint8", device="cuda")
        cls._requires_closed = create_tensor_from_list(requires_closed, "uint8", device="cuda")
        cls._requires_inside = create_tensor_from_list(requires_inside, "uint8", device="cuda")
        cls._requires_on_fire = create_tensor_from_list(requires_on_fire, "uint8", device="cuda")
        cls._ignition_temperatures = create_tensor_from_list(ignition_temperatures, "float32", device="cuda")
        cls._link_flat_idx = create_tensor_from_list(link_flat_idx, "int32", device="cuda")
        # vec3 has no scalar-only helper — wp.array reinterprets the (N, 3) float32 CPU buffer as (N,) vec3.
        cls._link_local_offset = wp.array(link_local_offset, dtype=wp.vec3, device="cuda")

        # Gate index maps are deferred to pre_update — OnFire's view has not been rebuilt yet
        # at this point (it initializes after Temperature, which initializes after this state).
        cls._self_toggle_idx = None
        cls._self_open_idx = None
        cls._self_onfire_idx = None

    @classmethod
    def _rebuild_cross_state_maps(cls):
        """
        Build (or rebuild) the index tables into the gate states' OBJ_IDXS. Called from
        pre_update when the captured graph is being rebuilt, because OnFire's view initializes
        AFTER this state's.
        """
        # Local import to avoid a module-level cycle
        from omnigibson.object_states.on_fire import OnFire

        N = len(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else 0
        if N == 0:
            return

        toggle_map = ToggledOn.OBJ_IDXS or {}
        open_map = Open.OBJ_IDXS or {}
        onfire_map = OnFire.OBJ_IDXS or {}

        self_toggle_idx = th.full((N,), -1, dtype=th.int32)
        self_open_idx = th.full((N,), -1, dtype=th.int32)
        self_onfire_idx = th.full((N,), -1, dtype=th.int32)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            self_toggle_idx[obj_idx] = toggle_map.get(rel_path, -1)
            self_open_idx[obj_idx] = open_map.get(rel_path, -1)
            self_onfire_idx[obj_idx] = onfire_map.get(rel_path, -1)

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list
        cls._self_toggle_idx = create_tensor_from_list(self_toggle_idx, "int32", device="cuda")
        cls._self_open_idx = create_tensor_from_list(self_open_idx, "int32", device="cuda")
        cls._self_onfire_idx = create_tensor_from_list(self_onfire_idx, "int32", device="cuda")

    @classmethod
    def pre_update(cls, dt=0.0):
        super().pre_update(dt)
        # Only rebuild when the captured graph is being rebuilt. graph_dirty is the single
        # trigger: every event that invalidates the index tables goes through some state's
        # initialize_view, which sets graph_dirty. Rebuild reallocates GPU buffers that must be
        # baked into the freshly-captured graph, so rebuild and recapture must stay coupled.
        if TensorizedState.graph_dirty:
            cls._rebuild_cross_state_maps()

    @classmethod
    def _update_values(cls, values):
        # Local import to avoid a module-level cycle
        from omnigibson.object_states.on_fire import OnFire

        if cls.VALUES_WP is None or cls._self_toggle_idx is None:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return

        # An optional gate state may track no objects; swap in a cached placeholder wp.array to
        # satisfy the kernel's signature, with a scene count of 0 so it is never read.
        def gate_values(state):
            values_wp = state.VALUES_WP
            if values_wp is None:
                return cls._placeholder_values, 0
            return values_wp, state.VALUES.shape[0]

        toggle_values_wp, n_toggle_scenes = gate_values(ToggledOn)
        open_values_wp, n_open_scenes = gate_values(Open)
        # NOTE: lag-1 read — OnFire runs after this state in the captured graph, so these are
        # the previous step's values. This is the deliberate one-step delay that breaks the
        # OnFire → Temperature → HeatSourceOrSink dependency cycle.
        onfire_values_wp, n_onfire_scenes = gate_values(OnFire)

        wp.launch(
            kernel=_heatsource_is_active_kernel,
            dim=(S, N),
            inputs=[
                cls._requires_toggled_on,
                cls._requires_closed,
                cls._requires_on_fire,
                cls._self_toggle_idx,
                cls._self_open_idx,
                cls._self_onfire_idx,
                toggle_values_wp,
                open_values_wp,
                onfire_values_wp,
                wp.int32(n_toggle_scenes),
                wp.int32(n_open_scenes),
                wp.int32(n_onfire_scenes),
                cls.VALUES_WP,
            ],
            device="cuda",
        )

    def _get_value(self):
        # Raw read of the CPU mirror. Freshness is handled by the public get_value() wrapper
        # (TensorizedState.get_value -> maybe_refresh_caches), matching AABB/ToggledOn. Refreshing
        # here would also fire on the _dump_state path (e.g. prim_base.initialize's state-size
        # probe), forcing a mid-init cache refresh + post_update.
        if self.OBJ_IDXS is None or self.obj.relative_prim_path not in self.OBJ_IDXS:
            return False
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return bool(self.VALUES_CPU[s, obj_idx].item())

    def _set_value(self, new_value):
        # Boolean gate; setter mostly used for testing / state reset. Write both CPU + GPU
        # so the next read is consistent without waiting for the next graph pass.
        s = self.obj.scene.idx
        if self.obj.relative_prim_path not in self.OBJ_IDXS:
            return False
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        self.VALUES[s, obj_idx] = bool(new_value)
        self.VALUES_CPU[s, obj_idx] = bool(new_value)
        return True

    def affects_obj(self, obj):
        """
        Whether this heat source / sink is currently heating @obj.

        Args:
            obj (StatefulObject): Object whose temperature delta is being queried.

        Returns:
            bool
        """
        # Local import to avoid circular dependency at module load time
        from omnigibson.object_states.temperature import Temperature

        return Temperature.is_influenced_by(self.obj, obj)

    # Nothing needs to be done to save/load HeatSource
