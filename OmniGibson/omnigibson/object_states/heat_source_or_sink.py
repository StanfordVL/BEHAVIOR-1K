import torch as th
import warp as wp

from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.object_states.tensorized_state import TensorizedState, _wp_from_torch
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.HEATSOURCE_META_LINK_TYPE = "heatsource"
m.HEATING_ELEMENT_MARKER_SCALE = [1.0] * 3

# TODO: Delete default values for this and make them required.
m.DEFAULT_TEMPERATURE = 200
m.DEFAULT_HEATING_RATE = 0.04
m.DEFAULT_DISTANCE_THRESHOLD = 0.2


@wp.kernel
def _hss_gate_kernel(
    requires_toggled_on: wp.array(dtype=wp.uint8),  # (N_hss,)
    requires_closed: wp.array(dtype=wp.uint8),  # (N_hss,)
    toggle_idx: wp.array(dtype=wp.int32),  # (N_hss,) into ToggledOn.OBJ_IDXS, -1 if missing
    open_idx: wp.array(dtype=wp.int32),  # (N_hss,) into Open.OBJ_IDXS, -1 if missing
    toggle_values: wp.array2d(dtype=wp.uint8),  # (S, N_toggle)
    open_values: wp.array2d(dtype=wp.uint8),  # (S, N_open)
    has_toggle: wp.int32,  # 1 if toggle_values is non-empty
    has_open: wp.int32,
    out_values: wp.array2d(dtype=wp.uint8),  # (S, N_hss) — set to 1 if gate passes
):
    """
    Per (scene, hss) thread: compute the activation gate.

    Active iff:
      - !requires_toggled_on OR ToggledOn[scene, toggle_idx[h]] is True
      - !requires_closed OR Open[scene, open_idx[h]] is False
    """
    s, h = wp.tid()
    active = wp.uint8(1)
    if requires_toggled_on[h] != wp.uint8(0):
        ti = toggle_idx[h]
        if ti < wp.int32(0) or has_toggle == wp.int32(0):
            active = wp.uint8(0)
        elif toggle_values[s, ti] == wp.uint8(0):
            active = wp.uint8(0)
    if active == wp.uint8(1) and requires_closed[h] != wp.uint8(0):
        oi = open_idx[h]
        if oi < wp.int32(0) or has_open == wp.int32(0):
            active = wp.uint8(0)
        elif open_values[s, oi] != wp.uint8(0):
            active = wp.uint8(0)
    out_values[s, h] = active


@wp.kernel
def _hss_propagate_kernel(
    hss_values: wp.array2d(dtype=wp.uint8),  # (S, N_hss)
    requires_inside: wp.array(dtype=wp.uint8),  # (N_hss,)
    temperatures: wp.array(dtype=wp.float32),  # (N_hss,)
    heating_rates: wp.array(dtype=wp.float32),  # (N_hss,)
    distance_thresholds: wp.array(dtype=wp.float32),  # (N_hss,)
    self_temp_idx: wp.array(dtype=wp.int32),  # (N_hss,) into Temperature N — to skip self
    self_inside_idx: wp.array(dtype=wp.int32),  # (N_hss,) into Inside N (for requires_inside)
    link_flat_idx: wp.array(dtype=wp.int32),  # (N_hss,) into POSE_MATRICES — -1 if N/A
    link_local_offset: wp.array(dtype=wp.vec3),  # (N_hss,) heat element offset in link local frame
    temp_to_aabb_idx: wp.array(dtype=wp.int32),  # (N_temp,) Temperature N → AABB N
    temp_to_inside_idx: wp.array(dtype=wp.int32),  # (N_temp,) Temperature N → Inside N
    pose_matrices: wp.array(dtype=wp.mat44),  # RigidBodyViewAPI.POSE_MATRICES
    aabb_values: wp.array3d(dtype=wp.float32),  # AABB (S, N_aabb, 6)
    inside_values: wp.array3d(dtype=wp.uint8),  # Inside (S, N_inside, N_inside) — uint8 view of bool
    temperature_values: wp.array2d(dtype=wp.float32),  # Temperature (S, N_temp)
    has_inside: wp.int32,
    affected_mask: wp.array3d(dtype=wp.uint8),  # (S, N_hss, N_temp) — out
    incoming_heat_rate: wp.array2d(dtype=wp.float32),  # Temperature.INCOMING_HEAT_RATE (S, N_temp) — out
):
    """
    Per (scene, hss, target) thread: if HSS h is active and target n is in range, write
    AFFECTED_MASK[s, h, n] = 1 and atomic_add (T_h - T_n) * rate into INCOMING_HEAT_RATE[s, n].
    """
    s, h, n = wp.tid()
    if hss_values[s, h] == wp.uint8(0):
        return
    if self_temp_idx[h] == n:
        return
    a_n = temp_to_aabb_idx[n]
    if a_n < wp.int32(0):
        return

    if requires_inside[h] != wp.uint8(0):
        if has_inside == wp.int32(0):
            return
        ihi = self_inside_idx[h]
        ini = temp_to_inside_idx[n]
        if ihi < wp.int32(0) or ini < wp.int32(0):
            return
        # Inside[s, target, container]: True iff target's AABB center lies in container's volume
        if inside_values[s, ini, ihi] == wp.uint8(0):
            return
    else:
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

    affected_mask[s, h, n] = wp.uint8(1)
    delta = (temperatures[h] - temperature_values[s, n]) * heating_rates[h]
    wp.atomic_add(incoming_heat_rate, s, n, delta)


class HeatSourceOrSink(TensorizedAbsoluteState, LinkBasedStateMixin):
    """
    Boolean state representing whether a heat source / sink is currently active. Active means
    the activation gates (`requires_toggled_on`, `requires_closed`) are satisfied; spatial
    affecting of specific objects is queried via `affects_obj(obj)` against the AFFECTED_MASK.

    Computation runs inside the captured Warp graph:
      1. Per-HSS gate kernel writes VALUES based on ToggledOn / Open.
      2. Per-(scene, hss, target) propagate kernel writes AFFECTED_MASK and atomic_adds
         (T_h - T_n) * rate into Temperature.INCOMING_HEAT_RATE.

    The Temperature decay kernel then consumes the accumulated rate and zeros the buffer
    (see temperature.py for the consume-and-zero rationale). Cloth targets are handled by a
    CPU post-pass in `_update` because cloth is not tracked by AABB / Inside.
    """

    # Per-HSS config (N_hss,) — uploaded once in initialize_view from each instance.
    TEMPERATURES = None
    TEMPERATURES_WP = None
    HEATING_RATES = None
    HEATING_RATES_WP = None
    DISTANCE_THRESHOLDS = None
    DISTANCE_THRESHOLDS_WP = None
    REQUIRES_TOGGLED_ON = None  # uint8 (N_hss,)
    REQUIRES_TOGGLED_ON_WP = None
    REQUIRES_CLOSED = None
    REQUIRES_CLOSED_WP = None
    REQUIRES_INSIDE = None
    REQUIRES_INSIDE_WP = None
    LINK_FLAT_IDX = None  # int32 (N_hss,) into RigidBodyViewAPI.POSE_MATRICES — -1 if requires_inside
    LINK_FLAT_IDX_WP = None
    LINK_LOCAL_OFFSET = None  # float32 (N_hss, 3) — offset of heat element in link frame
    LINK_LOCAL_OFFSET_WP = None

    # Cross-state index maps — rebuilt in pre_update because Temperature.initialize_view
    # runs after HSS.initialize_view (HSS is a dep of Temperature in the topo).
    SELF_TEMP_IDX = None  # int32 (N_hss,) into Temperature.OBJ_IDXS
    SELF_TEMP_IDX_WP = None
    SELF_INSIDE_IDX = None  # int32 (N_hss,) into Inside.OBJ_IDXS
    SELF_INSIDE_IDX_WP = None
    TOGGLE_IDX = None  # int32 (N_hss,) into ToggledOn.OBJ_IDXS
    TOGGLE_IDX_WP = None
    OPEN_IDX = None  # int32 (N_hss,) into Open.OBJ_IDXS
    OPEN_IDX_WP = None
    TEMP_TO_AABB_IDX = None  # int32 (N_temp,) Temperature N → AABB N
    TEMP_TO_AABB_IDX_WP = None
    TEMP_TO_INSIDE_IDX = None  # int32 (N_temp,) Temperature N → Inside N
    TEMP_TO_INSIDE_IDX_WP = None

    # AFFECTED_MASK: (S, N_hss, N_temp) bool — set by the propagate kernel, consumed via
    # affects_obj() on CPU.
    AFFECTED_MASK = None
    AFFECTED_MASK_WP = None
    AFFECTED_MASK_CPU = None
    AFFECTED_MASK_CPU_WP = None

    # Tracks the (N_hss, N_temp) shape used to allocate AFFECTED_MASK so pre_update can
    # detect when Temperature's count changes and resize.
    _CACHED_N_HSS = 0
    _CACHED_N_TEMP = 0

    # Placeholder wp.arrays used as kernel arguments when an optional dependency state has
    # zero tracked objects. The kernel branches behind `has_*` flags so it never indexes them,
    # but Warp requires a valid wp.array of the correct dtype for the signature. Allocated
    # once in global_initialize so the captured graph never touches torch / wp.from_torch.
    _PLACEHOLDER_VALUES = None
    _PLACEHOLDER_VALUES_WP = None
    _PLACEHOLDER_INSIDE = None
    _PLACEHOLDER_INSIDE_WP = None

    def __init__(
        self,
        obj,
        temperature=None,
        heating_rate=None,
        distance_threshold=None,
        requires_toggled_on=False,
        requires_closed=False,
        requires_inside=False,
    ):
        """
        Args:
            obj (StatefulObject): The object with the heat source ability.
            temperature (float): The temperature of the heat source.
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
        """
        super().__init__(obj)
        self._temperature = temperature if temperature is not None else m.DEFAULT_TEMPERATURE
        self._heating_rate = heating_rate if heating_rate is not None else m.DEFAULT_HEATING_RATE
        self.distance_threshold = distance_threshold if distance_threshold is not None else m.DEFAULT_DISTANCE_THRESHOLD

        if requires_toggled_on:
            assert ToggledOn in self.obj.states
        self.requires_toggled_on = requires_toggled_on

        if requires_closed:
            assert Open in self.obj.states
        self.requires_closed = requires_closed

        self.requires_inside = requires_inside

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
        # No meta link required if inside
        return not kwargs.get("requires_inside", False)

    @property
    def _default_link(self):
        # Only supported if we require inside
        return self.obj.root_link if self.requires_inside else super()._default_link

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
        deps.update({AABB, Inside})
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
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

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls.TEMPERATURES = None
        cls.TEMPERATURES_WP = None
        cls.HEATING_RATES = None
        cls.HEATING_RATES_WP = None
        cls.DISTANCE_THRESHOLDS = None
        cls.DISTANCE_THRESHOLDS_WP = None
        cls.REQUIRES_TOGGLED_ON = None
        cls.REQUIRES_TOGGLED_ON_WP = None
        cls.REQUIRES_CLOSED = None
        cls.REQUIRES_CLOSED_WP = None
        cls.REQUIRES_INSIDE = None
        cls.REQUIRES_INSIDE_WP = None
        cls.LINK_FLAT_IDX = None
        cls.LINK_FLAT_IDX_WP = None
        cls.LINK_LOCAL_OFFSET = None
        cls.LINK_LOCAL_OFFSET_WP = None
        cls.SELF_TEMP_IDX = None
        cls.SELF_TEMP_IDX_WP = None
        cls.SELF_INSIDE_IDX = None
        cls.SELF_INSIDE_IDX_WP = None
        cls.TOGGLE_IDX = None
        cls.TOGGLE_IDX_WP = None
        cls.OPEN_IDX = None
        cls.OPEN_IDX_WP = None
        cls.TEMP_TO_AABB_IDX = None
        cls.TEMP_TO_AABB_IDX_WP = None
        cls.TEMP_TO_INSIDE_IDX = None
        cls.TEMP_TO_INSIDE_IDX_WP = None
        cls.AFFECTED_MASK = None
        cls.AFFECTED_MASK_WP = None
        cls.AFFECTED_MASK_CPU = None
        cls.AFFECTED_MASK_CPU_WP = None
        cls._CACHED_N_HSS = 0
        cls._CACHED_N_TEMP = 0

        # Single-cell placeholders for use as kernel arguments when an optional dep state
        # is empty. The kernel never indexes them — flags gate the branches.
        cls._PLACEHOLDER_VALUES = th.zeros((1, 1), dtype=th.uint8, device="cuda")
        cls._PLACEHOLDER_VALUES_WP = wp.from_torch(cls._PLACEHOLDER_VALUES, dtype=wp.uint8)
        cls._PLACEHOLDER_INSIDE = th.zeros((1, 1, 1), dtype=th.uint8, device="cuda")
        cls._PLACEHOLDER_INSIDE_WP = wp.from_torch(cls._PLACEHOLDER_INSIDE, dtype=wp.uint8)

    @classmethod
    def initialize_view(cls):
        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES
        super().initialize_view()

        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls.TEMPERATURES_WP = None
            cls.HEATING_RATES_WP = None
            cls.DISTANCE_THRESHOLDS_WP = None
            cls.REQUIRES_TOGGLED_ON_WP = None
            cls.REQUIRES_CLOSED_WP = None
            cls.REQUIRES_INSIDE_WP = None
            cls.LINK_FLAT_IDX_WP = None
            cls.LINK_LOCAL_OFFSET_WP = None
            cls.SELF_TEMP_IDX_WP = None
            cls.SELF_INSIDE_IDX_WP = None
            cls.TOGGLE_IDX_WP = None
            cls.OPEN_IDX_WP = None
            cls.TEMP_TO_AABB_IDX_WP = None
            cls.TEMP_TO_INSIDE_IDX_WP = None
            cls.AFFECTED_MASK = None
            cls.AFFECTED_MASK_WP = None
            cls.AFFECTED_MASK_CPU = None
            cls.AFFECTED_MASK_CPU_WP = None
            cls._CACHED_N_HSS = 0
            cls._CACHED_N_TEMP = 0
            return

        # Per-HSS config — same across all scenes (HSS instances are matched by relative_prim_path).
        # We pick the first non-None instance for each obj_idx to read configuration from.
        temperatures = th.zeros(N, dtype=th.float32)
        heating_rates = th.zeros(N, dtype=th.float32)
        distance_thresholds = th.zeros(N, dtype=th.float32)
        requires_toggled_on = th.zeros(N, dtype=th.uint8)
        requires_closed = th.zeros(N, dtype=th.uint8)
        requires_inside = th.zeros(N, dtype=th.uint8)
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
            temperatures[obj_idx] = float(inst._temperature)
            heating_rates[obj_idx] = float(inst._heating_rate)
            distance_thresholds[obj_idx] = float(inst.distance_threshold)
            requires_toggled_on[obj_idx] = 1 if inst.requires_toggled_on else 0
            requires_closed[obj_idx] = 1 if inst.requires_closed else 0
            requires_inside[obj_idx] = 1 if inst.requires_inside else 0
            if not inst.requires_inside and inst._links:
                # Meta link exists; use its flat idx. Heat element pos == link world pos
                # (offset is zero since the meta link IS the heat element).
                link = inst.link
                flat = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                if flat is not None:
                    link_flat_idx[obj_idx] = int(flat)

        cls.TEMPERATURES = temperatures.cuda()
        cls.HEATING_RATES = heating_rates.cuda()
        cls.DISTANCE_THRESHOLDS = distance_thresholds.cuda()
        cls.REQUIRES_TOGGLED_ON = requires_toggled_on.cuda()
        cls.REQUIRES_CLOSED = requires_closed.cuda()
        cls.REQUIRES_INSIDE = requires_inside.cuda()
        cls.LINK_FLAT_IDX = link_flat_idx.cuda()
        cls.LINK_LOCAL_OFFSET = link_local_offset.cuda()

        cls.TEMPERATURES_WP = wp.from_torch(cls.TEMPERATURES)
        cls.HEATING_RATES_WP = wp.from_torch(cls.HEATING_RATES)
        cls.DISTANCE_THRESHOLDS_WP = wp.from_torch(cls.DISTANCE_THRESHOLDS)
        cls.REQUIRES_TOGGLED_ON_WP = wp.from_torch(cls.REQUIRES_TOGGLED_ON, dtype=wp.uint8)
        cls.REQUIRES_CLOSED_WP = wp.from_torch(cls.REQUIRES_CLOSED, dtype=wp.uint8)
        cls.REQUIRES_INSIDE_WP = wp.from_torch(cls.REQUIRES_INSIDE, dtype=wp.uint8)
        cls.LINK_FLAT_IDX_WP = wp.from_torch(cls.LINK_FLAT_IDX)
        cls.LINK_LOCAL_OFFSET_WP = wp.from_torch(cls.LINK_LOCAL_OFFSET, dtype=wp.vec3)

        # Cross-state index maps + AFFECTED_MASK are deferred to pre_update because
        # Temperature / Inside / ToggledOn / Open initialize_views may not have been called yet
        # in the order Temperature depends on HSS.
        cls.SELF_TEMP_IDX_WP = None
        cls.SELF_INSIDE_IDX_WP = None
        cls.TOGGLE_IDX_WP = None
        cls.OPEN_IDX_WP = None
        cls.TEMP_TO_AABB_IDX_WP = None
        cls.TEMP_TO_INSIDE_IDX_WP = None
        cls.AFFECTED_MASK = None
        cls.AFFECTED_MASK_WP = None
        cls.AFFECTED_MASK_CPU = None
        cls.AFFECTED_MASK_CPU_WP = None
        cls._CACHED_N_HSS = N
        cls._CACHED_N_TEMP = 0

    @classmethod
    def _rebuild_cross_state_maps(cls):
        """
        Build (or rebuild) the index tables that reference *other* states' OBJ_IDXS, and
        allocate AFFECTED_MASK with the current N_temp. Called from pre_update when needed.
        """
        # Local import to avoid circular dependency at module load time
        from omnigibson.object_states.temperature import Temperature

        N = len(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else 0
        S = len(cls.IDX_OBJS) if cls.IDX_OBJS is not None else 0
        N_temp = len(Temperature.OBJ_IDXS) if Temperature.OBJ_IDXS is not None else 0

        if N == 0:
            return

        # Per-HSS lookups into other states' N spaces
        self_temp_idx = th.full((N,), -1, dtype=th.int32)
        self_inside_idx = th.full((N,), -1, dtype=th.int32)
        toggle_idx = th.full((N,), -1, dtype=th.int32)
        open_idx = th.full((N,), -1, dtype=th.int32)

        temp_map = Temperature.OBJ_IDXS or {}
        inside_map = Inside.OBJ_IDXS or {}
        toggle_map = ToggledOn.OBJ_IDXS or {}
        open_map = Open.OBJ_IDXS or {}

        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            self_temp_idx[obj_idx] = temp_map.get(rel_path, -1)
            self_inside_idx[obj_idx] = inside_map.get(rel_path, -1)
            toggle_idx[obj_idx] = toggle_map.get(rel_path, -1)
            open_idx[obj_idx] = open_map.get(rel_path, -1)

        cls.SELF_TEMP_IDX = self_temp_idx.cuda()
        cls.SELF_INSIDE_IDX = self_inside_idx.cuda()
        cls.TOGGLE_IDX = toggle_idx.cuda()
        cls.OPEN_IDX = open_idx.cuda()
        cls.SELF_TEMP_IDX_WP = wp.from_torch(cls.SELF_TEMP_IDX)
        cls.SELF_INSIDE_IDX_WP = wp.from_torch(cls.SELF_INSIDE_IDX)
        cls.TOGGLE_IDX_WP = wp.from_torch(cls.TOGGLE_IDX)
        cls.OPEN_IDX_WP = wp.from_torch(cls.OPEN_IDX)

        # Per-Temperature lookups into AABB / Inside N spaces
        aabb_map = AABB.OBJ_IDXS or {}
        temp_to_aabb = th.full((N_temp,), -1, dtype=th.int32)
        temp_to_inside = th.full((N_temp,), -1, dtype=th.int32)
        for rel_path, t_idx in temp_map.items():
            temp_to_aabb[t_idx] = aabb_map.get(rel_path, -1)
            temp_to_inside[t_idx] = inside_map.get(rel_path, -1)
        if N_temp > 0:
            cls.TEMP_TO_AABB_IDX = temp_to_aabb.cuda()
            cls.TEMP_TO_INSIDE_IDX = temp_to_inside.cuda()
            cls.TEMP_TO_AABB_IDX_WP = wp.from_torch(cls.TEMP_TO_AABB_IDX)
            cls.TEMP_TO_INSIDE_IDX_WP = wp.from_torch(cls.TEMP_TO_INSIDE_IDX)
        else:
            cls.TEMP_TO_AABB_IDX_WP = None
            cls.TEMP_TO_INSIDE_IDX_WP = None

        # (Re)allocate AFFECTED_MASK if the shape changed
        if S > 0 and N > 0 and N_temp > 0:
            cls.AFFECTED_MASK = th.zeros((S, N, N_temp), dtype=th.bool, device="cuda")
            cls.AFFECTED_MASK_WP = _wp_from_torch(cls.AFFECTED_MASK)
            cls.AFFECTED_MASK_CPU = th.zeros((S, N, N_temp), dtype=th.bool).pin_memory()
            cls.AFFECTED_MASK_CPU_WP = _wp_from_torch(cls.AFFECTED_MASK_CPU)
        else:
            cls.AFFECTED_MASK = None
            cls.AFFECTED_MASK_WP = None
            cls.AFFECTED_MASK_CPU = None
            cls.AFFECTED_MASK_CPU_WP = None
        cls._CACHED_N_TEMP = N_temp

    @classmethod
    def pre_update(cls):
        super().pre_update()
        # Local import to avoid circular dependency at module load time
        from omnigibson.object_states.temperature import Temperature

        N_temp_now = len(Temperature.OBJ_IDXS) if Temperature.OBJ_IDXS is not None else 0
        # Rebuild cross-state maps when the captured graph is being rebuilt or N_temp changed.
        # `graph_dirty=True` indicates initialize_view has run since the last refresh.
        if TensorizedState.graph_dirty or cls._CACHED_N_TEMP != N_temp_now or cls.SELF_TEMP_IDX_WP is None:
            cls._rebuild_cross_state_maps()

        # Zero AFFECTED_MASK every step so the propagate kernel only OR-writes hits.
        if cls.AFFECTED_MASK is not None:
            cls.AFFECTED_MASK.zero_()

    @classmethod
    def _update_values(cls, values):
        # Local imports to keep module-load order safe
        from omnigibson.object_states.temperature import Temperature

        if cls.VALUES_WP is None or cls.SELF_TEMP_IDX_WP is None:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return

        toggle_values_wp = ToggledOn.VALUES_WP
        open_values_wp = Open.VALUES_WP
        # When an optional gate state has no tracked objects, swap in the cached placeholder
        # wp.array (allocated in global_initialize) so the kernel signature is satisfied;
        # the `has_*` flags below gate the actual reads.
        if toggle_values_wp is None:
            toggle_values_wp = cls._PLACEHOLDER_VALUES_WP
            has_toggle = 0
        else:
            has_toggle = 1
        if open_values_wp is None:
            open_values_wp = cls._PLACEHOLDER_VALUES_WP
            has_open = 0
        else:
            has_open = 1

        # 1) Gate kernel: write VALUES
        wp.launch(
            kernel=_hss_gate_kernel,
            dim=(S, N),
            inputs=[
                cls.REQUIRES_TOGGLED_ON_WP,
                cls.REQUIRES_CLOSED_WP,
                cls.TOGGLE_IDX_WP,
                cls.OPEN_IDX_WP,
                toggle_values_wp,
                open_values_wp,
                wp.int32(has_toggle),
                wp.int32(has_open),
                cls.VALUES_WP,
            ],
            device="cuda",
        )

        # 2) Propagate kernel: write AFFECTED_MASK + atomic_add INCOMING_HEAT_RATE
        if (
            cls.AFFECTED_MASK_WP is None
            or Temperature.VALUES_WP is None
            or Temperature.INCOMING_HEAT_RATE_WP is None
            or cls.TEMP_TO_AABB_IDX_WP is None
            or AABB.VALUES_WP is None
        ):
            return
        N_temp = Temperature.VALUES.shape[1]
        if N_temp == 0:
            return

        inside_values_wp = Inside.VALUES_WP
        if inside_values_wp is None:
            inside_values_wp = cls._PLACEHOLDER_INSIDE_WP
            has_inside = 0
        else:
            has_inside = 1

        wp.launch(
            kernel=_hss_propagate_kernel,
            dim=(S, N, N_temp),
            inputs=[
                cls.VALUES_WP,
                cls.REQUIRES_INSIDE_WP,
                cls.TEMPERATURES_WP,
                cls.HEATING_RATES_WP,
                cls.DISTANCE_THRESHOLDS_WP,
                cls.SELF_TEMP_IDX_WP,
                cls.SELF_INSIDE_IDX_WP,
                cls.LINK_FLAT_IDX_WP,
                cls.LINK_LOCAL_OFFSET_WP,
                cls.TEMP_TO_AABB_IDX_WP,
                cls.TEMP_TO_INSIDE_IDX_WP,
                RigidBodyViewAPI.POSE_MATRICES,
                AABB.VALUES_WP,
                inside_values_wp,
                Temperature.VALUES_WP,
                wp.int32(has_inside),
                cls.AFFECTED_MASK_WP,
                Temperature.INCOMING_HEAT_RATE_WP,
            ],
            device="cuda",
        )

        # Mirror AFFECTED_MASK → AFFECTED_MASK_CPU for affects_obj() CPU reads.
        if cls.AFFECTED_MASK_CPU_WP is not None:
            wp.copy(cls.AFFECTED_MASK_CPU_WP, cls.AFFECTED_MASK_WP)

    def _get_value(self):
        # Match base class semantics: bring caches into sync, then read CPU mirror
        TensorizedState.maybe_refresh_caches()
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

        # Lazy refresh so the read sees this-step's state.
        TensorizedState.maybe_refresh_caches()

        if not self.get_value():
            return False
        cls = type(self)
        if cls.OBJ_IDXS is None or self.obj.relative_prim_path not in cls.OBJ_IDXS:
            return False
        if Temperature.OBJ_IDXS is None or obj.relative_prim_path not in Temperature.OBJ_IDXS:
            return False
        if cls.AFFECTED_MASK_CPU is None:
            return False
        h = cls.OBJ_IDXS[self.obj.relative_prim_path]
        n = Temperature.OBJ_IDXS[obj.relative_prim_path]
        s = self.obj.scene.idx
        return bool(cls.AFFECTED_MASK_CPU[s, h, n].item())

    # Nothing needs to be done to save/load HeatSource
