import torch as th
import warp as wp

import omnigibson.lazy as lazy
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
def _heatsource_is_active_kernel(
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
    hss = heat_source_or_sink
    Per (scene, hss) thread: compute whether this hss is active. 0 not active. 1 active.

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
def _heatsource_can_influence_kernel(
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
    hss = heat_source_or_sink
    Per (scene, hss, target) thread: if HSS h is active and target n is in range, write
    affected_mask[s, h, n] = 1 and atomic_add (T_h - T_n) * rate into incoming_heat_rate[s, n].
    """
    s, h, n = wp.tid()
    if hss_values[s, h] == wp.uint8(0):  # early return if hss not active
        return
    if self_temp_idx[h] == n:  # do not influence self
        return
    target_aabb_idx = temp_to_aabb_idx[n]
    if target_aabb_idx < wp.int32(0):
        return

    # for require_inside items
    if requires_inside[h] != wp.uint8(0):
        if has_inside == wp.int32(0):  # check whether item is inside hss
            return
        hss_idx_in_inside = self_inside_idx[h]
        target_idx_in_inside = temp_to_inside_idx[n]
        if hss_idx_in_inside < wp.int32(0) or target_idx_in_inside < wp.int32(0):
            return
        # Check if target's AABB center lies in container's volume
        if inside_values[s, target_idx_in_inside, hss_idx_in_inside] == wp.uint8(0):
            return
    # for other items
    else:
        # compute hss world position
        li = link_flat_idx[h]
        if li < wp.int32(0):
            return
        link_pose = pose_matrices[li]
        local_offset = link_local_offset[h]
        hss_world_position = wp.mul(
            link_pose, wp.vec4(local_offset[0], local_offset[1], local_offset[2], wp.float32(1.0))
        )
        # get cx, cy, cz: center x, y, z of target's aabb bounding box
        cx = (aabb_values[s, target_aabb_idx, 0] + aabb_values[s, target_aabb_idx, 3]) * wp.float32(0.5)
        cy = (aabb_values[s, target_aabb_idx, 1] + aabb_values[s, target_aabb_idx, 4]) * wp.float32(0.5)
        cz = (aabb_values[s, target_aabb_idx, 2] + aabb_values[s, target_aabb_idx, 5]) * wp.float32(0.5)
        # get dx, dy, dz: delta of hss position and target's center on 3 directions
        dx = hss_world_position[0] - cx
        dy = hss_world_position[1] - cy
        dz = hss_world_position[2] - cz
        d2 = dx * dx + dy * dy + dz * dz
        threshold = distance_thresholds[h]
        if d2 > threshold * threshold:
            return

    affected_mask[s, h, n] = wp.uint8(1)
    delta = (temperatures[h] - temperature_values[s, n]) * heating_rates[h]
    wp.atomic_add(incoming_heat_rate, s, n, delta)


class HeatSourceOrSink(TensorizedAbsoluteState, LinkBasedStateMixin):
    """
    Boolean state representing whether a heat source / sink is currently active. Active means
    the activation gates (`requires_toggled_on`, `requires_closed`) are satisfied; spatial
    affecting of specific objects is queried via `affects_obj(obj)` against `_affected_mask`.

    Computation runs inside the captured Warp graph:
      1. Per-HSS `_heatsource_is_active_kernel` writes VALUES based on ToggledOn / Open.
      2. Per-(scene, hss, target) `_heatsource_can_influence_kernel` writes `_affected_mask`
         and atomic_adds (T_h - T_n) * rate into Temperature.INCOMING_HEAT_RATE.

    The Temperature decay kernel then consumes the accumulated rate and zeros INCOMING_HEAT_RATE
    (see temperature.py for the consume-and-zero rationale). Cloth targets are handled by a
    CPU post-pass in `_update` because cloth is not tracked by AABB / Inside and not supported now.
    """

    # Per-HSS config (N_hss,) — uploaded once in initialize_view as wp.arrays (single source of truth).
    _temperatures = None  # wp.array (N_hss,) float32
    _heating_rates = None  # wp.array (N_hss,) float32
    _distance_thresholds = None  # wp.array (N_hss,) float32
    _requires_toggled_on = None  # wp.array (N_hss,) uint8
    _requires_closed = None  # wp.array (N_hss,) uint8
    _requires_inside = None  # wp.array (N_hss,) uint8
    _link_flat_idx = None  # wp.array (N_hss,) int32 into RigidBodyViewAPI.POSE_MATRICES — -1 if requires_inside
    _link_local_offset = None  # wp.array (N_hss,) vec3 — offset of heat element in link frame

    # Cross-state index maps — rebuilt in pre_update because Temperature.initialize_view
    # runs after HSS.initialize_view (HSS is a dep of Temperature in the topo).
    _self_temp_idx = None  # wp.array (N_hss,) int32 into Temperature.OBJ_IDXS
    _self_inside_idx = None  # wp.array (N_hss,) int32 into Inside.OBJ_IDXS
    _self_toggle_idx = None  # wp.array (N_hss,) int32 into ToggledOn.OBJ_IDXS
    _self_open_idx = None  # wp.array (N_hss,) int32 into Open.OBJ_IDXS
    _temp_to_aabb_idx = None  # wp.array (N_temp,) int32 Temperature N → AABB N
    _temp_to_inside_idx = None  # wp.array (N_temp,) int32 Temperature N → Inside N

    # _affected_mask: (S, N_hss, N_temp) — set by the propagate kernel, consumed via affects_obj().
    # GPU is a uint8 wp.array (single source of truth); the CPU mirror keeps a torch
    # tensor (for .item() reads) plus a wp view (for the graph-safe wp.copy), mirroring how the
    # base class keeps VALUES_CPU + VALUES_CPU_WP.
    _affected_mask = None  # wp.array (S, N_hss, N_temp) uint8 — GPU
    _affected_mask_cpu = None  # torch bool (S, N_hss, N_temp), pinned — CPU mirror for affects_obj()
    _affected_mask_cpu_wp = None  # wp.array uint8 view of _affected_mask_cpu

    # Placeholder wp.array to fill into kernel arguments when an optional dependent state
    # (like toggle, inside) is empty. Warp requires a valid wp.array of the correct dtype
    # for the signature.
    _placeholder_values = None  # wp.array (1, 1) uint8
    _placeholder_inside = None  # wp.array (1, 1, 1) uint8

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
        cls._temperatures = None
        cls._heating_rates = None
        cls._distance_thresholds = None
        cls._requires_toggled_on = None
        cls._requires_closed = None
        cls._requires_inside = None
        cls._link_flat_idx = None
        cls._link_local_offset = None
        cls._self_temp_idx = None
        cls._self_inside_idx = None
        cls._self_toggle_idx = None
        cls._self_open_idx = None
        cls._temp_to_aabb_idx = None
        cls._temp_to_inside_idx = None
        cls._affected_mask = None
        cls._affected_mask_cpu = None
        cls._affected_mask_cpu_wp = None

        # Placeholder wp.array to fill into kernel arguments when an optional dependent state
        # (like toggle, inside) is empty.
        cls._placeholder_values = wp.zeros((1, 1), dtype=wp.uint8, device="cuda")
        cls._placeholder_inside = wp.zeros((1, 1, 1), dtype=wp.uint8, device="cuda")

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
            cls._link_flat_idx = None
            cls._link_local_offset = None
            cls._self_temp_idx = None
            cls._self_inside_idx = None
            cls._self_toggle_idx = None
            cls._self_open_idx = None
            cls._temp_to_aabb_idx = None
            cls._temp_to_inside_idx = None
            cls._affected_mask = None
            cls._affected_mask_cpu = None
            cls._affected_mask_cpu_wp = None
            return

        temperatures = th.zeros(N, dtype=th.float32)
        heating_rates = th.zeros(N, dtype=th.float32)
        distance_thresholds = th.zeros(N, dtype=th.float32)
        requires_toggled_on = th.zeros(N, dtype=th.uint8)
        requires_closed = th.zeros(N, dtype=th.uint8)
        requires_inside = th.zeros(N, dtype=th.uint8)
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
            if not hss_state_instance.requires_inside and hss_state_instance._links:
                # store information of this hss' link position so later we can compute distance of item with hss
                link_flat_idx[hss_obj_idx] = RigidBodyViewAPI.get_flat_idx(hss_state_instance.link.prim_path)

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list
        cls._temperatures = create_tensor_from_list(temperatures, "float32", device="cuda")
        cls._heating_rates = create_tensor_from_list(heating_rates, "float32", device="cuda")
        cls._distance_thresholds = create_tensor_from_list(distance_thresholds, "float32", device="cuda")
        cls._requires_toggled_on = create_tensor_from_list(requires_toggled_on, "uint8", device="cuda")
        cls._requires_closed = create_tensor_from_list(requires_closed, "uint8", device="cuda")
        cls._requires_inside = create_tensor_from_list(requires_inside, "uint8", device="cuda")
        cls._link_flat_idx = create_tensor_from_list(link_flat_idx, "int32", device="cuda")
        # vec3 has no scalar-only helper — wp.array reinterprets the (N, 3) float32 CPU buffer as (N,) vec3.
        cls._link_local_offset = wp.array(link_local_offset, dtype=wp.vec3, device="cuda")

        # Init of cross-state index maps + `_affected_mask` are deferred to pre_update because
        # Temperature / Inside / ToggledOn / Open initialize_views may not have been called yet
        # in the order Temperature depends on HSS.
        cls._self_temp_idx = None
        cls._self_inside_idx = None
        cls._self_toggle_idx = None
        cls._self_open_idx = None
        cls._temp_to_aabb_idx = None
        cls._temp_to_inside_idx = None
        cls._affected_mask = None
        cls._affected_mask_cpu = None
        cls._affected_mask_cpu_wp = None

    @classmethod
    def _rebuild_cross_state_maps(cls):
        """
        Build (or rebuild) the index tables that reference *other* states' OBJ_IDXS, and
        allocate `_affected_mask` with the current number of object with temperature state.
        Called from pre_update when needed.
        """
        # Local import to avoid circular dependency
        from omnigibson.object_states.temperature import Temperature

        N_hss = len(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else 0
        S_hss = len(cls.IDX_OBJS) if cls.IDX_OBJS is not None else 0
        N_temp = len(Temperature.OBJ_IDXS) if Temperature.OBJ_IDXS is not None else 0

        if N_hss == 0:
            return

        # lookup table into other states
        # hss' index in temp's obj_idx, to prevent influence self's temperature
        self_temp_idx = th.full((N_hss,), -1, dtype=th.int32)
        # hss' index in inside's obj_idx, to help find out whether an item is inside hss
        self_inside_idx = th.full((N_hss,), -1, dtype=th.int32)
        self_toggle_idx = th.full((N_hss,), -1, dtype=th.int32)
        self_open_idx = th.full((N_hss,), -1, dtype=th.int32)

        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            self_temp_idx[obj_idx] = Temperature.OBJ_IDXS.get(rel_path, -1)
            self_inside_idx[obj_idx] = Inside.OBJ_IDXS.get(rel_path, -1)
            self_toggle_idx[obj_idx] = ToggledOn.OBJ_IDXS.get(rel_path, -1)
            self_open_idx[obj_idx] = Open.OBJ_IDXS.get(rel_path, -1)

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list
        cls._self_temp_idx = create_tensor_from_list(self_temp_idx, "int32", device="cuda")
        cls._self_inside_idx = create_tensor_from_list(self_inside_idx, "int32", device="cuda")
        cls._self_toggle_idx = create_tensor_from_list(self_toggle_idx, "int32", device="cuda")
        cls._self_open_idx = create_tensor_from_list(self_open_idx, "int32", device="cuda")

        # Build Temperature to AABB / Inside look-up table
        temp_to_aabb = th.full((N_temp,), -1, dtype=th.int32)
        temp_to_inside = th.full((N_temp,), -1, dtype=th.int32)
        for rel_path, temperature_idx in Temperature.OBJ_IDXS.items():
            temp_to_aabb[temperature_idx] = AABB.OBJ_IDXS.get(rel_path, -1)
            temp_to_inside[temperature_idx] = Inside.OBJ_IDXS.get(rel_path, -1)
        if N_temp > 0:
            cls._temp_to_aabb_idx = create_tensor_from_list(temp_to_aabb, "int32", device="cuda")
            cls._temp_to_inside_idx = create_tensor_from_list(temp_to_inside, "int32", device="cuda")
        else:
            cls._temp_to_aabb_idx = None
            cls._temp_to_inside_idx = None

        # (Re)allocate `_affected_mask` if the shape changed
        if S_hss > 0 and N_hss > 0 and N_temp > 0:
            cls._affected_mask = wp.zeros((S_hss, N_hss, N_temp), dtype=wp.uint8, device="cuda")
            cls._affected_mask_cpu = th.zeros((S_hss, N_hss, N_temp), dtype=th.bool).pin_memory()
            cls._affected_mask_cpu_wp = _wp_from_torch(cls._affected_mask_cpu)
        else:
            cls._affected_mask = None
            cls._affected_mask_cpu = None
            cls._affected_mask_cpu_wp = None

    @classmethod
    def pre_update(cls):
        super().pre_update()
        # Make sure that only rebuild when graph is dirty
        if TensorizedState.graph_dirty:
            cls._rebuild_cross_state_maps()

        # Zero _affected_mask every step so the propagate kernel only OR-writes hits.
        if cls._affected_mask is not None:
            cls._affected_mask.zero_()

    @classmethod
    def _update_values(cls, values):
        # Local imports to keep module-load order safe
        from omnigibson.object_states.temperature import Temperature

        if cls.VALUES_WP is None or cls._self_temp_idx is None:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return

        toggle_values_wp = ToggledOn.VALUES_WP
        open_values_wp = Open.VALUES_WP
        # It's possible that an optional state has no tracked objects,
        # in that case we swap in a cached placeholder wp.array to satisfy
        # kernel's signature.
        if toggle_values_wp is None:
            toggle_values_wp = cls._placeholder_values
            has_toggle = 0
        else:
            has_toggle = 1
        if open_values_wp is None:
            open_values_wp = cls._placeholder_values
            has_open = 0
        else:
            has_open = 1

        # check whether HSS is active and update VALUES
        wp.launch(
            kernel=_heatsource_is_active_kernel,
            dim=(S, N),
            inputs=[
                cls._requires_toggled_on,
                cls._requires_closed,
                cls._self_toggle_idx,
                cls._self_open_idx,
                toggle_values_wp,
                open_values_wp,
                wp.int32(has_toggle),
                wp.int32(has_open),
                cls.VALUES_WP,
            ],
            device="cuda",
        )

        # check what TEMPERATURE obj will be influenced by hss
        # write `_affected_mask` + INCOMING_HEAT_RATE
        if (
            cls._affected_mask is None
            or Temperature.VALUES_WP is None
            or Temperature.INCOMING_HEAT_RATE_WP is None
            or cls._temp_to_aabb_idx is None
            or AABB.VALUES_WP is None
        ):
            return
        N_temp = Temperature.VALUES.shape[1]
        if N_temp == 0:
            return

        inside_values_wp = Inside.VALUES_WP
        if inside_values_wp is None:
            inside_values_wp = cls._placeholder_inside
            has_inside = 0
        else:
            has_inside = 1

        wp.launch(
            kernel=_heatsource_can_influence_kernel,
            dim=(S, N, N_temp),
            inputs=[
                cls.VALUES_WP,
                cls._requires_inside,
                cls._temperatures,
                cls._heating_rates,
                cls._distance_thresholds,
                cls._self_temp_idx,
                cls._self_inside_idx,
                cls._link_flat_idx,
                cls._link_local_offset,
                cls._temp_to_aabb_idx,
                cls._temp_to_inside_idx,
                RigidBodyViewAPI.POSE_MATRICES,
                AABB.VALUES_WP,
                inside_values_wp,
                Temperature.VALUES_WP,
                wp.int32(has_inside),
                cls._affected_mask,
                Temperature.INCOMING_HEAT_RATE_WP,
            ],
            device="cuda",
        )

        # Mirror `_affected_mask` → `_affected_mask_cpu` for affects_obj() CPU reads.
        if cls._affected_mask_cpu_wp is not None:
            wp.copy(cls._affected_mask_cpu_wp, cls._affected_mask)

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
        if cls._affected_mask_cpu is None:
            return False
        h = cls.OBJ_IDXS[self.obj.relative_prim_path]
        n = Temperature.OBJ_IDXS[obj.relative_prim_path]
        s = self.obj.scene.idx
        return bool(cls._affected_mask_cpu[s, h, n].item())

    # Nothing needs to be done to save/load HeatSource
