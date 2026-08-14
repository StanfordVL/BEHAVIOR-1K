import torch as th
import warp as wp

import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.object_states.tensorized_state import TensorizedState, _wp_from_torch
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


@wp.kernel
def _incoming_heat_kernel(
    source_values: wp.array2d(dtype=wp.uint8),  # HeatSourceOrSink.VALUES (S_hss, N_hss)
    requires_inside: wp.array(dtype=wp.uint8),  # (N_hss,)
    source_temperatures: wp.array(dtype=wp.float32),  # (N_hss,)
    heating_rates: wp.array(dtype=wp.float32),  # (N_hss,)
    distance_thresholds: wp.array(dtype=wp.float32),  # (N_hss,)
    hss_self_temp_idx: wp.array(dtype=wp.int32),  # (N_hss,) into Temperature N — to skip self
    hss_self_inside_idx: wp.array(dtype=wp.int32),  # (N_hss,) into Inside N (for requires_inside)
    link_flat_idx: wp.array2d(dtype=wp.int32),  # (S_hss, N_hss) into POSE_MATRICES — -1 if N/A
    link_local_offset: wp.array(dtype=wp.vec3),  # (N_hss,) heat element offset in link local frame
    temp_to_aabb_idx: wp.array(dtype=wp.int32),  # (N_temp,) Temperature N → AABB N
    temp_to_inside_idx: wp.array(dtype=wp.int32),  # (N_temp,) Temperature N → Inside N
    pose_matrices: wp.array(dtype=wp.mat44),  # RigidBodyViewAPI.POSE_MATRICES
    aabb_values: wp.array3d(dtype=wp.float32),  # AABB (S, N_aabb, 6)
    inside_values: wp.array3d(dtype=wp.uint8),  # Inside (S, N_inside, N_inside) — uint8 view of bool
    temperature_values: wp.array2d(dtype=wp.float32),  # (S_temp, N_temp)
    n_inside_scenes: wp.int32,  # scene rows in inside_values (0 if the state tracks nothing)
    influence_mask: wp.array3d(dtype=wp.uint8),  # (S_hss, N_hss, N_temp) — out
    incoming_heat_rate: wp.array2d(dtype=wp.float32),  # (S_temp, N_temp) — out
):
    """
    Per (scene, source, target) thread: if heat source / sink h is active and target n is in
    range, write influence_mask[s, h, n] = 1 and atomic_add (T_h - T_n) * rate into
    incoming_heat_rate[s, n]. Launched over s < min(S_hss, S_temp).
    """
    s, h, n = wp.tid()
    if source_values[s, h] == wp.uint8(0):  # early return if source not active
        return
    if hss_self_temp_idx[h] == n:  # do not influence self
        return
    target_aabb_idx = temp_to_aabb_idx[n]
    if target_aabb_idx < wp.int32(0):
        return

    # containment sources (e.g. ovens): target must be inside the source
    if requires_inside[h] != wp.uint8(0):
        if s >= n_inside_scenes:
            return
        src_idx_in_inside = hss_self_inside_idx[h]
        target_idx_in_inside = temp_to_inside_idx[n]
        if src_idx_in_inside < wp.int32(0) or target_idx_in_inside < wp.int32(0):
            return
        if inside_values[s, target_idx_in_inside, src_idx_in_inside] == wp.uint8(0):
            return
    # point sources: the target's AABB must be within the distance threshold of the heat element.
    # The element link is a different rigid body in every scene, so its pose index is per (s, h).
    else:
        li = link_flat_idx[s, h]
        if li < wp.int32(0):
            return
        link_pose = pose_matrices[li]
        local_offset = link_local_offset[h]
        source_world_position = wp.mul(
            link_pose, wp.vec4(local_offset[0], local_offset[1], local_offset[2], wp.float32(1.0))
        )
        # Distance from the heat element to the CLOSEST POINT of the target's AABB (element
        # position clamped into the box).
        #
        # This is an APPROXIMATION of the pre-vectorization behaviour, which ran a PhysX
        # overlap_sphere(threshold) scene query against the target's actual COLLISION GEOMETRY
        # (heat_source_or_sink.py on main). Since a target's AABB always contains its geometry,
        # distance-to-AABB <= distance-to-geometry, so this test is slightly MORE permissive than
        # overlap_sphere — most visibly for long, thin or diagonally-oriented targets whose AABB
        # is much larger than the mesh itself.
        #
        # It is nonetheless much closer than what it replaces: testing the AABB *center* shrinks
        # every source's reach by the target's own extent (e.g. a sausage in a pan on a burner
        # grazes the sphere, but its center is beyond the threshold, so it would never cook).
        # An exact port would need per-(source, target) mesh queries inside this kernel, which is
        # not available to Warp here; see the PR discussion for the trade-off.
        qx = wp.clamp(source_world_position[0], aabb_values[s, target_aabb_idx, 0], aabb_values[s, target_aabb_idx, 3])
        qy = wp.clamp(source_world_position[1], aabb_values[s, target_aabb_idx, 1], aabb_values[s, target_aabb_idx, 4])
        qz = wp.clamp(source_world_position[2], aabb_values[s, target_aabb_idx, 2], aabb_values[s, target_aabb_idx, 5])
        dx = source_world_position[0] - qx
        dy = source_world_position[1] - qy
        dz = source_world_position[2] - qz
        d2 = dx * dx + dy * dy + dz * dz
        threshold = distance_thresholds[h]
        if d2 > threshold * threshold:
            return

    influence_mask[s, h, n] = wp.uint8(1)
    delta = (source_temperatures[h] - temperature_values[s, n]) * heating_rates[h]
    wp.atomic_add(incoming_heat_rate, s, n, delta)


@wp.kernel
def _temperature_decay_kernel(
    values: wp.array2d(dtype=wp.float32),
    incoming_heat_rate: wp.array2d(dtype=wp.float32),
    default_temp: wp.float32,
    decay_rate: wp.float32,
    dt: wp.array(dtype=wp.float32),
):
    """
    Fused exponential decay toward `default_temp` plus consumption of the per-step incoming heat
    rate scratch buffer (written by _incoming_heat_kernel earlier in the same update pass).
    One thread per (scene, obj).
        values += (default_temp - values) * decay_rate * dt[0] + incoming_heat_rate * dt[0]
    Then zero the consumed entry so the scratch is fresh for the next step's gather.

    `dt` is a single-element warp array (read as `dt[0]`) rather than a scalar so the
    per-frame value written in pre_update is visible inside the captured graph without
    re-capturing it (time-dependent-state mechanism).
    """
    s, o = wp.tid()
    values[s, o] = values[s, o] + (default_temp - values[s, o]) * decay_rate * dt[0] + incoming_heat_rate[s, o] * dt[0]
    incoming_heat_rate[s, o] = wp.float32(0.0)


@wp.kernel
def _self_heating_clamp_kernel(
    source_values: wp.array2d(dtype=wp.uint8),  # HeatSourceOrSink.VALUES (S_hss, N_hss)
    requires_on_fire: wp.array(dtype=wp.uint8),  # (N_hss,)
    source_temperatures: wp.array(dtype=wp.float32),  # (N_hss,)
    ignition_temperatures: wp.array(dtype=wp.float32),  # (N_hss,)
    hss_self_temp_idx: wp.array(dtype=wp.int32),  # (N_hss,) into Temperature N
    temperature_values: wp.array2d(dtype=wp.float32),  # (S_temp, N_temp) — out
):
    """
    Fire sources sustain their own temperature: an active requires_on_fire source holds its
    object at the source (fire) temperature. Gated on the temperature still being at or above
    ignition, so deliberately cooling the object below ignition (extinguishing) is not
    overridden — the fire then dies out on the next OnFire pass instead. Launched over
    s < min(S_hss, S_temp).
    """
    s, h = wp.tid()
    if requires_on_fire[h] == wp.uint8(0):
        return
    if source_values[s, h] == wp.uint8(0):
        return
    n = hss_self_temp_idx[h]
    if n < wp.int32(0):
        return
    t = temperature_values[s, n]
    if t < ignition_temperatures[h]:
        return
    if source_temperatures[h] > t:
        temperature_values[s, n] = source_temperatures[h]


class Temperature(TensorizedAbsoluteState):
    """
    Continuous per-object temperature (°C).

    Temperature owns every write into its own tensors. Each step, `_update_values` launches,
    in order:
      1. `_incoming_heat_kernel` — gathers heat from active HeatSourceOrSink entries (whose
         activation gates were computed earlier this step; HeatSourceOrSink is a dependency)
         into the private INCOMING_HEAT_RATE scratch, recording who-heats-whom in
         INFLUENCE_MASK (served to HeatSourceOrSink.affects_obj via is_influenced_by()).
      2. `_temperature_decay_kernel` — integrates ambient decay plus the gathered rate, then
         zeroes the scratch for the next step.
      3. `_self_heating_clamp_kernel` — objects on fire (heat sources with requires_on_fire)
         are held at their fire temperature.

    Note the deliberate one-step lag in the fire feedback loop: OnFire (which depends on
    Temperature) flips True the step temperature crosses ignition; the object's
    HeatSourceOrSink gate reads OnFire's previous-step values, so the fire starts heating
    (including the self-clamp) on the following step.
    """

    # (S, N) float32 — private per-step rate accumulator; written by _incoming_heat_kernel and
    # consumed + zeroed by _temperature_decay_kernel within the same _update_values pass.
    INCOMING_HEAT_RATE = None  # wp.array (S, N) float32 — GPU-only scratch (single source of truth)

    # (S_hss, N_hss, N_temp) — which heat source influenced which object this step. The GPU
    # uint8 wp.array is the single source of truth; the CPU mirror keeps a torch bool tensor
    # (for .item() reads in is_influenced_by) plus a wp view for the graph-safe wp.copy,
    # mirroring how the base class keeps VALUES_CPU + VALUES_CPU_WP.
    INFLUENCE_MASK = None  # wp.array (S_hss, N_hss, N_temp) uint8 — GPU
    INFLUENCE_MASK_CPU = None  # torch bool (S_hss, N_hss, N_temp), pinned — CPU mirror
    INFLUENCE_MASK_CPU_WP = None  # wp.array uint8 view of INFLUENCE_MASK_CPU

    # Index maps into other states' N dimensions. Built in initialize_view — safe because all
    # referenced states (HeatSourceOrSink, AABB, Inside) initialize before Temperature in
    # dependency order.
    _hss_self_temp_idx = None  # wp.array (N_hss,) int32 — HeatSourceOrSink N → Temperature N
    _hss_self_inside_idx = None  # wp.array (N_hss,) int32 — HeatSourceOrSink N → Inside N
    _temp_to_aabb_idx = None  # wp.array (N_temp,) int32 — Temperature N → AABB N
    _temp_to_inside_idx = None  # wp.array (N_temp,) int32 — Temperature N → Inside N

    # Placeholder wp.array to satisfy the kernel signature when Inside tracks no objects.
    _placeholder_inside = None  # wp.array (1, 1, 1) uint8

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        # Optional because objects without HeatSourceOrSink (e.g. cookable food items) are still
        # eligible for a Temperature state — but the topo sort still places HeatSourceOrSink
        # before Temperature, so _incoming_heat_kernel reads this step's freshly-computed
        # activation gates.
        deps.add(HeatSourceOrSink)
        return deps

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls.INCOMING_HEAT_RATE = None
        cls.INFLUENCE_MASK = None
        cls.INFLUENCE_MASK_CPU = None
        cls.INFLUENCE_MASK_CPU_WP = None
        cls._hss_self_temp_idx = None
        cls._hss_self_inside_idx = None
        cls._temp_to_aabb_idx = None
        cls._temp_to_inside_idx = None
        cls._placeholder_inside = wp.zeros((1, 1, 1), dtype=wp.uint8, device="cuda")

    @classmethod
    def initialize_view(cls):
        # Snapshot which relative paths existed before the rebuild
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        # Initialize new VALUE slots (not carried over) to DEFAULT_TEMPERATURE
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(len(cls.IDX_OBJS)):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = m.DEFAULT_TEMPERATURE
                        cls.VALUES_CPU[s_idx, obj_idx] = m.DEFAULT_TEMPERATURE

        # Allocate the per-step heat-rate scratch buffer. No carry-over: a partial step from
        # the previous configuration would be applied to the wrong indices.
        if cls.VALUES.numel() > 0:
            cls.INCOMING_HEAT_RATE = wp.zeros(tuple(cls.VALUES.shape), dtype=wp.float32, device="cuda")
        else:
            cls.INCOMING_HEAT_RATE = None

        # Rebuild the maps into the states the heat kernels read.
        cls._rebuild_heat_source_maps()

    @classmethod
    def _rebuild_heat_source_maps(cls):
        """
        Rebuild the index maps into HeatSourceOrSink / AABB / Inside plus INFLUENCE_MASK.
        Called from initialize_view — safe because those states are (transitive) dependencies
        of Temperature, so their views are rebuilt before this one.
        """
        N_temp = len(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else 0
        hss_obj_idxs = HeatSourceOrSink.OBJ_IDXS or {}
        N_hss = len(hss_obj_idxs)
        S_hss = len(HeatSourceOrSink.IDX_OBJS) if HeatSourceOrSink.IDX_OBJS is not None else 0
        inside_map = Inside.OBJ_IDXS or {}
        aabb_map = AABB.OBJ_IDXS or {}

        if N_temp == 0 or N_hss == 0 or S_hss == 0:
            cls._hss_self_temp_idx = None
            cls._hss_self_inside_idx = None
            cls._temp_to_aabb_idx = None
            cls._temp_to_inside_idx = None
            cls.INFLUENCE_MASK = None
            cls.INFLUENCE_MASK_CPU = None
            cls.INFLUENCE_MASK_CPU_WP = None
            return

        create_tensor_from_list = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list

        hss_self_temp_idx = th.full((N_hss,), -1, dtype=th.int32)
        hss_self_inside_idx = th.full((N_hss,), -1, dtype=th.int32)
        for rel_path, h in hss_obj_idxs.items():
            hss_self_temp_idx[h] = cls.OBJ_IDXS.get(rel_path, -1)
            hss_self_inside_idx[h] = inside_map.get(rel_path, -1)
        cls._hss_self_temp_idx = create_tensor_from_list(hss_self_temp_idx, "int32", device="cuda")
        cls._hss_self_inside_idx = create_tensor_from_list(hss_self_inside_idx, "int32", device="cuda")

        temp_to_aabb = th.full((N_temp,), -1, dtype=th.int32)
        temp_to_inside = th.full((N_temp,), -1, dtype=th.int32)
        for rel_path, n in cls.OBJ_IDXS.items():
            temp_to_aabb[n] = aabb_map.get(rel_path, -1)
            temp_to_inside[n] = inside_map.get(rel_path, -1)
        cls._temp_to_aabb_idx = create_tensor_from_list(temp_to_aabb, "int32", device="cuda")
        cls._temp_to_inside_idx = create_tensor_from_list(temp_to_inside, "int32", device="cuda")

        cls.INFLUENCE_MASK = wp.zeros((S_hss, N_hss, N_temp), dtype=wp.uint8, device="cuda")
        cls.INFLUENCE_MASK_CPU = th.zeros((S_hss, N_hss, N_temp), dtype=th.bool).pin_memory()
        cls.INFLUENCE_MASK_CPU_WP = _wp_from_torch(cls.INFLUENCE_MASK_CPU)

    @classmethod
    def pre_update(cls, dt=0.0):
        super().pre_update(dt)
        # Zero the influence mask every step so _incoming_heat_kernel only OR-writes hits.
        if cls.INFLUENCE_MASK is not None:
            cls.INFLUENCE_MASK.zero_()

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None or cls.INCOMING_HEAT_RATE is None:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return

        hss = HeatSourceOrSink

        # 1) Gather incoming heat from active heat sources / sinks into our scratch + mask.
        if (
            cls.INFLUENCE_MASK is not None
            and cls._hss_self_temp_idx is not None
            and hss.VALUES_WP is not None
            and AABB.VALUES_WP is not None
        ):
            S_hss, N_hss = hss.VALUES.shape[:2]
            # Scenes beyond either state's row count hold no (source, target) pairs.
            S_common = min(S, S_hss)
            inside_values_wp = Inside.VALUES_WP
            n_inside_scenes = Inside.VALUES.shape[0] if inside_values_wp is not None else 0
            if inside_values_wp is None:
                inside_values_wp = cls._placeholder_inside
            if S_common > 0 and N_hss > 0:
                wp.launch(
                    kernel=_incoming_heat_kernel,
                    dim=(S_common, N_hss, N),
                    inputs=[
                        hss.VALUES_WP,
                        hss._requires_inside,
                        hss._temperatures,
                        hss._heating_rates,
                        hss._distance_thresholds,
                        cls._hss_self_temp_idx,
                        cls._hss_self_inside_idx,
                        hss._link_flat_idx,
                        hss._link_local_offset,
                        cls._temp_to_aabb_idx,
                        cls._temp_to_inside_idx,
                        RigidBodyViewAPI.POSE_MATRICES,
                        AABB.VALUES_WP,
                        inside_values_wp,
                        cls.VALUES_WP,
                        wp.int32(n_inside_scenes),
                        cls.INFLUENCE_MASK,
                        cls.INCOMING_HEAT_RATE,
                    ],
                    device="cuda",
                )
                # Mirror the mask for CPU reads (HeatSourceOrSink.affects_obj).
                if cls.INFLUENCE_MASK_CPU_WP is not None:
                    wp.copy(cls.INFLUENCE_MASK_CPU_WP, cls.INFLUENCE_MASK)

        # 2) Decay toward ambient + consume the gathered heat rate (also zeroes the scratch).
        #    dt is read from cls._dt at kernel-launch time inside the captured graph, so the
        #    per-frame value written in pre_update is visible without re-capturing the graph.
        wp.launch(
            kernel=_temperature_decay_kernel,
            dim=(S, N),
            inputs=[
                cls.VALUES_WP,
                cls.INCOMING_HEAT_RATE,
                wp.float32(m.DEFAULT_TEMPERATURE),
                wp.float32(m.TEMPERATURE_DECAY_SPEED),
                cls._dt,
            ],
            device="cuda",
        )

        # 3) Hold burning objects at their fire temperature (see kernel docstring for the
        #    ignition-threshold gate that lets deliberate cooling extinguish them).
        if cls._hss_self_temp_idx is not None and hss.VALUES_WP is not None:
            S_hss, N_hss = hss.VALUES.shape[:2]
            S_common = min(S, S_hss)
            if S_common > 0 and N_hss > 0:
                wp.launch(
                    kernel=_self_heating_clamp_kernel,
                    dim=(S_common, N_hss),
                    inputs=[
                        hss.VALUES_WP,
                        hss._requires_on_fire,
                        hss._temperatures,
                        hss._ignition_temperatures,
                        cls._hss_self_temp_idx,
                        cls.VALUES_WP,
                    ],
                    device="cuda",
                )

    @classmethod
    def is_influenced_by(cls, source_obj, target_obj):
        """
        Whether @source_obj's heat source / sink contributed heat to @target_obj's temperature
        on the most recent update pass.

        Args:
            source_obj (StatefulObject): Object with the HeatSourceOrSink state.
            target_obj (StatefulObject): Object with the Temperature state.

        Returns:
            bool
        """
        # Lazy refresh so the read sees this-step's state.
        TensorizedState.maybe_refresh_caches()
        if cls.INFLUENCE_MASK_CPU is None:
            return False
        if HeatSourceOrSink.OBJ_IDXS is None or source_obj.relative_prim_path not in HeatSourceOrSink.OBJ_IDXS:
            return False
        if cls.OBJ_IDXS is None or target_obj.relative_prim_path not in cls.OBJ_IDXS:
            return False
        s = source_obj.scene.idx
        if s >= cls.INFLUENCE_MASK_CPU.shape[0]:
            return False
        h = HeatSourceOrSink.OBJ_IDXS[source_obj.relative_prim_path]
        n = cls.OBJ_IDXS[target_obj.relative_prim_path]
        return bool(cls.INFLUENCE_MASK_CPU[s, h, n].item())

    @classproperty
    def value_name(cls):
        return "temperature"
