"""
ParticleViewAPI read the world position of every particle in every scene.
Particle positions are stored in one flat buffer on gpu.

An entry = one particle system in one scene, i.e. a (scene_index, system_name) pair. The same system,
say "water", running in 2 scenes counts as 2 separate entries. Particles are grouped by entry, and the
flat buffer has every entry's particles positions laid end to end:

    [ entry 0's particles ][ entry 1's particles ][ entry 2's particles ] ...
    ^ entry_start[0]        ^ entry_start[1]        ^ entry_start[2]

So each particle carries (a) its scene index and (b) its entry index (which scene+system it belongs
to), and entry_start[e] is where entry e's block begins in the flat buffer.

The flat output buffers: allocated once and reused; only the first PARTICLE_COUNT entries are valid
    PARTICLE_POSITIONS          (capacity,) vec3f  — every particle's world position
    PARTICLE_SCENE_INDEX        (capacity,) int32  — which scene each particle is in
    PARTICLE_ENTRY_INDEX        (capacity,) int32  — which entry (scene+system) each particle belongs to
    VISUAL_PARTICLE_ORIENTATION (capacity,) quat   — world orientation, written only for macro-visual particles
    PARTICLE_COUNT       (1,) int32 (GPU)   — how many particles are valid this step
The buffers only grow (never shrink), and only when the particle count exceeds the current capacity.

Particle systems come in three kinds, each stored very differently in the simulator, so each has its
own read that fills its slice of PARTICLE_POSITIONS:

  - micro-physical (water, rice): thousands of tiny particles, held per system as a single "point
    instancer" whose positions live Fabric. One SelectPrims
    call hands us every instancer's positions at once; then one 2-D kernel copies them all into the
    flat buffer (sized by the per-instancer particle counts).
  - macro-physical (diced apple, ...): each particle is a small rigid body. We build a rigid-body
    view over all such particles across all scenes, call get_transforms() once to read all their
    poses (M = the total number of these particles), then one kernel writes each into the flat buffer.
  - macro-visual (stains): each particle is glued onto a link (a body part) of some object. One kernel
    computes each particle's world position from its parent link's live pose plus a fixed local offset
    (K = the total number of such particles the kernel handles). The rare particles glued onto CLOTH
    can't use this and fall back to their own system's slower, one-system-at-a-time getter.

Each step refreshes positions in TWO explicit phases:
    prepare_step_host()        — OUT-OF-GRAPH phase. Processes pending metadata mutations, performs
                                backend calls that cannot be captured (Fabric SelectPrims / GetPaths,
                                PhysX get_transforms, cloth getters), and fills fixed host staging
                                buffers. Fabric's transient source is also scattered here because its
                                wrapper is not stable across graph replays.
    update_positions_gpu()     — CAPTURED phase. Copies the macro-physical pinned staging buffer to
                                CUDA, scatters macro-physical positions, and updates attached
                                macro-visual positions. It must run after RigidBodyViewAPI.update() so
                                visual particles read this step's parent-link matrices.

Important functions:
    rebuild_topology()          — rebuilds the complete registry, layout, and family metadata. It
                                returns with a usable view; the first position read does not finish
                                initialization lazily.
    initialize_view()           — compatibility alias for rebuild_topology().
    get_particle_positions(scene, system) — convenience: the slice of PARTICLE_POSITIONS for one entry.

Compatibility aliases read_particle_positions() and update() retain the old public API.
"""

import torch as th
import warp as wp

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.systems import MicroPhysicalParticleSystem
from omnigibson.systems.system_base import PhysicalParticleSystem, VisualParticleSystem
from omnigibson.utils.constants import PrimType
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log
from omnigibson.utils.usd_utils import RigidBodyViewAPI

log = create_module_logger(module_name=__name__)


FAMILY_MACRO_PHYSICAL = "macro_physical"
FAMILY_MACRO_VISUAL = "macro_visual"
FAMILY_MICRO_PHYSICAL = "micro_physical"

# Minimum capacity to allocate when growing the flat buffers (avoids frequent regrowth for small counts).
_MIN_BUFFER_CAPACITY = 1024
_MIN_VISUAL_METADATA_CAPACITY = 64


def _classify_family(system):
    """Return the family tag for a particle system, or None if unsupported (e.g. Cloth)."""
    if isinstance(system, MicroPhysicalParticleSystem):
        return FAMILY_MICRO_PHYSICAL
    if isinstance(system, PhysicalParticleSystem):
        return FAMILY_MACRO_PHYSICAL
    if isinstance(system, VisualParticleSystem):
        return FAMILY_MACRO_VISUAL
    return None


@wp.kernel
def _scatter_instancer_positions_kernel(
    instancer_positions: wp.fabricarrayarray(dtype=wp.vec3f),  # one Fabric entry per prim that has a "positions" attr
    instancer_destination_start: wp.array(
        dtype=wp.int32
    ),  # (num_instancers,) flat-buffer start per instancer, -1 = skip
    positions_out: wp.array(dtype=wp.vec3f),
):
    """Copy every micro instancer's Fabric positions into the flat buffer in a single 2-D launch:
    thread (instancer, particle) writes that instancer's particle to its destination slot. Instancers
    whose start is -1 (prims that aren't tracked micro instancers) are skipped."""
    instancer, particle = wp.tid()
    destination_start = instancer_destination_start[instancer]
    if destination_start < 0:
        return
    this_instancer = instancer_positions[instancer]
    if particle < this_instancer.shape[0]:
        positions_out[destination_start + particle] = this_instancer[particle]


@wp.kernel
def _get_macro_visual_positions_kernel(
    pose_matrices: wp.array(dtype=wp.mat44),  # RigidBodyViewAPI.POSE_MATRICES (rigid world, per link)
    parent_link_index: wp.array(dtype=wp.int32),  # (K,) per particle -> flat link idx (-1 if untracked)
    world_scale: wp.array(dtype=wp.vec3),  # (K,) per particle world-accumulated scale
    local_matrix: wp.array(dtype=wp.mat44),  # (K,) per particle static local pose
    particle_entry_index: wp.array(dtype=wp.int32),  # (K,) entry index into _entries order
    particle_local_index: wp.array(dtype=wp.int32),  # (K,) slot within the owning entry
    entry_start: wp.array(dtype=wp.int32),  # (num_entries,) start offset of each entry
    particle_count: wp.array(dtype=wp.int32),  # (1,) live number of valid rows in the metadata arrays
    positions_out: wp.array(dtype=wp.vec3f),
    orientations_out: wp.array(dtype=wp.quat),  # world orientation, written for these visual particles
):
    """Compute each attached visual particle's world position as
    POSE_MATRICES[parent] @ diag(world_scale) @ local_matrix and take the translation. Also write the
    world orientation (rotation part, columns normalized to strip scale) so downstream containment can
    offset the check point. Scatter both into the entry's slice at entry_start[entry] + local, so one
    launch covers every kernel-eligible particle."""
    particle = wp.tid()
    if particle >= particle_count[0]:
        return
    link = parent_link_index[particle]
    if link < 0:
        return
    s = world_scale[particle]
    scale_matrix = wp.mat44(
        s[0],
        0.0,
        0.0,
        0.0,
        0.0,
        s[1],
        0.0,
        0.0,
        0.0,
        0.0,
        s[2],
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    world = wp.mul(wp.mul(pose_matrices[link], scale_matrix), local_matrix[particle])
    destination = entry_start[particle_entry_index[particle]] + particle_local_index[particle]
    positions_out[destination] = wp.vec3f(world[0, 3], world[1, 3], world[2, 3])

    # Pure rotation from world's 3x3 via Gram-Schmidt on the columns — matches the getter's
    # decompose_mat exactly (strips both scale and shear from non-uniform scaling).
    col0 = wp.vec3(world[0, 0], world[1, 0], world[2, 0])
    col1 = wp.vec3(world[0, 1], world[1, 1], world[2, 1])
    col2 = wp.vec3(world[0, 2], world[1, 2], world[2, 2])
    e0 = wp.normalize(col0)
    e1 = wp.normalize(col1 - wp.dot(e0, col1) * e0)
    v2 = col2 - wp.dot(e0, col2) * e0
    v2 = v2 - wp.dot(e1, v2) * e1
    e2 = wp.normalize(v2)
    if wp.dot(e0, wp.cross(e1, e2)) < 0.0:  # negative determinant -> flip to a proper rotation
        e0 = -e0
        e1 = -e1
        e2 = -e2
    rotation = wp.mat33(
        e0[0],
        e1[0],
        e2[0],
        e0[1],
        e1[1],
        e2[1],
        e0[2],
        e1[2],
        e2[2],
    )
    orientations_out[destination] = wp.quat_from_matrix(rotation)


@wp.kernel
def _scatter_macro_physical_positions_kernel(
    transforms: wp.array2d(
        dtype=wp.float32
    ),  # (M,7) rows from the cross-scene rigid-body view: pos(0:3) + quat xyzw(3:7)
    particle_entry_index: wp.array(dtype=wp.int32),  # (M,) entry index into _entries order (-1 to skip)
    particle_local_index: wp.array(dtype=wp.int32),  # (M,) slot within the owning entry (system getter row order)
    particle_offset: wp.array(dtype=wp.vec3),  # (M,) owning system's _particle_offset
    entry_start: wp.array(dtype=wp.int32),  # (num_entries,) start offset of each entry
    positions_out: wp.array(dtype=wp.vec3f),
):
    """Replicate each system's getter (pos + quat2mat(ori) @ _particle_offset) and scatter each
    particle into its entry's slice: positions_out[entry_start[entry] + local] = world_pos."""
    particle = wp.tid()
    entry = particle_entry_index[particle]
    if entry < 0:
        return
    position = wp.vec3(transforms[particle, 0], transforms[particle, 1], transforms[particle, 2])
    orientation = wp.quat(
        transforms[particle, 3], transforms[particle, 4], transforms[particle, 5], transforms[particle, 6]
    )
    world = position + wp.quat_rotate(orientation, particle_offset[particle])
    positions_out[entry_start[entry] + particle_local_index[particle]] = world


@wp.kernel
def _fill_particle_labels_kernel(
    entry_start: wp.array(dtype=wp.int32),  # (num_entries,) start offset of each entry
    entry_scene_index: wp.array(dtype=wp.int32),  # (num_entries,) scene index of each entry
    num_entries: int,
    num_particles: int,
    scene_index_out: wp.array(dtype=wp.int32),
    entry_index_out: wp.array(dtype=wp.int32),
):
    """Per particle: find its entry (largest entry whose start <= particle) and write its
    scene index + entry index. num_entries is small so the linear scan is cheap."""
    particle = wp.tid()
    if particle >= num_particles:
        return
    entry = int(0)
    for k in range(num_entries):
        if entry_start[k] <= particle:
            entry = k
    scene_index_out[particle] = entry_scene_index[entry]
    entry_index_out[particle] = entry


class ParticleViewAPI:
    # Registry (rebuilt only on change in rebuild_topology).
    _entries = {}  # {(scene_idx, system_name): {"family", "system", "scene_idx"}}
    _family_keys = {}  # family -> ordered list of entry keys; avoids reconstructing family lists every step
    _entry_ranges = {}  # {(scene_idx, system_name): (start, count)} slices into the flat buffers

    # Per-entry metadata on GPU (tiny, sized to num_entries), read by _fill_particle_labels_kernel.
    # scene index is static (built once in initialize_view); start offsets are rebuilt on layout change.
    _entry_start = None  # wp.array (num_entries,) int32 cuda
    _entry_start_host = None  # fixed pinned CPU staging array for _entry_start
    _entry_scene = None  # wp.array (num_entries,) int32 cuda

    # Persistent flat GPU buffers (grow-on-demand, reused). Valid prefix is [:PARTICLE_COUNT].
    PARTICLE_POSITIONS = None  # wp.array (capacity,) wp.vec3f
    PARTICLE_SCENE_INDEX = None  # wp.array (capacity,) int32
    PARTICLE_ENTRY_INDEX = None  # wp.array (capacity,) int32
    # Per-particle world orientation, written ONLY for macro-visual particles (other families' slots
    # are unused). Downstream containment applies the visual-particle check offset along this
    # orientation; non-visual consumers ignore it.
    VISUAL_PARTICLE_ORIENTATION = None  # wp.array (capacity,) wp.quat
    _buffer_capacity = 0

    # The number of valid particles this step, as a 1-element GPU array (single source of truth) so
    # in-graph kernels launched over the fixed capacity can gate on the live count without a graph
    # re-capture. Only rewritten on a layout change (in _rebuild_flat_layout).
    PARTICLE_COUNT = None  # wp.array (1,) int32 cuda

    # Use this to detect if there is a layout change
    # value = tuple of per-entry particle counts the current layout was built for.
    _particle_counts_at_last_layout = None

    _micro_instancer_path_to_entry = {}  # instancer prim path -> (scene, system) key
    _micro_instancer_destination_start = (
        None  # wp int32 (num_instancers,) CUDA: start index in flat-buffer per instancer (-1 = skip)
    )
    _micro_instancer_destination_start_host = None  # wp int32 (num_instancers,) CPU pinned version of above
    _micro_instancer_capacity = 0  # allocated length of the two arrays above (grow-on-demand, never shrinks)
    _micro_max_particles_per_instancer = 0  # largest micro instancer's particle count

    _macro_visual_num_particles = 0  # K = valid rows in the capacity-sized visual metadata arrays
    _macro_visual_capacity = 0
    _macro_visual_count = None  # wp int32 (1,) live K, read by the fixed-capacity captured kernel
    _macro_visual_parent_link_index = None  # wp int32 (capacity,) -> parent link flat idx
    _macro_visual_world_scale = None  # wp vec3 (capacity,) -> parent link world-accumulated scale
    _macro_visual_local_matrix = None  # wp mat44 (capacity,) -> each particle's static local pose
    _macro_visual_particle_entry_index = None  # wp int32 (capacity,) -> entry index in _entries order
    _macro_visual_particle_local_index = None  # wp int32 (capacity,) -> slot within the owning entry
    _macro_visual_fallback_keys = []  # entries with cloth/untracked parents, they need to use system's own getter

    # macro-physical: ONE cross-scene rigid-body view over every macro-physical particle prim, plus
    # static per-particle tables (rebuilt only on topology in initialize_view). Per step this yields
    # one get_transforms() + one H2D copy + one scatter kernel (no per-system loop).
    _macro_physical_view = None  # rigid-body view over /World/scene_*/*/particles/*, or None
    _macro_physical_num_particles = 0  # M = number of view rows
    _macro_physical_particle_entry_index = None  # wp int32 (M,) -> entry index in _entries order (-1 to skip)
    _macro_physical_particle_local_index = None  # wp int32 (M,) -> slot within the owning entry (system getter order)
    _macro_physical_particle_offset = None  # wp vec3 (M,) -> owning system's _particle_offset
    _macro_physical_transforms_host = None  # pinned CPU staging buffer filled by get_transforms()
    _macro_physical_transforms = None  # wp float32 (M,7) GPU, copied/scattered inside the graph

    @classmethod
    def rebuild_topology(cls):
        """Rebuild the complete registry, flat layout, and family metadata.

        Unlike the old lazy lifecycle, this method returns with all metadata tables valid. Runtime
        mutations are still coalesced and processed by prepare_step_host() before the next consumer.
        """
        cls.clear()
        entries = {}
        for scene_idx, scene in enumerate(og.sim.scenes):
            if scene is None:
                continue
            for system_name, system in scene.active_systems.items():
                if not system.initialized:
                    continue
                family = _classify_family(system)
                if family is None:
                    continue
                entries[(scene_idx, system_name)] = {"family": family, "system": system, "scene_idx": scene_idx}
        cls._entries = entries
        cls._family_keys = {
            family: [key for key, entry in entries.items() if entry["family"] == family]
            for family in (FAMILY_MICRO_PHYSICAL, FAMILY_MACRO_PHYSICAL, FAMILY_MACRO_VISUAL)
        }

        if len(entries) > 0:
            # These per-entry arrays keep stable pointers until the registry itself is rebuilt.
            scene_indices = th.tensor([entry["scene_idx"] for entry in entries.values()], dtype=th.int32)
            cls._entry_scene = wp.array(scene_indices, dtype=wp.int32, device="cuda")
            cls._entry_start_host = wp.zeros(len(entries), dtype=wp.int32, device="cpu", pinned=True)
            cls._entry_start = wp.zeros(len(entries), dtype=wp.int32, device="cuda")

        # Count scalars have stable pointers and let captured kernels use capacity-sized launches.
        cls.PARTICLE_COUNT = wp.zeros(1, dtype=wp.int32, device="cuda")
        cls._macro_visual_count = wp.zeros(1, dtype=wp.int32, device="cuda")

        counts = [entry["system"].n_particles for entry in entries.values()]
        cls._rebuild_flat_layout(counts, tuple(counts))
        cls._rebuild_micro_metadata()
        cls._rebuild_macro_physical_view()
        cls._rebuild_macro_visual_tables()
        cls._clear_all_tracked_metadata_dirty()
        cls._mark_graph_dirty()

    @classmethod
    def initialize_view(cls):
        """Compatibility alias for rebuild_topology()."""
        cls.rebuild_topology()

    @classmethod
    def _rebuild_macro_physical_view(cls):
        """Build cross-scene rigid-body view over every macro-physical particle prim across all
        scenes, plus static per-particle (entry index, local slot, offset) tables. Runs only on
        topology (macro-physical particle add/remove goes through og.sim.update_handles). Leaves the
        view as None when no macro-physical particles exist (an empty pattern would fail to create)."""
        cls._macro_physical_view = None
        cls._macro_physical_num_particles = 0
        cls._macro_physical_particle_entry_index = None
        cls._macro_physical_particle_local_index = None
        cls._macro_physical_particle_offset = None
        cls._macro_physical_transforms_host = None
        cls._macro_physical_transforms = None
        macro_physical_keys = [
            key for key in cls._family_keys[FAMILY_MACRO_PHYSICAL] if cls._entries[key]["system"].n_particles > 0
        ]
        if len(macro_physical_keys) == 0:
            cls._clear_family_metadata_dirty(FAMILY_MACRO_PHYSICAL)
            return

        entry_index_by_key = {key: i for i, key in enumerate(cls._entries.keys())}

        # abs particle prim path -> (entry_index, local_index, offset). local_index is the row order of
        # the system's OWN particles_view -> matches its getter exactly, giving bit-for-bit parity.
        path_to_particle = {}
        for key in macro_physical_keys:
            system = cls._entries[key]["system"]
            offset = system._particle_offset.to(th.float32)
            for local_index, abs_path in enumerate(system.particles_view.prim_paths):
                path_to_particle[abs_path] = (entry_index_by_key[key], local_index, offset)

        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            cls._macro_physical_view = og.sim.physics_sim_view.create_rigid_body_view(
                pattern="/World/scene_*/*/particles/*"
            )
        view_paths = list(cls._macro_physical_view.prim_paths)
        cls._macro_physical_num_particles = len(view_paths)
        if cls._macro_physical_num_particles == 0:
            cls._clear_family_metadata_dirty(FAMILY_MACRO_PHYSICAL)
            return

        entry_indices, local_indices, offsets = [], [], []
        for abs_path in view_paths:
            match = path_to_particle.get(abs_path)
            if match is None:  # pattern over-match (not a tracked macro-physical particle) -> skip in kernel
                entry_indices.append(-1)
                local_indices.append(0)
                offsets.append(th.zeros(3, dtype=th.float32))
            else:
                entry_index, local_index, offset = match
                entry_indices.append(entry_index)
                local_indices.append(local_index)
                offsets.append(offset)

        cls._macro_physical_particle_entry_index = wp.array(
            th.tensor(entry_indices, dtype=th.int32), dtype=wp.int32, device="cuda"
        )
        cls._macro_physical_particle_local_index = wp.array(
            th.tensor(local_indices, dtype=th.int32), dtype=wp.int32, device="cuda"
        )
        cls._macro_physical_particle_offset = wp.array(th.stack(offsets), dtype=wp.vec3, device="cuda")
        cls._macro_physical_transforms_host = wp.zeros(
            (cls._macro_physical_num_particles, 7), dtype=wp.float32, device="cpu", pinned=True
        )
        cls._macro_physical_transforms = wp.zeros(
            (cls._macro_physical_num_particles, 7), dtype=wp.float32, device="cuda"
        )
        cls._clear_family_metadata_dirty(FAMILY_MACRO_PHYSICAL)

    @classmethod
    def prepare_step_host(cls):
        """Prepare non-capturable sources and metadata before the per-step GPU graph runs."""
        cls._refresh_dirty_metadata()
        cls._prepare_micro_positions_outside_graph()
        cls._stage_macro_physical_transforms()
        cls._prepare_macro_visual_fallback()

    @classmethod
    def read_particle_positions(cls):
        """Compatibility alias for prepare_step_host()."""
        cls.prepare_step_host()

    @classmethod
    def update_positions_gpu(cls):
        """Run all capture-safe particle position copies and kernels.

        MUST be invoked from inside the captured simulation graph, AFTER RigidBodyViewAPI.update() (so
        POSE_MATRICES holds this step's link poses) and BEFORE the tensorized particle states'
        global_update(). Macro-physical H2D and scatter are captured here. Micro remains outside the
        graph because its Fabric wrapper is transient across steps.
        """
        if cls._macro_physical_num_particles > 0:
            wp.copy(cls._macro_physical_transforms, cls._macro_physical_transforms_host)
            wp.launch(
                _scatter_macro_physical_positions_kernel,
                dim=cls._macro_physical_num_particles,
                inputs=[
                    cls._macro_physical_transforms,
                    cls._macro_physical_particle_entry_index,
                    cls._macro_physical_particle_local_index,
                    cls._macro_physical_particle_offset,
                    cls._entry_start,
                    cls.PARTICLE_POSITIONS,
                ],
                device="cuda",
            )

        if (
            RigidBodyViewAPI.POSE_MATRICES is not None
            and cls._macro_visual_capacity > 0
            and cls.PARTICLE_POSITIONS is not None
        ):
            wp.launch(
                _get_macro_visual_positions_kernel,
                dim=cls._macro_visual_capacity,
                inputs=[
                    RigidBodyViewAPI.POSE_MATRICES,
                    cls._macro_visual_parent_link_index,
                    cls._macro_visual_world_scale,
                    cls._macro_visual_local_matrix,
                    cls._macro_visual_particle_entry_index,
                    cls._macro_visual_particle_local_index,
                    cls._entry_start,
                    cls._macro_visual_count,
                    cls.PARTICLE_POSITIONS,
                    cls.VISUAL_PARTICLE_ORIENTATION,
                ],
                device="cuda",
            )

    @classmethod
    def update(cls):
        """Compatibility alias for update_positions_gpu()."""
        cls.update_positions_gpu()

    @classmethod
    def _refresh_dirty_metadata(cls):
        """Process dirty per-system metadata changes once, outside the captured graph."""
        dirty_families = {
            family: [key for key in keys if cls._entries[key]["system"].particle_metadata_dirty]
            for family, keys in cls._family_keys.items()
        }
        if not any(dirty_families.values()):
            return

        counts = [entry["system"].n_particles for entry in cls._entries.values()]
        particle_counts = tuple(counts)
        if particle_counts != cls._particle_counts_at_last_layout:
            cls._rebuild_flat_layout(counts, particle_counts)

        if dirty_families[FAMILY_MICRO_PHYSICAL]:
            cls._rebuild_micro_metadata()
        if dirty_families[FAMILY_MACRO_PHYSICAL]:
            # Normal macro-physical topology mutations reach rebuild_topology() through update_handles().
            # Keep this defensive cold path for callers that explicitly refresh without doing so.
            cls._rebuild_macro_physical_view()
            cls._mark_graph_dirty()
        if dirty_families[FAMILY_MACRO_VISUAL]:
            # TODO(vector): Keep the full-family logical rebuild for correctness. A future packed layout
            # can update stable per-system slices, but should retain the capacity-sized device buffers.
            cls._rebuild_macro_visual_tables()
            cls._clear_family_metadata_dirty(FAMILY_MACRO_VISUAL)

    @classmethod
    def _rebuild_flat_layout(cls, counts, particle_counts):
        """Recompute per-entry offsets + total, grow buffers if needed, and refill the label
        buffers. Runs only when the per-entry particle counts change."""
        cls._entry_ranges = {}
        entry_starts = []
        offset = 0
        for i, key in enumerate(cls._entries.keys()):
            cls._entry_ranges[key] = (offset, counts[i])
            entry_starts.append(offset)
            offset += counts[i]
        total = offset
        # Mirror the valid count to the GPU (single source of truth; only changes here, on layout change).
        cls.PARTICLE_COUNT.fill_(total)

        # Grow the flat buffers only when capacity is exceeded.
        if total > cls._buffer_capacity:
            cls._buffer_capacity = max(total, cls._buffer_capacity * 2, _MIN_BUFFER_CAPACITY)
            cls.PARTICLE_POSITIONS = wp.zeros(cls._buffer_capacity, dtype=wp.vec3f, device="cuda")
            cls.PARTICLE_SCENE_INDEX = wp.zeros(cls._buffer_capacity, dtype=wp.int32, device="cuda")
            cls.PARTICLE_ENTRY_INDEX = wp.zeros(cls._buffer_capacity, dtype=wp.int32, device="cuda")
            cls.VISUAL_PARTICLE_ORIENTATION = wp.zeros(cls._buffer_capacity, dtype=wp.quat, device="cuda")
            cls._mark_graph_dirty()

        cls._particle_counts_at_last_layout = particle_counts

        # Update the existing per-entry buffers in place. Count changes no longer replace _entry_start
        # or force graph capture when the flat output capacity is still sufficient.
        if len(entry_starts) > 0:
            wp.to_torch(cls._entry_start_host).copy_(th.tensor(entry_starts, dtype=th.int32))
            wp.copy(cls._entry_start, cls._entry_start_host)
        if total > 0:
            wp.launch(
                _fill_particle_labels_kernel,
                dim=total,
                inputs=[
                    cls._entry_start,
                    cls._entry_scene,
                    len(cls._entries),
                    total,
                    cls.PARTICLE_SCENE_INDEX,
                    cls.PARTICLE_ENTRY_INDEX,
                ],
                device="cuda",
            )

    @classmethod
    def _rebuild_micro_metadata(cls):
        """Rebuild micro instancer identity mapping and the maximum 2-D scatter width."""
        cls._micro_instancer_path_to_entry = {}
        for key in cls._family_keys[FAMILY_MICRO_PHYSICAL]:
            entry = cls._entries[key]
            if entry["system"].n_particles == 0:
                continue
            instancer = entry["system"].particle_instancer
            if instancer is not None:
                cls._micro_instancer_path_to_entry[instancer.prim_path] = key
        cls._micro_max_particles_per_instancer = max(
            (cls._entry_ranges[key][1] for key in cls._family_keys[FAMILY_MICRO_PHYSICAL]), default=0
        )
        cls._clear_family_metadata_dirty(FAMILY_MICRO_PHYSICAL)

    @classmethod
    def _prepare_micro_positions_outside_graph(cls):
        """Resolve Fabric's transient rows and scatter all micro positions outside the graph."""
        if cls._micro_instancer_path_to_entry:
            Usd, Sdf = lazy.usdrt.Usd, lazy.usdrt.Sdf
            selection = og.sim.usdrt_stage.SelectPrims(
                require_attrs=[(Sdf.ValueTypeNames.Point3fArray, "positions", Usd.Access.Read)],
                device="cuda:0",
                want_paths=True,
            )
            instancer_positions_fabric = wp.fabricarrayarray(data=selection, attrib="positions", dtype=wp.vec3f)
            paths = selection.GetPaths()
            num_instancers = len(paths)
            if num_instancers > 0 and cls._micro_max_particles_per_instancer > 0:
                # Map each Fabric row (instancer) to its destination slice. The path -> entry lookup
                # table is cached and only rebuilt on layout change; this per-step loop is just tiny
                # dict lookups (one per active micro instancer, i.e. per active micro system across
                # scenes) into it, NOT per-particle work. It can't be fully cached away because
                # SelectPrims' GetPaths() row order is not guaranteed stable across steps, so we
                # re-resolve the row->destination array each step. Particle positions themselves still
                # come from the single Fabric selection above + one batched scatter launch below.
                cls._ensure_micro_instancer_capacity(num_instancers)
                destination_start = cls._micro_instancer_destination_start_host.numpy()
                for instancer, path in enumerate(paths):
                    key = cls._micro_instancer_path_to_entry.get(str(path))
                    destination_start[instancer] = cls._entry_ranges[key][0] if key is not None else -1
                wp.copy(
                    cls._micro_instancer_destination_start[:num_instancers],
                    cls._micro_instancer_destination_start_host[:num_instancers],
                )
                wp.launch(
                    _scatter_instancer_positions_kernel,
                    dim=(num_instancers, cls._micro_max_particles_per_instancer),
                    inputs=[instancer_positions_fabric, cls._micro_instancer_destination_start, cls.PARTICLE_POSITIONS],
                    device="cuda",
                )

    @classmethod
    def _stage_macro_physical_transforms(cls):
        """Read one cross-scene PhysX batch into fixed pinned host memory."""
        if cls._macro_physical_view is not None and cls._macro_physical_num_particles > 0:
            transforms = cls._macro_physical_view.get_transforms()  # (M,7) CPU torch, single batched read
            wp.to_torch(cls._macro_physical_transforms_host).copy_(transforms)

    @classmethod
    def _prepare_macro_visual_fallback(cls):
        """Refresh visual entries whose cloth/untracked parents cannot use the captured rigid kernel."""
        # Cloth/untracked entries (rare) fall back to the per-entry getter. If POSE_MATRICES isn't ready
        # yet (transient startup), every visual entry falls back for that step.
        if RigidBodyViewAPI.POSE_MATRICES is None:
            fallback_keys = cls._family_keys[FAMILY_MACRO_VISUAL]
        else:
            fallback_keys = cls._macro_visual_fallback_keys
        for key in fallback_keys:
            start, count = cls._entry_ranges[key]
            if count == 0:
                continue
            # Copy BOTH position and orientation: downstream containment applies the visual-particle
            # check offset along VISUAL_PARTICLE_ORIENTATION, so fallback slots must carry it too
            # (otherwise the offset uses a stale/zero quat).
            positions, orientations = cls._entries[key]["system"].get_particles_position_orientation()
            wp.copy(
                cls.PARTICLE_POSITIONS[start : start + count],
                wp.from_torch(positions.contiguous(), dtype=wp.vec3f),
            )
            wp.copy(
                cls.VISUAL_PARTICLE_ORIENTATION[start : start + count],
                wp.from_torch(orientations.contiguous(), dtype=wp.quat),
            )

    @classmethod
    def _ensure_micro_instancer_capacity(cls, num_instancers):
        """Make sure the reused per-instancer destination buffers can hold `num_instancers` entries.
        They grow (never shrink) only when a step presents more instancers than any previous step."""
        if num_instancers <= cls._micro_instancer_capacity:
            return
        cls._micro_instancer_capacity = max(num_instancers, cls._micro_instancer_capacity * 2)
        cls._micro_instancer_destination_start = wp.zeros(cls._micro_instancer_capacity, dtype=wp.int32, device="cuda")
        cls._micro_instancer_destination_start_host = wp.zeros(
            cls._micro_instancer_capacity, dtype=wp.int32, device="cpu", pinned=True
        )

    @classmethod
    def _clear_family_metadata_dirty(cls, family):
        for key in cls._family_keys[family]:
            cls._entries[key]["system"].clear_particle_metadata_dirty()

    @classmethod
    def _clear_all_tracked_metadata_dirty(cls):
        for family in cls._family_keys:
            cls._clear_family_metadata_dirty(family)

    @classmethod
    def _rebuild_macro_visual_tables(cls):
        """Collect, for every kernel-eligible visual particle across all scenes/systems, its parent
        link index, world scale, static local matrix, and (entry, slot) destination into flat arrays
        for one kernel launch. An entry whose particles are cloth-parented or lack a rigid-body link is
        recorded as a getter fallback instead (all-or-nothing per entry).

        TODO(vector): Runtime parent-object rescaling remains unsupported because the parent owns that
        mutation and has no hook to invalidate attached systems' cached world scales.
        """
        entry_index_by_key = {key: i for i, key in enumerate(cls._entries.keys())}
        parent_link_index, world_scale, local_matrix = [], [], []
        particle_entry_index, particle_local_index = [], []
        fallback_keys = []
        for key in cls._family_keys[FAMILY_MACRO_VISUAL]:
            entry = cls._entries[key]
            system = entry["system"]
            names = list((system.particles or {}).keys())  # None until first particle added
            if len(names) == 0:
                continue
            # Gather this entry's particles into temporaries; commit only if the whole entry is eligible.
            entry_links, entry_scales, entry_matrices = [], [], []
            eligible = True
            for name in names:
                info = system._particles_info[name]
                if info["obj"].prim_type == PrimType.CLOTH:
                    eligible = False  # cloth-parented particles aren't in the rigid-body view
                    break
                flat_index = RigidBodyViewAPI.get_flat_idx(info["link"].prim_path)
                if flat_index is None:
                    eligible = False
                    break
                scaled = info["link"].scaled_transform  # (4,4) world w/ scale
                column_scale = th.linalg.norm(scaled[:3, :3], dim=0)  # column norms = world-accumulated scale
                entry_links.append(flat_index)
                entry_scales.append(column_scale.to(th.float32))
                entry_matrices.append(system._particles_local_mat[name].to(th.float32))
            if not eligible:
                fallback_keys.append(key)
                continue
            ei = entry_index_by_key[key]
            for local_index in range(len(names)):
                parent_link_index.append(entry_links[local_index])
                world_scale.append(entry_scales[local_index])
                local_matrix.append(entry_matrices[local_index])
                particle_entry_index.append(ei)
                particle_local_index.append(local_index)

        cls._macro_visual_fallback_keys = fallback_keys
        cls._macro_visual_num_particles = len(parent_link_index)
        reserve = _MIN_VISUAL_METADATA_CAPACITY if cls._family_keys[FAMILY_MACRO_VISUAL] else 0
        cls._ensure_macro_visual_capacity(max(cls._macro_visual_num_particles, reserve))
        cls._macro_visual_count.fill_(cls._macro_visual_num_particles)
        if cls._macro_visual_num_particles == 0:
            return

        wp.copy(
            cls._macro_visual_parent_link_index[: cls._macro_visual_num_particles],
            wp.from_torch(th.tensor(parent_link_index, dtype=th.int32)),
        )
        wp.copy(
            cls._macro_visual_world_scale[: cls._macro_visual_num_particles],
            wp.from_torch(th.stack(world_scale).contiguous(), dtype=wp.vec3),
        )
        wp.copy(
            cls._macro_visual_local_matrix[: cls._macro_visual_num_particles],
            wp.from_torch(th.stack(local_matrix).contiguous(), dtype=wp.mat44),
        )
        wp.copy(
            cls._macro_visual_particle_entry_index[: cls._macro_visual_num_particles],
            wp.from_torch(th.tensor(particle_entry_index, dtype=th.int32)),
        )
        wp.copy(
            cls._macro_visual_particle_local_index[: cls._macro_visual_num_particles],
            wp.from_torch(th.tensor(particle_local_index, dtype=th.int32)),
        )

    @classmethod
    def _ensure_macro_visual_capacity(cls, required_capacity):
        if required_capacity <= cls._macro_visual_capacity:
            return
        cls._macro_visual_capacity = max(required_capacity, cls._macro_visual_capacity * 2)
        cls._macro_visual_parent_link_index = wp.zeros(cls._macro_visual_capacity, dtype=wp.int32, device="cuda")
        cls._macro_visual_world_scale = wp.zeros(cls._macro_visual_capacity, dtype=wp.vec3, device="cuda")
        cls._macro_visual_local_matrix = wp.zeros(cls._macro_visual_capacity, dtype=wp.mat44, device="cuda")
        cls._macro_visual_particle_entry_index = wp.zeros(cls._macro_visual_capacity, dtype=wp.int32, device="cuda")
        cls._macro_visual_particle_local_index = wp.zeros(cls._macro_visual_capacity, dtype=wp.int32, device="cuda")
        cls._mark_graph_dirty()

    @staticmethod
    def _mark_graph_dirty():
        from omnigibson.object_states.tensorized_state import TensorizedState

        TensorizedState.graph_dirty = True

    @classmethod
    def get_particle_positions(cls, scene_idx, system_name):
        """Convenience/test accessor: the (count,) wp.vec3f slice of PARTICLE_POSITIONS for one
        (scene, system), or None if untracked."""
        range = cls._entry_ranges.get((scene_idx, system_name))
        if range is None or cls.PARTICLE_POSITIONS is None:
            return None
        start, count = range
        return cls.PARTICLE_POSITIONS[start : start + count]

    @classmethod
    def get_family(cls, scene_idx, system_name):
        entry = cls._entries.get((scene_idx, system_name))
        return None if entry is None else entry["family"]

    @classmethod
    def entries(cls):
        return list(cls._entries.keys())

    @classmethod
    def clear(cls):
        cls._entries = {}
        cls._family_keys = {}
        cls._entry_ranges = {}
        cls._entry_start = None
        cls._entry_start_host = None
        cls._entry_scene = None
        cls.PARTICLE_POSITIONS = None
        cls.PARTICLE_SCENE_INDEX = None
        cls.PARTICLE_ENTRY_INDEX = None
        cls.VISUAL_PARTICLE_ORIENTATION = None
        cls.PARTICLE_COUNT = None
        cls._buffer_capacity = 0
        cls._particle_counts_at_last_layout = None
        cls._micro_instancer_path_to_entry = {}
        cls._micro_instancer_destination_start = None
        cls._micro_instancer_destination_start_host = None
        cls._micro_instancer_capacity = 0
        cls._micro_max_particles_per_instancer = 0
        cls._macro_visual_num_particles = 0
        cls._macro_visual_capacity = 0
        cls._macro_visual_count = None
        cls._macro_visual_parent_link_index = None
        cls._macro_visual_world_scale = None
        cls._macro_visual_local_matrix = None
        cls._macro_visual_particle_entry_index = None
        cls._macro_visual_particle_local_index = None
        cls._macro_visual_fallback_keys = []
        cls._macro_physical_view = None
        cls._macro_physical_num_particles = 0
        cls._macro_physical_particle_entry_index = None
        cls._macro_physical_particle_local_index = None
        cls._macro_physical_particle_offset = None
        cls._macro_physical_transforms_host = None
        cls._macro_physical_transforms = None
