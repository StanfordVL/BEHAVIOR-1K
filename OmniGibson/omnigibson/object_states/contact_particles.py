import torch as th
import warp as wp

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.tensorized_object_system_state import TensorizedObjectSystemState
from omnigibson.utils.particle_view_utils import FAMILY_MACRO_PHYSICAL, FAMILY_MICRO_PHYSICAL, ParticleViewAPI
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI, rigid_inverse_mat44

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Distance tolerance for detecting contact
m.CONTACT_AABB_TOLERANCE = 2.5e-2
m.CONTACT_TOLERANCE = 5e-3


@wp.func
def _particle_contacts_obj(
    p_world: wp.vec3,
    aabb_row: wp.int32,  # this object's row in AABB.VALUES_WP, or -1 (skip prefilter)
    aabb_values: wp.array3d(dtype=wp.float32),  # AABB.VALUES_WP (S, O, 6)
    scene: wp.int32,
    aabb_margin: wp.float32,  # particle_radius + CONTACT_AABB_TOLERANCE
    link_start: wp.int32,
    link_count: wp.int32,
    link_flat_idx: wp.array(dtype=wp.int32),  # flat link indices of this object's links
    pose_matrices: wp.array(dtype=wp.mat44),  # RigidBodyViewAPI.POSE_MATRICES
    mesh_ids: wp.array(dtype=wp.uint64),  # RigidBodyViewAPI.LINK_MESH_IDS (0 = no collision mesh)
    radius: wp.float32,  # particle_contact_radius + CONTACT_TOLERANCE
):
    """Return 1 if a sphere of `radius` at the particle overlaps any of the object's link collision
    solids (broad-phase: object AABB expanded by aabb_margin), else 0. Mirrors overlap_sphere:
    overlap == the point is inside the solid OR within `radius` of its surface."""
    # Broad-phase: reject particles outside the object's (expanded) AABB.
    if aabb_row >= 0:
        if (
            p_world[0] < aabb_values[scene, aabb_row, 0] - aabb_margin
            or p_world[1] < aabb_values[scene, aabb_row, 1] - aabb_margin
            or p_world[2] < aabb_values[scene, aabb_row, 2] - aabb_margin
            or p_world[0] > aabb_values[scene, aabb_row, 3] + aabb_margin
            or p_world[1] > aabb_values[scene, aabb_row, 4] + aabb_margin
            or p_world[2] > aabb_values[scene, aabb_row, 5] + aabb_margin
        ):
            return wp.int32(0)
    # Narrow-phase: bring the particle into each link's local frame and test its collision mesh.
    for k in range(link_start, link_start + link_count):
        link = link_flat_idx[k]
        mesh_id = mesh_ids[link]
        if mesh_id == wp.uint64(0):
            continue
        local_homogeneous = wp.mul(
            rigid_inverse_mat44(pose_matrices[link]), wp.vec4(p_world[0], p_world[1], p_world[2], 1.0)
        )
        local = wp.vec3(local_homogeneous[0], local_homogeneous[1], local_homogeneous[2])
        # Big max_dist so even a deep-interior point resolves a closest face (=> its sign is readable);
        # the broad-phase AABB gate keeps this query restricted to nearby particles.
        query = wp.mesh_query_point(mesh_id, local, 1.0e6)
        if query.result:
            if query.sign < 0.0:
                return wp.int32(1)  # inside the solid (any depth) — sphere overlaps it
            closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            if wp.length(local - closest) <= radius:
                return wp.int32(1)  # within `radius` of the surface
    return wp.int32(0)


@wp.kernel
def _count_contact_particles_kernel(
    particle_positions: wp.array(dtype=wp.vec3f),  # ParticleViewAPI flat positions (capacity,)
    particle_scene_index: wp.array(dtype=wp.int32),
    particle_entry_index: wp.array(dtype=wp.int32),
    particle_count: wp.array(dtype=wp.int32),  # (1,) valid particle count this step
    entry_sys_index: wp.array(dtype=wp.int32),  # entry -> N_sys column, -1 to skip (incl. all visual)
    entry_contact_radius: wp.array(dtype=wp.float32),  # per entry: particle_contact_radius + CONTACT_TOLERANCE
    entry_aabb_margin: wp.array(dtype=wp.float32),  # per entry: particle_radius + CONTACT_AABB_TOLERANCE
    n_obj: int,
    aabb_idx: wp.array(dtype=wp.int32),  # (N_obj,) our obj -> AABB row
    aabb_values: wp.array3d(dtype=wp.float32),  # AABB.VALUES_WP (S, O, 6)
    obj_link_start: wp.array(dtype=wp.int32),  # (S*N_obj,)
    obj_link_count: wp.array(dtype=wp.int32),  # (S*N_obj,)
    link_flat_idx: wp.array(dtype=wp.int32),
    pose_matrices: wp.array(dtype=wp.mat44),
    mesh_ids: wp.array(dtype=wp.uint64),
    values: wp.array3d(dtype=wp.int32),  # (S, N_obj, N_sys) count output
):
    """One thread per (particle_slot, object). +1 to values[scene, obj, sys] if the particle contacts
    the object (within radius of any of its collision meshes)."""
    particle, obj = wp.tid()
    if particle >= particle_count[0]:
        return
    entry = particle_entry_index[particle]
    system_column = entry_sys_index[entry]
    if system_column < 0:
        return
    scene = particle_scene_index[particle]
    base = scene * n_obj + obj
    link_count = obj_link_count[base]
    if link_count == 0:
        return
    if _particle_contacts_obj(
        particle_positions[particle],
        aabb_idx[obj],
        aabb_values,
        scene,
        entry_aabb_margin[entry],
        obj_link_start[base],
        link_count,
        link_flat_idx,
        pose_matrices,
        mesh_ids,
        entry_contact_radius[entry],
    ) == wp.int32(1):
        wp.atomic_add(values, scene, obj, system_column, 1)


@wp.kernel
def _contact_mask_kernel(
    particle_positions: wp.array(dtype=wp.vec3f),  # ONE (scene, system) entry's particle slice (count,)
    aabb_row: wp.int32,
    aabb_values: wp.array3d(dtype=wp.float32),
    scene: wp.int32,
    aabb_margin: wp.float32,
    link_start: wp.int32,
    link_count: wp.int32,
    link_flat_idx: wp.array(dtype=wp.int32),
    pose_matrices: wp.array(dtype=wp.mat44),
    mesh_ids: wp.array(dtype=wp.uint64),
    radius: wp.float32,
    mask_out: wp.array(dtype=wp.int32),  # (count,) — 1 if that particle contacts the object
):
    """On-demand per-particle contact mask for ONE object over ONE (scene, system) entry's slice.
    Uses the SAME _particle_contacts_obj as the count kernel, so particle_indices agree with count."""
    i = wp.tid()
    mask_out[i] = _particle_contacts_obj(
        particle_positions[i],
        aabb_row,
        aabb_values,
        scene,
        aabb_margin,
        link_start,
        link_count,
        link_flat_idx,
        pose_matrices,
        mesh_ids,
        radius,
    )


class ContactParticlesData:
    """Result of ContactParticles.get_value(system[, link]):
        count (int): number of the system's particles in contact with the object.
        particle_indices (set[int]): system-local indices of those contacting particles.

    `count` comes from the tensor (fast, per step). `particle_indices` is computed lazily by the SAME
    GPU contact test over that (scene, system)'s particle slice, so the two stay consistent. A
    link-specific query derives count from len(particle_indices)."""

    def __init__(self, state, system, link, count):
        self._state = state
        self._system = system
        self._link = link
        self._count = count  # None for a link-specific query -> derive from particle_indices
        self._particle_indices = None
        self._particle_indices_computed = False

    @property
    def particle_indices(self):
        if not self._particle_indices_computed:
            self._particle_indices = self._state._compute_contact_particle_indices(self._system, self._link)
            self._particle_indices_computed = True
        return self._particle_indices

    @property
    def count(self):
        return len(self.particle_indices) if self._count is None else self._count


class ContactParticles(TensorizedObjectSystemState, KinematicsMixin):
    """
    Object state that handles contact checking between rigid bodies and individual particles.
    Calculate number of a physical system's particles in contact with an object (within
    particle_contact_radius of its collision geometry).
    """

    # Rebuilt on topology in initialize_view. All reuse RigidBodyViewAPI / AABB GPU state (no own mesh table).
    _entry_sys_index = None  # wp int32 (num_entries,) -> N_sys column (physical only; -1 = visual, skip)
    _entry_contact_radius = None  # wp float32 (num_entries,) particle_contact_radius + CONTACT_TOLERANCE
    _entry_aabb_margin = None  # wp float32 (num_entries,) particle_radius + CONTACT_AABB_TOLERANCE
    _aabb_idx = None  # wp int32 (N_obj,) our obj -> AABB.OBJ_IDXS row
    _obj_link_start = None  # wp int32 (S*N_obj,) first link for (scene, obj)
    _obj_link_count = None  # wp int32 (S*N_obj,)
    _link_flat_idx = None  # wp int32 (L,) flat link indices, grouped by (scene, obj)

    @classproperty
    def value_type(cls):
        return th.int32

    @classproperty
    def value_name(cls):
        return "contact_particles"

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)  # we read AABB.VALUES_WP / AABB.OBJ_IDXS; ensures AABB initializes first
        return deps

    @classmethod
    def _reset_tables(cls):
        cls._entry_sys_index = None
        cls._entry_contact_radius = None
        cls._entry_aabb_margin = None
        cls._aabb_idx = None
        cls._obj_link_start = None
        cls._obj_link_count = None
        cls._link_flat_idx = None

    @classmethod
    def initialize_view(cls):
        # Builds OBJ_IDXS / SYS_IDXS / IDX_OBJS / VALUES (S, N_obj, N_sys).
        super().initialize_view()
        cls._reset_tables()
        S = len(cls.IDX_OBJS)
        n_obj = len(cls.OBJ_IDXS)
        n_sys = len(cls.SYS_IDXS)
        if S == 0 or n_obj == 0 or n_sys == 0:
            return

        # Per ParticleViewAPI entry: system column (physical only) + contact radius + AABB margin.
        pv_entries = ParticleViewAPI.entries()
        if len(pv_entries) > 0:
            entry_sys, entry_radius, entry_margin = [], [], []
            for scene_idx, system_name in pv_entries:
                is_physical = ParticleViewAPI.get_family(scene_idx, system_name) in (
                    FAMILY_MICRO_PHYSICAL,
                    FAMILY_MACRO_PHYSICAL,
                )
                if is_physical:
                    system = og.sim.scenes[scene_idx].get_system(system_name, force_init=False)
                    entry_sys.append(cls.SYS_IDXS.get(system_name, -1))
                    entry_radius.append(float(system.particle_contact_radius) + m.CONTACT_TOLERANCE)
                    entry_margin.append(float(system.particle_radius) + m.CONTACT_AABB_TOLERANCE)
                else:
                    entry_sys.append(-1)  # ContactParticles is physical-only
                    entry_radius.append(0.0)
                    entry_margin.append(0.0)
            cls._entry_sys_index = wp.array(th.tensor(entry_sys, dtype=th.int32), dtype=wp.int32, device="cuda")
            cls._entry_contact_radius = wp.array(
                th.tensor(entry_radius, dtype=th.float32), dtype=wp.float32, device="cuda"
            )
            cls._entry_aabb_margin = wp.array(
                th.tensor(entry_margin, dtype=th.float32), dtype=wp.float32, device="cuda"
            )

        # Our obj -> AABB row (like inside.py).
        aabb_map = AABB.OBJ_IDXS or {}
        aabb_idx = [aabb_map.get(rel_path, -1) for rel_path in cls.OBJ_IDXS.keys()]
        cls._aabb_idx = wp.array(th.tensor(aabb_idx, dtype=th.int32), dtype=wp.int32, device="cuda")

        # Per (scene, obj) link range into a flat link-index table (reuses RigidBodyViewAPI flat indices).
        link_flat, obj_link_start, obj_link_count = [], [0] * (S * n_obj), [0] * (S * n_obj)
        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            for obj_idx, obj in enumerate(scene_row):
                if obj is None or cls not in obj.states:
                    continue
                base = scene_idx * n_obj + obj_idx
                obj_link_start[base] = len(link_flat)
                for link in obj.links.values():
                    flat = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                    if flat is None:
                        continue
                    link_flat.append(flat)
                    obj_link_count[base] += 1
        if len(link_flat) == 0:
            return
        cls._link_flat_idx = wp.array(th.tensor(link_flat, dtype=th.int32), dtype=wp.int32, device="cuda")
        cls._obj_link_start = wp.array(th.tensor(obj_link_start, dtype=th.int32), dtype=wp.int32, device="cuda")
        cls._obj_link_count = wp.array(th.tensor(obj_link_count, dtype=th.int32), dtype=wp.int32, device="cuda")

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None:
            return
        cls.VALUES_WP.zero_()
        if (
            cls._link_flat_idx is None
            or cls._entry_sys_index is None
            or RigidBodyViewAPI.POSE_MATRICES is None
            or RigidBodyViewAPI.LINK_MESH_IDS is None
            or AABB.VALUES_WP is None
            or ParticleViewAPI.PARTICLE_POSITIONS is None
        ):
            return
        S, n_obj, _ = values.shape
        capacity = ParticleViewAPI.PARTICLE_POSITIONS.shape[0]
        wp.launch(
            _count_contact_particles_kernel,
            dim=(capacity, n_obj),
            inputs=[
                ParticleViewAPI.PARTICLE_POSITIONS,
                ParticleViewAPI.PARTICLE_SCENE_INDEX,
                ParticleViewAPI.PARTICLE_ENTRY_INDEX,
                ParticleViewAPI.PARTICLE_COUNT,
                cls._entry_sys_index,
                cls._entry_contact_radius,
                cls._entry_aabb_margin,
                n_obj,
                cls._aabb_idx,
                AABB.VALUES_WP,
                cls._obj_link_start,
                cls._obj_link_count,
                cls._link_flat_idx,
                RigidBodyViewAPI.POSE_MATRICES,
                RigidBodyViewAPI.LINK_MESH_IDS,
                cls.VALUES_WP,
            ],
            device="cuda",
        )

    def _get_value(self, system, link=None):
        """
        Args:
            system (PhysicalParticleSystem): System whose contact particle info should be aggregated
            link (None or RigidPrim): If specified, the specific link to check for particles' contact

        Returns:
            Return a ContactParticlesData (`.count` from the tensor, `.particle_indices` lazily from
            the GPU mask kernel). `link` (optional) restricts the query to one link of this object.
        """
        # Make sure system is valid
        assert self.obj.scene.is_physical_particle_system(
            system_name=system.name
        ), "Can only get ContactParticles for a PhysicalParticleSystem!"
        # Whole-object count is the tensor cell; a link-specific count is derived from its particle indices.
        count = None if link is not None else int(super()._get_value(system))
        return ContactParticlesData(state=self, system=system, link=link, count=count)

    def _compute_contact_particle_indices(self, system, link):
        """Launch _contact_mask_kernel over this object's (scene, system) particle slice: it runs the
        same contact test (_particle_contacts_obj) as the count kernel on every particle in the slice
        and writes a 0/1 mask, which we turn into the set of contacting particles' indices. Checks all
        of the object's links, or only `link` if one is given.

        Args:
            system (PhysicalParticleSystem): system whose particles to test against this object.
            link (None or RigidPrim): if given, only test contact against this one link.

        Returns:
            set of int: system-local indices of the particles in contact (the slice order equals the
                system's own particle order, per ParticleViewAPI, so a mask position is a particle
                index). Empty set if the slice or the object's collision geometry is unavailable.
        """
        scene_idx = self.obj.scene.idx
        positions = ParticleViewAPI.get_particle_positions(scene_idx, system.name)
        if (
            positions is None
            or positions.shape[0] == 0
            or RigidBodyViewAPI.POSE_MATRICES is None
            or RigidBodyViewAPI.LINK_MESH_IDS is None
            or AABB.VALUES_WP is None
        ):
            return set()
        links = [link] if link is not None else list(self.obj.links.values())
        link_flats = [f for f in (RigidBodyViewAPI.get_flat_idx(lk.prim_path) for lk in links) if f is not None]
        if len(link_flats) == 0:
            return set()
        count = positions.shape[0]
        mask = wp.zeros(count, dtype=wp.int32, device="cuda")
        wp.launch(
            _contact_mask_kernel,
            dim=count,
            inputs=[
                positions,
                int((AABB.OBJ_IDXS or {}).get(self.obj.relative_prim_path, -1)),
                AABB.VALUES_WP,
                scene_idx,
                float(system.particle_radius) + m.CONTACT_AABB_TOLERANCE,
                0,
                len(link_flats),
                wp.array(th.tensor(link_flats, dtype=th.int32), dtype=wp.int32, device="cuda"),
                RigidBodyViewAPI.POSE_MATRICES,
                RigidBodyViewAPI.LINK_MESH_IDS,
                float(system.particle_contact_radius) + m.CONTACT_TOLERANCE,
                mask,
            ],
            device="cuda",
        )
        return set(wp.to_torch(mask).nonzero().flatten().tolist())

    def _set_value(self, system, new_value):
        raise NotImplementedError("ContactParticles state currently does not support setting.")

    def _cache_is_valid(self, get_value_args):
        # Cache is never valid since particles always change poses
        return False
