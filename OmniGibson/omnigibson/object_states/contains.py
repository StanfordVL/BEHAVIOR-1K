import torch as th
import warp as wp

import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.object_states.tensorized_object_system_state import TensorizedObjectSystemState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.particle_view_utils import FAMILY_MACRO_VISUAL, ParticleViewAPI
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI, rigid_inverse_mat44

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.CONTAINER_META_LINK_TYPES = ["fillable", "openfillable"]
m.VISUAL_PARTICLE_OFFSET = 0.01  # Offset to visual particles' poses when checking overlaps with container volume


class ContainedParticlesData:
    """
    Result of ContainedParticles.get_value(system) with the following fields:
        n_in_volume (int): number of @system particles in the container volume
        positions (th.tensor): (N, 3) raw global particle positions.
        in_volume (th.tensor): (N,) boolean, whether each particle is inside the container volume.

    Reading n_in_volume alone is cheap: it comes straight from the tensor and does NOT build the
    positions / in_volume arrays. positions/in_volume are computed LAZILY the original per-object way
    (system getter + link.check_points_in_volume) only when accessed. Building them re-derives
    n_in_volume from the in_volume mask (its sum), so the count and the per-particle mask always
    describe the same state — building the arrays is what makes the count reflect that same pass.
    """

    def __init__(self, n_in_volume, state, system):
        self.n_in_volume = n_in_volume
        self._state = state
        self._system = system
        self._positions = None
        self._in_volume = None
        self._computed = False

    def _compute(self):
        if not self._computed:
            self._positions, self._in_volume = self._state._compute_positions_in_volume(self._system)
            # Building the mask defines the count; keep n_in_volume consistent with it (same state).
            self.n_in_volume = int(self._in_volume.sum().item())
            self._computed = True

    @property
    def positions(self):
        self._compute()
        return self._positions

    @property
    def in_volume(self):
        self._compute()
        return self._in_volume


@wp.kernel
def _compute_mesh_world_to_local_kernel(
    pose_matrices: wp.array(dtype=wp.mat44),  # (L,) link world transforms (rigid)
    mesh_parent_link: wp.array(dtype=wp.int32),  # (M,) flat link idx
    mesh_inv_local_w_scale: wp.array(dtype=wp.mat44),  # (M,) static inv of mesh->link-local-w-scale
    world_to_mesh_local: wp.array(dtype=wp.mat44),  # (M,) output: world -> mesh-local-unscaled
):
    """Per mesh: compose static inv-local with the link's current rigid world inverse.
    (Duplicated from inside.py's _inside_inv_world_kernel; dedupe in Phase 5.)"""
    mesh = wp.tid()
    parent = mesh_parent_link[mesh]
    if parent < 0:
        return
    world_to_mesh_local[mesh] = wp.mul(mesh_inv_local_w_scale[mesh], rigid_inverse_mat44(pose_matrices[parent]))


@wp.kernel
def _count_contained_particles_kernel(
    particle_positions: wp.array(dtype=wp.vec3f),  # (capacity,) ParticleViewAPI flat positions
    particle_scene_index: wp.array(dtype=wp.int32),  # (capacity,)
    particle_entry_index: wp.array(dtype=wp.int32),  # (capacity,)
    particle_orientation: wp.array(dtype=wp.quat),  # (capacity,) valid only for visual particles
    particle_count: wp.array(dtype=wp.int32),  # (1,) number of valid particles this step
    entry_sys_index: wp.array(dtype=wp.int32),  # (num_entries,) entry -> system column (-1 = skip)
    entry_is_visual: wp.array(dtype=wp.int32),  # (num_entries,) 1 if the entry is a visual system
    visual_offset: float,  # VISUAL_PARTICLE_OFFSET
    n_obj: int,
    obj_mesh_start: wp.array(dtype=wp.int32),  # (S*N_obj,) first mesh index for (scene, obj)
    obj_mesh_count: wp.array(dtype=wp.int32),  # (S*N_obj,) number of meshes for (scene, obj)
    mesh_face_start: wp.array(dtype=wp.int32),  # (M,) first face index for each mesh
    mesh_face_count: wp.array(dtype=wp.int32),  # (M,) number of faces for each mesh
    world_to_mesh_local: wp.array(dtype=wp.mat44),  # (M,) world -> mesh-local-unscaled
    face_centroid: wp.array(dtype=wp.vec3),  # (F,)
    face_normal: wp.array(dtype=wp.vec3),  # (F,)
    values: wp.array3d(dtype=wp.int32),  # (S, N_obj, N_sys) count output (atomic_add target)
):
    """One thread per (particle, container object). If the particle lies inside ANY of that
    container's fillable meshes, add 1 to
    values[scene, obj, system]. Union across a container's meshes is handled by the break."""
    particle, obj = wp.tid()
    if particle >= particle_count[0]:
        return
    entry = particle_entry_index[particle]
    system_column = entry_sys_index[entry]
    if system_column < 0:
        return
    scene = particle_scene_index[particle]
    base = scene * n_obj + obj
    mesh_start = obj_mesh_start[base]
    mesh_count = obj_mesh_count[base]
    if mesh_count == 0:
        return

    # check point is the position we use to check against whether it's inside container
    # for physical particle we use raw position
    # we add an offset along the particle orientation for visual particles
    check_point = particle_positions[particle]
    if entry_is_visual[entry] == 1:
        check_point = check_point + wp.quat_rotate(particle_orientation[particle], wp.vec3(0.0, 0.0, visual_offset))

    # int flags (not bool literals) so Warp treats them as dynamic vars mutable inside the loops
    contained = int(0)
    for mesh in range(mesh_start, mesh_start + mesh_count):
        point_local = wp.transform_point(world_to_mesh_local[mesh], check_point)
        inside_this_mesh = int(1)
        face_begin = mesh_face_start[mesh]
        face_end = face_begin + mesh_face_count[mesh]
        for face in range(face_begin, face_end):
            if wp.dot(point_local - face_centroid[face], face_normal[face]) >= 0.0:
                inside_this_mesh = int(0)
                break
        if inside_this_mesh == 1:
            contained = int(1)
            break
    if contained == 1:
        wp.atomic_add(values, scene, obj, system_column, 1)


class ContainedParticles(TensorizedObjectSystemState, LinkBasedStateMixin):
    """
    Number of particles of a given system contained in this object's fillable container volume,
    computed for every (scene, container, system) at once via a single Warp kernel over
    ParticleViewAPI's flat particle buffer.
    """

    # Belows are to store information for fillable faces table
    # (DUPLICATED from inside.py; Phase-5 cleanup: extract a shared container_hull
    # module).
    # Rebuilt on topology in initialize_view.
    _mesh_parent_link = None  # wp int32 (M,) every mesh's fillable link's index in RigidBodyViewAPI.POSE_MATRICES
    _mesh_inv_local_w_scale = None  # wp mat44 (M,) static world->local-w-scale inverse at init
    _mesh_face_start = None  # wp int32 (M,) every mesh's face's end index in table
    _mesh_face_count = None  # wp int32 (M,) every mesh's face's count in table
    _face_centroid = None  # wp vec3 (F,) local-unscaled
    _face_normal = None  # wp vec3 (F,) local-unscaled
    _obj_mesh_start = None  # wp int32 (S*N_obj,) first mesh for (scene, obj)
    _obj_mesh_count = None  # wp int32 (S*N_obj,)
    _world_to_mesh_local = None  # wp mat44 (M,) per-step scratch, world->mesh-local

    # ParticleViewAPI entry -> our system column / visual flag (indexed by PARTICLE_ENTRY_INDEX).
    _entry_sys_index = None  # wp int32 (num_entries,) -> N_sys column, -1 if untracked
    _entry_is_visual = None  # wp int32 (num_entries,) 1 if a visual system

    @classproperty
    def value_type(cls):
        return th.int32

    @classproperty
    def value_name(cls):
        return "contained_particles"

    @classproperty
    def meta_link_types(cls):
        return m.CONTAINER_META_LINK_TYPES

    @classmethod
    def _reset_hull_tables(cls):
        cls._mesh_parent_link = None
        cls._mesh_inv_local_w_scale = None
        cls._mesh_face_start = None
        cls._mesh_face_count = None
        cls._face_centroid = None
        cls._face_normal = None
        cls._obj_mesh_start = None
        cls._obj_mesh_count = None
        cls._world_to_mesh_local = None
        cls._entry_sys_index = None
        cls._entry_is_visual = None

    @classmethod
    def initialize_view(cls):
        # Builds OBJ_IDXS / SYS_IDXS / IDX_OBJS / VALUES (S, N_obj, N_sys).
        super().initialize_view()
        cls._reset_hull_tables()

        S = len(cls.IDX_OBJS)
        n_obj = len(cls.OBJ_IDXS)
        n_sys = len(cls.SYS_IDXS)
        if S == 0 or n_obj == 0 or n_sys == 0:
            return

        # Map each ParticleViewAPI entry -> our system column + whether it's a visual system.
        pv_entries = ParticleViewAPI.entries()
        if len(pv_entries) > 0:
            entry_sys = [cls.SYS_IDXS.get(system_name, -1) for (_, system_name) in pv_entries]
            entry_visual = [
                1 if ParticleViewAPI.get_family(scene_idx, system_name) == FAMILY_MACRO_VISUAL else 0
                for (scene_idx, system_name) in pv_entries
            ]
            cls._entry_sys_index = wp.array(th.tensor(entry_sys, dtype=th.int32), dtype=wp.int32, device="cuda")
            cls._entry_is_visual = wp.array(th.tensor(entry_visual, dtype=th.int32), dtype=wp.int32, device="cuda")

        # Build the flat fillable-mesh face table, grouped by (scene, obj) then mesh.
        mesh_parent_link, mesh_inv_local, mesh_face_start, mesh_face_count = [], [], [], []
        face_centroids, face_normals = [], []
        obj_mesh_start = [0] * (S * n_obj)
        obj_mesh_count = [0] * (S * n_obj)
        face_cursor = 0
        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            for obj_idx, obj in enumerate(scene_row):
                if obj is None or cls not in obj.states or obj.prim_type == PrimType.CLOTH:
                    continue
                link = obj.states[cls].link
                parent_flat = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                if parent_flat is None:
                    continue
                link_world_inv_init = th.linalg.inv(T.pose2mat(link.get_position_orientation()))
                base = scene_idx * n_obj + obj_idx
                obj_mesh_start[base] = len(mesh_parent_link)
                for mesh in link.visual_meshes.values():
                    if mesh._mesh_type != "Mesh":
                        continue
                    centroids = mesh.mesh_face_centroids  # (F, 3) local-unscaled
                    normals = mesh.mesh_face_normals  # (F, 3) local-unscaled
                    face_count = centroids.shape[0]
                    if face_count == 0:
                        continue
                    inv_local_w_scale = th.linalg.inv(link_world_inv_init @ mesh.scaled_transform)
                    mesh_parent_link.append(parent_flat)
                    mesh_inv_local.append(inv_local_w_scale.to(th.float32))
                    mesh_face_start.append(face_cursor)
                    mesh_face_count.append(face_count)
                    face_centroids.append(centroids.to(th.float32))
                    face_normals.append(normals.to(th.float32))
                    face_cursor += face_count
                    obj_mesh_count[base] += 1

        num_meshes = len(mesh_parent_link)
        if num_meshes == 0:
            return
        cls._mesh_parent_link = wp.array(th.tensor(mesh_parent_link, dtype=th.int32), dtype=wp.int32, device="cuda")
        cls._mesh_inv_local_w_scale = wp.array(th.stack(mesh_inv_local), dtype=wp.mat44, device="cuda")
        cls._mesh_face_start = wp.array(th.tensor(mesh_face_start, dtype=th.int32), dtype=wp.int32, device="cuda")
        cls._mesh_face_count = wp.array(th.tensor(mesh_face_count, dtype=th.int32), dtype=wp.int32, device="cuda")
        cls._face_centroid = wp.array(th.cat(face_centroids, dim=0).contiguous(), dtype=wp.vec3, device="cuda")
        cls._face_normal = wp.array(th.cat(face_normals, dim=0).contiguous(), dtype=wp.vec3, device="cuda")
        cls._obj_mesh_start = wp.array(th.tensor(obj_mesh_start, dtype=th.int32), dtype=wp.int32, device="cuda")
        cls._obj_mesh_count = wp.array(th.tensor(obj_mesh_count, dtype=th.int32), dtype=wp.int32, device="cuda")
        cls._world_to_mesh_local = wp.zeros(num_meshes, dtype=wp.mat44, device="cuda")

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None:
            return
        cls.VALUES_WP.zero_()  # counts accumulate via atomic_add; start from zero each step
        if (
            cls._mesh_parent_link is None
            or cls._entry_sys_index is None
            or RigidBodyViewAPI.POSE_MATRICES is None
            or ParticleViewAPI.PARTICLE_POSITIONS is None
        ):
            return
        _, n_obj, _ = values.shape
        num_meshes = cls._mesh_parent_link.shape[0]
        capacity = ParticleViewAPI.PARTICLE_POSITIONS.shape[0]

        # Refresh per-mesh world->local from current link poses.
        wp.launch(
            _compute_mesh_world_to_local_kernel,
            dim=num_meshes,
            inputs=[
                RigidBodyViewAPI.POSE_MATRICES,
                cls._mesh_parent_link,
                cls._mesh_inv_local_w_scale,
                cls._world_to_mesh_local,
            ],
            device="cuda",
        )
        # One thread per (particle slot, container) — launch over the fixed capacity, gate on the count.
        wp.launch(
            _count_contained_particles_kernel,
            dim=(capacity, n_obj),
            inputs=[
                ParticleViewAPI.PARTICLE_POSITIONS,
                ParticleViewAPI.PARTICLE_SCENE_INDEX,
                ParticleViewAPI.PARTICLE_ENTRY_INDEX,
                ParticleViewAPI.VISUAL_PARTICLE_ORIENTATION,
                ParticleViewAPI.PARTICLE_COUNT,
                cls._entry_sys_index,
                cls._entry_is_visual,
                float(m.VISUAL_PARTICLE_OFFSET),
                n_obj,
                cls._obj_mesh_start,
                cls._obj_mesh_count,
                cls._mesh_face_start,
                cls._mesh_face_count,
                cls._world_to_mesh_local,
                cls._face_centroid,
                cls._face_normal,
                cls.VALUES_WP,
            ],
            device="cuda",
        )

    def _compute_positions_in_volume(self, system):
        """Original per-object computation of raw positions + in-volume mask (the getter path).
        Used lazily by ContainedParticlesData.positions / .in_volume.

        TODO(vector): this per-object CPU recompute is still the source of positions/in_volume for the
        rare callers (transition rules, Contains.set_value). Vectorize/optimize it when we tackle
        transition_rules (e.g. have the containment kernel also emit the per-particle in-volume mask so
        these callers stop recomputing on CPU)."""
        raw_positions, checked_positions, in_volume = th.empty(0).reshape(0, 3), th.empty(0).reshape(0, 3), th.empty(0)
        if system.n_particles > 0:
            if self.obj.scene.is_visual_particle_system(system_name=system.name):
                raw_positions, quats = system.get_particles_position_orientation()
                unit_z = th.zeros((len(raw_positions), 3, 1))
                unit_z[:, -1, :] = m.VISUAL_PARTICLE_OFFSET
                checked_positions = (T.quat2mat(quats) @ unit_z).reshape(-1, 3) + raw_positions
            elif self.obj.scene.is_physical_particle_system(system_name=system.name):
                raw_positions = system.get_particles_position_orientation()[0]
                checked_positions = raw_positions
            else:
                raise ValueError(
                    f"Invalid system {system} received for getting ContainedParticles state!"
                    f"Currently, only VisualParticleSystems and PhysicalParticleSystems are supported."
                )
        if len(checked_positions) > 0:
            in_volume = self.link.check_points_in_volume(checked_positions)
        return raw_positions, in_volume

    def _get_value(self, system):
        n_in_volume = int(super()._get_value(system))
        return ContainedParticlesData(n_in_volume=n_in_volume, state=self, system=system)

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()


class Contains(RelativeObjectState, BooleanStateMixin):
    def get_value(self, system):
        # ContainedParticles.VALUES_CPU is the cache. Do not memoize this cheap interpretation
        # separately because that could hide a same-step particle metadata change.
        assert self._initialized
        return self._get_value(system)

    def _get_value(self, system):
        # Grab value from Contains state; True if value is greater than 0
        return self.obj.states[ContainedParticles].get_value(system=system).n_in_volume > 0

    def _set_value(self, system, new_value):
        if new_value:
            # Cannot set contains = True, only False
            raise NotImplementedError(f"{self.__class__.__name__} does not support set_value(system, True)")
        else:
            # Remove all particles from inside the volume
            system.remove_particles(idxs=self.obj.states[ContainedParticles].get_value(system).in_volume.nonzero())

        return True

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ContainedParticles)
        return deps
