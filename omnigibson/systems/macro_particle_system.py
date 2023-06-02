import os
import matplotlib.pyplot as plt
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdPhysics, PhysxSchema
import trimesh

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.systems.system_base import BaseSystem, VisualParticleSystem, PhysicalParticleSystem, REGISTERED_SYSTEMS
from omnigibson.utils.constants import SemanticClass
from omnigibson.utils.python_utils import classproperty, subclass_factory, snake_case_to_camel_case
from omnigibson.utils.sampling_utils import sample_cuboid_on_object_symmetric_bimodal_distribution
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import FlatcacheAPI
from omnigibson.prims.geom_prim import VisualGeomPrim, CollisionVisualGeomPrim
import numpy as np
from scipy.spatial.transform import Rotation as R
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log

# Create module logger
log = create_module_logger(module_name=__name__)


class MacroParticleSystem(BaseSystem):
    """
    Global system for modeling "macro" level particles, e.g.: dirt, dust, etc.
    """
    # Template object to use -- this should be some instance of BasePrim. This will be the
    # object that symbolizes a single particle, and will be duplicated to generate the particle system.
    # Note that this object is NOT part of the actual particle system itself!
    _particle_object = None

    # dict, array of particle objects, mapped by their prim names
    particles = None

    # Counter to increment monotonically as we add more particles
    _particle_counter = None

    # Color associated with this system (NOTE: external queries should call cls.color)
    _color = None

    @classmethod
    def initialize(cls):
        # Run super method first
        super().initialize()

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        cls.particles = dict()
        cls._particle_counter = 0

        # Create the system prim -- this is merely a scope prim
        og.sim.stage.DefinePrim(f"/World/{cls.name}", "Scope")

        # Load the particle template, and make it kinematic only because it's not interacting with anything
        particle_template = cls._create_particle_template()
        og.sim.import_object(obj=particle_template, register=False)
        particle_template.kinematic_only = True

        # Make sure there is no ambiguity about which mesh to use as the particle from this template
        assert len(particle_template.links) == 1, "MacroParticleSystem particle template has more than one link"
        assert len(particle_template.root_link.visual_meshes) == 1, "MacroParticleSystem particle template has more than one visual mesh"

        # Class particle objet is assumed to be the first and only visual mesh belonging to the root link
        template = list(particle_template.root_link.visual_meshes.values())[0]
        template.material.shader_force_populate(render=True)
        cls.set_particle_template_object(obj=template)

    @classproperty
    def particle_idns(cls):
        """
        Returns:
            set: idn of all the particles across all groups.
        """
        return {cls.particle_name2idn(particle_name) for particle_name in cls.particles}

    @classproperty
    def next_available_particle_idn(cls):
        """
        Returns:
            int: the next available particle idn across all groups.
        """
        return cls._particle_counter

    @classmethod
    def _create_particle_template(cls):
        """
        Creates the particle template to be used for this system.

        NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
            visual mesh attached to its root link, since this will be the actual visual mesh used

        Returns:
            EntityPrim: Particle template that will be duplicated when generating future particle groups
        """
        raise NotImplementedError()

    @classmethod
    def remove_all_particles(cls):
        # Use list explicitly to prevent mid-loop mutation of dict
        for particle_name in tuple(cls.particles.keys()):
            cls.remove_particle_by_name(name=particle_name)

    @classmethod
    def reset(cls):
        # Call super first
        super().reset()

        # Reset the particle counter
        cls._particle_counter = 0

    @classmethod
    def clear(cls):
        # Call super first
        super().clear()

        # Clear all internal state
        cls._particle_object = None
        cls.particles = None
        cls.min_scale = None
        cls.max_scale = None
        cls._color = None

    @classproperty
    def n_particles(cls):
        return len(cls.particles)

    @classproperty
    def material(cls):
        return cls._particle_object.material

    @classproperty
    def particle_name_prefix(cls):
        """
        Returns:
            str: Naming prefix used for all generated particles. This is coupled with the unique particle ID to generate
                the full particle name
        """
        return f"{cls.name}Particle"

    @classproperty
    def state_size(cls):
        # In additon to super, we have:
        # scale (3*n), and particle counter (1)
        return super().state_size + 3 * cls.n_particles + 1

    @classmethod
    def _dump_state(cls):
        state = super()._dump_state()

        state["scales"] = np.array([particle.scale for particle in cls.particles.values()])
        state["particle_counter"] = cls._particle_counter

        return state

    @classmethod
    def _load_state(cls, state):
        # Run super first
        super()._load_state(state=state)

        # Set particle scales
        for particle, scale in zip(cls.particles.values(), state["scales"]):
            particle.scale = scale

        # Set particle counter
        cls._particle_counter = state["particle_counter"]

    @classmethod
    def _serialize(cls, state):
        # Run super first
        states_flat = super()._serialize(state=state)

        # Add particle scales, then the template info
        return np.concatenate([
            states_flat,
            state["scales"].flatten(),
            [state["particle_counter"]],
        ], dtype=float)

    @classmethod
    def _deserialize(cls, state):
        # Run super first
        state_dict, idx = super()._deserialize(state=state)

        # Infer how many scales we have, then deserialize
        n_particles = state_dict["n_particles"]
        len_scales = n_particles * 3
        state_dict["scales"] = state[idx:idx+len_scales].reshape(-1, 3)
        state_dict["particle_counter"] = int(state[idx+len_scales])

        return state_dict, idx + len_scales + 1

    @classmethod
    def set_particle_template_object(cls, obj):
        """
        Sets the template particle object that will be used for duplication purposes. Note that this automatically
        adds @obj itself to the ongoing array of particles!

        Args:
            obj (BasePrim): Object to serve as template
        """
        # Update color if it exists and store particle object
        color = np.ones(3)
        if obj.has_material():
            diffuse_texture = obj.material.diffuse_texture
            color = plt.imread(diffuse_texture).mean(axis=(0, 1)) if diffuse_texture else obj.material.diffuse_color_constant
        cls._color = color
        cls._particle_object = obj

    @classmethod
    def add_particle(cls, prim_path, idn=None):
        """
        Adds a particle to this system.

        Args:
            prim_path (str): Absolute path to the newly created particle, minus the name for this particle
            idn (None or int): If specified, should be unique identifier to assign to this particle. If not, will
                automatically generate a new unique one

        Returns:
            XFormPrim: Newly created particle instance, which is added internally as well
        """
        # Generate the new particle
        name = cls.particle_idn2name(idn=cls.next_available_particle_idn if idn is None else idn)
        # Make sure name doesn't already exist
        assert name not in cls.particles.keys(), f"Cannot create particle with name {name} because it already exists!"
        new_particle = cls._load_new_particle(prim_path=f"{prim_path}/{name}", name=name)

        # Make sure the particle is visible
        new_particle.visible = True

        # Track this particle as well
        cls.particles[new_particle.name] = new_particle

        # Increment counter
        cls._particle_counter += 1

        return new_particle

    @classmethod
    def remove_particle_by_name(cls, name):
        assert name in cls.particles, f"Got invalid name for particle to remove {name}"
        particle = cls.particles.pop(name)
        particle.remove()

    @classmethod
    def remove_particles(
            cls,
            idxs,
            **kwargs,
    ):
        particle_names = tuple(cls.particles.keys())
        for idx in idxs:
            cls.remove_particle_by_name(particle_names[idx])

    @classmethod
    def generate_particles(
            cls,
            positions,
            velocities=None,
            orientations=None,
            scales=None,
            **kwargs,
    ):
        # Grab pre-existing tfs
        current_positions, current_orientations = cls.get_particles_position_orientation()

        # Update the tensors
        n_particles = len(positions)
        orientations = R.random(num=n_particles).as_quat() if orientations is None else orientations
        scales = cls.sample_scales(n=n_particles) if scales is None else scales

        positions = np.concatenate([current_positions, positions], axis=0)
        orientations = np.concatenate([current_orientations, orientations], axis=0)

        # Add particles
        for scale in scales:
            particle = cls.add_particle(prim_path=f"{cls.prim_path}/particles")
            particle.scale = scale

        # Set the tfs
        cls.set_particles_position_orientation(positions=positions, orientations=orientations)

    @classmethod
    def _load_new_particle(cls, prim_path, name):
        """
        Loads a new particle into the current stage, leveraging @cls._particle_object as a template for the new particle
        to load. This function should be implemented by any subclasses.

        Args:
            prim_path (str): The absolute stage path at which to create the new particle
            name (str): The name to assign to this new particle at the path

        Returns:
            XFormPrim: Loaded particle
        """
        raise NotImplementedError()

    @classmethod
    def particle_name2idn(cls, name):
        """
        Args:
            name (str): Particle name to grab its corresponding unique id number for

        Returns:
            int: Unique ID assigned to the particle based on its name
        """
        assert cls.particle_name_prefix in name, \
            f"Particle name should have '{cls.particle_name_prefix}' in it when checking ID! Got: {name}"
        return int(name.split(cls.particle_name_prefix)[-1])

    @classmethod
    def particle_idn2name(cls, idn):
        """
        Args:
            idn (int): Unique ID number assigned to the particle to grab the name for

        Returns:
            str: Particle name corresponding to its unique id number
        """
        assert isinstance(idn, int), \
            f"Particle idn must be an integer when checking name! Got: {idn}. Type: {type(idn)}"
        return f"{cls.particle_name_prefix}{idn}"

    @classproperty
    def color(cls):
        return np.array(cls._color)


class MacroVisualParticleSystem(MacroParticleSystem, VisualParticleSystem):
    """
    Particle system class that procedurally generates individual particles that are not subject to physics
    """
    # Maps particle name to dict of {obj, link}
    _particles_info = None

    # Pre-cached information about visual particles so that we have efficient runtime computations
    # Maps particle name to local pose matrix for computing global poses for the particle
    _particles_local_mat = None

    # Default behavior for this class -- whether to clip generated particles halfway into objects when sampling
    # their locations on the surface of the given object
    _CLIP_INTO_OBJECTS = False

    # Default parameters for sampling particle locations
    # See omnigibson/utils/sampling_utils.py for how they are used.
    _SAMPLING_AXIS_PROBABILITIES = (0.25, 0.25, 0.5)
    _SAMPLING_AABB_OFFSET = 0.01
    _SAMPLING_BIMODAL_MEAN_FRACTION = 0.9
    _SAMPLING_BIMODAL_STDEV_FRACTION = 0.2
    _SAMPLING_MAX_ATTEMPTS = 20

    @classmethod
    def initialize(cls):
        # Run super method first
        super().initialize()

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        cls._particles_info = dict()
        cls._particles_local_mat = dict()

    @classproperty
    def particle_object(cls):
        return cls._particle_object

    @classmethod
    def _load_new_particle(cls, prim_path, name):
        # We copy the template prim and generate the new object if the prim doesn't already exist, otherwise we
        # reference the pre-existing one
        if not get_prim_at_path(prim_path):
            omni.kit.commands.execute(
                "CopyPrim",
                path_from=cls._particle_object.prim_path,
                path_to=prim_path,
            )
        return VisualGeomPrim(prim_path=prim_path, name=name)

    @classmethod
    def set_particle_template_object(cls, obj):
        # Sanity check to make sure the added object is an instance of VisualGeomPrim
        assert isinstance(obj, VisualGeomPrim), \
            f"Particle template object for {cls.name} must be a VisualGeomPrim instance!"

        # Run super method
        super().set_particle_template_object(obj=obj)

    @classmethod
    def clear(cls):
        # Run super method first
        super().clear()

        # Clear all groups as well
        cls._particles_info = dict()
        cls._particles_local_mat = dict()

    @classmethod
    def remove_particle_by_name(cls, name):
        # Run super first
        super().remove_particle_by_name(name=name)

        # Remove this particle from its respective group as well
        cls._group_particles[cls._particles_info[name]["obj"].name].pop(name)
        cls._particles_info.pop(name)
        cls._particles_local_mat.pop(name)

    @classmethod
    def generate_group_particles(
            cls,
            group,
            positions,
            orientations=None,
            scales=None,
            link_prim_paths=None,
    ):
        # Make sure the group exists
        cls._validate_group(group=group)

        # Standardize orientations and links
        obj = cls._group_objects[group]
        n_particles = positions.shape[0]
        if orientations is None:
            orientations = np.zeros((n_particles, 4))
            orientations[:, -1] = 1.0
        link_prim_paths = [obj.root_link.prim_path] * n_particles if link_prim_paths is None else link_prim_paths

        scales = cls.sample_scales_by_group(group=group, n=n_particles) if scales is None else scales
        bbox_extents_local = [(cls._particle_object.aabb_extent * scale).tolist() for scale in scales]

        # If we're using flatcache, we need to update the object's pose on the USD manually
        if gm.ENABLE_FLATCACHE:
            FlatcacheAPI.sync_raw_object_transforms_in_usd(prim=obj)

        # Generate particles
        z_up = np.zeros((3, 1))
        z_up[-1] = 1.0
        for position, orientation, scale, bbox_extent_local, link_prim_path in \
                zip(positions, orientations, scales, bbox_extents_local, link_prim_paths):
            link_name = link_prim_path.split("/")[-1]
            link = obj.links[link_name]
            # Possibly shift the particle slightly away from the object if we're not clipping into objects
            if cls._CLIP_INTO_OBJECTS:
                # Shift the particle halfway down
                base_to_center = bbox_extent_local[2] / 2.0
                normal = (T.quat2mat(orientation) @ z_up).flatten()
                position -= normal * base_to_center

            # Create particle
            particle = cls.add_particle(prim_path=link_prim_path)

            # Add to group
            cls._group_particles[group][particle.name] = particle
            cls._particles_info[particle.name] = dict(obj=cls._group_objects[group], link=link)

            # Set the pose and scale
            cls.set_particle_position_orientation(idx=-1, position=position, orientation=orientation)
            particle.scale = scale

    @classmethod
    def generate_group_particles_on_object(cls, group, max_samples, min_samples_for_success=1):
        assert max_samples >= min_samples_for_success, "number of particles to sample should exceed the min for success"

        # Make sure the group exists
        cls._validate_group(group=group)

        # Remove all stale particles
        cls.remove_all_group_particles(group=group)

        # Generate requested number of particles
        obj = cls._group_objects[group]

        # Sample scales and corresponding bbox extents
        scales = cls.sample_scales_by_group(group=group, n=max_samples)
        # For sampling particle positions, we need the global bbox extents, NOT the local extents
        # which is what we would get naively if we directly use @scales
        avg_scale = np.cbrt(np.product(obj.scale))
        bbox_extents_global = scales * cls._particle_object.aabb_extent.reshape(1, 3) * avg_scale

        # Sample locations for all particles
        # TODO: Does simulation need to play at this point in time? Answer: yes
        results = sample_cuboid_on_object_symmetric_bimodal_distribution(
            obj=obj,
            num_samples=max_samples,
            cuboid_dimensions=bbox_extents_global,
            bimodal_mean_fraction=cls._SAMPLING_BIMODAL_MEAN_FRACTION,
            bimodal_stdev_fraction=cls._SAMPLING_BIMODAL_STDEV_FRACTION,
            axis_probabilities=cls._SAMPLING_AXIS_PROBABILITIES,
            undo_cuboid_bottom_padding=True,
            verify_cuboid_empty=False,
            aabb_offset=cls._SAMPLING_AABB_OFFSET,
            max_sampling_attempts=cls._SAMPLING_MAX_ATTEMPTS,
            refuse_downwards=True,
        )

        # Use sampled points
        positions, orientations, particle_scales, link_prim_paths = [], [], [], []
        for result, scale in zip(results, scales):
            position, normal, quaternion, hit_link, reasons = result
            if position is not None:
                positions.append(position)
                orientations.append(quaternion)
                particle_scales.append(scale)
                link_prim_paths.append(hit_link)

        success = len(positions) >= min_samples_for_success
        # If we generated a sufficient number of points, generate them in the simulator
        if success:
            cls.generate_group_particles(
                group=group,
                positions=np.array(positions),
                orientations=np.array(orientations),
                scales=np.array(scales),
                link_prim_paths=link_prim_paths,
            )

        return success

    @classmethod
    def _compute_batch_particles_position_orientation(cls, particles, local=False):
        """
        Computes all @particles' positions and orientations

        Args:
            particles (Iterable of str): Names of particles to compute batched position orientation for
            local (bool): Whether to compute particles' poses in local frame or not

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        n_particles = len(particles)
        if local:
            global_poses = np.zeros((n_particles, 4, 4))
            for i, name in enumerate(particles):
                global_poses[i] = T.pose2mat(cls.particles[name].get_local_pose())
        else:
            # Iterate over all particles and compute link tfs programmatically, then batch the matrix transform
            link_tfs = dict()
            link_tfs_batch = np.zeros((n_particles, 4, 4))
            particle_local_poses_batch = np.zeros_like(link_tfs_batch)
            for i, name in enumerate(particles):
                link = cls._particles_info[name]["link"]
                if link not in link_tfs:
                    link_tfs[link] = T.pose2mat(link.get_position_orientation())
                link_tfs_batch[i] = link_tfs[link]
                particle_local_poses_batch[i] = cls._particles_local_mat[name]

            # Compute once
            global_poses = np.matmul(link_tfs_batch, particle_local_poses_batch)

        # Decompose back into positions and orientations
        return global_poses[:, :3, 3], T.mat2quat(global_poses[:, :3, :3])

    @classmethod
    def get_particles_position_orientation(cls):
        return cls._compute_batch_particles_position_orientation(particles=cls.particles, local=False)

    @classmethod
    def get_particles_local_pose(cls):
        return cls._compute_batch_particles_position_orientation(particles=cls.particles, local=True)

    @classmethod
    def get_group_particles_position_orientation(cls, group):
        return cls._compute_batch_particles_position_orientation(particles=cls._group_particles[group], local=False)

    @classmethod
    def get_group_particles_local_pose(cls, group):
        return cls._compute_batch_particles_position_orientation(particles=cls._group_particles[group], local=True)

    @classmethod
    def get_particle_position_orientation(cls, idx):
        name = list(cls.particles.keys())[idx]
        # First, get local pose, scale it by the parent link's scale, and then convert into a matrix
        parent_link = cls._particles_info[name]["link"]
        local_mat = cls._particles_local_mat[name]
        link_tf = T.pose2mat(parent_link.get_position_orientation())

        # Multiply the local pose by the link's global transform, then return as pos, quat tuple
        return T.mat2pose(link_tf @ local_mat)

    @classmethod
    def get_particle_local_pose(cls, idx):
        name = list(cls.particles.keys())[idx]
        return cls.particles[name].get_local_pose()

    @classmethod
    def _modify_batch_particles_position_orientation(cls, particles, positions=None, orientations=None, local=False):
        """
        Modifies all @particles' positions and orientations with @positions and @orientations

        Args:
            particles (Iterable of str): Names of particles to compute batched position orientation for
            local (bool): Whether to set particles' poses in local frame or not

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        if positions is None or orientations is None:
            pos, ori = cls._compute_batch_particles_position_orientation(particles=particles, local=local)
            positions = pos if positions is None else positions
            orientations = ori if orientations is None else orientations
        lens = np.array([len(particles), len(positions), len(orientations)])
        assert lens.min() == lens.max(), "Got mismatched particles, positions, and orientations!"

        particle_local_poses_batch = np.zeros((cls.n_particles, 4, 4))
        particle_local_poses_batch[:, -1, -1] = 1.0
        particle_local_poses_batch[:, :3, 3] = positions
        particle_local_poses_batch[:, :3, :3] = T.quat2mat(orientations)

        if not local:
            # Iterate over all particles and compute link tfs programmatically, then batch the matrix transform
            link_tfs = dict()
            link_tfs_batch = np.zeros((cls.n_particles, 4, 4))
            for i, name in enumerate(particles):
                link = cls._particles_info[name]["link"]
                if link in link_tfs:
                    link_tf = link_tfs[link]
                else:
                    link_tf = T.pose2mat(link.get_position_orientation())
                    link_tfs[link] = link_tf
                link_tfs_batch[i] = link_tf

            particle_local_poses_batch = np.matmul(np.linalg.inv(link_tfs_batch), particle_local_poses_batch)

        for i, name in enumerate(particles):
            cls._modify_particle_local_mat(name=name, mat=particle_local_poses_batch[i], ignore_scale=local)

    @classmethod
    def set_particles_position_orientation(cls, positions=None, orientations=None):
        return cls._modify_batch_particles_position_orientation(particles=cls.particles, positions=positions, orientations=orientations, local=False)

    @classmethod
    def set_particles_local_pose(cls, positions=None, orientations=None):
        return cls._modify_batch_particles_position_orientation(particles=cls.particles, positions=positions, orientations=orientations, local=True)

    @classmethod
    def set_group_particles_position_orientation(cls, group, positions=None, orientations=None):
        return cls._modify_batch_particles_position_orientation(particles=cls._group_particles[group], positions=positions, orientations=orientations, local=False)

    @classmethod
    def set_group_particles_local_pose(cls, group, positions=None, orientations=None):
        return cls._modify_batch_particles_position_orientation(particles=cls._group_particles[group], positions=positions, orientations=orientations, local=True)

    @classmethod
    def set_particle_position_orientation(cls, idx, position=None, orientation=None):
        if position is None or orientation is None:
            pos, ori = cls.get_particle_position_orientation(idx=idx)
            position = pos if position is None else position
            orientation = ori if orientation is None else orientation

        name = list(cls.particles.keys())[idx]
        global_mat = np.zeros((4, 4))
        global_mat[-1, -1] = 1.0
        global_mat[:3, 3] = position
        global_mat[:3, :3] = T.quat2mat(orientation)
        # First, get global pose, scale it by the parent link's scale, and then convert into a matrix
        parent_link = cls._particles_info[name]["link"]
        link_tf = T.pose2mat(parent_link.get_position_orientation())
        local_mat = np.linalg.inv(link_tf) @ global_mat

        cls._modify_particle_local_mat(name=name, mat=local_mat, ignore_scale=False)

    @classmethod
    def set_particle_local_pose(cls, idx, position=None, orientation=None):
        if position is None or orientation is None:
            pos, ori = cls.get_particle_local_pose(idx=idx)
            position = pos if position is None else position
            orientation = ori if orientation is None else orientation

        name = list(cls.particles.keys())[idx]
        local_mat = np.zeros((4, 4))
        local_mat[-1, -1] = 1.0
        local_mat[:3, 3] = position
        local_mat[:3, :3] = T.quat2mat(orientation)
        cls._modify_particle_local_mat(name=name, mat=local_mat, ignore_scale=True)

    @classmethod
    def _compute_particle_local_mat(cls, name, ignore_scale=False):
        """
        Computes particle @name's local transform as a homogeneous 4x4 matrix

        Args:
            name (str): Name of the particle to compute local transform matrix for
            ignore_scale (bool): Whether to ignore the parent_link scale when computing the local transform

        Returns:
            np.array: (4, 4) homogeneous transform matrix
        """
        particle = cls.particles[name]
        parent_link = cls._particles_info[name]["link"]
        local_pos, local_quat = particle.get_local_pose()
        local_pos = local_pos if ignore_scale else local_pos * parent_link.scale
        return T.pose2mat((local_pos, local_quat))

    @classmethod
    def _modify_particle_local_mat(cls, name, mat, ignore_scale=False):
        """
        Sets particle @name's local transform as a homogeneous 4x4 matrix

        Args:
            name (str): Name of the particle to compute local transform matrix for
            mat (n-array): (4, 4) homogeneous transform matrix
            ignore_scale (bool): Whether to ignore the parent_link scale when setting the local transform
        """
        particle = cls.particles[name]
        parent_link = cls._particles_info[name]["link"]
        local_pos, local_quat = T.mat2pose(mat)
        local_pos = local_pos if ignore_scale else local_pos / parent_link.scale
        particle.set_local_pose(local_pos, local_quat)

        # Store updated value
        cls._particles_local_mat[name] = mat

    @classmethod
    def _sync_particle_groups(cls, group_objects, particle_idns, particle_attached_link_names):
        """
        Synchronizes the particle groups based on desired identification numbers @group_idns

        Args:
            group_objects (list of None or BaseObject): Desired unique group objects that should be active for
            this particle system. Any objects that aren't found will be skipped over
            particle_idns (list of list of int): Per-group unique id numbers for the particles assigned to that group.
                List should be same length as @group_idns with sub-entries corresponding to the desired number of
                particles assigned to that group
            particle_attached_link_names (list of list of str): Per-group link names corresponding to the specific
                links each particle is attached for each group. List should be same length as @group_idns with
                sub-entries corresponding to the desired number of particles assigned to that group
        """
        # We have to be careful here -- some particle groups may have been deleted / are mismatched, so we need
        # to update accordingly, potentially deleting stale groups and creating new groups as needed
        name_to_info_mapping = {obj.name: {
            "n_particles": len(p_idns),
            "particle_idns": p_idns,
            "link_names": link_names,
        }
            for obj, p_idns, link_names in
            zip(group_objects, particle_idns, particle_attached_link_names)}

        current_group_names = cls.groups
        desired_group_names = set(obj.name for obj in group_objects)
        groups_to_delete = current_group_names - desired_group_names
        groups_to_create = desired_group_names - current_group_names
        common_groups = current_group_names.intersection(desired_group_names)

        # Sanity check the common groups, we will recreate any where there is a mismatch
        for name in common_groups:
            info = name_to_info_mapping[name]
            if cls.num_group_particles(group=name) != info["n_particles"]:
                log.debug(f"Got mismatch in particle group {name} when syncing, "
                                f"deleting and recreating group now.")
                # Add this group to both the delete and creation pile
                groups_to_delete.add(name)
                groups_to_create.add(name)

        # Delete any groups we no longer want
        for name in groups_to_delete:
            cls.remove_attachment_group(group=name)

        # Create any groups we don't already have
        for name in groups_to_create:
            obj = og.sim.scene.object_registry("name", name)
            info = name_to_info_mapping[name]
            cls.create_attachment_group(obj=obj)

            for particle_idn, link_name in zip(info["particle_idns"], info["link_names"]):
                # Create the necessary particles
                particle = cls.add_particle(prim_path=f"{obj.prim_path}/{link_name}", idn=int(particle_idn))
                cls._group_particles[name][particle.name] = particle
                cls._particles_info[particle.name] = dict(obj=obj, link=obj.links[link_name])

    @classmethod
    def create(cls, name, create_particle_template, min_scale=None, max_scale=None, scale_relative_to_parent=False, **kwargs):
        """
        Utility function to programmatically generate monolithic visual particle system classes.

        Note: If using super() calls in any functions, we have to use slightly esoteric syntax in order to
        accommodate this procedural method for using super calls
        cf. https://stackoverflow.com/questions/22403897/what-does-it-mean-by-the-super-object-returned-is-unbound-in-python
            Use: super(cls).__get__(cls).<METHOD_NAME>(<KWARGS>)

        Args:
            name (str): Name of the visual particles, in snake case.
            min_scale (None or 3-array): If specified, sets the minumum bound for the visual particles' relative scale.
                Else, defaults to 1
            max_scale (None or 3-array): If specified, sets the maximum bound for the visual particles' relative scale.
                Else, defaults to 1
            scale_relative_to_parent (bool): If True, will scale generated particles relative to the corresponding
                group's object
            create_particle_template (function): Method for generating the visual particle template that will be duplicated
                when generating groups of particles.
                Expected signature:

                create_particle_template(prim_path: str, name: str) --> EntityPrim

                where @prim_path and @name are the parameters to assign to the generated EntityPrim.
                NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
                    visual mesh attached to its root link, since this will be the actual visual mesh used

            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)

        Returns:
            VisualParticleSystem: Generated visual particle system class
        """
        # Override the necessary parameters
        @classproperty
        def cp_register_system(cls):
            # We should register this system since it's an "actual" system (not an intermediate class)
            return True

        @classproperty
        def cp_scale_relative_to_parent(cls):
            return scale_relative_to_parent

        @classmethod
        def cm_initialize(cls):
            # Potentially override the min / max scales
            if min_scale is not None:
                cls.min_scale = np.array(min_scale)
            if max_scale is not None:
                cls.max_scale = np.array(max_scale)

            # Run super (we have to use a bit esoteric syntax in order to accommodate this procedural method for
            # using super calls -- cf. https://stackoverflow.com/questions/22403897/what-does-it-mean-by-the-super-object-returned-is-unbound-in-python
            super(cls).__get__(cls).initialize()

        @classmethod
        def cm_create_particle_template(cls):
            return create_particle_template(prim_path=f"{cls.prim_path}/template", name=f"{cls.name}_template")

        # Add to any other params specified
        kwargs["_register_system"] = cp_register_system
        kwargs["scale_relative_to_parent"] = cp_scale_relative_to_parent
        kwargs["initialize"] = cm_initialize
        kwargs["_create_particle_template"] = cm_create_particle_template

        # Run super
        return super().create(name=name, **kwargs)

    @classmethod
    def _dump_state(cls):
        state = super()._dump_state()

        # Add in per-group information
        groups_dict = dict()
        for group_name, group_particles in cls._group_particles.items():
            groups_dict[group_name] = dict(
                particle_attached_obj_uuid=cls._group_objects[group_name].uuid,
                n_particles=cls.num_group_particles(group=group_name),
                particle_idns=[cls.particle_name2idn(name=name) for name in group_particles.keys()],
                particle_attached_link_names=[cls._particles_info[name]["link"].prim_path.split("/")[-1] for name in group_particles.keys()],
            )

        state["n_groups"] = len(cls._group_particles)
        state["groups"] = groups_dict

        return state

    @classmethod
    def _load_state(cls, state):
        # First, we sync our particle systems
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (dict): Keyword-mapped states of this object to set
        """
        # Synchronize particle groups
        cls._sync_particle_groups(
            group_objects=[og.sim.scene.object_registry("uuid", info["particle_attached_obj_uuid"])
                           for info in state["groups"].values()],
            particle_idns=[info["particle_idns"] for info in state["groups"].values()],
            particle_attached_link_names=[info["particle_attached_link_names"] for info in state["groups"].values()],
        )

        # Run super
        super()._load_state(state=state)

    @classmethod
    def _serialize(cls, state):
        # Run super first
        state_flat = super()._serialize(state=state)

        groups_dict = state["groups"]
        state_group_flat = [[state["n_groups"]]]
        for group_name, group_dict in groups_dict.items():
            group_obj_link2id = {link_name: i for i, link_name in enumerate(cls._group_objects[group_name].links.keys())}
            state_group_flat += [
                [group_dict["particle_attached_obj_uuid"]],
                [group_dict["n_particles"]],
                group_dict["particle_idns"],
                [group_obj_link2id[link_name] for link_name in group_dict["particle_attached_link_names"]],
            ]

        return np.concatenate([*state_group_flat, state_flat]).astype(float)

    @classmethod
    def _deserialize(cls, state):
        # Synchronize the particle groups
        n_groups = int(state[0])
        groups_dict = dict()
        group_objs = []
        # Index starts at 1 because index 0 is n_groups
        idx = 1
        for i in range(n_groups):
            obj_uuid, n_particles = int(state[idx]), int(state[idx + 1])
            obj = og.sim.scene.object_registry("uuid", obj_uuid)
            group_obj_id2link = {i: link_name for i, link_name in enumerate(obj.links.keys())}
            group_objs.append(obj)
            groups_dict[obj.name] = dict(
                particle_attached_obj_uuid=obj_uuid,
                n_particles=n_particles,
                particle_idns=[int(idn) for idn in state[idx + 2 : idx + 2 + n_particles]], # Idx + 2 because the first two are obj_uuid and n_particles
                particle_attached_link_names=[group_obj_id2link[int(idn)] for idn in state[idx + 2 + n_particles : idx + 2 + n_particles * 2]],
            )
            idx += 2 + n_particles * 2
        log.debug(f"Syncing {cls.name} particles with {n_groups} groups..")
        cls._sync_particle_groups(
            group_objects=group_objs,
            particle_idns=[group_info["particle_idns"] for group_info in groups_dict.values()],
            particle_attached_link_names=[group_info["particle_attached_link_names"] for group_info in groups_dict.values()],
        )

        # Get super method
        state_dict, idx_super = super()._deserialize(state=state[idx:])
        state_dict["groups"] = groups_dict

        return state_dict, idx + idx_super


class MacroPhysicalParticleSystem(PhysicalParticleSystem, MacroParticleSystem):
    """
    Particle system class that procedurally generates individual particles that are subject to physics
    """
    # Physics rigid body view for keeping track of all particles' state
    particles_view = None

    # Approximate radius of the macro particle, and distance from particle frame to approximate center
    _particle_radius = None
    _particle_offset = None

    @classmethod
    def initialize(cls):
        # Run super method first
        super().initialize()

        # Create the particles head prim -- this is merely a scope prim
        og.sim.stage.DefinePrim(f"{cls.prim_path}/particles", "Scope")

        # A new view needs to be created every time once sim is playing, so we add a callback now
        og.sim.add_callback_on_play(name=f"{cls.name}_particles_view", callback=cls._refresh_particles_view)

        # If sim is already playing, refresh particles immediately
        if og.sim.is_playing():
            cls._refresh_particles_view()

    @classmethod
    def _load_new_particle(cls, prim_path, name):
        # We copy the template prim and generate the new object if the prim doesn't already exist, otherwise we
        # reference the pre-existing one
        if not get_prim_at_path(prim_path):
            omni.kit.commands.execute(
                "CopyPrim",
                path_from=cls._particle_object.prim_path,
                path_to=prim_path,
            )
            # Apply RigidBodyAPI to it so it is subject to physics
            prim = get_prim_at_path(prim_path)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            # PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        return CollisionVisualGeomPrim(prim_path=prim_path, name=name)

    @classmethod
    def set_particle_template_object(cls, obj):
        # Run super method
        super().set_particle_template_object(obj=obj)

        # Make sure min and max scale are identical
        assert np.all(cls.min_scale == cls.max_scale), \
            "Min and max scale should be identical for MacroPhysicalParticleSystem!"

        # Compute particle radius
        vertices = np.array(cls._particle_object.get_attribute("points")) * cls.max_scale.reshape(1, 3)
        cls._particle_offset, cls._particle_radius = trimesh.nsphere.minimum_nsphere(trimesh.Trimesh(vertices=vertices))

    @classmethod
    def _refresh_particles_view(cls):
        """
        Internal helper method to refresh the particles' rigid body view to grab state

        Should be called every time sim.play() is called
        """
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            cls.particles_view = og.sim.physics_sim_view.create_rigid_body_view(pattern=f"{cls.prim_path}/particles/*")

    @classmethod
    def clear(cls):
        # Run super method first
        super().clear()

        # Clear internal variables
        cls.particles_view = None

    @classmethod
    def remove_particle_by_name(cls, name):
        # Run super first
        super().remove_particle_by_name(name=name)

        # Refresh particles view
        cls._refresh_particles_view()

    @classmethod
    def add_particle(cls, prim_path, idn=None):
        # Run super first
        super().add_particle(prim_path=prim_path, idn=idn)

        # Refresh particles view
        cls._refresh_particles_view()

    @classmethod
    def get_particles_position_orientation(cls):
        if cls.n_particles > 0:
            tfs = cls.particles_view.get_transforms()
            pos, ori = tfs[:, :3], tfs[:, 3:]
            pos = pos + T.quat2mat(ori) @ cls._particle_offset
        else:
            pos, ori = np.array([]).reshape(0, 3), np.array([]).reshape(0, 4)
        return pos, ori

    @classmethod
    def get_particles_local_pose(cls):
        return cls.get_particles_position_orientation()

    @classmethod
    def get_particle_position_orientation(cls, idx):
        positions, orientations = cls.get_particles_position_orientation()
        return (positions[idx], orientations[idx]) if cls.n_particles > 0 else (positions, orientations)

    @classmethod
    def get_particle_local_pose(cls, idx):
        return cls.get_particle_position_orientation(idx=idx)

    @classmethod
    def set_particles_position_orientation(cls, positions=None, orientations=None):
        if positions is None or orientations is None:
            pos, ori = cls.get_particles_position_orientation()
            orientations = ori if orientations is None else orientations
            positions = pos if positions is None else (positions - T.quat2mat(orientations) @ cls._particle_offset)
        cls.particles_view.set_transforms(np.concatenate([positions, orientations], axis=1), indices=np.arange(len(positions)))

    @classmethod
    def set_particles_local_pose(cls, positions=None, orientations=None):
        cls.set_particles_position_orientation(positions=positions, orientations=orientations)

    @classmethod
    def set_particle_position_orientation(cls, idx, position=None, orientation=None):
        if position is None or orientation is None:
            pos, ori = cls.get_particle_position_orientation(idx=idx)
            orientation = ori if orientation is None else orientation
            position = pos if position is None else (position - T.quat2mat(orientation) @ cls._particle_offset)
        cls.particles_view.set_transforms(np.concatenate([position, orientation]).reshape(1, -1), indices=np.array([idx]))

    @classmethod
    def set_particle_local_pose(cls, idx, position=None, orientation=None):
        cls.set_particle_position_orientation(idx=idx, position=position, orientation=orientation)

    @classmethod
    def get_particles_velocities(cls):
        """
        Grab particles' global linear and angular velocities

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x, y, z) linear velocities in the world frame
                - (n, 3)-array: per-particle (ax, ay, az) angular velocities in the world frame
        """
        if cls.n_particles > 0:
            vels = cls.particles_view.get_velocities()
            lin_vel, ang_vel = vels[:, :3], vels[:, 3:]
        else:
            lin_vel, ang_vel = np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        return lin_vel, ang_vel

    @classmethod
    def get_particle_velocities(cls, idx):
        """
        Grab particle @idx's global linear and angular velocities

        Returns:
            2-tuple:
                - 3-array: particle (x, y, z) linear velocity in the world frame
                - 3-array: particle (ax, ay, az) angular velocity in the world frame
        """
        lin_vel, ang_vel = cls.get_particles_velocities()
        return (lin_vel[idx], ang_vel[idx]) if cls.n_particles > 0 else lin_vel, ang_vel

    @classmethod
    def set_particles_velocities(cls, lin_vels=None, ang_vels=None):
        if lin_vels is None or ang_vels is None:
            l_vels, a_vels = cls.get_particles_velocities()
            lin_vels = l_vels if lin_vels is None else lin_vels
            ang_vels = a_vels if ang_vels is None else ang_vels
        cls.particles_view.set_velocities(np.concatenate([lin_vels, ang_vels], axis=1), indices=np.arange(len(lin_vels)))

    @classmethod
    def set_particle_velocities(cls, idx, lin_vel=None, ang_vel=None):
        if lin_vel is None or ang_vel is None:
            l_vel, a_vel = cls.get_particles_velocities()
            lin_vel = l_vel if lin_vel is None else lin_vel
            ang_vel = a_vel if ang_vel is None else ang_vel
        cls.particles_view.set_velocities(np.concatenate([lin_vel, ang_vel]).reshape(1, -1), indices=np.array([idx]))

    @classproperty
    def particle_radius(cls):
        return cls._particle_radius

    @classproperty
    def particle_contact_radius(cls):
        # This is simply the normal radius
        return cls.particle_radius

    @classmethod
    def generate_particles(
            cls,
            positions,
            velocities=None,
            angular_velocities=None,
            orientations=None,
            scales=None,
            **kwargs,
    ):
        """
        Generates new particles

        Args:
            positions (np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions
            velocities (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
                If not specified, all will be set to 0
            angular_velocities (None or np.array): (n_particles, 3) shaped array specifying per-particle (ax,ay,az)
                angular velocities. If not specified, all will be set to 0
            orientations (None or np.array): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be sampled randomly
            scales (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (cls.min_scale, cls.max_scale)
            **kwargs (dict): Any additional keyword-specific arguments required by subclass implementation
        """
        # Call super first
        super().generate_particles(
            positions=positions,
            velocities=velocities,
            orientations=orientations,
            scales=scales,
            **kwargs,
        )

        # Grab pre-existing vels -- note that this already includes the newly included particles, so we will only
        # keep the first (N - n_new) values
        current_lin_vels, current_ang_vels = cls.get_particles_velocities()

        # Update the tensors
        n_particles = len(positions)
        velocities = np.zeros((n_particles, 3)) if velocities is None else velocities
        angular_velocities = np.zeros_like(velocities) if angular_velocities is None else angular_velocities

        velocities = np.concatenate([current_lin_vels[:-n_particles], velocities], axis=0)
        angular_velocities = np.concatenate([current_ang_vels[:-n_particles], angular_velocities], axis=0)

        # Set the vels
        cls.set_particles_velocities(lin_vels=velocities, ang_vels=angular_velocities)

    @classmethod
    def create(cls, name, create_particle_template, particle_density, scale, **kwargs):
        """
        Utility function to programmatically generate monolithic visual particle system classes.

        Note: If using super() calls in any functions, we have to use slightly esoteric syntax in order to
        accommodate this procedural method for using super calls
        cf. https://stackoverflow.com/questions/22403897/what-does-it-mean-by-the-super-object-returned-is-unbound-in-python
            Use: super(cls).__get__(cls).<METHOD_NAME>(<KWARGS>)

        Note: Scale is automatically inferred from the particle object that is loaded!

        Args:
            name (str): Name of the macro physical particles, in snake case.
            particle_density (float): Particle density for the generated system
            create_particle_template (function): Method for generating the visual particle template that will be duplicated
                when generating groups of particles.
                Expected signature:

                create_particle_template(prim_path: str, name: str) --> EntityPrim

                where @prim_path and @name are the parameters to assign to the generated EntityPrim.
                NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
                    visual mesh attached to its root link, since this will be the actual mesh used for duplication
            scale (None or 3-array): If specified, sets the scaling factor for the particles' relative scale.
                Else, defaults to 1

            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)

        Returns:
            VisualParticleSystem: Generated visual particle system class
        """
        # Override the necessary parameters
        @classproperty
        def cp_register_system(cls):
            # We should register this system since it's an "actual" system (not an intermediate class)
            return True

        @classproperty
        def cp_particle_density(cls):
            return particle_density

        @classmethod
        def cm_initialize(cls):
            # Potentially override the min / max scales
            if scale is not None:
                cls.min_scale = np.array(scale)
                cls.max_scale = np.array(scale)

            # Run super (we have to use a bit esoteric syntax in order to accommodate this procedural method for
            # using super calls -- cf. https://stackoverflow.com/questions/22403897/what-does-it-mean-by-the-super-object-returned-is-unbound-in-python
            super(cls).__get__(cls).initialize()

        @classmethod
        def cm_create_particle_template(cls):
            return create_particle_template(prim_path=f"{cls.prim_path}/template", name=f"{cls.name}_template")

        # Add to any other params specified
        kwargs["_register_system"] = cp_register_system
        kwargs["particle_density"] = cp_particle_density
        kwargs["initialize"] = cm_initialize
        kwargs["_create_particle_template"] = cm_create_particle_template

        # Run super
        return super().create(name=name, **kwargs)

    @classproperty
    def state_size(cls):
        # In additon to super, we have:
        # velocities (6*n)
        return super().state_size + 6 * cls.n_particles

    @classmethod
    def _dump_state(cls):
        state = super()._dump_state()

        # Store all particles' velocities as well
        state["lin_velocities"], state["ang_velocities"] = cls.get_particles_velocities()

        return state

    @classmethod
    def _load_state(cls, state):
        super()._load_state(state=state)

        # Make sure we update all the velocities
        cls.set_particles_velocities(state["lin_velocities"], state["ang_velocities"])

    @classmethod
    def _serialize(cls, state):
        # Run super first
        state_flat = super()._serialize(state=state)

        # Add velocities
        return np.concatenate([state_flat, state["lin_velocities"].flatten(), state["ang_velocities"].flatten()], dtype=float)

    @classmethod
    def _deserialize(cls, state):
        # Run super first
        state_dict, idx = super()._deserialize(state=state)

        # Deserialize velocities
        len_velocities = 3 * state_dict["n_particles"]
        for vel in ("lin_velocities", "ang_velocities"):
            state_dict[vel] = state[idx:idx+len_velocities].reshape(-1, 3)
            idx += len_velocities

        return state_dict, idx


MacroVisualParticleSystem.create(
    name="dust",
    scale_relative_to_parent=False,
    create_particle_template=lambda prim_path, name: og.objects.PrimitiveObject(
        prim_path=prim_path,
        primitive_type="Cube",
        name=name,
        class_id=SemanticClass.DIRT,
        size=0.01,
        rgba=[0.2, 0.2, 0.1, 1.0],
        visible=False,
        fixed_base=False,
        visual_only=True,
        include_default_states=False,
    )
)


MacroVisualParticleSystem.create(
    name="stain",
    scale_relative_to_parent=True,
    create_particle_template=lambda prim_path, name: og.objects.USDObject(
        prim_path=prim_path,
        usd_path=os.path.join(gm.ASSET_PATH, "models", "stain", "stain.usd"),
        name=name,
        class_id=SemanticClass.DIRT,
        visible=False,
        fixed_base=False,
        visual_only=True,
        include_default_states=False,
    ),
)


MacroPhysicalParticleSystem.create(
    name="raspberry",
    particle_density=800.0,
    create_particle_template=lambda prim_path, name: og.objects.DatasetObject(
        prim_path=prim_path,
        # usd_path=os.path.join(gm.ASSET_PATH, "models", "stain", "stain.usd"),
        name=name,
        # class_id=SemanticClass.DIRT,
        visible=False,
        fixed_base=False,
        visual_only=True,
        include_default_states=False,
        category="raspberry",
        model="spkers",
    ),
    scale=np.ones(3) * 5.0,
)


# GrassSystem = VisualParticleSystem.create(
#     name="Grass",
#     create_particle_template=lambda prim_path, name: og.objects.DatasetObject(
#         prim_path=prim_path,
#         name=name,
#         category="grass_patch",
#         model="kqhokv",
#         class_id=SemanticClass.GRASS,
#         visible=False,
#         fixed_base=False,
#         visual_only=True,
#         include_default_states=False,
#     ),
#     # Also need to override how we sample particles, since grass should only point upwards and placed on "top"
#     # parts of surfaces!
#     _SAMPLING_AXIS_PROBABILITIES=(0, 0, 1.0),
# )
