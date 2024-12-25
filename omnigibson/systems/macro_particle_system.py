import cv2
import torch as th
import trimesh

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.geom_prim import CollisionVisualGeomPrim, VisualGeomPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.systems.system_base import BaseSystem, PhysicalParticleSystem, VisualParticleSystem
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import torch_delete
from omnigibson.utils.sampling_utils import sample_cuboid_on_object_symmetric_bimodal_distribution
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log
from omnigibson.utils.usd_utils import (
    FlatcacheAPI,
    absolute_prim_path_to_scene_relative,
    scene_relative_prim_path_to_absolute,
)

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.MIN_PARTICLE_RADIUS = (
    0.01  # Minimum particle radius for physical macro particles -- this reduces the chance of omni physx crashing
)


class MacroParticleSystem(BaseSystem):
    """
    Global system for modeling "macro" level particles, e.g.: dirt, dust, etc.
    """

    def __init__(self, name, **kwargs):
        # Template object to use -- class particle objet is assumed to be the first and only visual mesh belonging to the
        # root link of this template object, which symbolizes a single particle, and will be duplicated to generate the
        # particle system. Note that this object is NOT part of the actual particle system itself!
        self._particle_template = None

        # dict, array of particle objects, mapped by their prim names
        self.particles = None

        # Counter to increment monotonically as we add more particles
        self._particle_counter = 0

        # Color associated with this system (NOTE: external queries should call self.color)
        self._color = None
        return super().__init__(name=name, **kwargs)

    def initialize(self, scene):
        # Run super method first
        super().initialize(scene)

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        self._particle_counter = 0

        # Create the system prim -- this is merely a scope prim
        og.sim.stage.DefinePrim(f"/World/scene_{scene.idx}/{self.name}", "Scope")

        # Load the particle template, and make it kinematic only because it's not interacting with anything
        particle_template = self._create_particle_template()
        scene.add_object(particle_template, register=False)

        # Make sure template scaling is [1, 1, 1] -- any particle scaling should be done via self.min/max_scale
        assert th.all(particle_template.scale == 1.0)

        # Make sure there is no ambiguity about which mesh to use as the particle from this template
        assert len(particle_template.links) == 1, "MacroParticleSystem particle template has more than one link"
        assert (
            len(particle_template.root_link.visual_meshes) == 1
        ), "MacroParticleSystem particle template has more than one visual mesh"

        self._particle_template = particle_template

        # Class particle objet is assumed to be the first and only visual mesh belonging to the root link
        self.particle_object.material.shader_force_populate(render=True)
        self.process_particle_object()

    @property
    def particle_object(self):
        return list(self._particle_template.root_link.visual_meshes.values())[0]

    @property
    def particle_idns(self):
        """
        Returns:
            set: idn of all the particles across all groups.
        """
        return {self.particle_name2idn(particle_name) for particle_name in self.particles}

    @property
    def next_available_particle_idn(self):
        """
        Returns:
            int: the next available particle idn across all groups.
        """
        return self._particle_counter

    def _create_particle_template(self):
        """
        Creates the particle template to be used for this system.

        NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
            visual mesh attached to its root link, since this will be the actual visual mesh used

        Returns:
            EntityPrim: Particle template that will be duplicated when generating future particle groups
        """
        raise NotImplementedError()

    def remove_all_particles(self):
        # Use list explicitly to prevent mid-loop mutation of dict
        for particle_name in tuple(self.particles.keys()):
            self.remove_particle_by_name(name=particle_name)

    def reset(self):
        # Call super first
        super().reset()

        # Reset the particle counter
        self._particle_counter = 0

    def _clear(self):
        # Clear all internal state
        self.scene.remove_object(self._particle_template)

        super()._clear()

        self._particle_template = None
        self.particles = None
        self._color = None

    @property
    def n_particles(self):
        return len(self.particles) if self.particles is not None else 0

    @property
    def material(self):
        return self.particle_object.material

    @property
    def particle_name_prefix(self):
        """
        Returns:
            str: Naming prefix used for all generated particles. This is coupled with the unique particle ID to generate
                the full particle name
        """
        return f"{self.name}Particle"

    def _dump_state(self):
        state = super()._dump_state()
        state["scales"] = (
            th.stack([particle.scale for particle in self.particles.values()])
            if self.particles is not None and self.particles != {}
            else th.empty(0)
        )
        state["particle_counter"] = self._particle_counter
        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Set particle scales
        if self.particles is not None:
            for particle, scale in zip(self.particles.values(), state["scales"]):
                particle.scale = scale

        # Set particle counter
        self._particle_counter = state["particle_counter"]

    def serialize(self, state):
        # Run super first
        states_flat = super().serialize(state=state)

        # Add particle scales, then the template info
        return th.cat(
            [
                states_flat,
                state["scales"].flatten(),
                th.tensor([state["particle_counter"]], dtype=th.float32),
            ]
        )

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Infer how many scales we have, then deserialize
        n_particles = state_dict["n_particles"]
        len_scales = n_particles * 3
        state_dict["scales"] = state[idx : idx + len_scales].reshape(-1, 3)
        state_dict["particle_counter"] = int(state[idx + len_scales])

        return state_dict, idx + len_scales + 1

    def process_particle_object(self):
        """
        Perform any necessary processing on the particle object to extract further information.
        """
        # Update color if the particle object has any material
        color = th.ones(3)
        if self.particle_object.has_material():
            if self.particle_object.material.is_glass:
                color = self.particle_object.material.glass_color
            else:
                diffuse_texture = self.particle_object.material.diffuse_texture
                color = (
                    cv2.imread(diffuse_texture).mean()
                    if diffuse_texture
                    else self.particle_object.material.diffuse_color_constant
                )
        self._color = color

    def add_particle(self, relative_prim_path, scale, idn=None):
        """
        Adds a particle to this system.

        Args:
            relative_prim_path (str): scene-local prim path to the newly created particle, minus the name for this particle
            scale (3-array): (x,y,z) scale to set for the added particle
            idn (None or int): If specified, should be unique identifier to assign to this particle. If not, will
                automatically generate a new unique one

        Returns:
            XFormPrim: Newly created particle instance, which is added internally as well
        """
        # Generate the new particle
        name = self.particle_idn2name(idn=self.next_available_particle_idn if idn is None else idn)
        # Make sure name doesn't already exist
        assert (
            self.particles is None or name not in self.particles.keys()
        ), f"Cannot create particle with name {name} because it already exists!"
        new_particle = self._load_new_particle(relative_prim_path=f"{relative_prim_path}/{name}", name=name)

        # Set the scale and make sure the particle is visible
        new_particle.scale *= scale
        new_particle.visible = True

        # Track this particle as well
        if self.particles is None:
            self.particles = dict()
        self.particles[new_particle.name] = new_particle

        # Increment counter
        self._particle_counter += 1

        return new_particle

    def remove_particle_by_name(self, name):
        assert name in self.particles, f"Got invalid name for particle to remove {name}"
        particle = self.particles.pop(name)
        og.sim.remove_prim(particle)

    def remove_particles(
        self,
        idxs,
        **kwargs,
    ):
        particle_names = tuple(self.particles.keys()) if self.particles else []
        for idx in idxs:
            self.remove_particle_by_name(particle_names[idx])

    def generate_particles(
        self,
        positions,
        orientations=None,
        scales=None,
        **kwargs,
    ):
        # Grab pre-existing tfs
        current_positions, current_orientations = self.get_particles_position_orientation()

        # Update the tensors
        n_particles = len(positions)
        orientations = T.random_quaternion(n_particles) if orientations is None else orientations
        scales = self.sample_scales(n=n_particles) if scales is None else scales

        positions = th.cat([current_positions, positions], dim=0)
        orientations = th.cat([current_orientations, orientations], dim=0)

        # Add particles
        for scale in scales:
            self.add_particle(relative_prim_path=f"{self.relative_prim_path}/particles", scale=scale)

        # Set the tfs
        self.set_particles_position_orientation(positions=positions, orientations=orientations)

    def _load_new_particle(self, relative_prim_path, name):
        """
        Loads a new particle into the current stage, leveraging @self.particle_object as a template for the new particle
        to load. This function should be implemented by any subclasses.

        Args:
            relative_prim_path (str): scene-local prim path at which to create the new particle
            name (str): The name to assign to this new particle at the path

        Returns:
            XFormPrim: Loaded particle
        """
        raise NotImplementedError()

    def particle_name2idn(self, name):
        """
        Args:
            name (str): Particle name to grab its corresponding unique id number for

        Returns:
            int: Unique ID assigned to the particle based on its name
        """
        assert (
            self.particle_name_prefix in name
        ), f"Particle name should have '{self.particle_name_prefix}' in it when checking ID! Got: {name}"
        return int(name.split(self.particle_name_prefix)[-1])

    def particle_idn2name(self, idn):
        """
        Args:
            idn (int): Unique ID number assigned to the particle to grab the name for

        Returns:
            str: Particle name corresponding to its unique id number
        """
        assert isinstance(
            idn, int
        ), f"Particle idn must be an integer when checking name! Got: {idn}. Type: {type(idn)}"
        return f"{self.particle_name_prefix}{idn}"

    @property
    def color(self):
        return th.tensor(self._color)


class MacroVisualParticleSystem(MacroParticleSystem, VisualParticleSystem):
    """
    Particle system class that procedurally generates individual particles that are not subject to physics
    """

    def __init__(
        self,
        name,
        create_particle_template,
        min_scale=None,
        max_scale=None,
        scale_relative_to_parent=False,
        sampling_axis_probabilities=(0.25, 0.25, 0.5),
        sampling_aabb_offset=0.01,
        sampling_bimodal_mean_fraction=0.9,
        sampling_bimodal_stdev_fraction=0.2,
        sampling_max_attempts=20,
        sampling_hit_proportion=0.4,
        **kwargs,
    ):
        """
        Args:
            name (str): Name of the visual particles, in snake case.
            create_particle_template (function): Method for generating the visual particle template that will be duplicated
                when generating groups of particles.
                Expected signature:

                create_particle_template(prim_path: str, name: str) --> EntityPrim

                where @prim_path and @name are the parameters to assign to the generated EntityPrim.
                NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
                    visual mesh attached to its root link, since this will be the actual visual mesh used
            min_scale (None or 3-array): If specified, sets the minumum bound for the visual particles' relative scale.
                Else, defaults to 1
            max_scale (None or 3-array): If specified, sets the maximum bound for the visual particles' relative scale.
                Else, defaults to 1
            scale_relative_to_parent (bool): If True, will scale generated particles relative to the corresponding
                group's object
            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)
        """
        self._scale_relative_to_parent = scale_relative_to_parent
        self._create_particle_template_fcn = create_particle_template

        # Maps particle name to dict of {obj, link, face_id}
        # NOTE: link will only exist for particles on rigid bodies
        # NOTE: face_id will only exist for particles on cloths
        self._particles_info = None

        # Pre-cached information about visual particles so that we have efficient runtime computations
        # Maps particle name to local pose matrix for computing global poses for the particle
        self._particles_local_mat = None

        # Maps group name to array of face_ids where particles are located if the group object is a cloth type
        # Maps group name to th.tensor of face IDs (int) that particles are attached to
        self._cloth_face_ids = None

        # Default behavior for this class -- whether to clip generated particles halfway into objects when sampling
        # their locations on the surface of the given object
        self._CLIP_INTO_OBJECTS = False

        # Default parameters for sampling particle locations
        # See omnigibson/utils/sampling_utils.py for how they are used.
        self._SAMPLING_AXIS_PROBABILITIES = sampling_axis_probabilities
        self._SAMPLING_AABB_OFFSET = sampling_aabb_offset
        self._SAMPLING_BIMODAL_MEAN_FRACTION = sampling_bimodal_mean_fraction
        self._SAMPLING_BIMODAL_STDEV_FRACTION = sampling_bimodal_stdev_fraction
        self._SAMPLING_MAX_ATTEMPTS = sampling_max_attempts
        self._SAMPLING_HIT_PROPORTION = sampling_hit_proportion
        return super().__init__(name=name, min_scale=min_scale, max_scale=max_scale, **kwargs)

    def initialize(self, scene):
        # Run super method first
        super().initialize(scene)

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        self._particles_info = dict()
        self._particles_local_mat = dict()
        self._cloth_face_ids = dict()

    def update(self):
        # Run super first
        super().update()

        z_extent = self.particle_object.aabb_extent[2]
        # Iterate over all objects, and update all particles belonging to any cloth objects
        for name, obj in self._group_objects.items():
            group = self.get_group_name(obj=obj)
            if obj.prim_type == PrimType.CLOTH and self.num_group_particles(group=group) > 0:
                # Update the transforms
                cloth = obj.root_link
                face_ids = self._cloth_face_ids[group]
                idxs = cloth.faces[face_ids].flatten()
                positions = cloth.compute_particle_positions(idxs=idxs).reshape(-1, 3, 3)
                normals = cloth.compute_face_normals_from_particle_positions(positions=positions)

                # The actual positions we want are the face centroids, or the mean of all the positions
                positions = positions.mean(dim=1)
                # Orientations are the normals
                z_up = th.zeros_like(normals)
                z_up[:, 2] = 1.0
                orientations = T.axisangle2quat(T.vecs2axisangle(z_up, normals))
                if not self._CLIP_INTO_OBJECTS and z_extent > 0:
                    z_offsets = (
                        th.tensor([z_extent * particle.scale[2] for particle in self._group_particles[group].values()])
                        / 2.0
                    )
                    # Shift the particles halfway up
                    positions += normals * z_offsets.reshape(-1, 1)

                # Set the group particle poses
                self.set_group_particles_position_orientation(
                    group=group, positions=positions, orientations=orientations
                )

    def _load_new_particle(self, relative_prim_path, name):
        # We copy the template prim and generate the new object if the prim doesn't already exist, otherwise we
        # reference the pre-existing one
        prim_path = scene_relative_prim_path_to_absolute(self.scene, relative_prim_path)
        if not lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path):
            lazy.omni.kit.commands.execute(
                "CopyPrim",
                path_from=self.particle_object.prim_path,
                path_to=prim_path,
            )
            prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
            lazy.omni.isaac.core.utils.semantics.add_update_semantics(
                prim=prim,
                semantic_label=self.name,
                type_label="class",
            )
        result = VisualGeomPrim(relative_prim_path=relative_prim_path, name=name)
        result.load(self.scene)
        return result

    def _clear(self):
        # Run super method first
        super()._clear()

        # Clear all groups as well
        self._particles_info = dict()
        self._particles_local_mat = dict()
        self._cloth_face_ids = dict()

    def remove_attachment_group(self, group):
        # Call super first
        super().remove_attachment_group(group=group)

        # If the group is a cloth, also remove the cloth face ids
        if group in self._cloth_face_ids:
            self._cloth_face_ids.pop(group)

        return group

    def remove_particle_by_name(self, name):
        # Run super first
        super().remove_particle_by_name(name=name)

        # Remove this particle from its respective group as well
        parent_obj = self._particles_info[name]["obj"]
        group = self.get_group_name(obj=parent_obj)
        self._group_particles[group].pop(name)
        self._particles_local_mat.pop(name)
        particle_info = self._particles_info.pop(name)
        if self._is_cloth_obj(obj=parent_obj):
            # Also remove from cloth face ids
            face_ids = self._cloth_face_ids[group]
            idx_mapping = {face_id: i for i, face_id in enumerate(face_ids)}
            self._cloth_face_ids[group] = torch_delete(face_ids, idx_mapping[particle_info["face_id"]])

    def generate_group_particles(
        self,
        group,
        positions,
        orientations=None,
        scales=None,
        link_prim_paths=None,
    ):
        # Make sure the group exists
        self._validate_group(group=group)

        # Standardize orientations and links
        obj = self._group_objects[group]
        is_cloth = self._is_cloth_obj(obj=obj)

        # If cloth, run the following sanity checks:
        # (1) make sure link prim paths are not specified -- we can ONLY apply particles under the object xform prim
        # (2) make sure object prim path exists at /World/<NAME> -- global pose inference assumes this is the case
        if is_cloth:
            assert link_prim_paths is None, "link_prim_paths should not be specified for cloth object group!"
            assert (
                obj.prim.GetParent().GetPath().pathString == "/World"
            ), "cloth object should exist as direct child of /World prim!"

        n_particles = len(positions)
        if orientations is None:
            orientations = th.zeros((n_particles, 4))
            orientations[:, -1] = 1.0
        link_prim_paths = [None] * n_particles if is_cloth else link_prim_paths

        scales = self.sample_scales_by_group(group=group, n=n_particles) if scales is None else scales

        bbox_extents_local = [(self.particle_object.aabb_extent * scale).tolist() for scale in scales]

        # If we're using flatcache, we need to update the object's pose on the USD manually
        if gm.ENABLE_FLATCACHE:
            FlatcacheAPI.sync_raw_object_transforms_in_usd(prim=obj)

        # Generate particles
        z_up = th.zeros((3, 1))
        z_up[-1] = 1.0
        for position, orientation, scale, bbox_extent_local, link_prim_path in zip(
            positions, orientations, scales, bbox_extents_local, link_prim_paths
        ):
            link = None if is_cloth else obj.links[link_prim_path.split("/")[-1]]
            # Possibly shift the particle slightly away from the object if we're not clipping into objects
            # Note: For particles tied to rigid objects, the given position is on the surface of the object,
            # so clipping would move the particle INTO the object surface, whereas for particles tied to cloth objects,
            # the given position is at the particle location (i.e.: already clipped), so NO clipping would move the
            # particle AWAY from the object surface
            if (is_cloth and not self._CLIP_INTO_OBJECTS) or (not is_cloth and self._CLIP_INTO_OBJECTS):
                # Shift the particle halfway down
                base_to_center = bbox_extent_local[2] / 2.0
                normal = (T.quat2mat(orientation) @ z_up).flatten()
                offset = normal * base_to_center if is_cloth else -normal * base_to_center
                position += offset

            # Create particle
            particle_prim_path = obj.prim_path if is_cloth else link_prim_path
            particle = self.add_particle(
                relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, particle_prim_path),
                scale=scale,
            )

            # Add to group
            self._group_particles[group][particle.name] = particle
            self._particles_info[particle.name] = dict(obj=self._group_objects[group], link=link)

            # Set the pose
            self.set_particle_position_orientation(idx=-1, position=position, orientation=orientation)

    def generate_group_particles_on_object(self, group, max_samples=None, min_samples_for_success=1):
        # This function does not support max_samples=None. Must be explicitly specified
        assert (
            max_samples is not None
        ), f"max_samples must be specified for {self.name}'s generate_group_particles_on_object!"
        assert max_samples >= min_samples_for_success, "number of particles to sample should exceed the min for success"

        # Make sure the group exists
        self._validate_group(group=group)

        # Remove all stale particles
        self.remove_all_group_particles(group=group)

        # Generate requested number of particles
        obj = self._group_objects[group]

        # Sample scales and corresponding bbox extents
        scales = self.sample_scales_by_group(group=group, n=max_samples)
        # For sampling particle positions, we need the global bbox extents, NOT the local extents
        # which is what we would get naively if we directly use @scales
        avg_scale = th.pow(th.prod(obj.scale), 1 / 3)

        bbox_extents_global = scales * self.particle_object.aabb_extent.reshape(1, 3) * avg_scale

        if obj.prim_type == PrimType.CLOTH:
            # Sample locations based on randomly sampled keyfaces
            cloth = obj.root_link
            n_faces = len(cloth.faces)
            face_ids = th.randperm(n_faces)[: min(max_samples, n_faces)]
            # Positions are the midpoints of each requested face
            normals = cloth.compute_face_normals(face_ids=face_ids)
            positions = (
                cloth.compute_particle_positions(idxs=cloth.faces[face_ids].flatten()).reshape(-1, 3, 3).mean(dim=1)
            )
            # Orientations are the normals
            z_up = th.zeros_like(normals)
            z_up[:, 2] = 1.0
            orientations = th.tensor(T.axisangle2quat(T.vecs2axisangle(z_up, normals)))
            link_prim_paths = None
            self._cloth_face_ids[group] = face_ids
        else:
            # Sample locations for all particles
            results = sample_cuboid_on_object_symmetric_bimodal_distribution(
                obj=obj,
                num_samples=max_samples,
                cuboid_dimensions=bbox_extents_global,
                bimodal_mean_fraction=self._SAMPLING_BIMODAL_MEAN_FRACTION,
                bimodal_stdev_fraction=self._SAMPLING_BIMODAL_STDEV_FRACTION,
                axis_probabilities=self._SAMPLING_AXIS_PROBABILITIES,
                undo_cuboid_bottom_padding=True,
                verify_cuboid_empty=False,
                aabb_offset=self._SAMPLING_AABB_OFFSET,
                max_sampling_attempts=self._SAMPLING_MAX_ATTEMPTS,
                refuse_downwards=True,
                hit_proportion=self._SAMPLING_HIT_PROPORTION,
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
            scales = particle_scales

        success = len(positions) >= min_samples_for_success
        # If we generated a sufficient number of points, generate them in the simulator
        if success:
            self.generate_group_particles(
                group=group,
                positions=positions,
                orientations=orientations,
                scales=scales,
                link_prim_paths=link_prim_paths,
            )
            # If we're a cloth, store the face_id as well
            if obj.prim_type == PrimType.CLOTH:
                for particle_name, face_id in zip(self._group_particles[group].keys(), self._cloth_face_ids[group]):
                    self._particles_info[particle_name]["face_id"] = int(face_id)

        return success

    def _compute_batch_particles_position_orientation(self, particles, local=False):
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
        n_particles = len(particles) if particles else 0
        if n_particles == 0:
            return (th.empty(0).reshape(0, 3), th.empty(0).reshape(0, 4))

        if local:
            poses = th.zeros((n_particles, 4, 4))
            for i, name in enumerate(particles):
                poses[i] = T.pose2mat(self.particles[name].get_position_orientation(frame="parent"))
        else:
            # Iterate over all particles and compute link tfs programmatically, then batch the matrix transform
            link_tfs = dict()
            link_tfs_batch = th.zeros((n_particles, 4, 4))
            particle_local_poses_batch = th.zeros_like(link_tfs_batch)
            for i, name in enumerate(particles):
                obj = self._particles_info[name]["obj"]
                is_cloth = self._is_cloth_obj(obj=obj)
                if is_cloth:
                    if obj not in link_tfs:
                        # We want World --> obj transform, NOT the World --> root_link transform, since these particles
                        # do NOT exist under a link but rather the object prim itself. So we use XFormPrim to directly
                        # get the transform, and not obj.get_position_orientation(frame="parent") which will give us the local pose of the
                        # root link!
                        link_tfs[obj] = T.pose2mat(XFormPrim.get_position_orientation(obj, frame="parent"))
                    link = obj
                else:
                    link = self._particles_info[name]["link"]
                    if link not in link_tfs:
                        link_tfs[link] = T.pose2mat(link.get_position_orientation())
                link_tfs_batch[i] = link_tfs[link]
                particle_local_poses_batch[i] = self._particles_local_mat[name]

            # Compute once
            poses = link_tfs_batch @ particle_local_poses_batch

        # Decompose back into positions and orientations
        return poses[:, :3, 3], T.mat2quat(poses[:, :3, :3])

    def get_particles_position_orientation(self):
        return self._compute_batch_particles_position_orientation(particles=self.particles, local=False)

    def get_particles_local_pose(self):
        return self._compute_batch_particles_position_orientation(particles=self.particles, local=True)

    def get_group_particles_position_orientation(self, group):
        return self._compute_batch_particles_position_orientation(particles=self._group_particles[group], local=False)

    def get_group_particles_local_pose(self, group):
        return self._compute_batch_particles_position_orientation(particles=self._group_particles[group], local=True)

    def get_particle_position_orientation(self, idx):
        name = list(self.particles.keys())[idx]
        # First, get local pose, scale it by the parent link's scale, and then convert into a matrix
        # Note that particles_local_mat already takes the parent scale into account when computing the transform!
        parent_obj = self._particles_info[name]["obj"]
        is_cloth = self._is_cloth_obj(obj=parent_obj)
        local_mat = self._particles_local_mat[name]
        link_tf = (
            T.pose2mat(XFormPrim.get_position_orientation(parent_obj, frame="parent"))
            if is_cloth
            else T.pose2mat(self._particles_info[name]["link"].get_position_orientation())
        )

        # Multiply the local pose by the link's global transform, then return as pos, quat tuple
        return T.mat2pose(link_tf @ local_mat)

    def get_particle_local_pose(self, idx):
        name = list(self.particles.keys())[idx]
        return self.particles[name].get_position_orientation(frame="parent")

    def _modify_batch_particles_position_orientation(self, particles, positions=None, orientations=None, local=False):
        """
        Modifies all @particles' positions and orientations with @positions and @orientations

        Args:
            particles (Iterable of str): Names of particles to modify
            positions (None or (n, 3)-array): New positions to set for the particles
            orientations (None or (n, 4)-array): New orientations to set for the particles
            local (bool): Whether to modify particles' poses in local frame or not

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        n_particles = len(particles) if particles is not None else 0
        if n_particles == 0:
            return

        if positions is None or orientations is None:
            pos, ori = self._compute_batch_particles_position_orientation(particles=particles, local=local)
            positions = pos if positions is None else positions
            orientations = ori if orientations is None else orientations
        lens = th.tensor([len(particles), len(positions), len(orientations)])
        assert lens.min() == lens.max(), "Got mismatched particles, positions, and orientations!"

        particle_local_poses_batch = th.zeros((n_particles, 4, 4))
        particle_local_poses_batch[:, -1, -1] = 1.0
        particle_local_poses_batch[:, :3, 3] = positions
        particle_local_poses_batch[:, :3, :3] = T.quat2mat(orientations)

        if not local:
            # Iterate over all particles and compute link tfs programmatically, then batch the matrix transform
            link_tfs = dict()
            link_tfs_batch = th.zeros((n_particles, 4, 4))
            for i, name in enumerate(particles):
                obj = self._particles_info[name]["obj"]
                is_cloth = self._is_cloth_obj(obj=obj)
                if is_cloth:
                    if obj not in link_tfs:
                        # We want World --> obj transform, NOT the World --> root_link transform, since these particles
                        # do NOT exist under a link but rather the object prim itself. So we use XFormPrim to directly
                        # get the transform, and not obj.get_position_orientation(frame="parent") which will give us the local pose of the
                        # root link!
                        link_tfs[obj] = T.pose2mat(XFormPrim.get_position_orientation(obj, frame="parent"))
                    link_tf = link_tfs[obj]
                else:
                    link = self._particles_info[name]["link"]
                    if link not in link_tfs:
                        link_tfs[link] = T.pose2mat(link.get_position_orientation())
                    link_tf = link_tfs[link]
                link_tfs_batch[i] = link_tf

            # particle_local_poses_batch = th.linalg.inv_ex(link_tfs_batch).inverse @ particle_local_poses_batch
            particle_local_poses_batch = th.linalg.solve(link_tfs_batch, particle_local_poses_batch)

        for i, name in enumerate(particles):
            self._modify_particle_local_mat(name=name, mat=particle_local_poses_batch[i], ignore_scale=local)

    def set_particles_position_orientation(self, positions=None, orientations=None):
        return self._modify_batch_particles_position_orientation(
            particles=self.particles, positions=positions, orientations=orientations, local=False
        )

    def set_particles_local_pose(self, positions=None, orientations=None):
        return self._modify_batch_particles_position_orientation(
            particles=self.particles, positions=positions, orientations=orientations, local=True
        )

    def set_group_particles_position_orientation(self, group, positions=None, orientations=None):
        return self._modify_batch_particles_position_orientation(
            particles=self._group_particles[group], positions=positions, orientations=orientations, local=False
        )

    def set_group_particles_local_pose(self, group, positions=None, orientations=None):
        return self._modify_batch_particles_position_orientation(
            particles=self._group_particles[group], positions=positions, orientations=orientations, local=True
        )

    def set_particle_position_orientation(self, idx, position=None, orientation=None):
        if position is None or orientation is None:
            pos, ori = self.get_particle_position_orientation(idx=idx)
            position = pos if position is None else position
            orientation = ori if orientation is None else orientation

        position = position if isinstance(position, th.Tensor) else th.tensor(position, dtype=th.float32)
        orientation = orientation if isinstance(orientation, th.Tensor) else th.tensor(orientation, dtype=th.float32)

        name = list(self.particles.keys())[idx]
        global_mat = th.zeros((4, 4))
        global_mat[-1, -1] = 1.0
        global_mat[:3, 3] = position
        global_mat[:3, :3] = T.quat2mat(orientation)
        # First, get global pose, scale it by the parent link's scale, and then convert into a matrix
        parent_obj = self._particles_info[name]["obj"]
        is_cloth = self._is_cloth_obj(obj=parent_obj)
        link_tf = (
            T.pose2mat(XFormPrim.get_position_orientation(parent_obj, frame="parent"))
            if is_cloth
            else T.pose2mat(self._particles_info[name]["link"].get_position_orientation())
        )
        local_mat = th.linalg.inv_ex(link_tf).inverse @ global_mat

        self._modify_particle_local_mat(name=name, mat=local_mat, ignore_scale=False)

    def set_particle_local_pose(self, idx, position=None, orientation=None):
        if position is None or orientation is None:
            pos, ori = self.get_particle_local_pose(idx=idx)
            position = pos if position is None else position
            orientation = ori if orientation is None else orientation

        position = position if isinstance(position, th.Tensor) else th.tensor(position, dtype=th.float32)
        orientation = orientation if isinstance(orientation, th.Tensor) else th.tensor(orientation, dtype=th.float32)

        name = list(self.particles.keys())[idx]
        local_mat = th.zeros((4, 4))
        local_mat[-1, -1] = 1.0
        local_mat[:3, 3] = position
        local_mat[:3, :3] = T.quat2mat(orientation)
        self._modify_particle_local_mat(name=name, mat=local_mat, ignore_scale=True)

    def _is_cloth_obj(self, obj):
        """
        Checks whether object @obj is a cloth or not

        Args:
            obj (BaseObject): Object to check

        Returns:
            bool: True if the object is cloth type, otherwise False
        """
        return obj.prim_type == PrimType.CLOTH

    def _compute_particle_local_mat(self, name, ignore_scale=False):
        """
        Computes particle @name's local transform as a homogeneous 4x4 matrix

        Args:
            name (str): Name of the particle to compute local transform matrix for
            ignore_scale (bool): Whether to ignore the parent_link scale when computing the local transform

        Returns:
            th.tensor: (4, 4) homogeneous transform matrix
        """
        particle = self.particles[name]
        parent_obj = self._particles_info[name]["obj"]
        is_cloth = self._is_cloth_obj(obj=parent_obj)
        scale = th.ones(3) if is_cloth else self._particles_info[name]["link"].scale
        local_pos, local_quat = particle.get_position_orientation(frame="parent")
        local_pos = local_pos if ignore_scale else local_pos * scale
        return T.pose2mat((local_pos, local_quat))

    def _modify_particle_local_mat(self, name, mat, ignore_scale=False):
        """
        Sets particle @name's local transform as a homogeneous 4x4 matrix

        Args:
            name (str): Name of the particle to compute local transform matrix for
            mat (n-array): (4, 4) homogeneous transform matrix
            ignore_scale (bool): Whether to ignore the parent_link scale when setting the local transform
        """
        particle = self.particles[name]
        parent_obj = self._particles_info[name]["obj"]
        is_cloth = self._is_cloth_obj(obj=parent_obj)
        scale = th.ones(3) if is_cloth else self._particles_info[name]["link"].scale
        local_pos, local_quat = T.mat2pose(mat)
        local_pos = local_pos if ignore_scale else local_pos / scale
        particle.set_position_orientation(position=local_pos, orientation=local_quat, frame="parent")

        # Store updated value
        self._particles_local_mat[name] = mat

    def _sync_particle_groups(
        self,
        group_objects,
        particle_idns,
        particle_attached_references,
    ):
        """
        Synchronizes the particle groups based on desired identification numbers @group_idns

        Args:
            group_objects (list of BaseObject): Desired unique group objects that should be active for
            this particle system.
            particle_idns (list of list of int): Per-group unique id numbers for the particles assigned to that group.
                List should be same length as @group_idns with sub-entries corresponding to the desired number of
                particles assigned to that group
            particle_attached_references (list of list of str or int): Per-group reference info relevant for each
                particle. List should be same length as @group_idns with sub-entries corresponding to the desired
                number of particles assigned to that group. If a given group is a cloth object, the entries should be
                integers corresponding to the individual face IDs that each particle is attached to for the group.
                Otherwise, the group is assumed to be a rigid object, in which case the entries should be link
                names corresponding to the specific links each particle is attached for each group.
        """
        # We have to be careful here -- some particle groups may have been deleted / are mismatched, so we need
        # to update accordingly, potentially deleting stale groups and creating new groups as needed
        name_to_info_mapping = {
            obj.name: {
                "n_particles": len(p_idns),
                "particle_idns": p_idns,
                "references": references,
            }
            for obj, p_idns, references in zip(group_objects, particle_idns, particle_attached_references)
        }

        current_group_names = self.groups
        desired_group_names = set(obj.name for obj in group_objects)
        groups_to_delete = current_group_names - desired_group_names
        groups_to_create = desired_group_names - current_group_names
        common_groups = current_group_names.intersection(desired_group_names)

        # Sanity check the common groups, we will recreate any where there is a mismatch
        for name in common_groups:
            info = name_to_info_mapping[name]
            if self.num_group_particles(group=name) != info["n_particles"]:
                log.debug(f"Got mismatch in particle group {name} when syncing, " f"deleting and recreating group now.")
                # Add this group to both the delete and creation pile
                groups_to_delete.add(name)
                groups_to_create.add(name)

        # Delete any groups we no longer want
        for name in groups_to_delete:
            self.remove_attachment_group(group=name)

        # Create any groups we don't already have
        for name in groups_to_create:
            obj = self.scene.object_registry("name", name)
            info = name_to_info_mapping[name]
            self.create_attachment_group(obj=obj)
            is_cloth = self._is_cloth_obj(obj=obj)
            for particle_idn, reference in zip(info["particle_idns"], info["references"]):
                # Reference is either the face ID (int) if cloth group or link name (str) if rigid body group
                # Create the necessary particles
                # Use scale (1,1,1) since it will get overridden anyways when loading state
                particle_prim_path = obj.prim_path if is_cloth else obj.links[reference].prim_path
                particle = self.add_particle(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, particle_prim_path),
                    scale=th.ones(3),
                    idn=int(particle_idn),
                )
                self._group_particles[name][particle.name] = particle
                self._particles_info[particle.name] = dict(obj=obj)
                # Add face_id if is_cloth, otherwise, add link
                if is_cloth:
                    self._particles_info[particle.name]["face_id"] = int(reference)
                else:
                    self._particles_info[particle.name]["link"] = obj.links[reference]

            # Also store the cloth face IDs as a vector
            if is_cloth:
                self._cloth_face_ids[self.get_group_name(obj)] = th.tensor(
                    [self._particles_info[particle_name]["face_id"] for particle_name in self._group_particles[name]]
                )

    @property
    def _register_system(self):
        return True

    def _create_particle_template(self):
        return self._create_particle_template_fcn(
            relative_prim_path=f"/{self.name}/template", name=f"{self.name}_template"
        )

    def _dump_state(self):
        state = super()._dump_state()
        particle_names = list(self.particles.keys()) if self.particles else []
        # Add in per-group information
        groups_dict = dict()
        name2idx = {name: idx for idx, name in enumerate(particle_names)}
        for group_name, group_particles in self._group_particles.items():
            obj = self._group_objects[group_name]
            is_cloth = self._is_cloth_obj(obj=obj)
            groups_dict[group_name] = dict(
                particle_attached_obj_uuid=obj.uuid,
                n_particles=self.num_group_particles(group=group_name),
                particle_idns=[self.particle_name2idn(name=name) for name in group_particles.keys()],
                particle_indices=[name2idx[name] for name in group_particles.keys()],
                # If the attached object is a cloth, store the face_id, otherwise, store the link name
                particle_attached_references=(
                    [self._particles_info[name]["face_id"] for name in group_particles.keys()]
                    if is_cloth
                    else [
                        self._particles_info[name]["link"].prim_path.split("/")[-1] for name in group_particles.keys()
                    ]
                ),
            )

        state["n_groups"] = len(self._group_particles)
        state["groups"] = groups_dict

        return state

    def _load_state(self, state):
        # First, we sync our particle systems
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (dict): Keyword-mapped states of this object to set
        """
        # Synchronize particle groups
        group_objects = []
        particle_idns = []
        particle_attached_references = []

        indices_to_remove = th.empty(0, dtype=int)
        for info in state["groups"].values():
            obj = self.scene.object_registry("uuid", info["particle_attached_obj_uuid"])
            # obj will be None if an object with an attachment group is removed between dump_state() and load_state()
            if obj is not None:
                group_objects.append(obj)
                particle_idns.append(info["particle_idns"])
                particle_attached_references.append(info["particle_attached_references"])
            else:
                indices_to_remove = th.cat((indices_to_remove, th.tensor(info["particle_indices"], dtype=int)))
        self._sync_particle_groups(
            group_objects=group_objects,
            particle_idns=particle_idns,
            particle_attached_references=particle_attached_references,
        )
        state["n_particles"] -= len(indices_to_remove)
        state["positions"] = torch_delete(state["positions"], indices_to_remove, dim=0)
        state["orientations"] = torch_delete(state["orientations"], indices_to_remove, dim=0)
        state["scales"] = torch_delete(state["scales"], indices_to_remove, dim=0)

        # Run super
        super()._load_state(state=state)

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        groups_dict = state["groups"]
        state_group_flat = [th.tensor([state["n_groups"]], dtype=th.float32)]
        for group_name, group_dict in groups_dict.items():
            obj = self._group_objects[group_name]
            is_cloth = self._is_cloth_obj(obj=obj)
            group_obj_link2id = {link_name: i for i, link_name in enumerate(obj.links.keys())}
            state_group_flat += [
                th.tensor([group_dict["particle_attached_obj_uuid"]], dtype=th.float32),
                th.tensor([group_dict["n_particles"]], dtype=th.float32),
                th.tensor(group_dict["particle_idns"], dtype=th.float32),
                th.tensor(group_dict["particle_indices"], dtype=th.float32),
                th.tensor(
                    (
                        group_dict["particle_attached_references"]
                        if is_cloth
                        else [group_obj_link2id[reference] for reference in group_dict["particle_attached_references"]]
                    ),
                    dtype=th.float32,
                ),
            ]

        return th.cat([*state_group_flat, state_flat])

    def deserialize(self, state):
        # Synchronize the particle groups
        n_groups = int(state[0])
        groups_dict = dict()
        group_objs = []
        # Index starts at 1 because index 0 is n_groups
        idx = 1
        for i in range(n_groups):
            obj_uuid, n_particles = int(state[idx]), int(state[idx + 1])
            obj = self.scene.object_registry("uuid", obj_uuid)
            assert obj is not None, f"Object with UUID {obj_uuid} not found in the scene"
            is_cloth = self._is_cloth_obj(obj=obj)
            group_obj_id2link = {i: link_name for i, link_name in enumerate(obj.links.keys())}
            group_objs.append(obj)
            groups_dict[obj.name] = dict(
                particle_attached_obj_uuid=obj_uuid,
                n_particles=n_particles,
                particle_idns=[
                    int(idn) for idn in state[idx + 2 : idx + 2 + n_particles]
                ],  # Idx + 2 because the first two are obj_uuid and n_particles
                particle_indices=[int(idn) for idn in state[idx + 2 + n_particles : idx + 2 + n_particles * 2]],
                particle_attached_references=(
                    [int(idn) for idn in state[idx + 2 + n_particles * 2 : idx + 2 + n_particles * 3]]
                    if is_cloth
                    else [
                        group_obj_id2link[int(idn)]
                        for idn in state[idx + 2 + n_particles * 2 : idx + 2 + n_particles * 3]
                    ]
                ),
            )
            idx += 2 + n_particles * 3

        log.debug(f"Syncing {self.name} particles with {n_groups} groups..")
        self._sync_particle_groups(
            group_objects=group_objs,
            particle_idns=[group_info["particle_idns"] for group_info in groups_dict.values()],
            particle_attached_references=[
                group_info["particle_attached_references"] for group_info in groups_dict.values()
            ],
        )

        # Get super method
        state_dict, idx_super = super().deserialize(state=state[idx:])
        state_dict["n_groups"] = n_groups
        state_dict["groups"] = groups_dict

        return state_dict, idx + idx_super


class MacroPhysicalParticleSystem(MacroParticleSystem, PhysicalParticleSystem):
    """
    Particle system class that procedurally generates individual particles that are subject to physics
    """

    def __init__(self, name, create_particle_template, particle_density, scale=None, **kwargs):
        """
        Args:
            name (str): Name of the macro physical particles, in snake case.
            create_particle_template (function): Method for generating the visual particle template that will be duplicated
                when generating groups of particles.
                Expected signature:

                create_particle_template(prim_path: str, name: str) --> EntityPrim

                where @prim_path and @name are the parameters to assign to the generated EntityPrim.
                NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
                    visual mesh attached to its root link, since this will be the actual mesh used for duplication
            particle_density (float): Particle density for the generated system
            scale (None or 3-array): If specified, sets the scaling factor for the particles' relative scale.
                Else, defaults to 1

            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)
        """
        # Run super
        super().__init__(name=name, min_scale=scale, max_scale=scale, **kwargs)

        self._create_particle_template_fcn = create_particle_template

        self._particle_density = particle_density

        # Physics rigid body view for keeping track of all particles' state
        self.particles_view = None

        # Approximate radius of the macro particle, and distance from particle frame to approximate center
        self._particle_radius = None
        self._particle_offset = None

    def initialize(self, scene):
        # Run super method first
        super().initialize(scene)

        # Create the particles head prim -- this is merely a scope prim
        og.sim.stage.DefinePrim(f"{self.prim_path}/particles", "Scope")

        # A new view needs to be created every time once sim is playing, so we add a callback now
        og.sim.add_callback_on_play(name=f"{self.name}_particles_view", callback=self.refresh_particles_view)

        # If sim is already playing, refresh particles immediately
        if og.sim.is_playing():
            self.refresh_particles_view()

    def _load_new_particle(self, relative_prim_path, name):
        # We copy the template prim and generate the new object if the prim doesn't already exist, otherwise we
        # reference the pre-existing one
        prim_path = scene_relative_prim_path_to_absolute(self.scene, relative_prim_path)
        if not lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path):
            lazy.omni.kit.commands.execute(
                "CopyPrim",
                path_from=self.particle_object.prim_path,
                path_to=prim_path,
            )
            # Apply RigidBodyAPI to it so it is subject to physics
            prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
            lazy.pxr.UsdPhysics.RigidBodyAPI.Apply(prim)
            lazy.omni.isaac.core.utils.semantics.add_update_semantics(
                prim=prim,
                semantic_label=self.name,
                type_label="class",
            )
        result = CollisionVisualGeomPrim(relative_prim_path=relative_prim_path, name=name)
        result.load(self.scene)
        return result

    def process_particle_object(self):
        # Run super method
        super().process_particle_object()

        # Compute particle radius
        vertices = (
            th.tensor(self.particle_object.get_attribute("points"))
            * self.particle_object.scale
            * self.max_scale.reshape(1, 3)
        )

        particle_offset, particle_radius = trimesh.nsphere.minimum_nsphere(trimesh.Trimesh(vertices=vertices))
        particle_offset = th.tensor(particle_offset, dtype=th.float32)
        particle_radius = th.tensor(particle_radius, dtype=th.float32)

        if particle_radius < m.MIN_PARTICLE_RADIUS:
            ratio = m.MIN_PARTICLE_RADIUS / particle_radius
            self.particle_object.scale *= ratio
            particle_offset *= ratio
            particle_radius = m.MIN_PARTICLE_RADIUS

        self._particle_offset = particle_offset
        self._particle_radius = particle_radius

    def refresh_particles_view(self):
        """
        Internal helper method to refresh the particles' rigid body view to grab state

        Should be called every time sim.play() is called
        """
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            self.particles_view = og.sim.physics_sim_view.create_rigid_body_view(
                pattern=f"{self.prim_path}/particles/*"
            )

    def _clear(self):
        # Run super method first
        super()._clear()

        # Clear internal variables
        self.particles_view = None
        self._particle_radius = None
        self._particle_offset = None

        # Remove callback
        og.sim.remove_callback_on_play(name=f"{self.name}_particles_view")

    def remove_particle_by_name(self, name):
        # Run super first
        super().remove_particle_by_name(name=name)

        # Refresh particles view
        self.refresh_particles_view()

    def add_particle(self, relative_prim_path, scale, idn=None):
        # Run super first
        particle = super().add_particle(relative_prim_path=relative_prim_path, scale=scale, idn=idn)

        # Refresh particles view
        self.refresh_particles_view()

        return particle

    def get_particles_position_orientation(self):
        # Note: This gets the center of the sphere approximation of the particles, NOT the actual particle frames!
        if self.n_particles > 0:
            tfs = self.particles_view.get_transforms()
            pos, ori = tfs[:, :3], tfs[:, 3:]
            pos = pos + T.quat2mat(ori) @ self._particle_offset
        else:
            pos, ori = th.empty(0).reshape(0, 3), th.empty(0).reshape(0, 4)
        return pos, ori

    def get_particles_local_pose(self):
        return self.get_particles_position_orientation()

    def get_particle_position_orientation(self, idx):
        assert (
            idx <= self.n_particles
        ), f"Got invalid idx for getting particle pose! N particles: {self.n_particles}, got idx: {idx}"
        positions, orientations = self.get_particles_position_orientation()
        return (positions[idx], orientations[idx]) if self.n_particles > 0 else (positions, orientations)

    def get_particle_local_pose(self, idx):
        return self.get_particle_position_orientation(idx=idx)

    def set_particles_position_orientation(self, positions=None, orientations=None):
        if self.n_particles == 0:
            return

        # Note: This sets the center of the sphere approximation of the particles, NOT the actual particle frames!
        if positions is None or orientations is None:
            pos, ori = self.get_particles_position_orientation()
            orientations = ori if orientations is None else orientations
            positions = pos if positions is None else (positions - T.quat2mat(orientations) @ self._particle_offset)
        self.particles_view.set_transforms(th.cat([positions, orientations], dim=1), indices=th.arange(len(positions)))

    def set_particles_local_pose(self, positions=None, orientations=None):
        self.set_particles_position_orientation(positions=positions, orientations=orientations)

    def set_particle_position_orientation(self, idx, position=None, orientation=None):
        assert (
            idx <= self.n_particles
        ), f"Got invalid idx for setting particle pose! N particles: {self.n_particles}, got idx: {idx}"
        if position is None or orientation is None:
            pos, ori = self.get_particle_position_orientation(idx=idx)
            orientation = ori if orientation is None else orientation
            position = pos if position is None else (position - T.quat2mat(orientation) @ self._particle_offset)
        self.particles_view.set_transforms(th.cat([position, orientation]).reshape(1, -1), indices=th.tensor([idx]))

    def set_particle_local_pose(self, idx, position=None, orientation=None):
        self.set_particle_position_orientation(idx=idx, position=position, orientation=orientation)

    def get_particles_velocities(self):
        """
        Grab particles' global linear and angular velocities

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x, y, z) linear velocities in the world frame
                - (n, 3)-array: per-particle (ax, ay, az) angular velocities in the world frame
        """
        if self.n_particles > 0:
            vels = self.particles_view.get_velocities()
            lin_vel, ang_vel = vels[:, :3], vels[:, 3:]
        else:
            lin_vel, ang_vel = th.empty(0).reshape(0, 3), th.empty(0).reshape(0, 3)
        return lin_vel, ang_vel

    def get_particle_velocities(self, idx):
        """
        Grab particle @idx's global linear and angular velocities

        Returns:
            2-tuple:
                - 3-array: particle (x, y, z) linear velocity in the world frame
                - 3-array: particle (ax, ay, az) angular velocity in the world frame
        """
        assert (
            idx <= self.n_particles
        ), f"Got invalid idx for getting particle velocity! N particles: {self.n_particles}, got idx: {idx}"
        lin_vel, ang_vel = self.get_particles_velocities()
        return (lin_vel[idx], ang_vel[idx]) if self.n_particles > 0 else lin_vel, ang_vel

    def set_particles_velocities(self, lin_vels=None, ang_vels=None):
        if self.n_particles == 0:
            return

        if lin_vels is None or ang_vels is None:
            l_vels, a_vels = self.get_particles_velocities()
            lin_vels = l_vels if lin_vels is None else lin_vels
            ang_vels = a_vels if ang_vels is None else ang_vels
        self.particles_view.set_velocities(th.cat([lin_vels, ang_vels], dim=1), indices=th.arange(len(lin_vels)))

    def set_particle_velocities(self, idx, lin_vel=None, ang_vel=None):
        assert (
            idx <= self.n_particles
        ), f"Got invalid idx for setting particle velocity! N particles: {self.n_particles}, got idx: {idx}"
        if lin_vel is None or ang_vel is None:
            l_vel, a_vel = self.get_particles_velocities()
            lin_vel = l_vel if lin_vel is None else lin_vel
            ang_vel = a_vel if ang_vel is None else ang_vel
        self.particles_view.set_velocities(th.cat([lin_vel, ang_vel]).reshape(1, -1), indices=th.tensor([idx]))

    @property
    def particle_radius(self):
        return self._particle_radius

    @property
    def particle_contact_radius(self):
        # This is simply the normal radius
        return self.particle_radius

    @property
    def particle_density(self):
        """
        Returns:
            float: Particle density for the generated system
        """
        return self._particle_density

    def generate_particles(
        self,
        positions,
        orientations=None,
        velocities=None,
        angular_velocities=None,
        scales=None,
        **kwargs,
    ):
        """
        Generates new particles

        Args:
            positions (th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions
            orientations (None or th.tensor): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be sampled randomly
            velocities (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
                If not specified, all will be set to 0
            angular_velocities (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (ax,ay,az)
                angular velocities. If not specified, all will be set to 0
            scales (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (self.min_scale, self.max_scale)
            **kwargs (dict): Any additional keyword-specific arguments required by subclass implementation
        """
        if not isinstance(positions, th.Tensor):
            positions = th.tensor(positions, dtype=th.float32)

        # Call super first
        super().generate_particles(
            positions=positions,
            orientations=orientations,
            scales=scales,
            **kwargs,
        )

        # Grab pre-existing vels -- note that this already includes the newly included particles, so we will only
        # keep the first (N - n_new) values
        current_lin_vels, current_ang_vels = self.get_particles_velocities()

        # Update the tensors
        n_particles = len(positions)
        velocities = th.zeros((n_particles, 3)) if velocities is None else velocities
        angular_velocities = th.zeros_like(velocities) if angular_velocities is None else angular_velocities

        velocities = th.cat([current_lin_vels[:-n_particles], velocities], dim=0)
        angular_velocities = th.cat([current_ang_vels[:-n_particles], angular_velocities], dim=0)

        # Set the vels
        self.set_particles_velocities(lin_vels=velocities, ang_vels=angular_velocities)

    @property
    def _register_system(self):
        return True

    def _create_particle_template(self):
        return self._create_particle_template_fcn(
            relative_prim_path=f"/{self.name}/template", name=f"{self.name}_template"
        )

    def _sync_particles(self, n_particles):
        """
        Synchronizes the number of particles seen in the scene with @n_particles

        Args:
            n_particles (int): Desired number of particles to force simulator to have
        """
        # Get the difference between current and desired particles
        n_particles_to_generate = n_particles - self.n_particles

        # If positive, add particles
        if n_particles_to_generate > 0:
            for i in range(n_particles_to_generate):
                # Min scale == max scale, so no need for sampling
                self.add_particle(relative_prim_path=f"{self.relative_prim_path}/particles", scale=self.max_scale)
        else:
            # Remove excess particles
            self.remove_particles(idxs=th.arange(-n_particles_to_generate))

    def _dump_state(self):
        state = super()._dump_state()

        # Store all particles' velocities as well
        state["lin_velocities"], state["ang_velocities"] = self.get_particles_velocities()

        return state

    def _load_state(self, state):
        # Sync the number of particles first
        self._sync_particles(n_particles=state["n_particles"])

        super()._load_state(state=state)

        if self.initialized:
            # Make sure view is refreshed
            self.refresh_particles_view()

        # Make sure we update all the velocities
        self.set_particles_velocities(state["lin_velocities"], state["ang_velocities"])

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Add velocities
        return th.cat([state_flat, state["lin_velocities"].flatten(), state["ang_velocities"].flatten()])

    def deserialize(self, state):
        # Sync the number of particles first
        self._sync_particles(n_particles=int(state[0]))

        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize velocities
        len_velocities = 3 * state_dict["n_particles"]
        for vel in ("lin_velocities", "ang_velocities"):
            state_dict[vel] = state[idx : idx + len_velocities].reshape(-1, 3)
            idx += len_velocities

        return state_dict, idx
