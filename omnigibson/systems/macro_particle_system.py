import os
import matplotlib.pyplot as plt
import omni
from omni.isaac.core.utils.prims import get_prim_at_path

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.systems.system_base import BaseSystem, REGISTERED_SYSTEMS
from omnigibson.utils.constants import SemanticClass
from omnigibson.utils.python_utils import classproperty, subclass_factory, snake_case_to_camel_case
from omnigibson.utils.sampling_utils import sample_cuboid_on_object_symmetric_bimodal_distribution
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import FlatcacheAPI
from omnigibson.prims.geom_prim import VisualGeomPrim
import numpy as np
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Parameters used if scaling particles relative to its parent object's scale
m.BBOX_LOWER_LIMIT_FRACTION_OF_AABB = 0.06
m.BBOX_LOWER_LIMIT_MIN = 0.01
m.BBOX_LOWER_LIMIT_MAX = 0.02
m.BBOX_UPPER_LIMIT_FRACTION_OF_AABB = 0.1
m.BBOX_UPPER_LIMIT_MIN = 0.02
m.BBOX_UPPER_LIMIT_MAX = 0.1


def is_visual_particle_system(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    return issubclass(system, VisualParticleSystem)


class MacroParticleSystem(BaseSystem):
    """
    Global system for modeling "macro" level particles, e.g.: dirt, dust, etc.
    """
    # Template object to use -- this should be some instance of BasePrim. This will be the
    # object that symbolizes a single particle, and will be duplicated to generate the particle system.
    # Note that this object is NOT part of the actual particle system itself!
    particle_object = None

    # dict, array of particle objects, mapped by their prim names
    particles = None

    # Scaling factor to sample from when generating a new particle
    min_scale = None              # (x,y,z) scaling
    max_scale = None              # (x,y,z) scaling

    # Color associated with this system (NOTE: external queries should call cls.color)
    _color = None

    @classmethod
    def initialize(cls):
        # Run super method first
        super().initialize()

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        cls.particles = dict()
        cls.min_scale = np.ones(3)
        cls.max_scale = np.ones(3)

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
        if cls.n_particles == 0:
            return 0
        else:
            # We don't fill in any holes, just simply use the next subsequent integer after the largest
            # current ID
            return max(cls.particle_idns) + 1

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
    def reset(cls):
        # Call super first
        super().reset()

        # Reset all internal variables
        cls.remove_all_particles()

    @classmethod
    def clear(cls):
        # Call super first
        super().clear()

        # Clear all internal state
        cls.particle_object = None
        cls.particles = None
        cls.min_scale = None
        cls.max_scale = None
        cls._color = None

    @classproperty
    def n_particles(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        return len(cls.particles)

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
        # We have n_particles (1), each particle pose (7*n), scale (3*n), and
        # possibly template pose (7), and template scale (3)
        state_size = 10 * cls.n_particles + 1
        return state_size if cls.particle_object is None else state_size + 10

    @classmethod
    def _dump_state(cls):
        return dict(
            n_particles=cls.n_particles,
            poses=[particle.get_local_pose() for particle in cls.particles.values()],
            scales=[particle.scale for particle in cls.particles.values()],
            template_pose=cls.particle_object.get_local_pose() if cls.particle_object is not None else None,
            template_scale=cls.particle_object.scale if cls.particle_object is not None else None,
        )

    @classmethod
    def _load_state(cls, state):
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (dict): Keyword-mapped states of this object to set
        """
        # Sanity check loading particles
        assert cls.n_particles == state["n_particles"], f"Inconsistent number of particles found when loading " \
                                                        f"particles state! Current number: {cls.n_particles}, " \
                                                        f"loaded number: {state['n_particles']}"
        # Load the poses and scales
        for particle, pose, scale in zip(cls.particles.values(), state["poses"], state["scales"]):
            particle.set_local_pose(*pose)
            particle.scale = scale

        # Load template pose and scale if it exists
        if state["template_pose"] is not None:
            cls.particle_object.set_local_pose(*state["template_pose"])
            cls.particle_object.scale = state["template_scale"]

    @classmethod
    def _serialize(cls, state):
        # Array is n_particles + poses for all particles, then the template info
        states_flat = [
            [state["n_particles"]],
            *[np.concatenate(pose) for pose in state["poses"]],
            *state["scales"]
        ]

        # Optionally add template pose and scale if it's not None
        if state["template_pose"] is not None:
            states_flat += [*state["template_pose"], state["template_scale"]]

        return np.concatenate(states_flat).astype(float)

    @classmethod
    def _deserialize(cls, state):
        # First index is number of particles, rest are the individual particle poses
        state_dict = dict()
        n_particles = int(state[0])
        state_dict["n_particles"] = n_particles

        poses, scales = [], []
        pose_offset_idx = 1                                 # This is where the pose info begins in the flattened array
        scale_offset_idx = n_particles * 7 + pose_offset_idx  # This is where the scale info begins in the flattened array
        for i in range(n_particles):
            poses.append([
                state[7*i + pose_offset_idx: 7*i + pose_offset_idx + 3],
                state[7*i + pose_offset_idx + 3: 7*(i+1) + pose_offset_idx]
            ])      # pos, ori
            scales.append(state[3*i + scale_offset_idx : 3*(i + 1) + scale_offset_idx])      # scale

        state_dict["poses"] = poses
        state_dict["scales"] = scales

        # Update idx - 1 for n_particles + 10*n_particles for pose + scale
        idx = 1 + n_particles * 10

        template_pose, template_scale = None, None
        # If our state size is larger than the current index we're at, this corresponds to the template info
        if cls.state_size > idx:
            template_pose = [
                state[idx: idx + 3],
                state[idx + 3: idx + 7],
            ]
            template_scale = state[idx + 7: idx + 10]
            idx += 10

        state_dict["template_pose"] = template_pose
        state_dict["template_scale"] = template_scale

        return state_dict, idx

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
        cls.particle_object = obj

    @classmethod
    def set_scale_limits(cls, minimum=None, maximum=None):
        """
        Set the min and / or max scaling limits that will be uniformly sampled from when generating new particles

        Args:
            minimum (None or 3-array): If specified, should be (x,y,z) minimum scaling factor to apply to generated
                particles
            maximum (None or 3-array): If specified, should be (x,y,z) maximum scaling factor to apply to generated
                particles
        """
        if minimum is not None:
            cls.min_scale = np.array(minimum)
        if maximum is not None:
            cls.max_scale = np.array(maximum)

    @classmethod
    def remove_all_particles(cls):
        """
        Removes all particles and deletes them from the simulator
        """
        # Use list explicitly to prevent mid-loop mutation of dict
        for particle_name in tuple(cls.particles.keys()):
            cls.remove_particle(name=particle_name)

    @classmethod
    def add_particle(cls, prim_path, idn=None, scale=None, position=None, orientation=None):
        """
        Adds a particle to this system.

        Args:
            prim_path (str): Absolute path to the newly created particle, minus the name for this particle
            idn (None or int): If specified, should be unique identifier to assign to this particle. If not, will
                automatically generate a new unique one
            scale (None or 3-array): Relative (x,y,z) scale of the particle, if any. If not specified, will
                automatically be sampled based on cls.min_scale and cls.max_scale
            position (None or 3-array): Global (x,y,z) position to set this particle to, if any
            orientation (None or 4-array): Global (x,y,z,w) quaternion orientation to set this particle to, if any

        Returns:
            XFormPrim: Newly created particle instance, which is added internally as well
        """
        # Generate the new particle
        name = cls.particle_idn2name(idn=cls.next_available_particle_idn if idn is None else idn)
        # Make sure name doesn't already exist
        assert name not in cls.particles.keys(), f"Cannot create particle with name {name} because it already exists!"
        new_particle = cls._load_new_particle(prim_path=f"{prim_path}/{name}", name=name)

        # Sample the scale and also make sure the particle is visible
        new_particle.scale *= np.random.uniform(cls.min_scale, cls.max_scale) if scale is None else scale
        new_particle.visible = True

        # Set the pose
        new_particle.set_position_orientation(position=position, orientation=orientation)

        # Track this particle as well
        cls.particles[new_particle.name] = new_particle

        return new_particle

    @classmethod
    def remove_particle(cls, name):
        """
        Remove particle with name @name from both the simulator as well as internally

        Args:
            name (str): Name of the particle to remove
        """
        assert name in cls.particles, f"Got invalid name for particle to remove {name}"

        particle = cls.particles.pop(name)
        particle.remove()

    @classmethod
    def _load_new_particle(cls, prim_path, name):
        """
        Loads a new particle into the current stage, leveraging @cls.particle_object as a template for the new particle
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


class VisualParticleSystem(MacroParticleSystem):
    """
    Particle system class that additionally includes sampling utilities for placing particles on specific objects
    """
    # Maps group name to the particles associated with it
    # This is an ordered dict of ordered dict (nested ordered dict maps particle names to particle instance)
    _group_particles = None

    # Maps group name to the parent object (the object with particles attached to it) of the group
    _group_objects = None

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
        cls._group_particles = dict()
        cls._group_objects = dict()
        cls._particles_info = dict()
        cls._particles_local_mat = dict()

    @classproperty
    def groups(cls):
        """
        Returns:
            set of str: Current attachment particle group names
        """
        return set(cls._group_particles.keys())

    @classproperty
    def scale_relative_to_parent(cls):
        """
        Returns:
            bool: Whether or not particles should be scaled relative to the group's parent object
        """
        return False

    @classproperty
    def state_size(cls):
        # Get super size first
        state_size = super().state_size

        # Additionally, we have n_groups (1), with m_particles for each group (n), attached_obj_uuids (n), and
        # particle ids and corresponding link info for each particle (m * 2)
        return state_size + 1 + 2 * len(cls._group_particles) + \
               sum(2 * cls.num_group_particles(group) for group in cls.groups)

    @classmethod
    def _load_new_particle(cls, prim_path, name):
        # We copy the template prim and generate the new object if the prim doesn't already exist, otherwise we
        # reference the pre-existing one
        if not get_prim_at_path(prim_path):
            omni.kit.commands.execute(
                "CopyPrim",
                path_from=cls.particle_object.prim_path,
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
        cls._group_particles = dict()
        cls._group_objects = dict()
        cls._particles_info = dict()
        cls._particles_local_mat = dict()

    @classmethod
    def remove_particle(cls, name):
        """
        Remove particle with name @name from both the simulator and internal state

        Args:
            name (str): Name of the particle to remove
        """
        # Run super first
        super().remove_particle(name=name)

        # Remove this particle from its respective group as well
        cls._group_particles[cls._particles_info[name]["obj"].name].pop(name)
        cls._particles_info.pop(name)
        cls._particles_local_mat.pop(name)

    @classmethod
    def remove_all_group_particles(cls, group):
        """
        Remove particle with name @name from both the simulator as well as internally

        Args:
            group (str): Name of the attachment group to remove all particles from
        """
        # Make sure the group exists
        cls._validate_group(group=group)
        # Remove all particles from the group
        for particle_name in tuple(cls._group_particles[group].keys()):
            cls.remove_particle(name=particle_name)

    @classmethod
    def num_group_particles(cls, group):
        """
        Gets the number of particles for the given group in the simulator

        Args:
            group (str): Name of the attachment group to remove all particles from.

        Returns:
            int: Number of particles allocated to this group in the scene. Note that if @group does not
                exist, this will return 0
        """
        # Make sure the group exists
        cls._validate_group(group=group)
        return len(cls._group_particles[group])

    @classmethod
    def get_group_name(cls, obj):
        """
        Grabs the corresponding group name for object @obj

        Args:
            obj (BaseObject): Object for which its procedurally generated particle attachment name should be grabbed

        Returns:
            str: Name of the attachment group to use when executing commands from this class on
                that specific attachment group
        """
        return obj.name

    @classmethod
    def create_attachment_group(cls, obj):
        """
        Creates an attachment group internally for object @obj. Note that this does NOT automatically generate particles
        for this object (should call generate_group_particles(...) ).

        Args:
            obj (BaseObject): Object for which a new particle attachment group will be created for

        Returns:
            str: Name of the attachment group to use when executing commands from this class on
                that specific attachment group
        """
        group = cls.get_group_name(obj=obj)
        # This should only happen once for a single attachment group, so we explicitly check to make sure the object
        # doesn't already exist
        assert group not in cls.groups, \
            f"Cannot create new attachment group because group with name {group} already exists!"

        # Create the group
        cls._group_particles[group] = dict()
        cls._group_objects[group] = obj

        return group

    @classmethod
    def remove_attachment_group(cls, group):
        """
        Removes an attachment group internally for object @obj. Note that this will automatically remove any particles
        currently assigned to that group

        Args:
            group (str): Name of the attachment group to remove

        Returns:
            str: Name of the attachment group to use when executing commands from this class on
                that specific attachment group
        """
        # Make sure the group exists
        cls._validate_group(group=group)

        # Remove all particles from the group
        cls.remove_all_group_particles(group=group)

        # Remove the actual groups
        cls._group_particles.pop(group)
        cls._group_objects.pop(group)

        return group

    @classmethod
    def update_particle_scaling(cls, group):
        """
        Update particle scaling for group @group before generating group particles. Default is a no-op
        (i.e.: returns the current cls.min_scale, cls.max_scale)

        Args:
            group (str): Specific group for which to modify the particle scaling

        Returns:
            2-tuple:
                - 3-array: min scaling factor to set
                - 3-array: max scaling factor to set
        """
        if cls.scale_relative_to_parent:
            # First set the bbox ranges -- depends on the object's bounding box
            obj = cls._group_objects[group]
            median_aabb_dim = np.median(obj.aabb_extent)

            # Compute lower and upper limits to bbox
            bbox_lower_limit_from_aabb = m.BBOX_LOWER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
            bbox_lower_limit = np.clip(
                bbox_lower_limit_from_aabb,
                m.BBOX_LOWER_LIMIT_MIN,
                m.BBOX_LOWER_LIMIT_MAX,
            )

            bbox_upper_limit_from_aabb = m.BBOX_UPPER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
            bbox_upper_limit = np.clip(
                bbox_upper_limit_from_aabb,
                m.BBOX_UPPER_LIMIT_MIN,
                m.BBOX_UPPER_LIMIT_MAX,
            )

            # Convert these into scaling factors for the x and y axes for our particle object
            particle_bbox = cls.particle_object.aabb_extent
            minimum = np.array([bbox_lower_limit / particle_bbox[0], bbox_lower_limit / particle_bbox[1], 1.0])
            maximum = np.array([bbox_upper_limit / particle_bbox[0], bbox_upper_limit / particle_bbox[1], 1.0])
        else:
            minimum, maximum = cls.min_scale, cls.max_scale

        return minimum, maximum

    @classmethod
    def sample_scales(cls, group, n):
        """
        Samples @n particle scales for group @group.

        Args:
            group (str): Specific group for which to sample scales
            n (int): Number of scales to sample

        Returns:
            (n, 3) array: Array of sampled scales
        """
        # Make sure the group exists
        cls._validate_group(group=group)

        # Update scaling and grab object
        cls.set_scale_limits(*cls.update_particle_scaling(group=group))
        obj = cls._group_objects[group]

        # Sample scales of the particles to generate
        # Since the particles will be placed under the object, it will be affected/stretched by obj.scale. In order to
        # preserve the absolute size of the particles, we need to scale the particle by obj.scale in some way. However,
        # since the particles have a relative rotation w.r.t the object, the scale between the two don't align. As a
        # heuristics, we divide it by the avg_scale, which is the cubic root of the product of the scales along 3 axes.
        avg_scale = np.cbrt(np.product(obj.scale))
        return np.random.uniform(cls.min_scale, cls.max_scale, (n, 3)) / avg_scale

    @classmethod
    def generate_group_particles(
            cls,
            group,
            positions,
            orientations=None,
            scales=None,
            link_prim_paths=None,
    ):
        """
        Generates new particle objects within group @group at the specified pose (@positions, @orientations) with
        corresponding scales @scales.

        NOTE: Assumes positions are the exact contact point on @group object's surface. If cls._CLIP_INTO_OBJECTS
            is not True, then the positions will be offset away from the object by half of its bbox

        Args:
            group (str): Object on which to sample particle locations
            positions (np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions
            orientations (None or np.array): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) scaling in its
                local frame. If not specified, all we randomly sampled based on @cls.min_scale and @cls.max_scale
            link_prim_paths (None or list of str): Determines which link each generated particle will
                be attached to. If not specified, all will be attached to the group object's root link
        """
        # Make sure the group exists
        cls._validate_group(group=group)

        # Update scaling
        cls.set_scale_limits(*cls.update_particle_scaling(group=group))

        # Standardize orientations and links
        obj = cls._group_objects[group]
        n_particles = positions.shape[0]
        if orientations is None:
            orientations = np.zeros((n_particles, 4))
            orientations[:, -1] = 1.0
        link_prim_paths = [obj.root_link.prim_path] * n_particles if link_prim_paths is None else link_prim_paths

        if scales is None:
            scales = cls.sample_scales(group=group, n=n_particles)
        bbox_extents_local = [(cls.particle_object.aabb_extent * scale).tolist() for scale in scales]

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
            particle = cls.add_particle(
                prim_path=link_prim_path,
                position=position,
                orientation=orientation,
                scale=scale,
            )

            # Add to group
            cls._group_particles[group][particle.name] = particle
            cls._particles_info[particle.name] = dict(obj=cls._group_objects[group], link=link)

            # Update particle local matrix
            cls._particles_local_mat[particle.name] = cls._compute_particle_local_mat(name=particle.name)

    @classmethod
    def generate_group_particles_on_object(cls, group, max_samples, min_samples_for_success=1):
        """
        Generates @max_samples new particle objects and samples their locations on the surface of object @obj. Note
        that if any particles are in the group already, they will be removed

        Args:
            group (str): Object on which to sample particle locations
            max_samples (int): Maximum number of particles to sample
            min_samples_for_success (int): Minimum number of particles required to be sampled successfully in order
                for this generation process to be considered successful

        Returns:
            bool: True if enough particles were generated successfully (number of successfully sampled points >=
                min_samples_for_success), otherwise False
        """

        assert max_samples >= min_samples_for_success, "number of particles to sample should exceed the min for success"

        # Make sure the group exists
        cls._validate_group(group=group)

        # Remove all stale particles
        cls.remove_all_group_particles(group=group)

        # Generate requested number of particles
        obj = cls._group_objects[group]

        # Sample scales and corresponding bbox extents
        scales = cls.sample_scales(group=group, n=max_samples)
        # For sampling particle positions, we need the global bbox extents, NOT the local extents
        # which is what we would get naively if we directly use @scales
        avg_scale = np.cbrt(np.product(obj.scale))
        bbox_extents_global = scales * cls.particle_object.aabb_extent.reshape(1, 3) * avg_scale

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
    def get_particles_position_orientation(cls, group=None):
        """
        Computes all particles' global positions and orientations that belong to this system

        Note: This is more optimized than doing a for loop with self.get_particle_position_orientation()

        Args:
            group (None or str): If specified, will only return subset of particle positions and orientations belonging
                to @group. None results in all particles being returned

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position in the world frame
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation in the world frame
        """
        # Iterate over all particles and compute link tfs programmatically, then batch the matrix transform
        link_tfs = dict()
        link_tfs_batch = np.zeros((cls.n_particles, 4, 4))
        particle_local_poses_batch = np.zeros_like(link_tfs_batch)
        particles = cls.particles if group is None else cls._group_particles[group]
        for i, name in enumerate(particles):
            link = cls._particles_info[name]["link"]
            if link in link_tfs:
                link_tf = link_tfs[link]
            else:
                link_tf = T.pose2mat(link.get_position_orientation())
                link_tfs[link] = link_tf
            link_tfs_batch[i] = link_tf
            particle_local_poses_batch[i] = cls._particles_local_mat[name]

        # Compute once
        global_poses = np.matmul(link_tfs_batch, particle_local_poses_batch)

        # Decompose back into positions and orientations
        return global_poses[:, :3, 3], T.mat2quat(global_poses[:, :3, :3])

    @classmethod
    def get_particle_position_orientation(cls, name):
        """
        Compute particle's global position and orientation. This automatically takes into account the relative
        pose w.r.t. its parent link and the global pose of that parent link.

        Args:
            name (str): Name of the particle to compute global position and orientation for

        Returns:
            2-tuple:
                - 3-array: (x,y,z) position in the world frame
                - 4-array: (x,y,z,w) quaternion orientation in the world frame
        """
        # First, get local pose, scale it by the parent link's scale, and then convert into a matrix
        parent_link = cls._particles_info[name]["link"]
        local_mat = cls._particles_local_mat[name]
        link_tf = T.pose2mat(parent_link.get_position_orientation())

        # Multiply the local pose by the link's global transform, then return as pos, quat tuple
        val = T.mat2pose(link_tf @ local_mat)
        return val

    @classmethod
    def _compute_particle_local_mat(cls, name):
        """
        Computes particle @name's local transform as a homogeneous 4x4 matrix

        Args:
            name (str): Name of the particle to compute local transform matrix for

        Returns:
            np.array: (4, 4) homogeneous transform matrix
        """
        particle = cls.particles[name]
        parent_link = cls._particles_info[name]["link"]
        local_pos, local_quat = particle.get_local_pose()
        return T.pose2mat((parent_link.scale * local_pos, local_quat))

    @classmethod
    def _validate_group(cls, group):
        """
        Checks if particle attachment group @group exists. (If not, can create the group via create_attachment_group).
        This will raise a ValueError if it doesn't exist.

        Args:
            group (str): Name of the group to check for
        """
        if group not in cls.groups:
            raise ValueError(f"Particle attachment group {group} does not exist!")

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
                particle = cls.add_particle(
                    prim_path=f"{obj.prim_path}/{link_name}",
                    idn=int(particle_idn),
                )
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
            # Run super first (we have to use a bit esoteric syntax in order to accommodate this procedural method for
            # using super calls -- cf. https://stackoverflow.com/questions/22403897/what-does-it-mean-by-the-super-object-returned-is-unbound-in-python
            super(cls).__get__(cls).initialize()

            # Potentially override the min / max scales
            if min_scale is not None:
                cls.min_scale = np.array(min_scale)
            if max_scale is not None:
                cls.max_scale = np.array(max_scale)

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

        # Sanity check loading particles
        assert cls.n_particles == state["n_particles"], f"Inconsistent number of particles found when loading " \
                                                        f"particles state! Current number: {cls.n_particles}, " \
                                                        f"loaded number: {state['n_particles']}"

        # Run super
        super()._load_state(state=state)

        # Make sure we update all the local transforms
        for name in cls.particles:
            cls._particles_local_mat[name] = cls._compute_particle_local_mat(name=name)

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


VisualParticleSystem.create(
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


VisualParticleSystem.create(
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
