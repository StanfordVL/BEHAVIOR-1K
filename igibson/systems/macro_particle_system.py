import os
from igibson import assets_path
from igibson.utils.usd_utils import create_joint
from igibson.systems.particle_system_base import BaseParticleSystem
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import classproperty
from igibson.utils.sampling_utils import sample_cuboid_on_object
from collections import OrderedDict
import numpy as np
from pxr import Gf
import logging


class MacroParticleSystem(BaseParticleSystem):
    """
    Global system for modeling "macro" level particles, e.g.: dirt, dust, etc.
    """
    # Template object to use -- this should be some instance of BasePrim. This will be the
    # object that symbolizes a single particle, and will be duplicated to generate the particle system.
    # Note that this object is NOT part of the actual particle system itself!
    particle_object = None

    # OrderedDict, array of particle objects, mapped by their prim names
    particles = None

    # Scaling factor to sample from when generating a new particle
    min_scale = None              # (x,y,z) scaling
    max_scale = None              # (x,y,z) scaling

    @classmethod
    def initialize(cls, simulator):
        # Run super method first
        super().initialize(simulator=simulator)

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        cls.particles = OrderedDict()
        cls.min_scale = np.ones(3)
        cls.max_scale = np.ones(3)

    @classproperty
    def n_particles(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        return len(cls.particles)

    @classproperty
    def state_size(cls):
        # We have n_particles (1), each particle pose (7*n), scale (3*n), and
        # possibly template pose (7), and template scale (3)
        state_size = 10 * cls.n_particles + 1
        return state_size if cls.particle_object is None else state_size + 10

    @classmethod
    def _dump_state(cls):
        return OrderedDict(
            n_particles=cls.n_particles,
            poses=[particle.get_position_orientation() for particle in cls.particles.values()],
            scales=[particle.scale for particle in cls.particles.values()],
            template_pose=cls.particle_object.get_position_orientation() if cls.particle_object is not None else None,
            template_scale=cls.particle_object.scale if cls.particle_object is not None else None,
        )

    @classmethod
    def _load_state(cls, state):
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (OrderedDict): Keyword-mapped states of this object to set
        """
        # Sanity check loading particles
        assert cls.n_particles == state["n_particles"], f"Inconsistent number of particles found when loading " \
                                                        f"particles state! Current number: {cls.n_particles}, " \
                                                        f"loaded number: {state['n_particles']}"

        # Load the poses and scales
        for particle, pose, scale in zip(cls.particles.values(), state["poses"], state["scales"]):
            particle.set_position_orientation(*pose)
            particle.scale = scale

        # Load template pose and scale if it exists
        if state["template_pose"] is not None:
            cls.particle_object.set_position_orientation(*state["template_pose"])
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

        return np.concatenate(states_flat)

    @classmethod
    def _deserialize(cls, state):
        # First index is number of particles, rest are the individual particle poses
        state_dict = OrderedDict()
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

        # Update idx -- one from n_particles + 10*n_particles for pose + scale
        idx = 1 + n_particles * 10

        template_pose, template_scale = None, None
        # If our state size is larger than the current index we're at, this corresponds to the template info
        if cls.state_size > idx:
            template_pose = [
                state[idx : idx + 3],
                state[idx + 3 : idx + 7],
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
        for particle in cls.particles.values():
            cls.simulator.stage.RemovePrim(particle.prim_path)

        cls.particles = OrderedDict()

    @classmethod
    def add_particle(cls, prim_path, scale=None, position=None, orientation=None):
        """
        Adds a particle to this system.

        Args:
            prim_path (str): Absolute path to the newly created particle
            scale (None or 3-array): Relative (x,y,z) scale of the particle, if any. If not specified, will
                automatically be sampled based on cls.min_scale and cls.max_scale
            position (None or 3-array): Global (x,y,z) position to set this particle to, if any
            orientation (None or 4-array): Global (x,y,z,w) quaternion orientation to set this particle to, if any

        Returns:
            BasePrim: Newly created particle instance, which is added internally as well
        """
        # Duplicate the prim template
        new_particle = cls.particle_object.duplicate(simulator=cls.simulator, prim_path=prim_path)

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
        # TODO: This causes segfaults UNLESS simulator is stopped
        # cls.simulator.stage.RemovePrim(particle.prim_path)


class VisualParticleSystem(MacroParticleSystem):
    """
    Particle system class that additionally includes sampling utilities for placing particles on specific objects
    """
    # Maps group name to the particles associated with it
    # This is an ordered dict of ordered dict (nested ordered dict maps particle names to particle instance)
    _group_particles = None

    # Maps group name to the parent object (the object with particles attached to it) of the group
    _group_objects = None

    # Default behavior for this class -- whether to clip generated particles halfway into objects when sampling
    # their locations on the surface of the given object
    _CLIP_INTO_OBJECTS = False

    # Default number of particles to sample per group
    _N_PARTICLES_PER_GROUP = 20

    # Default parameters for sampling particle locations
    # See igibson/utils/sampling_utils.py for how they are used.
    _SAMPLING_AXIS_PROBABILITIES = (0.25, 0.25, 0.5)
    _SAMPLING_AABB_OFFSET = 0.1
    _SAMPLING_BIMODAL_MEAN_FRACTION = 0.9
    _SAMPLING_BIMODAL_STDEV_FRACTION = 0.2
    _SAMPLING_MAX_ATTEMPTS = 20

    # List, keeps track of particles that need to be removed after the next sim step
    _particles_to_remove = None

    @classmethod
    def initialize(cls, simulator):
        # Run super method first
        super().initialize(simulator=simulator)

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        cls._group_particles = OrderedDict()
        cls._group_objects = OrderedDict()
        cls._particles_to_remove = []

    @classproperty
    def groups(cls):
        """
        Returns:
            set of str: Current attachment particle group names
        """
        return set(cls._group_particles.keys())

    @classmethod
    def update(cls):
        # Call super first
        super().update()

        # Remove any particles that have been requested to be removed
        for particle in cls._particles_to_remove:
            cls.remove_particle(name=particle.name)
        cls._particles_to_remove = []

    @classmethod
    def remove_all_particles(cls):
        # Run super method first
        super().remove_all_particles()

        # Clear all groups as well
        cls._group_particles = OrderedDict()
        cls._group_objects = OrderedDict()

        # Make sure particles to remove is empty
        cls._particles_to_remove = []

    @classmethod
    def remove_particle(cls, name):
        """
        Remove particle with name @name from both the simulator as well as internally

        Args:
            name (str): Name of the particle to remove
        """
        # Run super first
        super().remove_particle(name=name)

        #  Remove this particle from its respective group as well
        for group in cls._group_particles.values():
            # Maybe make this better? We have to manually search through the groups for this particle
            if name in group:
                group.pop(name)
                break

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
        for particle_name in cls._group_particles[group].keys():
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
        group = obj.name
        # This should only happen once for a single attachment group, so we explicitly check to make sure the object
        # doesn't already exist
        assert group not in cls.groups, \
            f"Cannot create new attachment group because group with name {group} already exists!"

        # Create the group
        cls._group_particles[group] = OrderedDict()
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
    def generate_group_particles(cls, group, n_particles=_N_PARTICLES_PER_GROUP, min_particles_for_success=1):
        """
        Generates @n_particles new particle objects and samples their locations on the surface of object @obj. Note
        that if any objects are in the group already, they will be removed

        Args:
            group (str): Object on which to sample particle locations
            n_particles (int): Number of particles to sample on the surface of @obj
            min_particles_for_success (int): Minimum number of particles required to be sampled successfully in order
                for this generation process to be considered successful

        Returns:
            bool: True if enough particles were generated successfully (number of successfully sampled points >=
                min_particles_for_success), otherwise False
        """
        # Make sure the group exists
        cls._validate_group(group=group)

        # Remove all stale particles
        cls.remove_all_group_particles(group=group)

        # Generate requested number of particles
        obj = cls._group_objects[group]
        bboxes = []
        for i in range(n_particles):
            # This already scales the particle to a random size
            particle = cls.add_particle(prim_path=f"{obj.prim_path}/particle{cls.n_particles}")
            # Add to group
            cls._group_particles[group][particle.name] = particle
            # Grab its bounding box
            bboxes.append(particle.bbox.tolist())

        # Sample locations for all particles
        # TODO: Does simulation need to play at this point in time?
        results = sample_cuboid_on_object(
            obj=obj,
            num_samples=n_particles,
            cuboid_dimensions=bboxes,
            bimodal_mean_fraction=cls._SAMPLING_BIMODAL_MEAN_FRACTION,
            bimodal_stdev_fraction=cls._SAMPLING_BIMODAL_STDEV_FRACTION,
            axis_probabilities=cls._SAMPLING_AXIS_PROBABILITIES,
            undo_padding=True,
            aabb_offset=cls._SAMPLING_AABB_OFFSET,
            max_sampling_attempts=cls._SAMPLING_MAX_ATTEMPTS,
            refuse_downwards=True,
        )

        print(f"RESULTS: \n{results}")

        # Get total number of sampled points
        n_success = sum(result[0] is not None for result in results)

        # If we aren't successful, then we will remove all particles and terminate early
        if n_success < min_particles_for_success:
            cls._particles_to_remove += list(cls._group_particles[group].values())
            group = None

        else:
            # Use sampled points
            for result, particle in zip(results, cls._group_particles[group].values()):
                position, normal, quaternion, hit_link, reasons = result

                # For now, we make sure all points were sampled successfully
                # TODO: Need to guarantee that all points are sampled correctly
                # assert position is not None, f"Unsuccessfully sampled some points!"
                if position is not None:
                    # Compute the point to stick the particle to.
                    surface_point = position
                    if cls._CLIP_INTO_OBJECTS:
                        # Shift the object halfway down.
                        cuboid_base_to_center = particle.bbox[2] / 2.0
                        surface_point -= normal * cuboid_base_to_center
                    # Set the pose for this particle
                    particle.set_position_orientation(position=position, orientation=quaternion)
                    # Fix to the hit link as well
                    joint_prim = create_joint(
                        prim_path=f"{particle.prim_path}/rootJoint",
                        joint_type="FixedJoint",
                        body0=hit_link,
                        body1=f"{particle.prim_path}/base_link",
                    )
                    # Make sure to offset this prim's position and orientation accordingly
                    # We use the raw local transforms from omni, NOT the inferred physical transforms
                    raw_relative_pos = particle.get_attribute("xformOp:translate")
                    raw_relative_quat = Gf.Quatf(particle.get_attribute("xformOp:orient"))
                    joint_prim.GetAttribute("physics:localPos0").Set(raw_relative_pos)
                    joint_prim.GetAttribute("physics:localRot0").Set(raw_relative_quat)
                else:
                    # Delete this object after we're done
                    # TODO: Figure out better way to do this
                    cls._particles_to_remove.append(particle)

        return group

    @classmethod
    def _validate_group(cls, group):
        """
        Checks if particle attachment group @group exists. (If not, can create the group via create_attachment_group).
        This will raise a ValueError if it doesn't exist.

        Args:
            group: Name of the group to check for
        """
        if group not in cls.groups:
            raise ValueError(f"Particle attachment group {group} does not exist!")


class DustSystem(VisualParticleSystem):
    """
    Particle system class to symbolize dust attached to objects
    """
    @classproperty
    def _register_system(cls):
        # We should register this system since it's an "actual" system (not an intermediate class)
        return True

    @classmethod
    def initialize(cls, simulator):
        # Run super first
        super().initialize(simulator=simulator)

        # Particle object will be overridden by default to be a small cuboid
        # We import now at runtime so prevent circular imports
        from igibson.objects.primitive_object import PrimitiveObject
        cls.particle_object = PrimitiveObject(
            prim_path=f"/World/{cls.name}/dust_template",
            primitive_type="Cube",
            name="dust_template",
            class_id=SemanticClass.DIRT,
            scale=np.array([0.015, 0.015, 0.015]),
            visible=False,
            fixed_base=False,
            visual_only=True,
        )

        # We also must load the particle object
        simulator.import_object(obj=cls.particle_object, register=False, auto_initialize=True)


class StainSystem(VisualParticleSystem):
    """
    Particle system class to symbolize stains attached to objects
    """
    # Default number of particles to sample per group
    _N_PARTICLES_PER_GROUP = 20

    # Default parameters for sampling particle sizes based on attachment group object size
    _BOUNDING_BOX_LOWER_LIMIT_FRACTION_OF_AABB = 0.06
    _BOUNDING_BOX_LOWER_LIMIT_MIN = 0.01
    _BOUNDING_BOX_LOWER_LIMIT_MAX = 0.02

    _BOUNDING_BOX_UPPER_LIMIT_FRACTION_OF_AABB = 0.1
    _BOUNDING_BOX_UPPER_LIMIT_MIN = 0.02
    _BOUNDING_BOX_UPPER_LIMIT_MAX = 0.1

    @classproperty
    def _register_system(cls):
        # We should register this system since it's an "actual" system (not an intermediate class)
        return True

    @classmethod
    def initialize(cls, simulator):
        # Run super first
        super().initialize(simulator=simulator)

        # Particle object will be overridden to by default be a specific USD file
        # We import now at runtime so prevent circular imports
        from igibson.objects.usd_object import USDObject
        cls.particle_object = USDObject(
            prim_path=f"/World/{cls.name}/stain_template",
            usd_path=os.path.join(assets_path, "models/stain/stain.usd"),
            name="stain_template",
            class_id=SemanticClass.DIRT,
            visible=False,
            fixed_base=False,
            visual_only=True,
        )

        # We also must load the particle object
        simulator.import_object(obj=cls.particle_object, register=False, auto_initialize=True)

    @classmethod
    def generate_group_particles(cls, group, n_particles=_N_PARTICLES_PER_GROUP, min_particles_for_success=1):
        # Make sure the group exists
        cls._validate_group(group=group)

        # First set the bbox ranges -- depends on the object's bounding box
        obj = cls._group_objects[group]
        median_aabb_dim = np.median(obj.bbox)

        # Compute lower and upper limits to bbox
        bbox_lower_limit_from_aabb = cls._BOUNDING_BOX_LOWER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
        bbox_lower_limit = np.clip(
            bbox_lower_limit_from_aabb,
            cls._BOUNDING_BOX_LOWER_LIMIT_MIN,
            cls._BOUNDING_BOX_LOWER_LIMIT_MAX,
        )

        bbox_upper_limit_from_aabb = cls._BOUNDING_BOX_UPPER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
        bbox_upper_limit = np.clip(
            bbox_upper_limit_from_aabb,
            cls._BOUNDING_BOX_UPPER_LIMIT_MIN,
            cls._BOUNDING_BOX_UPPER_LIMIT_MAX,
        )

        # Convert these into scaling factors for the x and y axes for our particle object
        particle_bbox = cls.particle_object.bbox
        cls.set_scale_limits(
            minimum=np.array([bbox_lower_limit / particle_bbox[0], bbox_lower_limit / particle_bbox[1], 1.0]),
            maximum=np.array([bbox_upper_limit / particle_bbox[0], bbox_upper_limit / particle_bbox[1], 1.0]),
        )

        # Run super method like normal to generate particles
        super().generate_group_particles(
            group=group,
            n_particles=n_particles,
            min_particles_for_success=min_particles_for_success
        )
