import os
import json
import numpy as np

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.utils.asset_utils import get_all_system_categories
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty, assert_valid_key, get_uuid, camel_case_to_snake_case, \
    snake_case_to_camel_case, subclass_factory, SerializableNonInstance, UniquelyNamedNonInstance
from omnigibson.utils.registry_utils import SerializableRegistry
from omnigibson.utils.sampling_utils import sample_cuboid_on_object_full_grid_topdown
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Parameters used if scaling particles relative to its parent object's scale
m.BBOX_LOWER_LIMIT_FRACTION_OF_AABB = 0.06
m.BBOX_LOWER_LIMIT_MIN = 0.002
m.BBOX_LOWER_LIMIT_MAX = 0.02
m.BBOX_UPPER_LIMIT_FRACTION_OF_AABB = 0.1
m.BBOX_UPPER_LIMIT_MIN = 0.01
m.BBOX_UPPER_LIMIT_MAX = 0.1


_CALLBACKS_ON_SYSTEM_INIT = dict()
_CALLBACKS_ON_SYSTEM_CLEAR = dict()


# Modifiers denoting a semantic difference in the system
SYSTEM_PREFIXES = {"diced", "cooked"}


class BaseSystem(SerializableNonInstance, UniquelyNamedNonInstance):
    """
    Base class for all systems. These are non-instance objects that should be used globally for a given environment.
    This is useful for items in a scene that are non-discrete / cannot be distinguished into individual instances,
    e.g.: water, particles, etc. While we keep the python convention of the system class name being camel case
    (e.g. StrawberrySmoothie), we adopt the snake case for the system registry to unify with the category of BaseObject.
    For example, get_system("strawberry_smoothie") will return the StrawberrySmoothie class.
    """
    # Scaling factor to sample from when generating a new particle
    min_scale = None              # (x,y,z) scaling
    max_scale = None              # (x,y,z) scaling

    # Whether this system has been initialized or not
    initialized = False

    # Internal variables used for bookkeeping
    _uuid = None
    _snake_case_name = None

    def __init_subclass__(cls, **kwargs):
        # While class names are camel case, we convert them to snake case to be consistent with object categories.
        name = camel_case_to_snake_case(cls.__name__)
        # Make sure prefixes preserve their double underscore
        for prefix in SYSTEM_PREFIXES:
            name = name.replace(f"{prefix}_", f"{prefix}__")
        cls._snake_case_name = name
        cls.min_scale = np.ones(3)
        cls.max_scale = np.ones(3)

        # Run super init
        super().__init_subclass__(**kwargs)

        # Register this system if requested
        if cls._register_system:
            global REGISTERED_SYSTEMS, UUID_TO_SYSTEMS
            REGISTERED_SYSTEMS[cls._snake_case_name] = cls
            cls._uuid = get_uuid(cls._snake_case_name)
            UUID_TO_SYSTEMS[cls._uuid] = cls

    @classproperty
    def name(cls):
        # Class name is the unique name assigned
        return cls._snake_case_name

    @classproperty
    def uuid(cls):
        return cls._uuid

    @classproperty
    def prim_path(cls):
        """
        Returns:
            str: Path to this system's prim in the scene stage
        """
        return f"/World/{cls.name}"

    @classproperty
    def n_particles(cls):
        """
        Returns:
            int: Number of particles belonging to this system
        """
        raise NotImplementedError()

    @classproperty
    def material(cls):
        """
        Returns:
            None or MaterialPrim: Material belonging to this system, if there is any
        """
        return None

    @classproperty
    def _register_system(cls):
        """
        Returns:
            bool: True if this system should be registered (i.e.: it is not an intermediate class but a "final" subclass
                representing a system we'd actually like to use, e.g.: water, dust, etc. Should be set by the subclass
        """
        # We assume we aren't registering by default
        return False

    @classproperty
    def _store_local_poses(cls):
        """
        Returns:
            bool: Whether to store local particle poses or not when state is saved. Default is False
        """
        return False

    @classmethod
    def initialize(cls):
        """
        Initializes this system
        """
        global _CALLBACKS_ON_SYSTEM_INIT

        assert not cls.initialized, f"Already initialized system {cls.name}!"
        og.sim.stage.DefinePrim(cls.prim_path, "Scope")

        cls.initialized = True

        # Add to registry
        SYSTEM_REGISTRY.add(obj=cls)
        # Make sure to refresh any transition rules that require this system
        # Import now to avoid circular imports
        from omnigibson.transition_rules import TransitionRuleAPI, RULES_REGISTRY
        system_rules = RULES_REGISTRY("required_systems", cls.name, default_val=[])
        TransitionRuleAPI.refresh_rules(rules=system_rules)

        # Run any callbacks
        for callback in _CALLBACKS_ON_SYSTEM_INIT.values():
            callback(cls)

    @classmethod
    def update(cls):
        """
        Executes any necessary system updates, once per og.sim._non_physics_step
        """
        # Default is no-op
        pass

    @classmethod
    def remove_all_particles(cls):
        """
        Removes all particles and deletes them from the simulator
        """
        raise NotImplementedError()

    @classmethod
    def remove_particles(
            cls,
            idxs,
            **kwargs,
    ):
        """
        Removes pre-existing particles

        Args:
            idxs (np.array): (n_particles,) shaped array specifying IDs of particles to delete
            **kwargs (dict): Any additional keyword-specific arguments required by subclass implementation
        """
        raise NotImplementedError()

    @classmethod
    def generate_particles(
            cls,
            positions,
            orientations=None,
            scales=None,
            **kwargs,
    ):
        """
        Generates new particles

        Args:
            positions (np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions
            orientations (None or np.array): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (cls.min_scale, cls.max_scale)
            **kwargs (dict): Any additional keyword-specific arguments required by subclass implementation
        """
        raise NotImplementedError()

    @classmethod
    def clear(cls):
        """
        Clears this system, so that it may possibly be re-initialized. Useful for, e.g., when loading from a new
        scene during the same sim instance
        """
        global SYSTEM_REGISTRY, _CALLBACKS_ON_SYSTEM_CLEAR

        if cls.initialized:
            # Run any callbacks
            for callback in _CALLBACKS_ON_SYSTEM_CLEAR.values():
                callback(cls)

            cls.reset()
            cls.initialized = False

            # Remove from active registry
            SYSTEM_REGISTRY.remove(obj=cls)
            # Make sure to refresh any transition rules that require this system
            # Import now to avoid circular imports
            from omnigibson.transition_rules import TransitionRuleAPI, RULES_REGISTRY
            system_rules = RULES_REGISTRY("required_systems", cls.name, default_val=[])
            TransitionRuleAPI.refresh_rules(rules=system_rules)

    @classmethod
    def reset(cls):
        """
        Reset this system
        """
        cls.remove_all_particles()

    @classmethod
    def create(cls, name, min_scale=None, max_scale=None, **kwargs):
        """
        Helper function to programmatically generate systems

        Args:
            name (str): Name of the visual particles, in snake case.
            min_scale (None or 3-array): If specified, sets the minumum bound for particles' relative scale.
                Else, defaults to 1
            max_scale (None or 3-array): If specified, sets the maximum bound for particles' relative scale.
                Else, defaults to 1
            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)

        Returns:
            BaseSystem: Generated system class given input arguments
        """
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

        kwargs["initialize"] = cm_initialize

        # Create and return the class
        return subclass_factory(name=snake_case_to_camel_case(name), base_classes=cls, **kwargs)

    @classmethod
    def get_active_systems(cls):
        """
        Returns:
            dict: Mapping from system name to system for all systems that are subclasses of this system AND active (initialized)
        """
        return {system.name: system for system in SYSTEM_REGISTRY.objects if issubclass(system, cls)}

    @classmethod
    def sample_scales(cls, n):
        """
        Samples scales uniformly based on @cls.min_scale and @cls.max_scale

        Args:
            n (int): Number of scales to sample

        Returns:
            (n, 3) array: Array of sampled scales
        """
        return np.random.uniform(cls.min_scale, cls.max_scale, (n, 3))

    @classmethod
    def get_particles_position_orientation(cls):
        """
        Computes all particles' positions and orientations that belong to this system in the world frame

        Note: This is more optimized than doing a for loop with self.get_particle_position_orientation()

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @classmethod
    def get_particle_position_orientation(cls, idx):
        """
        Compute particle's position and orientation. This automatically takes into account the relative
        pose w.r.t. its parent link and the global pose of that parent link.

        Args:
            idx (int): Index of the particle to compute position and orientation for. Note: this is
                equivalent to grabbing the corresponding idx'th entry from @get_particles_position_orientation()

        Returns:
            2-tuple:
                - 3-array: (x,y,z) position
                - 4-array: (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @classmethod
    def set_particles_position_orientation(cls, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to this system in the world frame

        Note: This is more optimized than doing a for loop with self.set_particle_position_orientation()

        Args:
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @classmethod
    def set_particle_position_orientation(cls, idx, position=None, orientation=None):
        """
        Sets particle's position and orientation. This automatically takes into account the relative
        pose w.r.t. its parent link and the global pose of that parent link.

        Args:
            idx (int): Index of the particle to set position and orientation for. Note: this is
                equivalent to setting the corresponding idx'th entry from @set_particles_position_orientation()
            position (3-array): particle (x,y,z) position
            orientation (4-array): particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @classmethod
    def get_particles_local_pose(cls):
        """
        Computes all particles' positions and orientations that belong to this system in the particles' parent frames

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @classmethod
    def get_particle_local_pose(cls, idx):
        """
        Compute particle's position and orientation in the particle's parent frame

        Args:
            idx (int): Index of the particle to compute position and orientation for. Note: this is
                equivalent to grabbing the corresponding idx'th entry from @get_particles_local_pose()

        Returns:
            2-tuple:
                - 3-array: (x,y,z) position
                - 4-array: (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @classmethod
    def set_particles_local_pose(cls, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to this system in the particles' parent frames

        Args:
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @classmethod
    def set_particle_local_pose(cls, idx, position=None, orientation=None):
        """
        Sets particle's position and orientation in the particle's parent frame

        Args:
            idx (int): Index of the particle to set position and orientation for. Note: this is
                equivalent to setting the corresponding idx'th entry from @set_particles_local_pose()
            position (3-array): particle (x,y,z) position
            orientation (4-array): particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    def __init__(self):
        raise ValueError("System classes should not be created!")

    @classproperty
    def state_size(cls):
        # We have n_particles (1), min / max scale (3*2), each particle pose (7*n)
        return 7 + 7 * cls.n_particles

    @classmethod
    def _dump_state(cls):
        positions, orientations = cls.get_particles_local_pose() if \
            cls._store_local_poses else cls.get_particles_position_orientation()
        return dict(
            n_particles=cls.n_particles,
            min_scale=cls.min_scale,
            max_scale=cls.max_scale,
            positions=positions,
            orientations=orientations,
        )

    @classmethod
    def _load_state(cls, state):
        # Sanity check loading particles
        assert cls.n_particles == state["n_particles"], f"Inconsistent number of particles found when loading " \
                                                        f"particles state! Current number: {cls.n_particles}, " \
                                                        f"loaded number: {state['n_particles']}"
        # Load scale
        cls.min_scale = state["min_scale"]
        cls.max_scale = state["max_scale"]

        # Load the poses
        setter = cls.set_particles_local_pose if cls._store_local_poses else cls.set_particles_position_orientation
        setter(positions=state["positions"], orientations=state["orientations"])

    @classmethod
    def _serialize(cls, state):
        # Array is n_particles, then min_scale and max_scale, then poses for all particles
        return np.concatenate([
            [state["n_particles"]],
            state["min_scale"],
            state["max_scale"],
            state["positions"].flatten(),
            state["orientations"].flatten(),
        ], dtype=float)

    @classmethod
    def _deserialize(cls, state):
        # First index is number of particles, then min_scale and max_scale, then the individual particle poses
        state_dict = dict()
        n_particles = int(state[0])
        len_positions = n_particles * 3
        len_orientations = n_particles * 4
        state_dict["n_particles"] = n_particles
        state_dict["min_scale"] = state[1:4]
        state_dict["max_scale"] = state[4:7]
        state_dict["positions"] = state[7:7+len_positions].reshape(-1, 3)
        state_dict["orientations"] = state[7+len_positions:7+len_positions+len_orientations].reshape(-1, 4)

        return state_dict, 7 + len_positions + len_orientations


# Global dict that contains mappings of all the systems
REGISTERED_SYSTEMS = dict()
UUID_TO_SYSTEMS = dict()

# Serializable registry of systems that are active on the stage (initialized)
SYSTEM_REGISTRY = SerializableRegistry(
    name="system_registry",
    class_types=BaseSystem,
    default_key="name",
    unique_keys=["name", "prim_path", "uuid"],
)


class VisualParticleSystem(BaseSystem):
    """
    Particle system class for generating particles not subject to physics, and are attached to individual objects
    """
    # Maps group name to the particles associated with it
    # This is an ordered dict of ordered dict (nested ordered dict maps particle names to particle instance)
    _group_particles = None

    # Maps group name to the parent object (the object with particles attached to it) of the group
    _group_objects = None

    # Maps group name to tuple (min_scale, max_scale) to apply to sampled particles for that group
    _group_scales = None

    @classmethod
    def initialize(cls):
        # Run super method first
        super().initialize()

        # Initialize mutable class variables so they don't automatically get overridden by children classes
        cls._group_particles = dict()
        cls._group_objects = dict()
        cls._group_scales = dict()

    @classproperty
    def particle_object(cls):
        """
        Returns:
            XFormPrim: Particle object to be used as a template for duplication
        """
        raise NotImplementedError()

    @classproperty
    def groups(cls):
        """
        Returns:
            set of str: Current attachment particle group names
        """
        return set(cls._group_particles.keys())

    @classproperty
    def _store_local_poses(cls):
        # Store local poses since particles are attached to moving bodies
        return True

    @classproperty
    def scale_relative_to_parent(cls):
        """
        Returns:
            bool: Whether or not particles should be scaled relative to the group's parent object. NOTE: If True,
                this will OVERRIDE cls.min_scale and cls.max_scale when sampling particles!
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
    def clear(cls):
        # Run super method first
        super().clear()

        # Clear all groups as well
        cls._group_particles = dict()
        cls._group_objects = dict()
        cls._group_scales = dict()

    @classmethod
    def remove_particle_by_name(cls, name):
        """
        Remove particle with name @name from both the simulator and internal state

        Args:
            name (str): Name of the particle to remove
        """
        raise NotImplementedError()

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
            cls.remove_particle_by_name(name=particle_name)

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

        # Compute the group scale if we're scaling relative to parent
        if cls.scale_relative_to_parent:
            cls._group_scales[group] = cls._compute_relative_group_scales(group=group)

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
        if cls.scale_relative_to_parent:
            cls._group_scales.pop(group)

        return group

    @classmethod
    def _compute_relative_group_scales(cls, group):
        """
        Computes relative particle scaling for group @group required when @cls.scale_relative_to_parent is True

        Args:
            group (str): Specific group for which to compute the relative particle scaling

        Returns:
            2-tuple:
                - 3-array: min scaling factor
                - 3-array: max scaling factor
        """
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

        return minimum, maximum

    @classmethod
    def sample_scales_by_group(cls, group, n):
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

        # Sample based on whether we're scaling relative to parent or not
        scales = np.random.uniform(*cls._group_scales[group], (n, 3)) if cls.scale_relative_to_parent else cls.sample_scales(n=n)

        # Since the particles will be placed under the object, it will be affected/stretched by obj.scale. In order to
        # preserve the absolute size of the particles, we need to scale the particle by obj.scale in some way. However,
        # since the particles have a relative rotation w.r.t the object, the scale between the two don't align. As a
        # heuristics, we divide it by the avg_scale, which is the cubic root of the product of the scales along 3 axes.
        obj = cls._group_objects[group]
        avg_scale = np.cbrt(np.product(obj.scale))
        return scales / avg_scale

    @classmethod
    def generate_particles(
            cls,
            positions,
            orientations=None,
            scales=None,
            **kwargs,
    ):
        # Should not be called, since particles must be tied to a group!
        raise ValueError("Cannot call generate_particles for a VisualParticleSystem! "
                         "Call generate_group_particles() instead.")

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
                be attached to. If not specified, all will be attached to the group object's prim, NOT a link
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @classmethod
    def get_group_particles_position_orientation(cls, group):
        """
        Computes all particles' positions and orientations that belong to @group

        Note: This is more optimized than doing a for loop with self.get_particle_position_orientation()

        Args:
            group (str): Group name whose particle positions and orientations should be computed

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError

    @classmethod
    def set_group_particles_position_orientation(cls, group, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to @group

        Note: This is more optimized than doing a for loop with self.set_particle_position_orientation()

        Args:
            group (str): Group name whose particle positions and orientations should be computed
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError

    @classmethod
    def get_group_particles_local_pose(cls, group):
        """
        Computes all particles' positions and orientations that belong to @group in the particles' parent frame

        Args:
            group (str): Group name whose particle positions and orientations should be computed

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError

    @classmethod
    def set_group_particles_local_pose(cls, group, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to @group in the particles' parent frame

        Args:
            group (str): Group name whose particle positions and orientations should be computed
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError

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


class PhysicalParticleSystem(BaseSystem):
    """
    System whose generated particles are subject to physics
    """
    @classmethod
    def initialize(cls):
        # Run super first
        super().initialize()

        # Make sure min and max scale are identical
        assert np.all(cls.min_scale == cls.max_scale), \
            "Min and max scale should be identical for PhysicalParticleSystem!"

    @classproperty
    def particle_density(cls):
        """
        Returns:
            float: The per-particle density, in kg / m^3
        """
        raise NotImplementedError()

    @classproperty
    def particle_radius(cls):
        """
        Returns:
            float: Radius for the particles to be generated, for the purpose of sampling
        """
        raise NotImplementedError()

    @classproperty
    def particle_contact_radius(cls):
        """
        Returns:
            float: Contact radius for the particles to be generated, for the purpose of estimating contacts
        """
        raise NotImplementedError()

    @classmethod
    def check_in_contact(cls, positions):
        """
        Checks whether each particle specified by @particle_positions are in contact with any rigid body.

        NOTE: This is a rough proxy for contact, given @positions. Should not be taken as ground truth.
        This is because for efficiency and underlying physics reasons, it's easier to treat particles as spheres
        for fast checking. For particles directly spawned from Omniverse's underlying ParticleSystem API, it is a
        rough proxy semantically, though it is accurate in sim-physics since all spawned particles interact as spheres.
        For particles spawned manually as rigid bodies, it is a rough proxy both semantically and physically, as the
        object physically interacts with its non-uniform geometry.

        Args:
            positions (np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions

        Returns:
            n-array: (n_particles,) boolean array, True if in contact, otherwise False
        """
        in_contact = np.zeros(len(positions), dtype=bool)
        for idx, pos in enumerate(positions):
            # TODO: Maybe multiply particle contact radius * 2?
            in_contact[idx] = og.sim.psqi.overlap_sphere_any(cls.particle_contact_radius, pos)
        return in_contact

    @classmethod
    def generate_particles_from_link(
            cls,
            obj,
            link,
            use_visual_meshes=True,
            mesh_name_prefixes=None,
            check_contact=True,
            sampling_distance=None,
            max_samples=5e5,
            **kwargs,
    ):
        """
        Generates a new particle instancer with unique identification number @idn, with particles sampled from the mesh
        located at @mesh_prim_path, and registers it internally. This will also check for collision with other rigid
        objects before spawning in individual particles

        Args:
            obj (EntityPrim): Object whose @link's visual meshes will be converted into sampled particles
            link (RigidPrim): @obj's link whose visual meshes will be converted into sampled particles
            use_visual_meshes (bool): Whether to use visual meshes of the link to generate particles
            mesh_name_prefixes (None or str): If specified, specifies the substring that must exist in @link's
                mesh names in order for that mesh to be included in the particle generator function.
                If None, no filtering will be used.
            check_contact (bool): If True, will only spawn in particles that do not collide with other rigid bodies
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            **kwargs (dict): Any additional keyword-mapped arguments required by subclass implementation
        """
        # Run sanity checks
        assert cls.initialized, "Must initialize system before generating particle instancers!"

        # Generate a checker function to see if particles are within the link's volumes
        check_in_volume, _ = generate_points_in_volume_checker_function(
            obj=obj,
            volume_link=link,
            use_visual_meshes=use_visual_meshes,
            mesh_name_prefixes=mesh_name_prefixes,
        )

        # Grab the link's AABB (or fallback to obj AABB if link does not have a valid AABB),
        # and generate a grid of points based on the sampling distance
        try:
            low, high = link.aabb
            extent = link.aabb_extent
        except ValueError:
            low, high = obj.aabb
            extent = obj.aabb_extent
        # We sample the range of each extent minus
        sampling_distance = 2 * cls.particle_radius if sampling_distance is None else sampling_distance
        n_particles_per_axis = (extent / sampling_distance).astype(int)
        assert np.all(n_particles_per_axis), f"link {link.name} is too small to sample any particle of radius {cls.particle_radius}."

        # 1e-10 is added because the extent might be an exact multiple of particle radius
        arrs = [np.arange(lo + cls.particle_radius, hi - cls.particle_radius + 1e-10, cls.particle_radius * 2)
                for lo, hi, n in zip(low, high, n_particles_per_axis)]
        # Generate 3D-rectangular grid of points
        particle_positions = np.stack([arr.flatten() for arr in np.meshgrid(*arrs)]).T
        # Check which points are inside the volume and only keep those
        particle_positions = particle_positions[np.where(check_in_volume(particle_positions))[0]]

        # Also prune any that in contact with anything if requested
        if check_contact:
            particle_positions = particle_positions[np.where(cls.check_in_contact(particle_positions) == 0)[0]]

        # Also potentially sub-sample if we're past our limit
        if len(particle_positions) > max_samples:
            particle_positions = particle_positions[
                np.random.choice(len(particle_positions), size=(max_samples,), replace=False)]

        return cls.generate_particles(
            positions=particle_positions,
            **kwargs,
        )

    @classmethod
    def generate_particles_on_object(
            cls,
            obj,
            sampling_distance=None,
            max_samples=5e5,
            min_samples_for_success=1,
            **kwargs,
    ):
        """
        Generates @n_particles new particle objects and samples their locations on the top surface of object @obj

        Args:
            obj (BaseObject): Object on which to generate a particle instancer with sampled particles on the object's
                top surface
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            min_samples_for_success (int): Minimum number of particles required to be sampled successfully in order
                for this generation process to be considered successful
            **kwargs (dict): Any additional keyword-mapped arguments required by subclass implementation

        Returns:
            bool: True if enough particles were generated successfully (number of successfully sampled points >=
                min_samples_for_success), otherwise False
        """
        assert max_samples >= min_samples_for_success, "number of particles to sample should exceed the min for success"

        # We densely sample a grid of points by ray-casting from top to bottom to find the valid positions
        radius = cls.particle_radius
        results = sample_cuboid_on_object_full_grid_topdown(
            obj,
            # the grid is fully dense - particles are sitting next to each other
            ray_spacing=radius * 2 if sampling_distance is None else sampling_distance,
            # assume the particles are extremely small - sample cuboids of size 0 for better performance
            cuboid_dimensions=np.zeros(3),
            # raycast start inside the aabb in x-y plane and outside the aabb in the z-axis
            aabb_offset=np.array([-radius, -radius, radius]),
            # bottom padding should be the same as the particle radius
            cuboid_bottom_padding=radius,
            # undo_cuboid_bottom_padding should be False - the sampled positions are above the surface by its radius
            undo_cuboid_bottom_padding=False,
        )
        particle_positions = np.array([result[0] for result in results if result[0] is not None])
        # Also potentially sub-sample if we're past our limit
        if len(particle_positions) > max_samples:
            particle_positions = particle_positions[
                np.random.choice(len(particle_positions), size=(max_samples,), replace=False)]

        n_particles = len(particle_positions)
        success = n_particles >= min_samples_for_success
        # If we generated a sufficient number of points, generate them in the simulator
        if success:
            cls.generate_particles(
                positions=particle_positions,
                **kwargs,
            )

        return success


def _create_system_from_metadata(system_name):
    """
    Internal helper function to programmatically create a system from dataset metadata

    NOTE: This only creates the system, and does NOT initialize the system

    Args:
        system_name (str): Name of the system to create, e.g.: "water", "stain", etc.

    Returns:
        BaseSystem: Created system class
    """
    # Avoid circular imports
    from omnigibson import systems

    # Search for the appropriate system, if not found, fallback
    # TODO: Once dataset is fully constructed, DON'T fallback, and assert False instead
    all_systems = set(get_all_system_categories())
    if system_name not in all_systems:
        # Use default config -- assume @system_name is a fluid that uses the same params as water
        return systems.__dict__["FluidSystem"].create(
            name=system_name.replace("-", "_"),
            particle_contact_offset=0.012,
            particle_density=500.0,
            is_viscous=False,
            material_mtl_name="DeepWater",
        )
    else:
        """
        This is not defined yet, but one proposal:
        
        Metadata = .json dict, with format:
        {
            "type": one of {"visual", "fluid", "granular"},
        }
        if visual, include:
            "relative_particle_scaling" : ...,
        
        if visual or granular, also includes:
            
            --> note: create_particle_template should be deterministic, configured via:
                lambda prim_path, name: og.objects.DatasetObject(
                    prim_path=prim_path,
                    name=name,
                    usd_path=os.path.join(gm.DATASET_PATH, "systems", system_name, f"{system_name}.usd"),
                    category=system_name,
                    visible=False,
                    fixed_base=False,
                    visual_only=True,
                    include_default_states=False,
                    abilities={},
                )
        
        if fluid / granular, also include:
            "particle_contact_offset": ...,
            "particle_density": ...,
        
        if fluid, also include:
            "is_viscous": bool
            "material_mtl_name": ...,       # Base material config to use
            "customize_particle_kwargs": {  # Maps property/ies from @MaterialPrim to value to set
                "opacity_constant": ...,
                "albedo_add": ...,
                "diffuse_color_constant": ...,
                ...,
            }
            
            --> This will be programmatically constructed into a function:
                def _customize_particle_material(mat: MaterialPrim): --> None
                    for attr, val in metadata["customize_particle_kwargs"].items():
                        mat.__setattr__(attr, val)
                        
        Then, compile the necessary kwargs and generate the requested system
        """
        # Parse information
        system_dir = os.path.join(gm.DATASET_PATH, "systems", system_name)
        with open(os.path.join(system_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        system_type = metadata["type"]
        system_kwargs = dict(name=system_name)

        particle_assets = set(os.listdir(system_dir))
        particle_assets.remove("metadata.json")
        has_asset = len(particle_assets) > 0
        if has_asset:
            model = sorted(particle_assets)[0]
            asset_path = os.path.join(system_dir, model, "usd", f"{model}.usd")
        else:
            asset_path = None

        if not has_asset:
            if system_type == "macro_visual_particle":
                # Fallback to stain asset
                asset_path = os.path.join(gm.DATASET_PATH, "systems", "stain", "ahkjul", "usd", "stain.usd")
                has_asset = True
        if has_asset:
            def generate_particle_template_fcn():
                return lambda prim_path, name: \
                    og.objects.USDObject(
                        prim_path=prim_path,
                        name=name,
                        usd_path=asset_path,
                        encrypted=True,
                        category=system_name,
                        visible=False,
                        fixed_base=False,
                        visual_only=True,
                        include_default_states=False,
                        abilities={},
                    )
        else:
            def generate_particle_template_fcn():
                return lambda prim_path, name: \
                    og.objects.PrimitiveObject(
                        prim_path=prim_path,
                        name=name,
                        primitive_type="Sphere",
                        category=system_name,
                        radius=0.015,
                        visible=False,
                        fixed_base=False,
                        visual_only=True,
                        include_default_states=False,
                        abilities={},
                    )

        def generate_customize_particle_material_fcn(mat_kwargs):
            def customize_mat(mat):
                for attr, val in mat_kwargs.items():
                    setattr(mat, attr, np.array(val) if isinstance(val, list) else val)
            return customize_mat

        if system_type == "macro_visual_particle":
            system_kwargs["create_particle_template"] = generate_particle_template_fcn()
            system_kwargs["scale_relative_to_parent"] = metadata["relative_particle_scaling"]
        elif system_type == "granular" or system_type == "macro_physical_particle":
            system_kwargs["create_particle_template"] = generate_particle_template_fcn()
            system_kwargs["particle_density"] = metadata["particle_density"]
        elif system_type == "fluid":
            system_kwargs["particle_contact_offset"] = metadata["particle_contact_offset"]
            system_kwargs["particle_density"] = metadata["particle_density"]
            system_kwargs["is_viscous"] = metadata["is_viscous"]
            system_kwargs["material_mtl_name"] = metadata["material_mtl_name"]
            system_kwargs["customize_particle_material"] = \
                generate_customize_particle_material_fcn(mat_kwargs=metadata["customize_material_kwargs"])
        else:
            raise ValueError(f"{system_name} system's type {system_type} is invalid! Must be one of "
                             f"{{ 'macro_visual_particle', 'macro_physical_particle', 'granular', or 'fluid' }}")

        # Generate the requested system
        system_cls = "".join([st.capitalize() for st in system_type.split("_")])
        return systems.__dict__[f"{system_cls}System"].create(**system_kwargs)


def import_og_systems():
    system_dir = os.path.join(gm.DATASET_PATH, "systems")
    if os.path.exists(system_dir):
        system_names = os.listdir(system_dir)
        for system_name in system_names:
            if system_name not in REGISTERED_SYSTEMS:
                _create_system_from_metadata(system_name=system_name)


def is_system_active(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    return system.initialized


def is_visual_particle_system(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    return issubclass(system, VisualParticleSystem)


def is_physical_particle_system(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    return issubclass(system, PhysicalParticleSystem)


def get_system(system_name, force_active=True):
    # Make sure scene exists
    assert og.sim.scene is not None, "Cannot get systems until scene is imported!"
    # If system_name is not in REGISTERED_SYSTEMS, create from metadata
    system = REGISTERED_SYSTEMS[system_name] if system_name in REGISTERED_SYSTEMS \
        else _create_system_from_metadata(system_name=system_name)
    if not system.initialized and force_active:
        system.initialize()
    return system


def clear_all_systems():
    global _CALLBACKS_ON_SYSTEM_INIT, _CALLBACKS_ON_SYSTEM_CLEAR
    _CALLBACKS_ON_SYSTEM_INIT = dict()
    _CALLBACKS_ON_SYSTEM_CLEAR = dict()
    for system in SYSTEM_REGISTRY.objects:
        system.clear()


def add_callback_on_system_init(name, callback):
    global _CALLBACKS_ON_SYSTEM_INIT
    _CALLBACKS_ON_SYSTEM_INIT[name] = callback


def add_callback_on_system_clear(name, callback):
    global _CALLBACKS_ON_SYSTEM_CLEAR
    _CALLBACKS_ON_SYSTEM_CLEAR[name] = callback


def remove_callback_on_system_init(name):
    global _CALLBACKS_ON_SYSTEM_INIT
    _CALLBACKS_ON_SYSTEM_INIT.pop(name)


def remove_callback_on_system_clear(name):
    global _CALLBACKS_ON_SYSTEM_CLEAR
    _CALLBACKS_ON_SYSTEM_CLEAR.pop(name)
