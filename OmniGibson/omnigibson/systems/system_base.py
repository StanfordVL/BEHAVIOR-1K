import json
import os
from functools import cache

import torch as th

import omnigibson as og
from omnigibson.macros import create_module_macros, gm
from omnigibson.utils.asset_utils import get_all_system_categories, get_dataset_path
from omnigibson.utils.python_utils import Serializable, get_uuid
from omnigibson.utils.sampling_utils import sample_cuboid_on_object_full_grid_topdown
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import scene_relative_prim_path_to_absolute, delete_or_deactivate_prim

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

# Global dict that contains mappings of all the systems
UUID_TO_SYSTEM_NAME = dict()


class BaseSystem(Serializable):
    """
    Base class for all systems. These are instanced objects that should be used globally for a given environment/scene.
    This is useful for items in a scene that are non-discrete / cannot be distinguished into individual instances,
    e.g.: water, particles, etc.
    """

    def __init__(self, name, min_scale=None, max_scale=None):
        self._name = name

        # Whether this system has been initialized or not
        self.initialized = False

        self.min_scale = min_scale if min_scale is not None else th.ones(3)
        self.max_scale = max_scale if max_scale is not None else th.ones(3)

        self._uuid = get_uuid(self.name)
        UUID_TO_SYSTEM_NAME[self._uuid] = self.name

        self._scene = None

    @property
    def name(self):
        # Class name is the unique name assigned
        return self._name

    @property
    def uuid(self):
        return self._uuid

    @property
    def prim_path(self):
        """
        Returns:
            str: Path to this system's prim in the scene stage
        """
        assert self._scene is not None, "Scene not set for system {self.name}!".format(self=self)
        return scene_relative_prim_path_to_absolute(self._scene, self.relative_prim_path)

    @property
    def relative_prim_path(self):
        """
        Returns:
            str: Path to this system's prim in the scene stage relative to the world
        """
        return f"/{self.name}"

    @property
    def n_particles(self):
        """
        Returns:
            int: Number of particles belonging to this system
        """
        raise NotImplementedError()

    @property
    def material(self):
        """
        Returns:
            None or MaterialPrim: Material belonging to this system, if there is any
        """
        return None

    @property
    def _register_system(self):
        """
        Returns:
            bool: True if this system should be registered (i.e.: it is not an intermediate class but a "final" subclass
                representing a system we'd actually like to use, e.g.: water, dust, etc. Should be set by the subclass
        """
        # We assume we aren't registering by default
        return False

    @property
    def _store_local_poses(self):
        """
        Returns:
            bool: Whether to store local particle poses or not when state is saved. Default is False
        """
        return False

    def initialize(self, scene):
        """
        Initializes this system
        """
        assert not self.initialized, f"Already initialized system {self.name}!"
        self._scene = scene
        self.initialized = True

        og.sim.stage.DefinePrim(self.prim_path, "Scope")

        if og.sim.is_playing() and gm.ENABLE_TRANSITION_RULES:
            scene.transition_rule_api.refresh_all_rules()

        # Run any callbacks
        for callback in og.sim.get_callbacks_on_system_init().values():
            callback(self)

    @property
    def scene(self):
        """
        Returns:
            Scene or None: Scene object that this prim is loaded into
        """
        assert self.initialized, f"System {self.name} has not been initialized yet!"
        return self._scene

    def update(self):
        """
        Executes any necessary system updates, once per og.sim._non_physics_step
        """
        # Default is no-op
        pass

    def remove_all_particles(self):
        """
        Removes all particles and deletes them from the simulator
        """
        raise NotImplementedError()

    def remove_particles(
        self,
        idxs,
        **kwargs,
    ):
        """
        Removes pre-existing particles

        Args:
            idxs (th.tensor): (n_particles,) shaped array specifying IDs of particles to delete
            **kwargs (dict): Any additional keyword-specific arguments required by subclass implementation
        """
        raise NotImplementedError()

    def generate_particles(
        self,
        positions,
        orientations=None,
        scales=None,
        **kwargs,
    ):
        """
        Generates new particles

        Args:
            positions (th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions
            orientations (None or th.tensor): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (self.min_scale, self.max_scale)
            **kwargs (dict): Any additional keyword-specific arguments required by subclass implementation
        """
        raise NotImplementedError()

    def clear(self):
        """
        Clears this system, so that it may possibly be re-initialized. Useful for, e.g., when loading from a new
        scene during the same sim instance
        """
        if self.initialized:
            self._clear()

    def _clear(self):
        for callback in og.sim.get_callbacks_on_system_clear().values():
            callback(self)

        self.reset()
        delete_or_deactivate_prim(self.prim_path)

        if og.sim.is_playing() and gm.ENABLE_TRANSITION_RULES:
            self.scene.transition_rule_api.prune_active_rules()

        self.initialized = False
        self._scene = None

    def reset(self):
        """
        Reset this system
        """
        self.remove_all_particles()

    def sample_scales(self, n):
        """
        Samples scales uniformly based on @self.min_scale and @self.max_scale

        Args:
            n (int): Number of scales to sample

        Returns:
            (n, 3) array: Array of sampled scales
        """
        return th.rand(n, 3) * (self.max_scale - self.min_scale) + self.min_scale

    def get_particles_position_orientation(self):
        """
        Computes all particles' positions and orientations that belong to this system in the world frame

        Note: This is more optimized than doing a for loop with self.get_particle_position_orientation()

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    def get_particle_position_orientation(self, idx):
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

    def set_particles_position_orientation(self, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to this system in the world frame

        Note: This is more optimized than doing a for loop with self.set_particle_position_orientation()

        Args:
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    def set_particle_position_orientation(self, idx, position=None, orientation=None):
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

    def get_particles_local_pose(self):
        """
        Computes all particles' positions and orientations that belong to this system in the particles' parent frames

        Returns:
            2-tuple:
                - (n, 3)-array: per-particle (x,y,z) position
                - (n, 4)-array: per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    def get_particle_local_pose(self, idx):
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

    def set_particles_local_pose(self, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to this system in the particles' parent frames

        Args:
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    def set_particle_local_pose(self, idx, position=None, orientation=None):
        """
        Sets particle's position and orientation in the particle's parent frame

        Args:
            idx (int): Index of the particle to set position and orientation for. Note: this is
                equivalent to setting the corresponding idx'th entry from @set_particles_local_pose()
            position (3-array): particle (x,y,z) position
            orientation (4-array): particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError()

    @property
    def state_size(self):
        # We have n_particles (1), min / max scale (3*2), each particle pose (7*n)
        return 7 + 7 * self.n_particles

    def _dump_state(self):
        positions, orientations = (
            self.get_particles_local_pose() if self._store_local_poses else self.get_particles_position_orientation()
        )
        return dict(
            n_particles=self.n_particles,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            positions=positions,
            orientations=orientations,
        )

    def _load_state(self, state):
        # Sanity check loading particles
        assert self.n_particles == state["n_particles"], (
            f"Inconsistent number of particles found when loading "
            f"particles state! Current number: {self.n_particles}, "
            f"loaded number: {state['n_particles']}"
        )
        # Load scale
        self.min_scale = state["min_scale"]
        self.max_scale = state["max_scale"]

        # Load the poses
        setter = self.set_particles_local_pose if self._store_local_poses else self.set_particles_position_orientation
        setter(positions=state["positions"], orientations=state["orientations"])

    def serialize(self, state):
        # Array is n_particles, then min_scale and max_scale, then poses for all particles
        return th.cat(
            [
                th.tensor([state["n_particles"]], dtype=th.float32),
                state["min_scale"],
                state["max_scale"],
                state["positions"].flatten(),
                state["orientations"].flatten(),
            ]
        )

    def deserialize(self, state):
        # First index is number of particles, then min_scale and max_scale, then the individual particle poses
        state_dict = dict()
        n_particles = int(state[0])
        len_positions = n_particles * 3
        len_orientations = n_particles * 4
        state_dict["n_particles"] = n_particles
        state_dict["min_scale"] = state[1:4]
        state_dict["max_scale"] = state[4:7]
        state_dict["positions"] = state[7 : 7 + len_positions].reshape(-1, 3)
        state_dict["orientations"] = state[7 + len_positions : 7 + len_positions + len_orientations].reshape(-1, 4)

        return state_dict, 7 + len_positions + len_orientations


class VisualParticleSystem(BaseSystem):
    """
    Particle system class for generating particles not subject to physics, and are attached to individual objects
    """

    def __init__(self, name, **kwargs):
        # Run super
        super().__init__(name=name, **kwargs)

        # Maps group name to the particles associated with it
        # This is an ordered dict of ordered dict (nested ordered dict maps particle names to particle instance)
        self._group_particles = {}

    def initialize(self, scene):
        # Run super method first
        super().initialize(scene)

        # Maps group name to the parent object (the object with particles attached to it) of the group
        self._group_objects = {}

        # Maps group name to tuple (min_scale, max_scale) to apply to sampled particles for that group
        self._group_scales = {}

    @property
    def particle_object(self):
        """
        Returns:
            XFormPrim: Particle object to be used as a template for duplication
        """
        raise NotImplementedError()

    @property
    def groups(self):
        """
        Returns:
            set of str: Current attachment particle group names
        """
        return set(self._group_particles.keys())

    @property
    def _store_local_poses(self):
        # Store local poses since particles are attached to moving bodies
        return True

    @property
    def scale_relative_to_parent(self):
        """
        Returns:
            bool: Whether or not particles should be scaled relative to the group's parent object. NOTE: If True,
                this will OVERRIDE self.min_scale and self.max_scale when sampling particles!
        """
        return False

    @property
    def state_size(self):
        # Get super size first
        state_size = super().state_size

        # Additionally, we have n_groups (1), with m_particles for each group (n), attached_obj_uuids (n), and
        # particle ids, particle indices, and corresponding link info for each particle (m * 3)
        return (
            state_size
            + 1
            + 2 * len(self._group_particles)
            + sum(3 * self.num_group_particles(group) for group in self.groups)
        )

    def _clear(self):
        super()._clear()

        # Clear all groups as well
        self._group_particles = dict()
        self._group_objects = dict()
        self._group_scales = dict()

    def remove_all_group_particles(self, group):
        """
        Remove particle with name @name from both the simulator as well as internally

        Args:
            group (str): Name of the attachment group to remove all particles from
        """
        # Make sure the group exists
        self._validate_group(group=group)
        # Remove all particles from the group
        for particle_name in tuple(self._group_particles[group].keys()):
            self.remove_particle_by_name(name=particle_name)

    def num_group_particles(self, group):
        """
        Gets the number of particles for the given group in the simulator

        Args:
            group (str): Name of the attachment group to remove all particles from.

        Returns:
            int: Number of particles allocated to this group in the scene. Note that if @group does not
                exist, this will return 0
        """
        # Make sure the group exists
        self._validate_group(group=group)
        return len(self._group_particles[group])

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

    def create_attachment_group(self, obj):
        """
        Creates an attachment group internally for object @obj. Note that this does NOT automatically generate particles
        for this object (should call generate_group_particles(...) ).

        Args:
            obj (BaseObject): Object for which a new particle attachment group will be created for

        Returns:
            str: Name of the attachment group to use when executing commands from this class on
                that specific attachment group
        """
        group = VisualParticleSystem.get_group_name(obj=obj)
        # This should only happen once for a single attachment group, so we explicitly check to make sure the object
        # doesn't already exist
        assert (
            group not in self.groups
        ), f"Cannot create new attachment group because group with name {group} already exists!"

        # Create the group
        self._group_particles[group] = dict()
        self._group_objects[group] = obj

        # Compute the group scale if we're scaling relative to parent
        if self._scale_relative_to_parent:
            self._group_scales[group] = self._compute_relative_group_scales(group=group)

        return group

    def remove_attachment_group(self, group):
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
        self._validate_group(group=group)

        # Remove all particles from the group
        self.remove_all_group_particles(group=group)

        # Remove the actual groups
        self._group_particles.pop(group)
        self._group_objects.pop(group)
        if self._scale_relative_to_parent:
            self._group_scales.pop(group)

        return group

    def _compute_relative_group_scales(self, group):
        """
        Computes relative particle scaling for group @group required when @self._scale_relative_to_parent is True

        Args:
            group (str): Specific group for which to compute the relative particle scaling

        Returns:
            2-tuple:
                - 3-array: min scaling factor
                - 3-array: max scaling factor
        """
        # First set the bbox ranges -- depends on the object's bounding box
        obj = self._group_objects[group]
        median_aabb_dim = th.median(obj.aabb_extent)

        # Compute lower and upper limits to bbox
        bbox_lower_limit_from_aabb = m.BBOX_LOWER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
        bbox_lower_limit = th.clip(
            bbox_lower_limit_from_aabb,
            m.BBOX_LOWER_LIMIT_MIN,
            m.BBOX_LOWER_LIMIT_MAX,
        )

        bbox_upper_limit_from_aabb = m.BBOX_UPPER_LIMIT_FRACTION_OF_AABB * median_aabb_dim
        bbox_upper_limit = th.clip(
            bbox_upper_limit_from_aabb,
            m.BBOX_UPPER_LIMIT_MIN,
            m.BBOX_UPPER_LIMIT_MAX,
        )

        # Convert these into scaling factors for the x and y axes for our particle object
        particle_bbox = self.particle_object.aabb_extent
        minimum = th.tensor([bbox_lower_limit / particle_bbox[0], bbox_lower_limit / particle_bbox[1], 1.0])
        maximum = th.tensor([bbox_upper_limit / particle_bbox[0], bbox_upper_limit / particle_bbox[1], 1.0])

        return minimum, maximum

    def sample_scales_by_group(self, group, n):
        """
        Samples @n particle scales for group @group.

        Args:
            group (str): Specific group for which to sample scales
            n (int): Number of scales to sample

        Returns:
            (n, 3) array: Array of sampled scales
        """
        # Make sure the group exists
        self._validate_group(group=group)

        # Sample based on whether we're scaling relative to parent or not
        scales = (
            th.rand(n, 3) * (self._group_scales[group][1] - self._group_scales[group][0]) + self._group_scales[group][0]
            if self._scale_relative_to_parent
            else self.sample_scales(n=n)
        )

        # Since the particles will be placed under the object, it will be affected/stretched by obj.scale. In order to
        # preserve the absolute size of the particles, we need to scale the particle by obj.scale in some way. However,
        # since the particles have a relative rotation w.r.t the object, the scale between the two don't align. As a
        # heuristics, we divide it by the avg_scale, which is the cubic root of the product of the scales along 3 axes.
        obj = self._group_objects[group]
        avg_scale = th.pow(th.prod(obj.scale), 1 / 3)
        return scales / avg_scale

    def generate_particles(
        self,
        positions,
        orientations=None,
        scales=None,
        **kwargs,
    ):
        # Should not be called, since particles must be tied to a group!
        raise ValueError(
            "Cannot call generate_particles for a VisualParticleSystem! Call generate_group_particles() instead."
        )

    def generate_group_particles(
        self,
        group,
        positions,
        orientations=None,
        scales=None,
        link_prim_paths=None,
    ):
        """
        Generates new particle objects within group @group at the specified pose (@positions, @orientations) with
        corresponding scales @scales.

        NOTE: Assumes positions are the exact contact point on @group object's surface. If self._CLIP_INTO_OBJECTS
            is not True, then the positions will be offset away from the object by half of its bbox

        Args:
            group (str): Object on which to sample particle locations
            positions (th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions
            orientations (None or th.tensor): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) scaling in its
                local frame. If not specified, all we randomly sampled based on @self.min_scale and @self.max_scale
            link_prim_paths (None or list of str): Determines which link each generated particle will
                be attached to. If not specified, all will be attached to the group object's prim, NOT a link
        """
        raise NotImplementedError

    def generate_group_particles_on_object(self, group, max_samples=None, min_samples_for_success=1):
        """
        Generates @max_samples new particle objects and samples their locations on the surface of object @obj. Note
        that if any particles are in the group already, they will be removed

        Args:
            group (str): Object on which to sample particle locations
            max_samples (None or int): If specified, maximum number of particles to sample
            min_samples_for_success (int): Minimum number of particles required to be sampled successfully in order
                for this generation process to be considered successful

        Returns:
            bool: True if enough particles were generated successfully (number of successfully sampled points >=
                min_samples_for_success), otherwise False
        """
        raise NotImplementedError

    def get_group_particles_position_orientation(self, group):
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

    def set_group_particles_position_orientation(self, group, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to @group

        Note: This is more optimized than doing a for loop with self.set_particle_position_orientation()

        Args:
            group (str): Group name whose particle positions and orientations should be computed
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError

    def get_group_particles_local_pose(self, group):
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

    def set_group_particles_local_pose(self, group, positions=None, orientations=None):
        """
        Sets all particles' positions and orientations that belong to @group in the particles' parent frame

        Args:
            group (str): Group name whose particle positions and orientations should be computed
            positions (n-array): (n, 3) per-particle (x,y,z) position
            orientations (n-array): (n, 4) per-particle (x,y,z,w) quaternion orientation
        """
        raise NotImplementedError

    def _validate_group(self, group):
        """
        Checks if particle attachment group @group exists. (If not, can create the group via create_attachment_group).
        This will raise a ValueError if it doesn't exist.

        Args:
            group (str): Name of the group to check for
        """
        if group not in self.groups:
            raise ValueError(f"Particle attachment group {group} does not exist!")


class PhysicalParticleSystem(BaseSystem):
    """
    System whose generated particles are subject to physics
    """

    def initialize(self, scene):
        # Run super first
        super().initialize(scene)

        # Make sure min and max scale are identical
        assert th.all(
            self.min_scale == self.max_scale
        ), "Min and max scale should be identical for PhysicalParticleSystem!"

    @property
    def particle_density(self):
        """
        Returns:
            float: The per-particle density, in kg / m^3
        """
        raise NotImplementedError()

    @property
    def particle_radius(self):
        """
        Returns:
            float: Radius for the particles to be generated, for the purpose of sampling
        """
        raise NotImplementedError()

    @property
    def particle_contact_radius(self):
        """
        Returns:
            float: Contact radius for the particles to be generated, for the purpose of estimating contacts
        """
        raise NotImplementedError()

    @property
    def particle_particle_rest_distance(self):
        """
        Returns:
            float: The minimum distance between individual particles at rest
        """
        return self.particle_radius * 2.0

    def check_in_contact(self, positions):
        """
        Checks whether each particle specified by @particle_positions are in contact with any rigid body.

        NOTE: This is a rough proxy for contact, given @positions. Should not be taken as ground truth.
        This is because for efficiency and underlying physics reasons, it's easier to treat particles as spheres
        for fast checking. For particles directly spawned from Omniverse's underlying ParticleSystem API, it is a
        rough proxy semantically, though it is accurate in sim-physics since all spawned particles interact as spheres.
        For particles spawned manually as rigid bodies, it is a rough proxy both semantically and physically, as the
        object physically interacts with its non-uniform geometry.

        Args:
            positions (th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions

        Returns:
            n-array: (n_particles,) boolean array, True if in contact, otherwise False
        """
        in_contact = th.zeros(len(positions), dtype=bool)
        for idx, pos in enumerate(positions):
            # TODO: Maybe multiply particle contact radius * 2?
            in_contact[idx] = og.sim.psqi.overlap_sphere_any(self.particle_contact_radius, pos.tolist())
        return in_contact

    def generate_particles_from_link(
        self,
        obj,
        link,
        use_visual_meshes=True,
        mesh_name_prefixes=None,
        check_contact=True,
        sampling_distance=None,
        max_samples=None,
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
            max_samples (None or int): If specified, maximum number of particles to sample
            **kwargs (dict): Any additional keyword-mapped arguments required by subclass implementation
        """
        # Run sanity checks
        assert self.initialized, "Must initialize system before generating particle instancers!"

        # Grab the link's AABB (or fallback to obj AABB if link does not have a valid AABB),
        # and generate a grid of points based on the sampling distance
        try:
            low, high = link.visual_aabb
            extent = link.visual_aabb_extent
        except ValueError:
            low, high = obj.aabb
            extent = obj.aabb_extent

        sampling_distance = self.particle_particle_rest_distance if sampling_distance is None else sampling_distance

        # This guarantees that h - self.particle_radius > l + self.particle_radius
        assert th.all(
            extent >= 2 * self.particle_radius
        ), f"link {link.name} is too small to sample any particle of radius {self.particle_radius}."

        arrs = [
            th.arange(l + self.particle_radius, h - self.particle_radius, sampling_distance) for l, h in zip(low, high)
        ]

        # Generate 3D-rectangular grid of points
        particle_positions = th.stack([arr.flatten() for arr in th.meshgrid(*arrs)]).T
        # Check which points are inside the volume and only keep those
        particle_positions = particle_positions[
            th.where(
                link.check_points_in_volume(
                    particle_positions_world=particle_positions,
                    use_visual_meshes=use_visual_meshes,
                )
            )[0]
        ]

        # Also prune any that in contact with anything if requested
        if check_contact:
            particle_positions = particle_positions[th.where(self.check_in_contact(particle_positions) == 0)[0]]

        # Also potentially sub-sample if we're past our limit
        if max_samples is not None and len(particle_positions) > max_samples:
            particle_positions = particle_positions[th.randperm(len(particle_positions))[: int(max_samples)]]

        return self.generate_particles(
            positions=particle_positions,
            **kwargs,
        )

    def generate_particles_on_object(
        self,
        obj,
        sampling_distance=None,
        max_samples=None,
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
            max_samples (None or int): If specified, maximum number of particles to sample
            min_samples_for_success (int): Minimum number of particles required to be sampled successfully in order
                for this generation process to be considered successful
            **kwargs (dict): Any additional keyword-mapped arguments required by subclass implementation

        Returns:
            bool: True if enough particles were generated successfully (number of successfully sampled points >=
                min_samples_for_success), otherwise False
        """
        assert max_samples >= min_samples_for_success, "number of particles to sample should exceed the min for success"

        # We densely sample a grid of points by ray-casting from top to bottom to find the valid positions
        radius = self.particle_radius
        results = sample_cuboid_on_object_full_grid_topdown(
            obj,
            # the grid is fully dense - particles are sitting next to each other
            ray_spacing=radius * 2 if sampling_distance is None else sampling_distance,
            # assume the particles are extremely small - sample cuboids of size 0 for better performance
            cuboid_dimensions=th.zeros(3),
            # raycast start inside the aabb in x-y plane and outside the aabb in the z-axis
            aabb_offset=th.tensor([-radius, -radius, radius]),
            # bottom padding should be the same as the particle radius
            cuboid_bottom_padding=radius,
            # undo_cuboid_bottom_padding should be False - the sampled positions are above the surface by its radius
            undo_cuboid_bottom_padding=False,
        )
        particle_positions = th.stack([result[0] for result in results if result[0] is not None])
        # Also potentially sub-sample if we're past our limit
        if max_samples is not None and len(particle_positions) > max_samples:
            particle_positions = particle_positions[th.randperm(len(particle_positions))[:max_samples]]

        n_particles = len(particle_positions)
        success = n_particles >= min_samples_for_success
        # If we generated a sufficient number of points, generate them in the simulator
        if success:
            self.generate_particles(
                positions=particle_positions,
                **kwargs,
            )

        return success


@cache
def get_all_system_names():
    """
    Gets all available systems from the BEHAVIOR-1K dataset

    Returns:
        set: Set of all available system names that can be created in OmniGibson
    """
    system_dir = os.path.join(get_dataset_path("behavior-1k-assets"), "systems")

    assert os.path.exists(system_dir), f"Path for OmniGibson systems not found! Attempted path: {system_dir}"
    return set(os.listdir(system_dir))


def create_system_from_metadata(system_name):
    """
    Internal helper function to programmatically create a system from dataset metadata.
    Args:
        system_name (str): Name of the system to create, e.g.: "water", "stain", etc.

    Returns:
        BaseSystem: Created system class
    """
    # Avoid circular imports
    from omnigibson import systems

    # Search for the appropriate system, if not found, fallback
    # TODO: Once dataset is fully constructed, DON'T fallback, and assert False instead
    all_systems = set(get_all_system_categories(include_cloth=True))
    if system_name not in all_systems:
        # Use default config -- assume @system_name is a fluid that uses the same params as water
        return systems.FluidSystem(
            name=system_name.replace("-", "_"),
            particle_contact_offset=0.012,
            particle_density=500.0,
            is_viscous=False,
            material_mtl_name="DeepWater",
        )
    else:
        # This is not defined yet, but one proposal:

        # Metadata = .json dict, with format:
        # {
        #     "type": one of {"visual", "fluid", "granular"},
        # }
        # if visual, include:
        #     "relative_particle_scaling" : ...,

        # if visual or granular, also includes:

        #     --> note: create_particle_template should be deterministic, configured via:
        #         lambda prim_path, name: og.objects.DatasetObject(
        #             prim_path=prim_path,
        #             name=name,
        #             usd_path=os.path.join(get_dataset_path("behavior-1k-assets"), "systems", system_name, f"{system_name}.usd"),
        #             category=system_name,
        #             visible=False,
        #             fixed_base=False,
        #             visual_only=True,
        #             include_default_states=False,
        #             abilities={},
        #         )

        # if fluid / granular, also include:
        #     "particle_contact_offset": ...,
        #     "particle_density": ...,

        # if fluid, also include:
        #     "is_viscous": bool
        #     "material_mtl_name": ...,       # Base material config to use
        #     "customize_particle_kwargs": {  # Maps property/ies from @MaterialPrim to value to set
        #         "opacity_constant": ...,
        #         "albedo_add": ...,
        #         "diffuse_color_constant": ...,
        #         ...,
        #     }

        #     --> This will be programmatically constructed into a function:
        #         def _customize_particle_material(mat: MaterialPrim): --> None
        #             for attr, val in metadata["customize_particle_kwargs"].items():
        #                 mat.__setattr__(attr, val)

        # Then, compile the necessary kwargs and generate the requested system
        # Parse information
        system_dir = os.path.join(get_dataset_path("behavior-1k-assets"), "systems", system_name)
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
                asset_path = os.path.join(
                    get_dataset_path("behavior-1k-assets"), "systems", "stain", "ahkjul", "usd", "ahkjul.usd"
                )
                has_asset = True
        if has_asset:

            def generate_particle_template_fcn():
                return lambda relative_prim_path, name: og.objects.USDObject(
                    relative_prim_path=relative_prim_path,
                    name=name,
                    usd_path=asset_path,
                    encrypted=True,
                    category=system_name,
                    visible=False,
                    fixed_base=True,
                    visual_only=True,
                    kinematic_only=True,
                    include_default_states=False,
                    abilities={},
                )

        else:

            def generate_particle_template_fcn():
                return lambda relative_prim_path, name: og.objects.PrimitiveObject(
                    relative_prim_path=relative_prim_path,
                    name=name,
                    primitive_type="Sphere",
                    category=system_name,
                    radius=0.015,
                    visible=False,
                    fixed_base=True,
                    visual_only=True,
                    kinematic_only=True,
                    include_default_states=False,
                    abilities={},
                )

        def generate_customize_particle_material_fcn(mat_kwargs):
            def customize_mat(mat):
                for attr, val in mat_kwargs.items():
                    setattr(mat, attr, th.tensor(val) if isinstance(val, list) else val)

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
            system_kwargs["customize_particle_material"] = generate_customize_particle_material_fcn(
                mat_kwargs=metadata["customize_material_kwargs"]
            )
        else:
            raise ValueError(
                f"{system_name} system's type {system_type} is invalid! Must be one of "
                f"{{ 'macro_visual_particle', 'macro_physical_particle', 'granular', or 'fluid' }}"
            )

        # Generate the requested system
        system_cls = "".join([st.capitalize() for st in system_type.split("_")])
        return getattr(systems, f"{system_cls}System")(**system_kwargs)
