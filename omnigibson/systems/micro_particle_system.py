import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.prims.prim_base import BasePrim
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.systems.system_base import BaseSystem, REGISTERED_SYSTEMS
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty, assert_valid_key, subclass_factory, snake_case_to_camel_case
from omnigibson.utils.sampling_utils import sample_cuboid_on_object_full_grid_topdown
from omnigibson.utils.usd_utils import array_to_vtarray
from omnigibson.utils.physx_utils import create_physx_particle_system, create_physx_particleset_pointinstancer
from omnigibson.utils.ui_utils import disclaimer, create_module_logger

import omni
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.physx.scripts import particleUtils
import numpy as np
from pxr import Gf, Vt, UsdShade, UsdGeom, PhysxSchema
from collections import defaultdict


# physics settins
from omni.physx.bindings._physx import (
    SETTING_UPDATE_TO_USD,
    SETTING_UPDATE_VELOCITIES_TO_USD,
    SETTING_NUM_THREADS,
    SETTING_UPDATE_PARTICLES_TO_USD,
)
import carb

# Create module logger
log = create_module_logger(module_name=__name__)


def is_physical_particle_system(system_name):
    assert system_name in REGISTERED_SYSTEMS, f"System {system_name} not in REGISTERED_SYSTEMS."
    system = REGISTERED_SYSTEMS[system_name]
    return issubclass(system, PhysicalParticleSystem)


def set_carb_settings_for_fluid_isosurface():
    """
    Sets relevant rendering settings in the carb settings in order to use isosurface effectively
    """
    # Settings for Isosurface
    isregistry = carb.settings.acquire_settings_interface()
    # disable grid and lights
    dOptions = isregistry.get_as_int("persistent/app/viewport/displayOptions")
    dOptions &= ~(1 << 6 | 1 << 8)
    isregistry.set_int("persistent/app/viewport/displayOptions", dOptions)
    isregistry.set_bool(SETTING_UPDATE_TO_USD, True)
    isregistry.set_int(SETTING_NUM_THREADS, 8)
    isregistry.set_bool(SETTING_UPDATE_VELOCITIES_TO_USD, True)
    isregistry.set_bool(SETTING_UPDATE_PARTICLES_TO_USD, True)     # TODO: Why does setting this value --> True result in no isosurface being rendered?
    isregistry.set_int("persistent/simulation/minFrameRate", 60)
    isregistry.set_bool("rtx-defaults/pathtracing/lightcache/cached/enabled", False)
    isregistry.set_bool("rtx-defaults/pathtracing/cached/enabled", False)
    isregistry.set_int("rtx-defaults/pathtracing/fireflyFilter/maxIntensityPerSample", 10000)
    isregistry.set_int("rtx-defaults/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse", 50000)
    isregistry.set_float("rtx-defaults/pathtracing/optixDenoiser/blendFactor", 0.09)
    isregistry.set_int("rtx-defaults/pathtracing/aa/op", 2)
    isregistry.set_int("rtx-defaults/pathtracing/maxBounces", 32)
    isregistry.set_int("rtx-defaults/pathtracing/maxSpecularAndTransmissionBounces", 16)
    isregistry.set_int("rtx-defaults/post/dlss/execMode", 1)
    isregistry.set_int("rtx-defaults/translucency/maxRefractionBounces", 12)


class PhysxParticleInstancer(BasePrim):
    """
    Simple class that wraps the raw omniverse point instancer prim and provides convenience functions for
    particle access
    """
    def __init__(self, prim_path, name, idn):
        """
        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            idn (int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation.
        """
        # Store inputs
        self._idn = idn

        # Values loaded at runtime
        self._n_particles = None

        # Run super method directly
        super().__init__(prim_path=prim_path, name=name)

    def _load(self):
        # We raise an error, this should NOT be created from scratch
        raise NotImplementedError("PhysxPointInstancer should NOT be loaded via this class! Should be created before.")

    def _post_load(self):
        # Run super
        super()._post_load()

        # Store how many particles we have
        self._n_particles = len(self.particle_positions)

    def add_particles(
            self,
            positions,
            velocities=None,
            orientations=None,
            scales=None,
            prototype_indices=None,
    ):
        """
        Adds particles to this particle instancer.

        positions (np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions.
        velocities (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
            If not specified, all will be set to 0
        orientations (None or np.array): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
            orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
        scales (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
            If not specified, will be scale [1, 1, 1] by default
        prototype_indices (None or list of int): If specified, should specify which prototype should be used for
            each particle. If None, will use all 0s (i.e.: the first prototype created)
        """
        n_new_particles = len(positions)

        velocities = np.zeros((n_new_particles, 3)) if velocities is None else velocities
        if orientations is None:
            orientations = np.zeros((n_new_particles, 4))
            orientations[:, -1] = 1.0
        scales = np.ones((n_new_particles, 3)) * np.ones((1, 3)) if scales is None else scales
        prototype_indices = np.zeros(n_new_particles, dtype=int) if prototype_indices is None else prototype_indices

        # Update the number of particles and update the values
        self._n_particles += n_new_particles
        self.particle_positions = np.vstack([self.particle_positions, positions])
        self.particle_velocities = np.vstack([self.particle_velocities, velocities])
        self.particle_orientations = np.vstack([self.particle_orientations, orientations])
        self.particle_scales = np.vstack([self.particle_scales, scales])
        self.particle_prototype_ids = np.hstack([self.particle_prototype_ids, prototype_indices])

    def remove_particles(self, idxs):
        """
        Remove particles from this instancer, specified by their indices @idxs in the data array

        Args:
            idxs (list or np.array of int): IDs corresponding to the indices of specific particles to remove from this
                instancer
        """
        if len(idxs) > 0:
            # Update the number of particles
            self._n_particles -= len(idxs)
            # Remove all requested indices and write to all the internal data arrays
            self.particle_positions = np.delete(self.particle_positions, idxs, axis=0)
            self.particle_velocities = np.delete(self.particle_velocities, idxs, axis=0)
            self.particle_orientations = np.delete(self.particle_orientations, idxs, axis=0)
            self.particle_scales = np.delete(self.particle_scales, idxs, axis=0)
            self.particle_prototype_ids = np.delete(self.particle_prototype_ids, idxs, axis=0)

    def remove_all_particles(self):
        """
        Removes all particles from this instancer, but does NOT delete this instancer
        """
        self.remove_particles(idxs=np.arange(self._n_particles))

    @property
    def n_particles(self):
        """
        Returns:
            int: Number of particles owned by this instancer
        """
        return self._n_particles

    @property
    def idn(self):
        """
        Returns:
            int: Identification number of this particle instancer
        """
        return self._idn

    @property
    def particle_group(self):
        """
        Returns:
            int: Particle group this instancer belongs to
        """
        return self.get_attribute(attr="physxParticle:particleGroup")

    @particle_group.setter
    def particle_group(self, group):
        """
        Args:
            group (int): Particle group this instancer belongs to
        """
        self.set_attribute(attr="physxParticle:particleGroup", val=group)

    @property
    def particle_positions(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return np.array(self.get_attribute(attr="positions"))

    @particle_positions.setter
    def particle_positions(self, pos):
        """
        Set the particle positions for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired positions are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        assert pos.shape[0] == self._n_particles, \
            f"Got mismatch in particle setting size: {pos.shape[0]}, vs. number of particles {self._n_particles}!"
        self.set_attribute(attr="positions", val=Vt.Vec3fArray.FromNumpy(pos.astype(float)))

    @property
    def particle_orientations(self):
        """
        Returns:
            np.array: (N, 4) numpy array, where each of the N particles' orientations are expressed in (x,y,z,w)
                quaternion coordinates relative to this instancer's parent prim
        """
        oris = self.get_attribute(attr="orientations")
        assert oris is not None, f"Orientations should be set for particle instancer {self.name}!"
        return np.array(oris)

    @particle_orientations.setter
    def particle_orientations(self, quat):
        """
        Set the particle positions for this instancer

        Args:
            np.array: (N, 4) numpy array, where each of the N particles' desired orientations are expressed in (x,y,z,w)
                quaternion coordinates relative to this instancer's parent prim
        """
        assert quat.shape[0] == self._n_particles, \
            f"Got mismatch in particle setting size: {quat.shape[0]}, vs. number of particles {self._n_particles}!"
        # Swap w position, since Quath takes (w,x,y,z)
        quat = quat.astype(float)
        quat = quat[:, [3, 0, 1, 2]]
        self.set_attribute(attr="orientations", val=Vt.QuathArray.FromNumpy(quat))

    @property
    def particle_velocities(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' velocities are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return np.array(self.get_attribute(attr="velocities"))

    @particle_velocities.setter
    def particle_velocities(self, vel):
        """
        Set the particle velocities for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired velocities are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        assert vel.shape[0] == self._n_particles, \
            f"Got mismatch in particle setting size: {vel.shape[0]}, vs. number of particles {self._n_particles}!"
        vel = vel.astype(float)
        self.set_attribute(attr="velocities", val=Vt.Vec3fArray.FromNumpy(vel))

    @property
    def particle_scales(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' scales are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        scales = self.get_attribute(attr="scales")
        return np.ones((self._n_particles, 3)) if scales is None else np.array(scales)

    @particle_scales.setter
    def particle_scales(self, scales):
        """
        Set the particle scales for this instancer

        Args:
            np.array: (N, 3) numpy array, where each of the N particles' desired scales are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        assert scales.shape[0] == self._n_particles, \
            f"Got mismatch in particle setting size: {scales.shape[0]}, vs. number of particles {self._n_particles}!"
        scales = scales.astype(float)
        self.set_attribute(attr="scales", val=Vt.Vec3fArray.FromNumpy(scales))

    @property
    def particle_prototype_ids(self):
        """
        Returns:
            np.array: (N,) numpy array, where each of the N particles' prototype_id (i.e.: which prototype is being used
                for that particle)
        """
        ids = self.get_attribute(attr="protoIndices")
        return np.zeros(self.n_particles) if ids is None else np.array(ids)

    @particle_prototype_ids.setter
    def particle_prototype_ids(self, prototype_ids):
        """
        Set the particle prototype_ids for this instancer

        Args:
            np.array: (N,) numpy array, where each of the N particles' desired prototype_id
                (i.e.: which prototype is being used for that particle)
        """
        assert prototype_ids.shape[0] == self._n_particles, \
            f"Got mismatch in particle setting size: {prototype_ids.shape[0]}, vs. number of particles {self._n_particles}!"
        self.set_attribute(attr="protoIndices", val=prototype_ids)

    @property
    def state_size(self):
        # idn (1), particle_group (1), n_particles (1), and the corresponding states for each particle
        # N * (pos (3) + vel (3) + orn (4) + scale (3) + prototype_id (1))
        return 3 + self._n_particles * 14

    def _dump_state(self):
        return dict(
            idn=self._idn,
            particle_group=self.particle_group,
            n_particles=self._n_particles,
            particle_positions=self.particle_positions,
            particle_velocities=self.particle_velocities,
            particle_orientations=self.particle_orientations,
            particle_scales=self.particle_scales,
            particle_prototype_ids=self.particle_prototype_ids,
        )

    def _load_state(self, state):
        # Sanity check the identification number and particle group
        assert self._idn == state["idn"], f"Got mismatch in identification number for this particle instancer when " \
            f"loading state! Should be: {self._idn}, got: {state['idn']}."
        assert self.particle_group == state["particle_group"], f"Got mismatch in particle group for this particle " \
            f"instancer when loading state! Should be: {self.particle_group}, got: {state['particle_group']}."

        # Set values appropriately
        self._n_particles = state["n_particles"]
        keys = ("particle_positions", "particle_velocities", "particle_orientations", "particle_scales", "particle_prototype_ids")
        for key in keys:
            # Make sure the loaded state is a numpy array, it could have been accidentally casted into a list during
            # JSON-serialization
            val = np.array(state[key]) if not isinstance(state[key], np.ndarray) else state[key]
            setattr(self, key, val)

    def _serialize(self, state):
        # Compress into a 1D array
         return np.concatenate([
             [state["idn"], state["particle_group"], state["n_particles"]],
             state["particle_positions"].reshape(-1),
             state["particle_velocities"].reshape(-1),
             state["particle_orientations"].reshape(-1),
             state["particle_scales"].reshape(-1),
             state["particle_prototype_ids"],
         ]).astype(float)

    def _deserialize(self, state):
        # Sanity check the identification number
        assert self._idn == state[0], f"Got mismatch in identification number for this particle instancer when " \
            f"deserializing state! Should be: {self._idn}, got: {state[0]}."
        assert self.particle_group == state[1], f"Got mismatch in particle group for this particle " \
            f"instancer when deserializing state! Should be: {self.particle_group}, got: {state[1]}."

        # De-compress from 1D array
        n_particles = int(state[2])
        state_dict = dict(
            idn=int(state[0]),
            particle_group=int(state[1]),
            n_particles=n_particles,
        )

        # Process remaining keys and reshape automatically
        keys = ("particle_positions", "particle_velocities", "particle_orientations", "particle_scales", "particle_prototype_ids")
        sizes = ((n_particles, 3), (n_particles, 3), (n_particles, 4), (n_particles, 3), (n_particles,))

        idx = 3
        for key, size in zip(keys, sizes):
            length = np.product(size)
            state_dict[key] = state[idx: idx + length].reshape(size)
            idx += length

        return state_dict, idx


class MicroParticleSystem(BaseSystem):
    """
    Global system for modeling "micro" level particles, e.g.: water, seeds, cloth. This system leverages
    Omniverse's native physx particle systems
    """
    # Particle system prim in the scene, should be generated at runtime
    system_prim = None

    @classmethod
    def initialize(cls):
        # Run super first
        super().initialize()
        if not gm.USE_GPU_DYNAMICS:
            raise ValueError(f"Failed to initialize {cls.name} system. Please set gm.USE_GPU_DYNAMICS to be True.")
        cls.system_prim = cls._create_particle_system()

    @classproperty
    def system_prim_path(cls):
        return f"{cls.prim_path}/system"

    @classproperty
    def visual_only(cls):
        """
        Returns:
            bool: Whether this particle system should be visual-only, i.e.: not subject to collisions and physics. If True,
                the generated particles will not move or collide
        """
        return False

    @classproperty
    def particle_contact_offset(cls):
        """
        Returns:
            float: Contact offset value to use for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#particle-system-configuration
                for more information
        """
        raise NotImplementedError()

    @classproperty
    def use_smoothing(cls):
        """
        Returns:
            bool: Whether to use smoothing or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#smoothing
                for more information
        """
        return False

    @classproperty
    def use_anisotropy(cls):
        """
        Returns:
            bool: Whether to use anisotropy or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#anisotropy
                for more information
        """
        return False

    @classproperty
    def use_isosurface(cls):
        """
        Returns:
            bool: Whether to use isosurface or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#isosurface
                for more information
        """
        return False

    @classproperty
    def particle_radius(cls):
        """
        Returns:
            float: Radius for the particles to be generated, since all fluids are composed of spheres
        """
        # Magic number from omni tutorials
        # See https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#offset-autocomputation
        return 0.99 * cls.particle_contact_offset

    @classmethod
    def _create_particle_system(cls):
        """
        Creates the single, global particle system. This should only be ever called once, and during initialize()

        Returns:
            Usd.Prim: Particle system prim created
        """
        return create_physx_particle_system(
            prim_path=cls.system_prim_path,
            physics_scene_path=og.sim.get_physics_context().get_current_physics_scene_prim().GetPrimPath().pathString,
            particle_contact_offset=cls.particle_contact_offset,
            visual_only=cls.visual_only,
            smoothing=cls.use_smoothing and gm.ENABLE_HQ_RENDERING,
            anisotropy=cls.use_anisotropy and gm.ENABLE_HQ_RENDERING,
            isosurface=cls.use_isosurface and gm.ENABLE_HQ_RENDERING,
        ).GetPrim()


class PhysicalParticleSystem(MicroParticleSystem):
    """
    Global system for modeling physical "micro" level particles, e.g.: water, seeds, rice, etc. This system leverages
    Omniverse's native physx particle systems
    """
    # Particle prototypes -- will be list of mesh prims to use as particle prototypes for this system
    particle_prototypes = None

    # Particle instancers -- maps name to particle instancer prims (dict)
    particle_instancers = None

    # Scaling factor to sample from when generating a new particle
    min_scale = None  # (x,y,z) scaling
    max_scale = None  # (x,y,z) scaling

    # Max particle instancer identification number -- this monotonically increases until reset() is called
    max_instancer_idn = None

    @classproperty
    def n_particles(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        return sum([instancer.n_particles for instancer in cls.particle_instancers.values()])

    @classproperty
    def n_instancers(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        return len(cls.particle_instancers)

    @classproperty
    def instancer_idns(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        return [inst.idn for inst in cls.particle_instancers.values()]

    @classproperty
    def mat_path(cls):
        """
        Returns:
            str: Path to this system's material in the scene stage
        """
        return f"{cls.prim_path}/material"

    @classproperty
    def mat_name(cls):
        """
        Returns:
            str: Name of this system's material
        """
        return f"{cls.name}:material"

    @classmethod
    def initialize(cls):
        # Run super first
        super().initialize()

        # Initialize class variables that are mutable so they don't get overridden by children classes
        cls.particle_instancers = dict()

        # Set the default scales
        cls.min_scale = np.ones(3)
        cls.max_scale = np.ones(3)

        # Initialize max instancer idn
        cls.max_instancer_idn = -1

        cls.particle_prototypes = cls._create_particle_prototypes()

    @classmethod
    def reset(cls):
        # Call super first
        super().reset()

        # Reset all internal variables
        cls.remove_all_particle_instancers()

    @classproperty
    def next_available_instancer_idn(cls):
        """
        Updates the max instancer identification number based on the current internal state
        """
        if cls.n_instancers == 0:
            return cls.default_instancer_idn
        else:
            for idn in range(max(cls.instancer_idns) + 2):
                if idn not in cls.instancer_idns:
                    return idn

    @classproperty
    def default_instancer_idn(cls):
        return 0

    @classproperty
    def state_size(cls):
        # We have the number of particle instancers (1), the instancer groups, particle groups, and,
        # number of particles in each instancer (3n),
        # and the corresponding states in each instancer (X)
        return 1 + 3 * len(cls.particle_instancers) + sum(inst.state_size for inst in cls.particle_instancers.values())

    @classproperty
    def default_particle_instancer(cls):
        """
        Returns:
            PhysxParticleInstancer: Default particle instancer for this particle system
        """
        # Default instancer is the 0th ID instancer
        name = cls.particle_instancer_idn_to_name(idn=cls.default_instancer_idn)
        # NOTE: Cannot use dict.get() call for some reason; it messes up IDE introspection
        return cls.particle_instancers[name] if name in cls.particle_instancers \
            else cls.generate_particle_instancer(n_particles=0, idn=cls.default_instancer_idn)

    @classproperty
    def color(cls):
        return cls._color

    @classproperty
    def is_fluid(cls):
        """
        Returns:
            bool: Whether this system is modeling fluid or not
        """
        raise NotImplementedError()

    @classproperty
    def particle_density(cls):
        """
        Returns:
            float: The per-particle density, in kg / m^3
        """
        raise NotImplementedError()

    @classmethod
    def _create_particle_prototypes(cls):
        """
        Creates any relevant particle prototypes to be used by this particle system.

        Returns:
            list of VisualGeomPrim: Visual mesh prim(s) to use as this system's particle prototype(s)
        """
        raise NotImplementedError()

    @classmethod
    def _create_particle_material_template(cls):
        """
        Creates the particle material template to be used for this particle system. Prim path does not matter,
        as it will be overridden internally such that it is a child prim of this particle system's prim.

        NOTE: This material is a template because it is loading an Omni material preset. It can then be customized (in
        addition to modifying its physical material properties) via @_customize_particle_material

        Returns:
            None or MaterialPrim: If specified, is the material to apply to all particles. If None, no material
                will be used. Default is None
        """
        return None

    @classmethod
    def _customize_particle_material(cls):
        """
        Modifies this particle system's particle material once it is loaded. Default is a no-op
        """
        pass

    @classmethod
    def generate_particles(
            cls,
            positions,
            instancer_idn=None,
            particle_group=0,
            velocities=None,
            orientations=None,
            scales=None,
            self_collision=True,
            prototype_indices=None,
    ):
        """
        Generates new particles, either as part of a pre-existing instancer corresponding to @instancer_idn or as part
            of a newly generated instancer.

        NOTE:

        Args:
            positions (np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions.
                If not specified, will be set to the origin by default
            instancer_idn (None or int): Unique identification number of the particle instancer to assign the generated
                particles to. This is used to deterministically reproduce individual particle instancer states
                dynamically, even if we delete / add additional ones at runtime during simulation. If there is no
                active instancer that matches the requested idn, a new one will be created.
                If None, this system will add particles to the default particle instancer
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by @self_collision
            velocities (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
                If not specified, all will be set to 0
            orientations (None or np.array): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (cls.min_scale, cls.max_scale)
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not
            prototype_indices (None or list of int): If specified, should specify which prototype should be used for
                each particle. If None, will use all 0s (i.e.: the first prototype created)

        Returns:
            PhysxParticleInstancer: Particle instancer that includes the generated particles
        """
        # Create a new particle instancer if a new idn is requested, otherwise use the pre-existing one
        inst = cls.default_particle_instancer if instancer_idn is None else \
            cls.particle_instancers.get(cls.particle_instancer_idn_to_name(idn=instancer_idn), None)

        if inst is None:
            inst = cls.generate_particle_instancer(
                idn=instancer_idn,
                particle_group=particle_group,
                n_particles=len(positions),
                positions=positions,
                velocities=velocities,
                orientations=orientations,
                scales=scales,
                prototype_indices=prototype_indices,
                self_collision=self_collision,
            )
        else:
            inst.add_particles(
                positions=positions,
                velocities=velocities,
                orientations=orientations,
                scales=scales,
                prototype_indices=prototype_indices,
            )

        return inst

    @classmethod
    def generate_particle_instancer(
            cls,
            n_particles,
            idn=None,
            particle_group=0,
            positions=None,
            velocities=None,
            orientations=None,
            scales=None,
            self_collision=True,
            prototype_indices=None,
    ):
        """
        Generates a new particle instancer with unique identification number @idn, and registers it internally

        Args:
            n_particles (int): Number of particles to generate for this instancer
            idn (None or int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation. If None, this system will generate a unique
                identifier automatically.
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by @self_collision
            positions (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions.
                If not specified, will be set to the origin by default
            velocities (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
                If not specified, all will be set to 0
            orientations (None or np.array): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or np.array): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (cls.min_scale, cls.max_scale)
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not
            prototype_indices (None or list of int): If specified, should specify which prototype should be used for
                each particle. If None, will use all 0s (i.e.: the first prototype created)

        Returns:
            PhysxParticleInstancer: Generated particle instancer
        """
        # Run sanity checks
        assert cls.initialized, "Must initialize system before generating particle instancers!"

        # Automatically generate an identification number for this instancer if none is specified
        if idn is None:
            idn = cls.next_available_instancer_idn

        assert idn not in cls.instancer_idns, f"instancer idn {idn} already exists."

        # Generate standardized prim path for this instancer
        name = cls.particle_instancer_idn_to_name(idn=idn)

        # Create the instancer
        instance = create_physx_particleset_pointinstancer(
            name=name,
            particle_system_path=cls.prim_path,
            physx_particle_system_path=cls.system_prim_path,
            particle_group=particle_group,
            positions=np.zeros((n_particles, 3)) if positions is None else positions,
            self_collision=self_collision,
            fluid=cls.is_fluid,
            particle_mass=None,
            particle_density=cls.particle_density,
            orientations=orientations,
            velocities=velocities,
            angular_velocities=None,
            scales=np.random.uniform(cls.min_scale, cls.max_scale, size=(n_particles, 3)) if scales is None else scales,
            prototype_prim_paths=[pp.prim_path for pp in cls.particle_prototypes],
            prototype_indices=prototype_indices,
            enabled=not cls.visual_only,
        )

        # Create the instancer object that wraps the raw prim
        instancer = PhysxParticleInstancer(
            prim_path=instance.GetPrimPath().pathString,
            name=name,
            idn=idn,
        )
        instancer.initialize()
        cls.particle_instancers[name] = instancer

        return instancer

    @classmethod
    def generate_particles_from_link(
            cls,
            obj,
            link,
            use_visual_meshes=True,
            mesh_name_prefixes=None,
            instancer_idn=None,
            particle_group=0,
            sampling_distance=None,
            max_samples=5e5,
            sample_volume=True,
            self_collision=True,
            prototype_indices_choices=None,
    ):
        """
        Generates a new particle instancer with unique identification number @idn, with particles sampled from the mesh
        located at @mesh_prim_path, and registers it internally

        Args:
            obj (EntityPrim): Object whose @link's visual meshes will be converted into sampled particles
            link (RigidPrim): @obj's link whose visual meshes will be converted into sampled particles
            use_visual_meshes (bool): Whether to use visual meshes of the link to generate particles
            mesh_name_prefixes (None or str): If specified, specifies the substring that must exist in @link's
                mesh names in order for that mesh to be included in the particle generator function.
                If None, no filtering will be used.
            instancer_idn (None or int): Unique identification number of the particle instancer to assign the generated
                particles to. This is used to deterministically reproduce individual particle instancer states
                dynamically, even if we delete / add additional ones at runtime during simulation. If there is no
                active instancer that matches the requested idn, a new one will be created.
                If None, this system will add particles to the default particle instancer
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by @self_collision.
                Only used if a new particle instancer is created!
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            sample_volume (bool): Whether to sample the particles at the mesh's surface or throughout its entire volume
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not. Only used if a new particle instancer is created!
            prototype_indices_choices (None or int or list of int): If specified, should specify which prototype(s)
                should be used for each particle. If None, will use all 0s (i.e.: the first prototype created). If a
                single number, will use that prototype ID for all sampled particles. If a list of int, will uniformly
                sample from those IDs for each particle.

        Returns:
            PhysxParticleInstancer: Particle instancer that includes the generated particles
        """
        # Run sanity checks
        assert cls.initialized, "Must initialize system before generating particle instancers!"
        # TODO: Implement!
        assert sample_volume, "Sampling surface of link for particles is not supported yet!"

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
        # Also potentially sub-sample if we're past our limit
        if len(particle_positions) > max_samples:
            particle_positions = particle_positions[
                np.random.choice(len(particle_positions), size=(max_samples,), replace=False)]

        # Get information about our sampled points
        n_particles = len(particle_positions)
        if prototype_indices_choices is not None:
            prototype_indices = np.ones(n_particles, dtype=int) * prototype_indices_choices if \
                isinstance(prototype_indices_choices, int) else \
                np.random.choice(prototype_indices_choices, size=(n_particles,))
        else:
            prototype_indices = None

        return cls.generate_particles(
            instancer_idn=instancer_idn,
            particle_group=particle_group,
            positions=particle_positions,
            self_collision=self_collision,
            prototype_indices=prototype_indices,
        )

    @classmethod
    def generate_particles_on_object(
            cls,
            obj,
            instancer_idn=None,
            particle_group=0,
            sampling_distance=None,
            max_samples=5e5,
            min_samples_for_success=1,
            self_collision=True,
            prototype_indices_choices=None,
    ):
        """
        Generates @n_particles new particle objects and samples their locations on the top surface of object @obj

        Args:
            obj (BaseObject): Object on which to generate a particle instancer with sampled particles on the object's
                top surface
            instancer_idn (None or int): Unique identification number of the particle instancer to assign the generated
                particles to. This is used to deterministically reproduce individual particle instancer states
                dynamically, even if we delete / add additional ones at runtime during simulation. If there is no
                active instancer that matches the requested idn, a new one will be created.
                If None, this system will add particles to the default particle instancer
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide.
                Only used if a new particle instancer is created!
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            min_samples_for_success (int): Minimum number of particles required to be sampled successfully in order
                for this generation process to be considered successful
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not. Only used if a new particle instancer is created!
            prototype_indices_choices (None or int or list of int): If specified, should specify which prototype(s)
                should be used for each particle. If None, will use all 0s (i.e.: the first prototype created). If a
                single number, will use that prototype ID for all sampled particles. If a list of int, will uniformly
                sample from those IDs for each particle.

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

        # Get information about our sampled points
        n_particles = len(particle_positions)
        if prototype_indices_choices is not None:
            prototype_indices = np.ones(n_particles, dtype=int) * prototype_indices_choices if \
                isinstance(prototype_indices_choices, int) else \
                np.random.choice(prototype_indices_choices, size=(n_particles,))
        else:
            prototype_indices = None

        success = n_particles >= min_samples_for_success
        # If we generated a sufficient number of points, generate them in the simulator
        if success:
            cls.generate_particles(
                instancer_idn=instancer_idn,
                particle_group=particle_group,
                positions=particle_positions,
                self_collision=self_collision,
                prototype_indices=prototype_indices,
            )

        return success

    @classmethod
    def remove_particle_instancer(cls, name):
        """
        Removes particle instancer with name @name from this system.

        Args:
            name (str): Particle instancer name to remove. If it does not exist, then an error will be raised
        """
        # Make sure the instancer actually exists
        assert_valid_key(key=name, valid_keys=cls.particle_instancers, name="particle instancer")
        # Remove instancer from our tracking and delete its prim
        instancer = cls.particle_instancers.pop(name)
        instancer.remove()
        og.sim.stage.RemovePrim(f"{cls.prim_path}/{name}")

    @classmethod
    def particle_instancer_name_to_idn(cls, name):
        """
        Args:
            name (str): Particle instancer name
        Returns:
            int: Particle instancer identification number
        """
        return int(name.split(f"{cls.name}Instancer")[-1])

    @classmethod
    def particle_instancer_idn_to_name(cls, idn):
        """
        Args:
            idn (idn): Particle instancer identification number
        Returns:
            str: Name of the particle instancer auto-generated from its unique identification number
        """
        return f"{cls.name}Instancer{idn}"

    @classmethod
    def _sync_particle_instancers(cls, idns, particle_groups, particle_counts):
        """
        Synchronizes the particle instancers based on desired identification numbers @idns

        Args:
            idns (list of int): Desired unique instancers that should be active for this particle system
            particle_groups (list of int): Desired particle groups that each instancer should be. Length of this
                list should be the same length as @idns
            particle_counts (list of int): Desired particle counts that should exist per instancer. Length of this
                list should be the same length as @idns
        """
        # We have to be careful here -- some particle instancers may have been deleted / are mismatched, so we need
        # to update accordingly, potentially deleting stale instancers and creating new instancers as needed
        idn_to_info_mapping = {idn: {"group": group, "count": count}
                               for idn, group, count in zip(idns, particle_groups, particle_counts)}
        current_instancer_names = set(cls.particle_instancers.keys())
        desired_instancer_names = set(cls.particle_instancer_idn_to_name(idn=idn) for idn in idns)
        instancers_to_delete = current_instancer_names - desired_instancer_names
        instancers_to_create = desired_instancer_names - current_instancer_names
        common_instancers = current_instancer_names.intersection(desired_instancer_names)

        # Sanity check the common instancers, we will recreate any where there is a mismatch
        for name in common_instancers:
            idn = cls.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            instancer = cls.particle_instancers[name]
            if instancer.particle_group != info["group"]:
                instancer.particle_group = info["group"]
            count_diff = info["count"] - instancer.n_particles
            if count_diff > 0:
                # We need to add more particles to this group
                instancer.add_particles(positions=np.zeros((count_diff, 3)))
            elif count_diff < 0:
                # We need to remove particles from this group
                instancer.remove_particles(idxs=np.arange(-count_diff))

        # Delete any instancers we no longer want
        for name in instancers_to_delete:
            cls.remove_particle_instancer(name=name)

        # Create any instancers we don't already have
        for name in instancers_to_create:
            idn = cls.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            cls.generate_particle_instancer(idn=idn, particle_group=info["group"], n_particles=info["count"])

    @classmethod
    def _dump_state(cls):
        return dict(
            n_instancers=cls.n_instancers,
            instancer_idns=cls.instancer_idns,
            instancer_particle_groups=[inst.particle_group for inst in cls.particle_instancers.values()],
            instancer_particle_counts=[inst.n_particles for inst in cls.particle_instancers.values()],
            particle_states=dict(((name, inst.dump_state(serialized=False))
                                  for name, inst in cls.particle_instancers.items())),
        )

    @classmethod
    def _load_state(cls, state):
        # Synchronize the particle instancers
        cls._sync_particle_instancers(
            idns=state["instancer_idns"],
            particle_groups=state["instancer_particle_groups"],
            particle_counts=state["instancer_particle_counts"],
        )

        # Iterate over all particle states and load their respective states
        for name, inst_state in state["particle_states"].items():
            cls.particle_instancers[name].load_state(inst_state, serialized=False)

    @classmethod
    def _serialize(cls, state):
        # Array is number of particle instancers, then the corresponding states for each particle instancer
        return np.concatenate([
            [state["n_instancers"]],
            state["instancer_idns"],
            state["instancer_particle_groups"],
            state["instancer_particle_counts"],
            *[cls.particle_instancers[name].serialize(inst_state)
              for name, inst_state in state["particle_states"].items()],
        ]).astype(float)

    @classmethod
    def _deserialize(cls, state):
        # Synchronize the particle instancers
        n_instancers = int(state[0])
        instancer_info = dict()
        idx = 1
        for info_name in ("instancer_idns", "instancer_particle_groups", "instancer_particle_counts"):
            instancer_info[info_name] = state[idx: idx + n_instancers].astype(int).tolist()
            idx += n_instancers

        # Syncing is needed so that each particle instancer can further deserialize its own state
        log.debug(f"Syncing {cls.name} particles with {n_instancers} instancers..")
        cls._sync_particle_instancers(
            idns=instancer_info["instancer_idns"],
            particle_groups=instancer_info["instancer_particle_groups"],
            particle_counts=instancer_info["instancer_particle_counts"],
        )

        # Procedurally deserialize the particle states
        particle_states = dict()
        for idn in instancer_info["instancer_idns"]:
            name = cls.particle_instancer_idn_to_name(idn=idn)
            state_size = cls.particle_instancers[name].state_size
            particle_states[name] = cls.particle_instancers[name].deserialize(state[idx: idx + state_size])
            idx += state_size

        return dict(
            n_instancers=n_instancers,
            **instancer_info,
            particle_states=particle_states,
        ), idx

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
    def remove_all_particle_instancers(cls):
        """
        Removes all particle instancers and deletes them from the simulator
        """
        cls._sync_particle_instancers(idns=[], particle_groups=[], particle_counts=[])

    @classmethod
    def create(
        cls,
        name,
        particle_contact_offset,
        particle_density,
        **kwargs,
    ):
        """
        Utility function to programmatically generate monolithic fluid system classes.

        Args:
            name (str): Name of the system, in snake case.
            particle_contact_offset (float): Contact offset for the generated system
            particle_density (float): Particle density for the generated system
            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)

        Returns:
            PhysicalParticleSystem: Generated system class
        """

        # Override the necessary parameters
        @classproperty
        def cp_register_system(cls):
            # We should register this system since it's an "actual" system (not an intermediate class)
            return True

        @classproperty
        def cp_particle_contact_offset(cls):
            return particle_contact_offset

        @classproperty
        def cp_particle_density(cls):
            return particle_density

        # Add to any other params specified
        kwargs["_register_system"] = cp_register_system
        kwargs["particle_contact_offset"] = cp_particle_contact_offset
        kwargs["particle_density"] = cp_particle_density

        # Create and return the class
        return subclass_factory(name=snake_case_to_camel_case(name), base_classes=cls, **kwargs)


class FluidSystem(PhysicalParticleSystem):
    """
    Particle system class simulating fluids, leveraging isosurface feature in omniverse to render nice PBR fluid
    texture. Individual particles are composed of spheres.
    """

    # Material -- either a MaterialPrim or None if no material is used for this particle system
    _material = None

    # Color associated with this system (NOTE: external queries should call cls.color)
    # The default color is blue.
    _color = np.array([0.0, 0.0, 1.0])

    @classmethod
    def initialize(cls):
        # Run super first
        super().initialize()

        # Create the particle material
        cls._material = cls._create_particle_material_template()
        # Load the material
        cls._material.load()
        # Bind the material to the particle system (for isosurface) and the prototypes (for non-isosurface)
        cls._material.bind(cls.system_prim_path)
        for prototype in cls.particle_prototypes:
            cls._material.bind(prototype.prim_path)
        # Also apply physics to this material
        particleUtils.add_pbd_particle_material(og.sim.stage, cls.mat_path)
        # Force populate inputs and outputs of the shader
        cls._material.shader_force_populate()
        # Potentially modify the material
        cls._customize_particle_material()

        # Compute the overall color of the fluid system
        base_color_weight = cls._material.diffuse_reflection_weight
        transmission_weight = cls._material.enable_specular_transmission * cls._material.specular_transmission_weight
        total_weight = base_color_weight + transmission_weight
        if total_weight == 0.0:
            # If the fluid doesn't have any color, we add a "blue" tint by default
            color = np.array([0.0, 0.0, 1.0])
        else:
            base_color_weight /= total_weight
            transmission_weight /= total_weight
            # Weighted sum of base color and transmission color
            color = base_color_weight * cls._material.diffuse_reflection_color + \
                    transmission_weight * (0.5 * cls._material.specular_transmission_color + \
                                           0.5 * cls._material.specular_transmission_scattering_color)
        cls._color = color

        # Set custom isosurface rendering settings if we are using high-quality rendering
        if gm.ENABLE_HQ_RENDERING:
            set_carb_settings_for_fluid_isosurface()
            # We also modify the grid smoothing radius to avoid "blobby" appearances
            cls.system_prim.GetAttribute("physxParticleIsosurface:gridSmoothingRadius").Set(0.0001)

    @classproperty
    def is_fluid(cls):
        return True

    @classproperty
    def use_isosurface(cls):
        return True

    @classproperty
    def material(cls):
        """
        Returns:
            None or MaterialPrim: The bound material to this prim, if there is one
        """
        return cls._material

    @classproperty
    def particle_radius(cls):
        """
        Returns:
            float: Radius for the particles to be generated, since all fluids are composed of spheres
        """
        # Magic number from omni tutorials
        # See https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#offset-autocomputation
        return 0.99 * 0.6 * cls.particle_contact_offset

    @classproperty
    def _material_mtl_name(cls):
        """
        Returns:
            None or str: Material mdl preset name to use for generating this fluid material. NOTE: Should be an
                entry from OmniSurfacePresets.mdl, minus the "OmniSurface_" string. If None if specified, will default
                to the generic OmniSurface material
        """
        return None

    @classmethod
    def _create_particle_prototypes(cls):
        # Simulate particles with simple spheres
        prototype = UsdGeom.Sphere.Define(og.sim.stage, f"{cls.prim_path}/prototype0")
        prototype.CreateRadiusAttr().Set(cls.particle_radius)
        prototype = VisualGeomPrim(prim_path=prototype.GetPath().pathString, name=prototype.GetPath().pathString)
        prototype.visible = False
        return [prototype]

    @classmethod
    def _create_particle_material_template(cls):
        # We use a template from OmniPresets if @_material_mtl_name is specified, else the default OmniSurface
        return MaterialPrim(
            prim_path=cls.mat_path,
            name=cls.mat_name,
            load_config={
                "mdl_name": f"OmniSurface{'' if cls._material_mtl_name is None else 'Presets'}.mdl",
                "mtl_name": f"OmniSurface{'' if cls._material_mtl_name is None else ('_' + cls._material_mtl_name)}"
            }
        )

    @classmethod
    def create(
        cls,
        name,
        particle_contact_offset,
        particle_density,
        material_mtl_name=None,
        customize_particle_material=None,
        **kwargs,
    ):
        """
        Utility function to programmatically generate monolithic fluid system classes.

        Args:
            name (str): Name of the system
            particle_contact_offset (float): Contact offset for the generated system
            particle_density (float): Particle density for the generated system
            material_mtl_name (None or str): Material mdl preset name to use for generating this fluid material.
                NOTE: Should be an entry from OmniSurfacePresets.mdl, minus the "OmniSurface_" string.
                If None if specified, will default to the generic OmniSurface material
            customize_particle_material (None or function): Method for customizing the particle material for the fluid
                after it has been loaded. Default is None, which will produce a no-op.
                If specified, expected signature:

                _customize_particle_material(mat: MaterialPrim) --> None

                where @MaterialPrim is the material to modify in-place

            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @classproperty or @classmethod decorators!)

        Returns:
            FluidSystem: Generated system class
        """
        @classproperty
        def cp_material_mtl_name(cls):
            return material_mtl_name

        @classmethod
        def cm_customize_particle_material(cls):
            if customize_particle_material is not None:
                customize_particle_material(mat=cls._material)

        # Add to any other params specified
        kwargs["_material_mtl_name"] = cp_material_mtl_name
        kwargs["_customize_particle_material"] = cm_customize_particle_material

        # Create and return the class
        return super().create(
            name=name,
            particle_contact_offset=particle_contact_offset,
            particle_density=particle_density,
            **kwargs,
        )

def customize_particle_material_factory(attr, value):
    def func(mat):
        setattr(mat, attr, np.array(value))
    return func


class GranularSystem(PhysicalParticleSystem):
    """
    Particle system class simulating granular materials. Individual particles are composed of custom USD objects.
    """

    @classproperty
    def is_fluid(cls):
        return False

    @classmethod
    def _create_particle_prototypes(cls):
        # Load the particle template
        particle_template = cls._create_particle_template()
        og.sim.import_object(obj=particle_template, register=False)

        # Make sure there is no ambiguity about which mesh to use as the particle from this template
        assert len(particle_template.links) == 1, "GranularSystem particle template has more than one link"
        assert len(particle_template.root_link.visual_meshes) == 1, "GranularSystem particle template has more than one visual mesh"

        # The prototype is assumed to be the first and only visual mesh belonging to the root link
        visual_geom = list(particle_template.root_link.visual_meshes.values())[0]

        # Copy it to the standardized prim path
        prototype_path = f"{cls.prim_path}/prototype0"
        omni.kit.commands.execute("CopyPrim", path_from=visual_geom.prim_path, path_to=prototype_path)

        # Wrap it with VisualGeomPrim with the correct scale
        prototype = VisualGeomPrim(prim_path=prototype_path, name=prototype_path)
        prototype.scale = (cls.particle_radius * 2) / prototype.aabb_extent
        prototype.visible = False
        return [prototype]

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
    def create(
        cls,
        name,
        particle_contact_offset,
        particle_density,
        create_particle_template,
        **kwargs,
    ):
        """
        Utility function to programmatically generate monolithic fluid system classes.

        Args:
            name (str): Name of the system
            particle_contact_offset (float): Contact offset for the generated system
            particle_density (float): Particle density for the generated system
            material_mtl_name (None or str): Material mdl preset name to use for generating this fluid material.
                NOTE: Should be an entry from OmniSurfacePresets.mdl, minus the "OmniSurface_" string.
                If None if specified, will default to the generic OmniSurface material
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
            GranularSystem: Generated granular system class
        """

        @classmethod
        def cm_create_particle_template(cls):
            return create_particle_template(prim_path=f"{cls.prim_path}/template", name=f"{cls.name}_template")

        # Add to any other params specified
        kwargs["_create_particle_template"] = cm_create_particle_template

        # Create and return the class
        return super().create(
            name=name,
            particle_contact_offset=particle_contact_offset,
            particle_density=particle_density,
            **kwargs,
        )

FluidSystem.create(
    name="water",
    particle_contact_offset=0.012,
    particle_density=500.0,
    material_mtl_name="DeepWater",
)

FluidSystem.create(
    name="milk",
    particle_contact_offset=0.008,
    particle_density=500.0,
    material_mtl_name="WholeMilk",
)

FluidSystem.create(
    name="strawberry_smoothie",
    particle_contact_offset=0.008,
    particle_density=500.0,
    material_mtl_name="SkimMilk",
    customize_particle_material=customize_particle_material_factory("specular_reflection_color", [1.0, 0.64, 0.64]),
)

GranularSystem.create(
    name="diced_apple",
    particle_contact_offset=0.015,
    particle_density=500.0,
    create_particle_template=lambda prim_path, name: og.objects.DatasetObject(
        prim_path=prim_path,
        name=name,
        category="apple",
        model="agveuv",
        visible=False,
        fixed_base=False,
        visual_only=True,
        include_default_states=False,
        abilities={},
    ),
)

GranularSystem.create(
    name="dango",
    particle_contact_offset=0.015,
    particle_density=500.0,
    create_particle_template=lambda prim_path, name: og.objects.PrimitiveObject(
        prim_path=prim_path,
        name=name,
        primitive_type="Sphere",
        radius=0.015,
        rgba=[1.0, 1.0, 1.0, 1.0],
        visible=False,
        fixed_base=False,
        visual_only=True,
        include_default_states=False,
    )
)


class Cloth(MicroParticleSystem):
    """
    Particle system class to simulate cloth.
    """

    @classproperty
    def _register_system(cls):
        # We should register this system since it's an "actual" system (not an intermediate class)
        return True

    @classproperty
    def particle_contact_offset(cls):
        # TODO (eric): figure out whether one offset can fit all
        return 0.005

    @classproperty
    def state_size(cls):
        # Default is no state
        return 0

    @classmethod
    def _dump_state(cls):
        # Empty by default
        return dict()

    @classmethod
    def _load_state(cls, state):
        # Nothing by default
        pass

    @classmethod
    def _serialize(cls, state):
        # Nothing by default
        return np.array([], dtype=float)

    @classmethod
    def _deserialize(cls, state):
        # Nothing by default
        return dict(), 0
