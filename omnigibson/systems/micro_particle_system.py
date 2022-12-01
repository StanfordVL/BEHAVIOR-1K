import os
import time

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.prims.prim_base import BasePrim
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.systems.system_base import SYSTEMS_REGISTRY
from omnigibson.systems.particle_system_base import BaseParticleSystem
from omnigibson.utils.constants import SemanticClass
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty, assert_valid_key, subclass_factory
from omnigibson.utils.sampling_utils import sample_cuboid_on_object_full_grid_topdown
from omnigibson.utils.usd_utils import create_joint, array_to_vtarray
from omnigibson.utils.ui_utils import disclaimer
from omnigibson.utils.physx_utils import create_physx_particle_system, create_physx_particleset_pointinstancer, \
    bind_material, get_prototype_path_from_particle_system_path
import omni
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.physx.scripts import particleUtils
from omni.physx import get_physx_scene_query_interface
from collections import OrderedDict
import numpy as np
from pxr import Gf, Vt, UsdShade, UsdGeom, PhysxSchema
from omni.isaac.core.utils.stage import get_current_stage
from collections import defaultdict


# physics settins
from omni.physx.bindings._physx import (
    SETTING_UPDATE_TO_USD,
    SETTING_UPDATE_VELOCITIES_TO_USD,
    SETTING_NUM_THREADS,
    SETTING_UPDATE_PARTICLES_TO_USD,
)
import carb


# Create settings for this module
m = create_module_macros(module_path=__file__)

# Garbage collect particle instancers where less than 25 percent of the particle are visible
m.GC_THRESHOLD = 0.25

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
    isregistry.set_bool(SETTING_UPDATE_VELOCITIES_TO_USD, False)
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


def get_fluid_systems():
    """
    Returns:
        OrderedDict: Mapping from fluid system name to fluid system
    """
    systems = OrderedDict()
    for system in SYSTEMS_REGISTRY.objects:
        if issubclass(system, FluidSystem):
            systems[system.name] = system

    return systems


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

    def _load(self, simulator=None):
        # We raise an error, this should NOT be created from scratch
        raise NotImplementedError("PhysxPointInstancer should NOT be loaded via this class! Should be created before.")

    def _post_load(self):
        # Run super
        super()._post_load()

        # Store how many particles we have
        self._n_particles = len(self.particle_positions)

        # Set the invisibleIds to be 0, and all particles to be 1
        self.set_attribute(attr="ids", val=np.ones(self._n_particles))
        self.set_attribute(attr="invisibleIds", val=[0])

    def _initialize(self):
        # Run super first
        super()._initialize()

    def update_default_state(self):
        pass

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

    @property
    def position(self):
        """
        Returns:
            3-array: (x,y,z) translation of this point instancer relative to its parent prim
        """
        return np.array(self.get_attribute(attr="xformOp:translate"))

    @position.setter
    def position(self, pos):
        """
        Sets this point instancer's (x,y,z) cartesian translation relative to its parent prim

        Args:
            pos (3-array): (x,y,z) relative position to set this prim relative to its parent prim
        """
        self.set_attribute(attr="xformOp:translate", val=Gf.Vec3f(*(pos.astype(float))))

    @property
    def particle_positions(self):
        """
        Returns:
            np.array: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return np.array(self.get_attribute(attr="positions")) + self.position

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
        pos = (pos - self.position).astype(float)
        self.set_attribute(attr="positions", val=array_to_vtarray(arr=pos, element_type=Gf.Vec3f))

    @property
    def particle_orientations(self):
        """
        Returns:
            np.array: (N, 4) numpy array, where each of the N particles' orientations are expressed in (x,y,z,w)
                quaternion coordinates relative to this instancer's parent prim
        """
        oris = self.get_attribute(attr="orientations")
        if oris is None:
            # Default orientations for all particles
            oris = np.zeros((self.n_particles, 4))
            oris[:, -1] = 1.0
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
        self.set_attribute(attr="orientations", val=array_to_vtarray(arr=quat, element_type=Gf.Quath))

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
        self.set_attribute(attr="velocities", val=array_to_vtarray(arr=vel, element_type=Gf.Vec3f))

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
        self.set_attribute(attr="scales", val=array_to_vtarray(arr=scales, element_type=Gf.Vec3f))

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
    def particle_visibilities(self):
        """
        Returns:
            np.array: (N,) numpy array, where each entry is the specific particle's visibility
                (1 if visible, 0 otherwise)
        """
        # We leverage the ids + invisibleIds prim fields to infer visibility
        # id = 1 means visible, id = 0 means invisible
        ids = self.get_attribute("ids")
        return np.ones(self.n_particles) if ids is None else np.array(ids)

    @particle_visibilities.setter
    def particle_visibilities(self, visibilities):
        """
        Set the particle visibilities for this instancer

        Args:
            np.array: (N,) numpy array, where each entry is the specific particle's desired visibility
                (1 if visible, 0 otherwise)
        """
        assert visibilities.shape[0] == self._n_particles, \
            f"Got mismatch in particle setting size: {visibilities.shape[0]}, vs. number of particles {self._n_particles}!"
        # We leverage the ids + invisibleIds prim fields to infer visibility
        # id = 1 means visible, id = 0 means invisible
        self.set_attribute(attr="ids", val=visibilities)

    def _dump_state(self):
        return OrderedDict(
            idn=self._idn,
            particle_group=self.particle_group,
            n_particles=self._n_particles,
            position=self.position,
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
        for attr in ("positions", "velocities", "orientations", "scales", "prototype_ids"):
            attr_name = f"particle_{attr}"
            # Make sure the loaded state is a numpy array, it could have been accidentally casted into a list during
            # JSON-serialization
            attr_val = np.array(state[attr_name]) if not isinstance(attr_name, np.ndarray) else state[attr_name]
            setattr(self, attr_name, attr_val)

    def _serialize(self, state):
        # Compress into a 1D array
         return np.concatenate([
             [state["idn"], state["particle_group"], state["n_particles"]],
             state["position"],
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
        state_dict = OrderedDict(
            idn=int(state[0]),
            particle_group=int(state[1]),
            n_particles=n_particles,
        )

        # Process remaining keys and reshape automatically
        keys = ("position", "particle_positions", "particle_velocities", "particle_orientations", "particle_scales", "particle_prototype_ids")
        sizes = ((3,), (n_particles, 3), (n_particles, 3), (n_particles, 4), (n_particles, 3), (n_particles,))

        idx = 3
        for key, size in zip(keys, sizes):
            length = np.product(size)
            state_dict[key] = state[idx: idx + length].reshape(size)
            idx += length

        return state_dict, idx


class MicroParticleSystem(BaseParticleSystem):
    """
    Global system for modeling "micro" level particles, e.g.: water, seeds, rice, etc. This system leverages
    Omniverse's native physx particle systems
    """
    # Particle system prim in the scene, should be generated at runtime
    prim = None

    # Particle prototypes -- will be list of mesh prims to use as particle prototypes for this system
    particle_prototypes = None

    # Particle instancers -- maps name to particle instancer prims (OrderedDict)
    particle_instancers = None

    # Material -- either a MaterialPrim or None if no material is used for this particle system
    _material = None

    # Scaling factor to sample from when generating a new particle
    min_scale = None                # (x,y,z) scaling
    max_scale = None                # (x,y,z) scaling

    # Max particle instancer identification number -- this monotonically increases until reset() is called
    max_instancer_idn = None

    # State cache -- OrderedDict, anything that should be updated exactly once between simulation step calls
    state_cache = None

    @classproperty
    def n_particles(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        return sum([instancer.n_particles for instancer in cls.particle_instancers.values()])

    @classproperty
    def prim_path(cls):
        """
        Returns:
            str: Path to this system's prim in the scene stage
        """
        return f"/World/{cls.name}"

    @classproperty
    def mat_path(cls):
        """
        Returns:
            str: Path to this system's material in the scene stage
        """
        return f"{cls.prim_path}/{cls.name}_material"

    @classproperty
    def mat_name(cls):
        """
        Returns:
            str: Name of this system's material
        """
        return f"{cls.name}:material"

    @classproperty
    def material(cls):
        """
        Returns:
            None or MaterialPrim: The bound material to this prim, if there is one
        """
        return cls._material

    @classmethod
    def initialize(cls, simulator):
        # Run super first
        super().initialize(simulator=simulator)

        # Initialize class variables that are mutable so they don't get overridden by children classes
        cls.particle_instancers = OrderedDict()

        # Set the default scales
        cls.min_scale = np.ones(3)
        cls.max_scale = np.ones(3)

        # Initialize max instancer idn
        cls.max_instancer_idn = -1

        # Initialize state cache
        cls.state_cache = OrderedDict()

        # Create the particle system if it doesn't already exist, otherwise sync with the pre-existing system
        prototype_path = get_prototype_path_from_particle_system_path(particle_system_path=cls.prim_path)

        if cls.particle_system_exists:
            cls.prim = get_prim_at_path(cls.prim_path)
            # If Material already exists on stage, just create a MaterialPrim wrapper around it (run its post_load())
            if is_prim_path_valid(cls.mat_path):
                cls._material = cls._create_particle_material_template()
                cls._material.shader_force_populate()
            prototype_dir_prim = get_prim_at_path(prototype_path)
            cls.particle_prototypes = [prototype_prim for prototype_prim in prototype_dir_prim.GetChildren()]

            # Also need to synchronize any instancers we have
            for prim in cls.prim.GetChildren():
                name = prim.GetName()
                if prim.GetPrimTypeInfo().GetTypeName() == "PointInstancer":
                    cls.particle_instancers[name] = PhysxParticleInstancer(
                        prim_path=prim.GetPrimPath().pathString,
                        name=name,
                        idn=cls.particle_instancer_name_to_idn(name=name),
                    )
        else:
            cls.prim = cls._create_particle_system()
            # Create the particle material (only if we're using high-quality rendering since this takes time)
            cls._material = cls._create_particle_material_template() if gm.ENABLE_HQ_RENDERING else None
            if cls._material is not None:
                # Load the material
                cls._material.load()
                # Bind the material to the particle system
                cls._material.bind(cls.prim_path)
                # Also apply physics to this material
                particleUtils.add_pbd_particle_material(cls.simulator.stage, cls.mat_path)
                # Force populate inputs and outputs of the shader
                cls._material.shader_force_populate()
                # Potentially modify the material
                cls._customize_particle_material()
            # Create the particle prototypes, move them to the appropriate directory, and make them all invisible
            prototypes = cls._create_particle_prototypes()
            cls.particle_prototypes = []
            cls.simulator.stage.DefinePrim(prototype_path, "Scope")
            cls.simulator.stage.DefinePrim(f"{prototype_path}/{cls.name}", "Scope")
            for i, prototype_prim in enumerate(prototypes):
                # TODO: Omni no longer likes prototypes being created in nested locations. Where to place now?
                path_from = prototype_prim.GetPrimPath().pathString
                path_to = f"{prototype_path}/{cls.name}/ParticlePrototype{i}"
                omni.kit.commands.execute("MovePrim", path_from=path_from, path_to=path_to)
                # path_to = prototype_prim.GetPrimPath().pathString
                prototype_prim_new = get_prim_at_path(path_to)
                UsdGeom.Imageable(prototype_prim_new).MakeInvisible()
                cls.particle_prototypes.append(prototype_prim_new)

        # Set custom rendering settings if we're using a fluid isosurface
        if cls.is_fluid and cls.use_isosurface and gm.ENABLE_HQ_RENDERING:
            set_carb_settings_for_fluid_isosurface()

            # We also modify the grid smoothing radius to avoid "blobby" appearances
            cls.prim.GetAttribute("physxParticleIsosurface:gridSmoothingRadius").Set(0.0001)

        # # Create dummy
        # create_physx_particleset_pointinstancer(
        #     name="dummy",
        #     particle_system_path=cls.prim_path,
        #     particle_group=0,
        #     positions=np.zeros((1, 3)),
        #     fluid=cls.is_fluid,
        #     particle_mass=0.001,
        #     particle_density=None,#cls.particle_density,
        #     prototype_prim_paths=[pp.GetPrimPath().pathString for pp in cls.particle_prototypes],
        #     enabled=not cls.visual_only,
        # )

    @classmethod
    def reset(cls):
        # Reset all internal variables
        cls.remove_all_particle_instancers()
        cls.max_instancer_idn = -1
        cls.state_cache = OrderedDict()

    @classproperty
    def state_size(cls):
        # We have the number of particle instancers (1), the instancer groups, particle groups, and,
        # number of particles in each instancer (3n),
        # and the corresponding states in each instancer (X)
        return 1 + 3 * len(cls.particle_instancers) + sum(inst.state_size for inst in cls.particle_instancers.values())

    @classproperty
    def particle_system_exists(cls):
        """
        Returns:
            bool: Whether this particle system already exists on the current stage. Useful for internal logic
                synchronization during, e.g., initialization, where a USD snapshot may have been loaded with a particle
                system already on the stage
        """
        return is_prim_path_valid(cls.prim_path)

    @classproperty
    def is_fluid(cls):
        """
        Returns:
            bool: Whether this system is modeling fluid or not
        """
        raise NotImplementedError()

    @classproperty
    def visual_only(cls):
        """
        Returns:
            bool: Whether this particle system should be visual-only, i.e.: not subject to collisions and physics. If True,
                the generated particles will not move or collide
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    @classproperty
    def use_anisotropy(cls):
        """
        Returns:
            bool: Whether to use anisotropy or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#anisotropy
                for more information
        """
        raise NotImplementedError()

    @classproperty
    def use_isosurface(cls):
        """
        Returns:
            bool: Whether to use isosurface or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#isosurface
                for more information
        """
        raise NotImplementedError()

    @classproperty
    def particle_radius(cls):
        """
        Returns:
            float: Radius for the particles to be generated, since all fluids are composed of spheres
        """
        # Magic number from omni tutorials
        # See https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#offset-autocomputation
        return 0.99 * 0.6 * cls.particle_contact_offset if cls.is_fluid else 0.99 * cls.particle_contact_offset

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
            list of Usd.Prim: Mesh prim(s) to use as this system's particle prototype(s)
        """
        raise NotImplementedError()

    @classmethod
    def _create_particle_material_template(cls):
        """
        Creates the particle material template to be used for this particle system. Prim path does not matter,
        as it will be overridden internally such that it is a child prim of this particle system's prim.

        NOTE: This material is a template because it is loading an Omni material present. It can then be customized (in
        addition to modifying its physical material properties) via @_modify_particle_material

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
    def _create_particle_system(cls):
        """
        Creates the single, global particle system. This should only be ever called once, and during initialize()

        Returns:
            Usd.Prim: Particle system prim created
        """
        return create_physx_particle_system(
            prim_path=cls.prim_path,
            physics_scene_path=cls.simulator.get_physics_context().get_current_physics_scene_prim().GetPrimPath().pathString,
            particle_contact_offset=cls.particle_contact_offset,
            visual_only=cls.visual_only,
            smoothing=cls.use_smoothing and gm.ENABLE_HQ_RENDERING,
            anisotropy=cls.use_anisotropy and gm.ENABLE_HQ_RENDERING,
            isosurface=cls.use_isosurface and gm.ENABLE_HQ_RENDERING,
        ).GetPrim()

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
            idn = cls.max_instancer_idn + 1
            # Also increment this counter
            cls.max_instancer_idn += 1

        # Generate standardized prim path for this instancer
        name = cls.particle_instancer_idn_to_name(idn=idn)

        # Create the instancer
        instance = create_physx_particleset_pointinstancer(
            name=name,
            particle_system_path=cls.prim_path,
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
            prototype_prim_paths=[pp.GetPrimPath().pathString for pp in cls.particle_prototypes],
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
    def generate_particle_instancer_from_link(
            cls,
            obj,
            link,
            mesh_name_prefixes=None,
            idn=None,
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
            mesh_name_prefixes (None or str): If specified, specifies the substring that must exist in @link's
            mesh names in order for that mesh to be included in the particle generator function. If None, no filtering
            will be used.
            idn (None or int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation. If None, this system will generate a unique
                identifier automatically.
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by @self_collision
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            sample_volume (bool): Whether to sample the particles at the mesh's surface or throughout its entire volume
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not
            prototype_indices_choices (None or int or list of int): If specified, should specify which prototype(s)
                should be used for each particle. If None, will use all 0s (i.e.: the first prototype created). If a
                single number, will use that prototype ID for all sampled particles. If a list of int, will uniformly
                sample from those IDs for each particle.

        Returns:
            PhysxParticleInstancer: Generated particle instancer
        """
        # Run sanity checks
        assert cls.initialized, "Must initialize system before generating particle instancers!"
        # TODO: Implement!
        assert sample_volume, "Sampling surface of link for particles is not supported yet!"

        # Generate a checker function to see if particles are within the link's volumes
        check_in_volume, _ = generate_points_in_volume_checker_function(
            obj=obj,
            volume_link=link,
            use_visual_meshes=True,
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
        n_particles_per_axis = ((extent - 2 * cls.particle_radius) / sampling_distance).astype(int) + 1
        arrs = [np.linspace(lo + cls.particle_radius, hi - cls.particle_radius, n) for lo, hi, n in zip(low, high, n_particles_per_axis)]
        # Generate 3D-rectangular grid of points
        particle_positions = np.stack([arr.flatten() for arr in np.meshgrid(*arrs)]).T
        # Check which points are inside the volume and only keep those
        particle_positions = particle_positions[np.where(check_in_volume(particle_positions))[0]]
        # Also potentially sub-sample if we're past our limit
        if len(particle_positions) > max_samples:
            particle_positions = particle_positions[np.random.choice(len(particle_positions), size=(max_samples,), replace=False)]

        # Get information about our sampled points
        n_particles = len(particle_positions)
        if prototype_indices_choices is not None:
            prototype_indices = np.ones(n_particles, dtype=int) * prototype_indices_choices if \
                isinstance(prototype_indices_choices, int) else \
                np.random.choice(prototype_indices_choices, size=(n_particles,))
        else:
            prototype_indices = None

        # Create and return the generated instancer
        return cls.generate_particle_instancer(
            idn=idn,
            particle_group=particle_group,
            n_particles=n_particles,
            positions=particle_positions,
            velocities=None,
            orientations=None,
            scales=None,
            self_collision=self_collision,
            prototype_indices=prototype_indices,
        )

    @classmethod
    def generate_particle_instancer_from_mesh(
            cls,
            mesh_prim_path,
            idn=None,
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
            mesh_prim_path (str): Stage path to the mesh prim which will be converted into sampled particles
            idn (None or int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation. If None, this system will generate a unique
                identifier automatically.
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by @self_collision
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            sample_volume (bool): Whether to sample the particles at the mesh's surface or throughout its entire volume
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not
            prototype_indices_choices (None or int or list of int): If specified, should specify which prototype(s)
                should be used for each particle. If None, will use all 0s (i.e.: the first prototype created). If a
                single number, will use that prototype ID for all sampled particles. If a list of int, will uniformly
                sample from those IDs for each particle.

        Returns:
            PhysxParticleInstancer: Generated particle instancer
        """
        # Run sanity checks
        assert cls.initialized, "Must initialize system before generating particle instancers!"
        assert cls.simulator.is_stopped(), "Can only sample particles from a mesh using Omni's API when simulator is stopped!"

        # Create points prim (this is used initially to generate the particles) and apply particle set API
        points_prim_path = f"{cls.prim_path}/tempSampledPoints"
        points = UsdGeom.Points.Define(cls.simulator.stage, points_prim_path).GetPrim()
        particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points)
        particle_set_api.CreateParticleSystemRel().SetTargets([cls.prim_path])

        # Apply the sampling API to our mesh prim and apply the sampling
        mesh_prim = get_prim_at_path(mesh_prim_path)
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(mesh_prim)
        sampling_api.CreateParticlesRel().AddTarget(points_prim_path)
        sampling_api.CreateSamplingDistanceAttr().Set(0 if sampling_distance is None else sampling_distance)
        sampling_api.CreateMaxSamplesAttr().Set(max_samples)
        sampling_api.CreateVolumeAttr().Set(sample_volume)

        # We apply 1 physics step to propagate the sampling (make sure to pause the sim since particles still propagate
        # forward even if we don't explicitly call sim.step())
        with cls.simulator.paused():
            cls.simulator.step_physics()
        time.sleep(0.1)                         # Empirically validated for ~2500 samples (0.02 time doesn't work)
        cls.simulator.render()                  # Rendering is needed after an initial, nontrivial amount of time for the particles to actually be sampled

        # Grab the actual positions, we will write this to a new instancer that's not tied to the sampler
        # The points / instancer tied to the sampled mesh seems to operate a bit weirdly, which is why we do it this way
        attr = "positions" if points.GetPrimTypeInfo().GetTypeName() == "PointInstancer" else "points"
        pos = points.GetAttribute(attr).Get()

        # Make sure sampling was successful
        assert pos is not None and len(pos) > 0, "Failed to sample particle points from mesh prim!"

        # Delete the points prim and sampling API, we don't need it anymore, and make the mesh prim invisible again
        cls.simulator.stage.RemovePrim(points_prim_path)
        mesh_prim.RemoveAPI(PhysxSchema.PhysxParticleSamplingAPI)
        UsdGeom.Imageable(mesh_prim).MakeInvisible()

        # Get information about our sampled points
        n_particles = len(pos)
        if prototype_indices_choices is not None:
            prototype_indices = np.ones(n_particles, dtype=int) * prototype_indices_choices if \
                isinstance(prototype_indices_choices, int) else \
                np.random.choice(prototype_indices_choices, size=(n_particles,))
        else:
            prototype_indices = None

        # Create and return the generated instancer
        return cls.generate_particle_instancer(
            idn=idn,
            particle_group=particle_group,
            n_particles=n_particles,
            positions=pos,
            velocities=None,
            orientations=None,
            scales=None,
            self_collision=self_collision,
            prototype_indices=prototype_indices,
        )

    @classmethod
    def generate_particle_instancer_on_object(
            cls,
            obj,
            idn=None,
            particle_group=0,
            sampling_distance=None,
            max_samples=5e5,
            self_collision=True,
            prototype_indices_choices=None,
    ):
        """
        Generates @n_particles new particle objects and samples their locations on the top surface of object @obj

        Args:
            obj (BaseObject): Object on which to generate a particle instancer with sampled particles on the object's
                top surface
            idn (None or int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation. If None, this system will generate a unique
                identifier automatically.
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (int): Maximum number of particles to sample
            self_collision (bool): Whether to enable particle-particle collision within the set
                (as defined by @particle_group) or not
            prototype_indices_choices (None or int or list of int): If specified, should specify which prototype(s)
                should be used for each particle. If None, will use all 0s (i.e.: the first prototype created). If a
                single number, will use that prototype ID for all sampled particles. If a list of int, will uniformly
                sample from those IDs for each particle.

        Returns:
            PhysxParticleInstancer: Generated particle instancer
        """
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
            # undo_padding should be False - the sampled positions are above the surface by its radius
            undo_padding=False,
        )
        particle_positions = np.array([result[0] for result in results if result[0] is not None])
        # Also potentially sub-sample if we're past our limit
        if len(particle_positions) > max_samples:
            particle_positions = particle_positions[np.random.choice(len(particle_positions), size=(max_samples,), replace=False)]

        # Get information about our sampled points
        n_particles = len(particle_positions)
        if prototype_indices_choices is not None:
            prototype_indices = np.ones(n_particles, dtype=int) * prototype_indices_choices if \
                isinstance(prototype_indices_choices, int) else \
                np.random.choice(prototype_indices_choices, size=(n_particles,))
        else:
            prototype_indices = None

        # Spawn the particles at the valid positions
        cls.generate_particle_instancer(
            n_particles=n_particles,
            positions=particle_positions,
            idn=idn,
            particle_group=particle_group,
            self_collision=self_collision,
            prototype_indices=prototype_indices,
        )

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
        instancer.remove(simulator=cls.simulator)
        cls.simulator.stage.RemovePrim(f"{get_prototype_path_from_particle_system_path(particle_system_path=cls.prim_path)}/{name}")

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
        # print(f"common: {common_instancers}")
        for name in common_instancers:
            idn = cls.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            instancer = cls.particle_instancers[name]
            if instancer.particle_group != info["group"] or instancer.n_particles != info["count"]:
                print(f"Got mismatch in particle instancer {name} when syncing, deleting and recreating instancer now.")
                # Add this instancer to both the delete and creation pile
                instancers_to_delete.add(name)
                instancers_to_create.add(name)

        # Delete any instancers we no longer want
        # print(f"del: {instancers_to_delete}")
        for name in instancers_to_delete:
            cls.remove_particle_instancer(name=name)

        # Create any instancers we don't already have
        # print(f"create: {instancers_to_create}")
        for name in instancers_to_create:
            idn = cls.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            cls.generate_particle_instancer(idn=idn, particle_group=info["group"], n_particles=info["count"])

    @classmethod
    def _dump_state(cls):
        return OrderedDict(
            n_particle_instancers=len(cls.particle_instancers),
            instancer_idns=[inst.idn for inst in cls.particle_instancers.values()],
            instancer_particle_groups=[inst.particle_group for inst in cls.particle_instancers.values()],
            instancer_particle_counts=[inst.n_particles for inst in cls.particle_instancers.values()],
            particle_states=OrderedDict(((name, inst.dump_state(serialized=False))
                                         for name, inst in cls.particle_instancers.items())),
        )

    @classmethod
    def _load_state(cls, state):
        # TODO: Verify this is no longer an issue once omni's fluid writing is ready
        if og.sim.is_playing():
            # Post disclaimer, we cannot write to omni particle states when sim is playing currently
            disclaimer(f"We are attempting to load the [{cls.name}], a particle system, state, which requires writing "
                       f"to individual particle positions.\n"
                       f"Currently, Omniverse does not respect writing to these "
                       f"states, but this should be resolved by the next public release.\n"
                       f"As a result, the fluid state may not be set correctly.\n"
                       f"Two explicit cases where a failure case will occur:\n"
                       f"    - Isosurfaces: Because an extra physics step is required to render the fluid correctly"
                       f"           when generating isosurfaces, we lose the ability to set positions afterwards\n"
                       f"    - Existing instance with different positions: Because this instance will not be deleted "
                       f"           and recreated in the simulator, we lose the ability to set positions.")

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
            [state["n_particle_instancers"]],
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
        instancer_info = OrderedDict()
        idx = 1
        for info_name in ("instancer_idns", "instancer_particle_groups", "instancer_particle_counts"):
            instancer_info[info_name] = state[idx: idx + n_instancers].astype(int).tolist()
            idx += n_instancers
        print(f"Syncing {cls.name} particles with {n_instancers} instancers..")
        cls._sync_particle_instancers(
            idns=instancer_info["instancer_idns"],
            particle_groups=instancer_info["instancer_particle_groups"],
            particle_counts=instancer_info["instancer_particle_counts"],
        )

        # Procedurally deserialize the particle states
        particle_states = OrderedDict()
        print(f"total state size: {len(state)}")
        for idn in instancer_info["instancer_idns"]:
            print(f"Deserializing {idn}...")
            name = cls.particle_instancer_idn_to_name(idn=idn)
            state_size = cls.particle_instancers[name].state_size
            print(f"state size: {state_size}")
            particle_states[name] = cls.particle_instancers[name].deserialize(state[idx: idx + state_size])
            idx += state_size

        return OrderedDict(
            n_particle_instancers=n_instancers,
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
    def cache(cls):
        """
        Cache the collision hits for all particle systems, to be used by the object state system to determine
        if a particle is in contact with a given object.

        Currently, state_cache includes the following entries:

        "obj_particle_contacts": {
            obj0: {
                inst0: {particle_idx0, ...},
                inst1: {...},
                ...
            },
            obj1: ...,
        },

        where obji is an instance of BaseObject

        "link_particle_contacts": {
            link0: {
                inst0: {particle_idx0, ...},
                inst1: {...},
                ...
            },
            link1: ...,
        },

        where linki is a string representing the link's (rigid body's) prim path

        """
        obj_particle_contacts = defaultdict(lambda: defaultdict(set))
        link_particle_contacts = defaultdict(lambda: defaultdict(set))

        particle_instancer = None
        particle_idx = 0

        def report_hit(hit):
            base, body = "/".join(hit.rigid_body.split("/")[:-1]), hit.rigid_body.split("/")[-1]
            obj = cls.simulator.scene.object_registry("prim_path", base)
            if obj is not None:
                link = obj.links[body]
                obj_particle_contacts[obj][particle_instancer].add(particle_idx)
                link_particle_contacts[link][particle_instancer].add(particle_idx)
            return True

        for inst_name, inst in cls.particle_instancers.items():
            visibilities = inst.particle_visibilities
            for idx, pos in enumerate(inst.particle_positions):
                particle_idx = idx
                particle_instancer = inst
                # Only run the scene query if the particle is not hidden
                if visibilities[idx]:
                    get_physx_scene_query_interface().overlap_sphere(cls.particle_contact_offset, pos, report_hit, False)

        cls.state_cache["obj_particle_contacts"] = obj_particle_contacts
        cls.state_cache["link_particle_contacts"] = link_particle_contacts


class FluidSystem(MicroParticleSystem):
    """
    Particle system class simulating fluids, leveraging isosurface feature in omniverse to render nice PBR fluid
    texture. Individual particles are composed of spheres.
    """

    @classproperty
    def is_fluid(cls):
        return True

    @classproperty
    def visual_only(cls):
        return False

    @classproperty
    def use_smoothing(cls):
        return False

    @classproperty
    def use_anisotropy(cls):
        return False

    @classproperty
    def use_isosurface(cls):
        return True

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
        prototype = UsdGeom.Sphere.Define(cls.simulator.stage, f"{cls.prim_path}/{cls.name}ParticlePrototype")
        prototype.CreateRadiusAttr().Set(cls.particle_radius)
        return [prototype.GetPrim()]

    @classmethod
    def update(cls):
        # For each particle instance, garbage collect particles if the number of visible particles
        # is below the garbage collection threshold (m.GC_THRESHOLD)
        instancer_names = list(cls.particle_instancers.keys())
        for inst_name in instancer_names: # type: ignore
            if np.mean(cls.particle_instancers[inst_name].particle_visibilities) <= m.GC_THRESHOLD:
                cls.remove_particle_instancer(inst_name)

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
        fluid_name,
        particle_contact_offset,
        particle_density,
        material_mtl_name=None,
        customize_particle_material=None,
        **kwargs,
    ):
        """
        Utility function to programmatically generate monolithic fluid system classes.

        Args:
            fluid_name (str): Name of the fluid
            particle_contact_offset (float): Contact offset for the generated fluid system
            particle_density (float): Particle density for the generated fluid system
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
            FluidSystem: Generated fluid system class
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

        @classproperty
        def cp_material_mtl_name(cls):
            return material_mtl_name

        @classmethod
        def cm_customize_particle_material(cls):
            if customize_particle_material is not None:
                customize_particle_material(mat=cls._material)

        # Add to any other params specified
        kwargs["_register_system"] = cp_register_system
        kwargs["particle_contact_offset"] = cp_particle_contact_offset
        kwargs["particle_density"] = cp_particle_density
        kwargs["_material_mtl_name"] = cp_material_mtl_name
        kwargs["_customize_particle_material"] = cm_customize_particle_material

        # Create and return the class
        return subclass_factory(name=f"{fluid_name}System", base_classes=FluidSystem, **kwargs)


# Generate fluid systems
def customize_strawberry_smoothie_material(mat):
    # We modify the specular reflection color
    mat.specular_reflection_color = np.array([1.0, 0.64, 0.64])


WaterSystem = FluidSystem.create(
    fluid_name="Water",
    particle_contact_offset=0.012,
    particle_density=500.0,
    material_mtl_name="DeepWater",
)

MilkSystem = FluidSystem.create(
    fluid_name="Milk",
    particle_contact_offset=0.008,
    particle_density=500.0,
    material_mtl_name="WholeMilk",
)

StrawberrySmoothieSystem = FluidSystem.create(
    fluid_name="StrawberrySmoothie",
    particle_contact_offset=0.008,
    particle_density=500.0,
    material_mtl_name="SkimMilk",
    customize_particle_material=customize_strawberry_smoothie_material,
)


class ClothSystem(MicroParticleSystem):
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
    def is_fluid(cls):
        return False

    @classproperty
    def visual_only(cls):
        return False

    @classproperty
    def use_smoothing(cls):
        return False

    @classproperty
    def use_anisotropy(cls):
        return False

    @classproperty
    def use_isosurface(cls):
        return False

    @classmethod
    def _create_particle_prototypes(cls):
        return []

