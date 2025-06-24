import math

import torch as th
import trimesh

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.prims.material_prim import OmniPBRMaterialPrim, OmniSurfaceMaterialPrim
from omnigibson.prims.prim_base import BasePrim
from omnigibson.systems.system_base import BaseSystem, PhysicalParticleSystem
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.physx_utils import create_physx_particle_system, create_physx_particleset_pointinstancer
from omnigibson.utils.python_utils import assert_valid_key, torch_delete
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import (
    absolute_prim_path_to_scene_relative,
    scene_relative_prim_path_to_absolute,
)

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.CLOTH_PARTICLE_CONTACT_OFFSET = 0.0075
m.CLOTH_STRETCH_STIFFNESS = 100.0
m.CLOTH_BEND_STIFFNESS = 50.0
m.CLOTH_SHEAR_STIFFNESS = 70.0
m.CLOTH_DAMPING = 0.02
m.CLOTH_FRICTION = 0.4
m.CLOTH_DRAG = 0.001
m.CLOTH_LIFT = 0.003
m.MIN_PARTICLE_CONTACT_OFFSET = 0.005  # Minimum particle contact offset for physical micro particles
m.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = None  # If set, the maximum particle velocity for micro particle systems


def set_carb_settings_for_fluid_isosurface():
    """
    Sets relevant rendering settings in the carb settings in order to use isosurface effectively
    """
    min_frame_rate = 60
    # Make sure we have at least 60 FPS before setting "persistent/simulation/minFrameRate" to 60
    assert (
        (1 / og.sim.get_rendering_dt()) >= min_frame_rate
    ), f"isosurface HQ rendering requires at least {min_frame_rate} FPS; consider increasing rendering_frequency of env_config to {min_frame_rate}."

    # Settings for Isosurface
    isregistry = lazy.carb.settings.acquire_settings_interface()
    # disable grid and lights
    dOptions = isregistry.get_as_int("persistent/app/viewport/displayOptions")
    dOptions &= ~(1 << 6 | 1 << 8)
    isregistry.set_int("persistent/app/viewport/displayOptions", dOptions)
    isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_NUM_THREADS, 8)
    isregistry.set_bool(lazy.omni.physx.bindings._physx.SETTING_UPDATE_VELOCITIES_TO_USD, True)
    isregistry.set_bool(lazy.omni.physx.bindings._physx.SETTING_UPDATE_PARTICLES_TO_USD, True)
    isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, min_frame_rate)
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

    def __init__(self, relative_prim_path, name, idn):
        """
        Args:
            relative_prim_path (str): scene-local prim path of the Instancer to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            idn (int): Unique identification number to assign to this particle instancer. This is used to
                deterministically reproduce individual particle instancer states dynamically, even if we
                delete / add additional ones at runtime during simulation.
        """
        # Store inputs
        self._idn = idn

        # Run super method directly
        super().__init__(relative_prim_path=relative_prim_path, name=name)

    def _load(self):
        # We raise an error, this should NOT be created from scratch
        raise NotImplementedError("PhysxPointInstancer should NOT be loaded via this class! Should be created before.")

    def remove(self):
        # We need to create this parent prim to avoid calling the low level omniverse delete prim method
        parent_absolute_path = self.prim.GetParent().GetPath().pathString
        parent_relative_path = absolute_prim_path_to_scene_relative(self.scene, parent_absolute_path)
        self._parent_prim = BasePrim(relative_prim_path=parent_relative_path, name=f"{self._name}_parent")
        self._parent_prim.load(self.scene)
        super().remove()
        self._parent_prim.remove()

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

        positions (th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions.
        velocities (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
            If not specified, all will be set to 0
        orientations (None or th.tensor): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
            orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
        scales (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
            If not specified, will be scale [1, 1, 1] by default
        prototype_indices (None or list of int): If specified, should specify which prototype should be used for
            each particle. If None, will use all 0s (i.e.: the first prototype created)
        """
        n_new_particles = len(positions)

        velocities = th.zeros((n_new_particles, 3)) if velocities is None else velocities
        if orientations is None:
            orientations = th.zeros((n_new_particles, 4))
            orientations[:, -1] = 1.0
        scales = th.ones((n_new_particles, 3)) * th.ones((1, 3)) if scales is None else scales
        prototype_indices = th.zeros(n_new_particles, dtype=int) if prototype_indices is None else prototype_indices

        self.particle_positions = (
            th.vstack([self.particle_positions, positions]) if self.particle_positions.numel() > 0 else positions
        )
        self.particle_velocities = (
            th.vstack([self.particle_velocities, velocities]) if self.particle_velocities.numel() > 0 else velocities
        )
        self.particle_orientations = (
            th.vstack([self.particle_orientations, orientations])
            if self.particle_orientations.numel() > 0
            else orientations
        )
        self.particle_scales = th.vstack([self.particle_scales, scales]) if self.particle_scales.numel() > 0 else scales
        self.particle_prototype_ids = (
            th.cat([self.particle_prototype_ids, prototype_indices])
            if self.particle_prototype_ids.numel() > 0
            else prototype_indices
        )

    def remove_particles(self, idxs):
        """
        Remove particles from this instancer, specified by their indices @idxs in the data array

        Args:
            idxs (list or th.tensor of int): IDs corresponding to the indices of specific particles to remove from this
                instancer
        """
        if len(idxs) > 0:
            # Remove all requested indices and write to all the internal data arrays
            self.particle_positions = torch_delete(self.particle_positions, idxs, dim=0)
            self.particle_velocities = torch_delete(self.particle_velocities, idxs, dim=0)
            self.particle_orientations = torch_delete(self.particle_orientations, idxs, dim=0)
            self.particle_scales = torch_delete(self.particle_scales, idxs, dim=0)
            self.particle_prototype_ids = torch_delete(self.particle_prototype_ids, idxs, dim=0)

    def remove_all_particles(self):
        self.remove_particles(idxs=th.arange(self.n_particles))

    @property
    def n_particles(self):
        """
        Returns:
            int: Number of particles owned by this instancer
        """
        return len(self.particle_positions)

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
            th.tensor: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return vtarray_to_torch(self.get_attribute(attr="positions"))

    @particle_positions.setter
    def particle_positions(self, pos):
        """
        Set the particle positions for this instancer

        Args:
            th.tensor: (N, 3) numpy array, where each of the N particles' desired positions are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        self.set_attribute(attr="positions", val=lazy.pxr.Vt.Vec3fArray(pos.tolist()))

    @property
    def particle_orientations(self):
        """
        Returns:
            th.tensor: (N, 4) numpy array, where each of the N particles' orientations are expressed in (x,y,z,w)
                quaternion coordinates relative to this instancer's parent prim
        """
        orientations = self.get_attribute(attr="orientations")
        if len(orientations) == 0:
            return th.empty(0, dtype=th.float32)
        else:
            return th.tensor(
                [[ori.imaginary[0], ori.imaginary[1], ori.imaginary[2], ori.real] for ori in orientations],
                dtype=th.float32,
            )

    @particle_orientations.setter
    def particle_orientations(self, quat):
        """
        Set the particle positions for this instancer

        Args:
            th.tensor: (N, 4) numpy array, where each of the N particles' desired orientations are expressed in (x,y,z,w)
                quaternion coordinates relative to this instancer's parent prim
        """
        assert (
            quat.shape[0] == self.n_particles
        ), f"Got mismatch in particle setting size: {quat.shape[0]}, vs. number of particles {self.n_particles}!"
        # If the number of particles is nonzero, swap w position, since Quath takes (w,x,y,z)
        quat = quat
        if self.n_particles > 0:
            quat = quat[:, [3, 0, 1, 2]]
        self.set_attribute(attr="orientations", val=lazy.pxr.Vt.QuathArray.FromNumpy(quat.cpu().numpy()))

    @property
    def particle_velocities(self):
        """
        Returns:
            th.tensor: (N, 3) numpy array, where each of the N particles' velocities are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return vtarray_to_torch(self.get_attribute(attr="velocities"))

    @particle_velocities.setter
    def particle_velocities(self, vel):
        """
        Set the particle velocities for this instancer

        Args:
            th.tensor: (N, 3) numpy array, where each of the N particles' desired velocities are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        assert (
            vel.shape[0] == self.n_particles
        ), f"Got mismatch in particle setting size: {vel.shape[0]}, vs. number of particles {self.n_particles}!"
        self.set_attribute(attr="velocities", val=lazy.pxr.Vt.Vec3fArray(vel.tolist()))

    @property
    def particle_scales(self):
        """
        Returns:
            th.tensor: (N, 3) numpy array, where each of the N particles' scales are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        return vtarray_to_torch(self.get_attribute(attr="scales"))

    @particle_scales.setter
    def particle_scales(self, scales):
        """
        Set the particle scales for this instancer

        Args:
            th.tensor: (N, 3) numpy array, where each of the N particles' desired scales are expressed in (x,y,z)
                cartesian coordinates relative to this instancer's parent prim
        """
        assert (
            scales.shape[0] == self.n_particles
        ), f"Got mismatch in particle setting size: {scales.shape[0]}, vs. number of particles {self.n_particles}!"
        self.set_attribute(attr="scales", val=lazy.pxr.Vt.Vec3fArray(scales.tolist()))

    @property
    def particle_prototype_ids(self):
        """
        Returns:
            th.tensor: (N,) numpy array, where each of the N particles' prototype_id (i.e.: which prototype is being used
                for that particle)
        """
        return vtarray_to_torch(self.get_attribute(attr="protoIndices"))

    @particle_prototype_ids.setter
    def particle_prototype_ids(self, prototype_ids):
        """
        Set the particle prototype_ids for this instancer

        Args:
            th.tensor: (N,) numpy array, where each of the N particles' desired prototype_id
                (i.e.: which prototype is being used for that particle)
        """
        assert (
            prototype_ids.shape[0] == self.n_particles
        ), f"Got mismatch in particle setting size: {prototype_ids.shape[0]}, vs. number of particles {self.n_particles}!"
        self.set_attribute(attr="protoIndices", val=prototype_ids.int().cpu().numpy())

    @property
    def state_size(self):
        # idn (1), particle_group (1), n_particles (1), and the corresponding states for each particle
        # N * (pos (3) + vel (3) + orn (4) + scale (3) + prototype_id (1))
        return 3 + self.n_particles * 14

    def _dump_state(self):
        if self.particle_positions.numel() == 0 and self.particle_orientations.numel() == 0:
            local_positions, local_orientations = [], []
        else:
            local_positions = []
            local_orientations = []
            for global_pos, global_ori in zip(self.particle_positions, self.particle_orientations):
                local_pos, local_ori = self.scene.convert_world_pose_to_scene_relative(
                    global_pos,
                    global_ori,
                )
                local_positions.append(local_pos)
                local_orientations.append(local_ori)
        return dict(
            idn=self._idn,
            particle_group=self.particle_group,
            n_particles=self.n_particles,
            particle_positions=th.stack(local_positions) if len(local_positions) > 0 else th.empty(0, dtype=th.float32),
            particle_velocities=self.particle_velocities,
            particle_orientations=(
                th.stack(local_orientations) if len(local_orientations) > 0 else th.empty(0, dtype=th.float32)
            ),
            particle_scales=self.particle_scales,
            particle_prototype_ids=self.particle_prototype_ids,
        )

    def _load_state(self, state):
        # Sanity check the identification number and particle group
        assert self._idn == state["idn"], (
            f"Got mismatch in identification number for this particle instancer when "
            f"loading state! Should be: {self._idn}, got: {state['idn']}."
        )
        assert self.particle_group == state["particle_group"], (
            f"Got mismatch in particle group for this particle "
            f"instancer when loading state! Should be: {self.particle_group}, got: {state['particle_group']}."
        )

        local_positions = state["particle_positions"]
        local_orientations = state["particle_orientations"]
        if local_positions.numel() == 0 and local_orientations.numel() == 0:
            global_positions, global_orientations = th.tensor([]), th.tensor([])
            setattr(self, "particle_positions", global_positions)
            setattr(self, "particle_orientations", global_orientations)
        else:
            global_positions, global_orientations = zip(
                *[
                    self.scene.convert_scene_relative_pose_to_world(local_pos, local_ori)
                    for local_pos, local_ori in zip(local_positions, local_orientations)
                ]
            )
            setattr(self, "particle_positions", th.stack(global_positions))
            setattr(self, "particle_orientations", th.stack(global_orientations))

        # Set values appropriately
        keys = (
            "particle_velocities",
            "particle_scales",
            "particle_prototype_ids",
        )
        for key in keys:
            # Make sure the loaded state is a numpy array, it could have been accidentally casted into a list during
            # JSON-serialization
            val = th.tensor(state[key]) if not isinstance(state[key], th.Tensor) else state[key]
            setattr(self, key, val)

    def serialize(self, state):
        # Compress into a 1D array
        return th.cat(
            [
                th.tensor([state["idn"], state["particle_group"], state["n_particles"]]),
                state["particle_positions"].reshape(-1),
                state["particle_velocities"].reshape(-1),
                state["particle_orientations"].reshape(-1),
                state["particle_scales"].reshape(-1),
                state["particle_prototype_ids"],
            ]
        )

    def deserialize(self, state):
        # Sanity check the identification number
        assert self._idn == state[0], (
            f"Got mismatch in identification number for this particle instancer when "
            f"deserializing state! Should be: {self._idn}, got: {state[0]}."
        )
        assert self.particle_group == state[1], (
            f"Got mismatch in particle group for this particle "
            f"instancer when deserializing state! Should be: {self.particle_group}, got: {state[1]}."
        )

        # De-compress from 1D array
        n_particles = int(state[2])
        state_dict = dict(
            idn=int(state[0]),
            particle_group=int(state[1]),
            n_particles=n_particles,
        )

        # Process remaining keys and reshape automatically
        keys = (
            "particle_positions",
            "particle_velocities",
            "particle_orientations",
            "particle_scales",
            "particle_prototype_ids",
        )
        sizes = ((n_particles, 3), (n_particles, 3), (n_particles, 4), (n_particles, 3), (n_particles,))

        idx = 3
        for key, size in zip(keys, sizes):
            length = math.prod(size)
            state_dict[key] = state[idx : idx + length].reshape(size)
            idx += length

        return state_dict, idx


class MicroParticleSystem(BaseSystem):
    """
    Global system for modeling "micro" level particles, e.g.: water, seeds, cloth. This system leverages
    Omniverse's native physx particle systems
    """

    def __init__(self, name, customize_particle_material=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # Particle system prim in the scene, should be generated at runtime
        self.system_prim = None

        # Material -- MaterialPrim associated with this particle system
        self._material = None

        self._customize_particle_material = customize_particle_material

        # Color of the generated material. Default is white [1.0, 1.0, 1.0]
        # (NOTE: external queries should call self.color)
        self._color = th.tensor([1.0, 1.0, 1.0])

    def initialize(self, scene):
        # Run super first
        super().initialize(scene)

        # Run sanity checks
        if not gm.USE_GPU_DYNAMICS:
            raise ValueError(f"Failed to initialize {self.name} system. Please set gm.USE_GPU_DYNAMICS=True.")

        self.system_prim = self._create_particle_system()
        # Get material
        material = self._get_particle_material_template()
        # Load the material if it's newly created and has never been loaded before
        if not material.loaded:
            material.load()
        material.add_user(self)
        self._material = material
        # Bind the material to the particle system (for isosurface) and the prototypes (for non-isosurface)
        self._material.bind(self.system_prim_path)
        # Also apply physics to this material
        lazy.omni.physx.scripts.particleUtils.add_pbd_particle_material(
            og.sim.stage, self.mat_path, **self._pbd_material_kwargs
        )
        # Potentially modify the material
        self._customize_particle_material() if self._customize_particle_material is not None else None

    def _clear(self):
        self._material.remove_user(self)

        super()._clear()

        self.system_prim = None
        self._material = None
        self._color = th.tensor([1.0, 1.0, 1.0])

    @property
    def particle_radius(self):
        # Magic number from omni tutorials
        # See https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#offset-autocomputation
        # Also https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/ParticleSystem.html#particle-system-configuration
        return 0.99 * self._particle_contact_offset

    @property
    def color(self):
        """
        Returns:
            None or 3-array: If @self._material exists, this will be its corresponding RGB color. Otherwise,
                will return None
        """
        return self._color

    @property
    def material(self):
        return self._material

    @property
    def mat_path(self):
        """
        Returns:
            str: Path to this system's material in the scene stage
        """
        return f"{self.prim_path}/material"

    @property
    def mat_name(self):
        """
        Returns:
            str: Name of this system's material
        """
        return f"{self.name}:material"

    @property
    def _pbd_material_kwargs(self):
        """
        Returns:
            dict: Any PBD material kwargs to pass to the PBD material method particleUtils.add_pbd_particle_material
                used to define physical properties associated with this particle system
        """
        # Default is empty dictionary
        return dict()

    def _get_particle_material_template(self):
        """
        Creates the particle material template to be used for this particle system. Prim path does not matter,
        as it will be overridden internally such that it is a child prim of this particle system's prim.

        NOTE: This material is a template because it is loading an Omni material preset. It can then be customized (in
        addition to modifying its physical material properties) via @_customize_particle_material

        Returns:
            MaterialPrim: The material to apply to all particles
        """
        # Default is PBR material
        return OmniPBRMaterialPrim.get_material(
            scene=self.scene,
            prim_path=self.mat_path,
            name=self.mat_name,
        )

    def _customize_particle_material(self):
        """
        Modifies this particle system's particle material once it is loaded. Default is a no-op
        """
        pass

    @property
    def system_prim_path(self):
        return f"{self.prim_path}/system"

    @property
    def visual_only(self):
        """
        Returns:
            bool: Whether this particle system should be visual-only, i.e.: not subject to collisions and physics. If True,
                the generated particles will not move or collide
        """
        return False

    @property
    def particle_contact_offset(self):
        """
        Returns:
            float: Contact offset value to use for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#particle-system-configuration
                for more information
        """
        raise NotImplementedError()

    @property
    def use_smoothing(self):
        """
        Returns:
            bool: Whether to use smoothing or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#smoothing
                for more information
        """
        return False

    @property
    def use_anisotropy(self):
        """
        Returns:
            bool: Whether to use anisotropy or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#anisotropy
                for more information
        """
        return False

    @property
    def use_isosurface(self):
        """
        Returns:
            bool: Whether to use isosurface or not for this particle system.
                See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#isosurface
                for more information
        """
        return False

    def _create_particle_system(self):
        """
        Creates the single, global particle system. This should only be ever called once, and during initialize()

        Returns:
            Usd.Prim: Particle system prim created
        """
        return create_physx_particle_system(
            prim_path=self.system_prim_path,
            physics_scene_path=og.sim.get_physics_context().get_current_physics_scene_prim().GetPrimPath().pathString,
            particle_contact_offset=self._particle_contact_offset,
            visual_only=self.visual_only,
            smoothing=self.use_smoothing and gm.ENABLE_HQ_RENDERING,
            anisotropy=self.use_anisotropy and gm.ENABLE_HQ_RENDERING,
            isosurface=self.use_isosurface and gm.ENABLE_HQ_RENDERING,
        ).GetPrim()


class MicroPhysicalParticleSystem(MicroParticleSystem, PhysicalParticleSystem):
    """
    Global system for modeling physical "micro" level particles, e.g.: water, seeds, rice, etc. This system leverages
    Omniverse's native physx particle systems
    """

    def __init__(
        self,
        name,
        particle_density,
        particle_contact_offset=None,
        is_viscous=None,
        material_mtl_name=None,
        customize_particle_material=None,
        min_scale=None,
        max_scale=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name of the system, in snake case.
            particle_density (float): Particle density for the generated system
            particle_contact_offset (float): Contact offset for the generated system
            is_viscous (bool): Whether or not the generated system should be viscous
            material_mtl_name (None or str): Material mdl preset name to use for generating this fluid material.
                    NOTE: Should be an entry from OmniSurfacePresets.mdl, minus the "OmniSurface_" string.
                    If None if specified, will default to the generic OmniSurface material
                customize_particle_material (None or function): Method for customizing the particle material for the fluid
                    after it has been loaded. Default is None, which will produce a no-op.
                    If specified, expected signature:

                    _customize_particle_material(mat: MaterialPrim) --> None

                    where @MaterialPrim is the material to modify in-place
            min_scale (None or 3-array): If specified, sets the minumum bound for particles' relative scale.
                Else, defaults to 1
            max_scale (None or 3-array): If specified, sets the maximum bound for particles' relative scale.
                Else, defaults to 1
            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @property or @classmethod decorators!)
        """

        # Store the particle density
        self._particle_density = particle_density

        # Particle prototypes -- will be list of mesh prims to use as particle prototypes for this system
        self.particle_prototypes = list()

        # Particle instancers -- maps name to particle instancer prims (dict)
        self.particle_instancers = dict()

        self._particle_contact_offset = particle_contact_offset

        self.is_viscous = is_viscous

        # Material mdl preset name to use for generating this fluid material. NOTE: Should be an entry from
        # OmniSurfacePresets.mdl, minus the "OmniSurface_" string. If None if specified, will default to the generic
        # OmniSurface material
        self._material_mtl_name = material_mtl_name

        self._customize_particle_material = customize_particle_material

        # Run super
        return super().__init__(name=name, min_scale=min_scale, max_scale=max_scale, **kwargs)

    @property
    def n_particles(self):
        return sum([instancer.n_particles for instancer in self.particle_instancers.values()])

    @property
    def n_instancers(self):
        """
        Returns:
            int: Number of active particles in this system
        """
        return len(self.particle_instancers)

    @property
    def instancer_idns(self):
        """
        Returns:
            list of int: Per-instancer number of active particles in this system
        """
        return [inst.idn for inst in self.particle_instancers.values()]

    @property
    def self_collision(self):
        """
        Returns:
            bool: Whether this system's particle should have self collisions enabled or not
        """
        # Default is True
        return True

    def _sync_particle_prototype_ids(self):
        """
        Synchronizes the particle prototype IDs across all particle instancers when sim is stopped.
        Omniverse has a bug where all particle positions, orientations, velocities, and scales are correctly reset
        when sim is stopped, but not the prototype IDs. This function is a workaround for that.
        """
        if self.initialized and self.particle_instancers is not None:
            for instancer in self.particle_instancers.values():
                instancer.particle_prototype_ids = th.zeros(instancer.n_particles, dtype=th.int32)

    def initialize(self, scene):
        self._scene = scene

        # Create prototype before running super!
        self.particle_prototypes = self._create_particle_prototypes()

        # Run super
        super().initialize(scene)

        # Potentially set system prim's max velocity value
        if m.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY is not None:
            self.system_prim.GetProperty("maxVelocity").Set(m.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY)

        # TODO: remove this hack once omniverse fixes the issue (now we assume prototype IDs are all 0 always)
        og.sim.add_callback_on_stop(
            name=f"{self.name}_sync_particle_prototype_ids", callback=self._sync_particle_prototype_ids
        )

    def _clear(self):
        for prototype in self.particle_prototypes:
            og.sim.remove_prim(prototype)

        super()._clear()

        self.particle_prototypes = list()
        self.particle_instancers = dict()

    @property
    def next_available_instancer_idn(self):
        """
        Updates the max instancer identification number based on the current internal state
        """
        if self.n_instancers == 0:
            return self.default_instancer_idn
        else:
            for idn in range(max(self.instancer_idns) + 2):
                if idn not in self.instancer_idns:
                    return idn

    @property
    def default_instancer_idn(self):
        return 0

    @property
    def state_size(self):
        # We have the number of particle instancers (1), the instancer groups, particle groups, and,
        # number of particles in each instancer (3n),
        # and the corresponding states in each instancer (X)
        return (
            1 + 3 * len(self.particle_instancers) + sum(inst.state_size for inst in self.particle_instancers.values())
        )

    @property
    def default_particle_instancer(self):
        """
        Returns:
            PhysxParticleInstancer: Default particle instancer for this particle system
        """
        # Default instancer is the 0th ID instancer
        name = self.particle_instancer_idn_to_name(idn=self.default_instancer_idn)
        # NOTE: Cannot use dict.get() call for some reason; it messes up IDE introspection
        return (
            self.particle_instancers[name]
            if name in self.particle_instancers
            else self.generate_particle_instancer(n_particles=0, idn=self.default_instancer_idn)
        )

    @property
    def particle_contact_radius(self):
        # This is simply the contact offset
        return self._particle_contact_offset

    @property
    def particle_density(self):
        """
        Returns:
            float: Particle density for the generated system
        """
        return self._particle_density

    @property
    def is_fluid(self):
        """
        Returns:
            bool: Whether this system is modeling fluid or not
        """
        raise NotImplementedError()

    def _create_particle_prototypes(self):
        """
        Creates any relevant particle prototypes to be used by this particle system.

        Returns:
            list of VisualGeomPrim: Visual mesh prim(s) to use as this system's particle prototype(s)
        """
        raise NotImplementedError()

    def remove_particles(
        self,
        idxs,
        instancer_idn=None,
    ):
        """
        Removes pre-existing particles from instancer @instancer_idn

        Args:
            idxs (th.tensor): (n_particles,) shaped array specifying IDs of particles to delete
            instancer_idn (None or int): Unique identification number of the particle instancer to delete the particles
                from. If None, this system will delete particles from the default particle instancer
        """
        # Create a new particle instancer if a new idn is requested, otherwise use the pre-existing one
        inst = (
            self.default_particle_instancer
            if instancer_idn is None
            else self.particle_instancers.get(self.particle_instancer_idn_to_name(idn=instancer_idn), None)
        )

        assert inst is not None, f"No instancer with ID {inst} exists!"

        inst.remove_particles(idxs=idxs)

    def generate_particles(
        self,
        positions,
        instancer_idn=None,
        particle_group=0,
        velocities=None,
        orientations=None,
        scales=None,
        prototype_indices=None,
    ):
        """
        Generates new particles, either as part of a pre-existing instancer corresponding to @instancer_idn or as part
            of a newly generated instancer.

        Args:
            positions (th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions
            instancer_idn (None or int): Unique identification number of the particle instancer to assign the generated
                particles to. This is used to deterministically reproduce individual particle instancer states
                dynamically, even if we delete / add additional ones at runtime during simulation. If there is no
                active instancer that matches the requested idn, a new one will be created.
                If None, this system will add particles to the default particle instancer
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by
                @self.self_collision
            velocities (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
                If not specified, all will be set to 0
            orientations (None or th.tensor): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (self.min_scale, self.max_scale)
            prototype_indices (None or list of int): If specified, should specify which prototype should be used for
                each particle. If None, will randomly sample from all available prototypes

        Returns:
            PhysxParticleInstancer: Particle instancer that includes the generated particles
        """
        if not isinstance(positions, th.Tensor):
            positions = th.tensor(positions, dtype=th.float32)

        # Create a new particle instancer if a new idn is requested, otherwise use the pre-existing one
        inst = (
            self.default_particle_instancer
            if instancer_idn is None
            else self.particle_instancers.get(self.particle_instancer_idn_to_name(idn=instancer_idn), None)
        )

        n_particles = len(positions)
        if prototype_indices is not None:
            prototype_indices = (
                th.ones(n_particles, dtype=int) * prototype_indices
                if isinstance(prototype_indices, int)
                else th.tensor(prototype_indices, dtype=int)
            )
        else:
            prototype_indices = th.randint(len(self.particle_prototypes), (n_particles,))

        if inst is None:
            inst = self.generate_particle_instancer(
                idn=instancer_idn,
                particle_group=particle_group,
                n_particles=len(positions),
                positions=positions,
                velocities=velocities,
                orientations=orientations,
                scales=scales,
                prototype_indices=prototype_indices,
            )
        else:
            inst.add_particles(
                positions=positions,
                velocities=velocities,
                orientations=orientations,
                scales=scales,
                prototype_indices=prototype_indices,
            )

        # Update semantics
        lazy.isaacsim.core.utils.semantics.add_update_semantics(
            prim=lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=self.prim_path),
            semantic_label=self.name,
            type_label="class",
        )

        return inst

    def generate_particle_instancer(
        self,
        n_particles,
        idn=None,
        particle_group=0,
        positions=None,
        velocities=None,
        orientations=None,
        scales=None,
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
                with each other. Particles in the same group will have collision behavior dictated by
                @self.self_collision
            positions (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) positions.
                If not specified, will be set to the origin by default
            velocities (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) velocities.
                If not specified, all will be set to 0
            orientations (None or th.tensor): (n_particles, 4) shaped array specifying per-particle (x,y,z,w) quaternion
                orientations. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
            scales (None or th.tensor): (n_particles, 3) shaped array specifying per-particle (x,y,z) scales.
                If not specified, will be uniformly randomly sampled from (self.min_scale, self.max_scale)
            prototype_indices (None or list of int): If specified, should specify which prototype should be used for
                each particle. If None, will use all 0s (i.e.: the first prototype created)

        Returns:
            PhysxParticleInstancer: Generated particle instancer
        """
        # Run sanity checks
        assert self.initialized, "Must initialize system before generating particle instancers!"

        # Multiple particle instancers is NOT supported currently, since there is no clear use case for multiple
        assert self.n_instancers == 0, (
            f"Cannot create multiple instancers for the same system! "
            f"There is already {self.n_instancers} pre-existing instancers."
        )

        # Automatically generate an identification number for this instancer if none is specified
        if idn is None:
            idn = self.next_available_instancer_idn

        assert idn not in self.instancer_idns, f"instancer idn {idn} already exists."

        # Generate standardized prim path for this instancer
        name = self.particle_instancer_idn_to_name(idn=idn)

        # Create the instancer
        instance = create_physx_particleset_pointinstancer(
            name=name,
            particle_system_path=self.prim_path,
            physx_particle_system_path=self.system_prim_path,
            particle_group=particle_group,
            positions=th.zeros((n_particles, 3)) if positions is None else positions,
            self_collision=self.self_collision,
            fluid=self.is_fluid,
            particle_mass=None,
            particle_density=self.particle_density,
            orientations=orientations,
            velocities=velocities,
            angular_velocities=None,
            scales=self.sample_scales(n=n_particles) if scales is None else scales,
            prototype_prim_paths=[pp.prim_path for pp in self.particle_prototypes],
            prototype_indices=prototype_indices,
            enabled=not self.visual_only,
        )

        # Create the instancer object that wraps the raw prim
        instancer = PhysxParticleInstancer(
            relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, instance.GetPrimPath().pathString),
            name=name,
            idn=idn,
        )
        instancer.load(self.scene)
        instancer.initialize()
        self.particle_instancers[name] = instancer

        return instancer

    def generate_particles_from_link(
        self,
        obj,
        link,
        use_visual_meshes=True,
        mesh_name_prefixes=None,
        check_contact=True,
        instancer_idn=None,
        particle_group=0,
        sampling_distance=None,
        max_samples=None,
        prototype_indices=None,
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
            instancer_idn (None or int): Unique identification number of the particle instancer to assign the generated
                particles to. This is used to deterministically reproduce individual particle instancer states
                dynamically, even if we delete / add additional ones at runtime during simulation. If there is no
                active instancer that matches the requested idn, a new one will be created.
                If None, this system will add particles to the default particle instancer
            particle_group (int): ID for this particle set. Particles from different groups will automatically collide
                with each other. Particles in the same group will have collision behavior dictated by
                @self.self_collision.
                Only used if a new particle instancer is created!
            sampling_distance (None or float): If specified, sets the distance between sampled particles. If None,
                a simulator autocomputed value will be used
            max_samples (None or int): If specified, maximum number of particles to sample
            prototype_indices (None or list of int): If specified, should specify which prototype should be used for
                each particle. If None, will randomly sample from all available prototypes
        """
        return super().generate_particles_from_link(
            obj=obj,
            link=link,
            use_visual_meshes=use_visual_meshes,
            mesh_name_prefixes=mesh_name_prefixes,
            check_contact=check_contact,
            instancer_idn=instancer_idn,
            particle_group=particle_group,
            sampling_distance=sampling_distance,
            max_samples=max_samples,
            prototype_indices=prototype_indices,
        )

    def generate_particles_on_object(
        self,
        obj,
        instancer_idn=None,
        particle_group=0,
        sampling_distance=None,
        max_samples=None,
        min_samples_for_success=1,
        prototype_indices=None,
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
            max_samples (None or int): If specified, maximum number of particles to sample
            min_samples_for_success (int): Minimum number of particles required to be sampled successfully in order
                for this generation process to be considered successful
            prototype_indices (None or list of int): If specified, should specify which prototype should be used for
                each particle. If None, will randomly sample from all available prototypes

        Returns:
            bool: True if enough particles were generated successfully (number of successfully sampled points >=
                min_samples_for_success), otherwise False
        """
        return super().generate_particles_on_object(
            obj=obj,
            instancer_idn=instancer_idn,
            particle_group=particle_group,
            sampling_distance=sampling_distance,
            max_samples=max_samples,
            min_samples_for_success=min_samples_for_success,
            prototype_indices=prototype_indices,
        )

    def remove_particle_instancer(self, name):
        """
        Removes particle instancer with name @name from this system.

        Args:
            name (str): Particle instancer name to remove. If it does not exist, then an error will be raised
        """
        # Make sure the instancer actually exists
        assert_valid_key(key=name, valid_keys=self.particle_instancers, name="particle instancer")
        # Remove instancer from our tracking and delete its prim
        instancer = self.particle_instancers.pop(name)
        og.sim.remove_prim(instancer)

    def particle_instancer_name_to_idn(self, name):
        """
        Args:
            name (str): Particle instancer name
        Returns:
            int: Particle instancer identification number
        """
        return int(name.split(f"{self.name}Instancer")[-1])

    def particle_instancer_idn_to_name(self, idn):
        """
        Args:
            idn (idn): Particle instancer identification number
        Returns:
            str: Name of the particle instancer auto-generated from its unique identification number
        """
        return f"{self.name}Instancer{idn}"

    def get_particles_position_orientation(self):
        return self.default_particle_instancer.particle_positions, self.default_particle_instancer.particle_orientations

    def get_particles_local_pose(self):
        return self.get_particles_position_orientation()

    def get_particle_position_orientation(self, idx):
        pos, ori = self.get_particles_position_orientation()
        return pos[idx], ori[idx]

    def get_particle_local_pose(self, idx):
        return self.get_particle_position_orientation(idx=idx)

    def set_particles_position_orientation(self, positions=None, orientations=None):
        if positions is not None:
            self.default_particle_instancer.particle_positions = positions
        if orientations is not None:
            self.default_particle_instancer.particle_orientations = orientations

    def set_particles_local_pose(self, positions=None, orientations=None):
        self.set_particles_position_orientation(positions=positions, orientations=orientations)

    def set_particle_position_orientation(self, idx, position=None, orientation=None):
        if position is not None:
            positions = self.default_particle_instancer.particle_positions
            positions[idx] = position
            self.default_particle_instancer.particle_positions = positions
        if orientation is not None:
            orientations = self.default_particle_instancer.particle_orientations
            orientations[idx] = orientation
            self.default_particle_instancer.particle_orientations = orientations

    def set_particle_local_pose(self, idx, position=None, orientation=None):
        self.set_particle_position_orientation(idx=idx, position=position, orientation=orientation)

    def _sync_particle_instancers(self, idns, particle_groups, particle_counts):
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
        idn_to_info_mapping = {
            idn: {"group": group, "count": count} for idn, group, count in zip(idns, particle_groups, particle_counts)
        }
        current_instancer_names = set(self.particle_instancers.keys())
        desired_instancer_names = set(self.particle_instancer_idn_to_name(idn=idn) for idn in idns)
        instancers_to_delete = current_instancer_names - desired_instancer_names
        instancers_to_create = desired_instancer_names - current_instancer_names
        common_instancers = current_instancer_names.intersection(desired_instancer_names)

        # Sanity check the common instancers, we will recreate any where there is a mismatch
        for name in common_instancers:
            idn = self.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            instancer = self.particle_instancers[name]
            if instancer.particle_group != info["group"]:
                instancer.particle_group = info["group"]
            count_diff = info["count"] - instancer.n_particles
            if count_diff > 0:
                # We need to add more particles to this group
                instancer.add_particles(positions=th.zeros((count_diff, 3)))
            elif count_diff < 0:
                # We need to remove particles from this group
                instancer.remove_particles(idxs=th.arange(-count_diff))

        # Delete any instancers we no longer want
        for name in instancers_to_delete:
            self.remove_particle_instancer(name=name)

        # Create any instancers we don't already have
        for name in instancers_to_create:
            idn = self.particle_instancer_name_to_idn(name=name)
            info = idn_to_info_mapping[idn]
            self.generate_particle_instancer(idn=idn, particle_group=info["group"], n_particles=info["count"])

    def _dump_state(self):
        return dict(
            n_instancers=self.n_instancers,
            instancer_idns=self.instancer_idns,
            instancer_particle_groups=[inst.particle_group for inst in self.particle_instancers.values()],
            instancer_particle_counts=[inst.n_particles for inst in self.particle_instancers.values()],
            particle_states=(
                dict(((name, inst.dump_state(serialized=False)) for name, inst in self.particle_instancers.items()))
            ),
        )

    def _load_state(self, state):
        # Synchronize the particle instancers
        self._sync_particle_instancers(
            idns=(
                state["instancer_idns"].int().tolist()
                if isinstance(state["instancer_idns"], th.Tensor)
                else state["instancer_idns"]
            ),
            particle_groups=(
                state["instancer_particle_groups"].int().tolist()
                if isinstance(state["instancer_particle_groups"], th.Tensor)
                else state["instancer_particle_groups"]
            ),
            particle_counts=(
                state["instancer_particle_counts"].int().tolist()
                if isinstance(state["instancer_particle_counts"], th.Tensor)
                else state["instancer_particle_counts"]
            ),
        )

        # Iterate over all particle states and load their respective states
        for name, inst_state in state["particle_states"].items():
            self.particle_instancers[name].load_state(inst_state, serialized=False)

    def serialize(self, state):
        # Array is number of particle instancers, then the corresponding states for each particle instancer
        return th.cat(
            [
                th.tensor([state["n_instancers"]]),
                th.tensor(state["instancer_idns"]),
                th.tensor(state["instancer_particle_groups"]),
                th.tensor(state["instancer_particle_counts"]),
                *[
                    self.particle_instancers[name].serialize(inst_state)
                    for name, inst_state in state["particle_states"].items()
                ],
            ]
        )

    def deserialize(self, state):
        # Synchronize the particle instancers
        n_instancers = int(state[0])
        instancer_info = dict()
        idx = 1
        for info_name in ("instancer_idns", "instancer_particle_groups", "instancer_particle_counts"):
            instancer_info[info_name] = state[idx : idx + n_instancers].int().tolist()
            idx += n_instancers

        # Syncing is needed so that each particle instancer can further deserialize its own state
        log.debug(f"Syncing {self.name} particles with {n_instancers} instancers...")
        self._sync_particle_instancers(
            idns=instancer_info["instancer_idns"],
            particle_groups=instancer_info["instancer_particle_groups"],
            particle_counts=instancer_info["instancer_particle_counts"],
        )

        # Procedurally deserialize the particle states
        particle_states = dict()
        for idn in instancer_info["instancer_idns"]:
            name = self.particle_instancer_idn_to_name(idn=idn)
            particle_states[name], deserialized_items = self.particle_instancers[name].deserialize(state[idx:])
            idx += deserialized_items

        return (
            dict(
                n_instancers=n_instancers,
                **instancer_info,
                particle_states=particle_states,
            ),
            idx,
        )

    def remove_all_particles(self):
        self._sync_particle_instancers(idns=[], particle_groups=[], particle_counts=[])


class FluidSystem(MicroPhysicalParticleSystem):
    """
    Particle system class simulating fluids, leveraging isosurface feature in omniverse to render nice PBR fluid
    texture. Individual particles are composed of spheres.
    """

    def __init__(
        self,
        name,
        particle_contact_offset,
        particle_density,
        is_viscous=False,
        material_mtl_name=None,
        customize_particle_material=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name of the system
            particle_contact_offset (float): Contact offset for the generated system
            particle_density (float): Particle density for the generated system
            is_viscous (bool): Whether or not the generated fluid system should be viscous
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
                (Note: These values should have either @property or @classmethod decorators!)
        """

        def cm_customize_particle_material():
            if customize_particle_material is not None:
                customize_particle_material(self._material)

        # Create and return the class
        return super().__init__(
            name=name,
            particle_density=particle_density,
            particle_contact_offset=particle_contact_offset,
            is_viscous=is_viscous,
            material_mtl_name=material_mtl_name,
            customize_particle_material=cm_customize_particle_material,
            **kwargs,
        )

    def initialize(self, scene):
        # Run super first
        super().initialize(scene)

        # Assert that the material is an OmniSurface material.
        assert isinstance(
            self._material, OmniSurfaceMaterialPrim
        ), "FluidSystem material must be an instance of MaterialPrim"

        # Bind the material to the particle system (for isosurface) and the prototypes (for non-isosurface)
        self._material.bind(self.system_prim_path)
        for prototype in self.particle_prototypes:
            self._material.bind(prototype.prim_path)
        # Apply the physical material preset based on whether or not this fluid is viscous
        apply_mat_physics = (
            lazy.omni.physx.scripts.particleUtils.AddPBDMaterialViscous
            if self.is_viscous
            else lazy.omni.physx.scripts.particleUtils.AddPBDMaterialWater
        )
        apply_mat_physics(p=self._material.prim)

        # Compute the overall color of the fluid system
        self._color = self._material.average_diffuse_color

        # Set custom isosurface rendering settings if we are using high-quality rendering
        if gm.ENABLE_HQ_RENDERING:
            set_carb_settings_for_fluid_isosurface()
            # We also modify the grid smoothing radius to avoid "blobby" appearances
            self.system_prim.GetAttribute("physxParticleIsosurface:gridSmoothingRadius").Set(0.0001)

    @property
    def is_fluid(self):
        return True

    @property
    def use_isosurface(self):
        return True

    @property
    def particle_radius(self):
        # Magic number from omni tutorials
        # See https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/physics-particles.html#offset-autocomputation
        return 0.99 * 0.6 * self._particle_contact_offset

    def _create_particle_prototypes(self):
        # Simulate particles with simple spheres
        prototype_prim_path = f"{scene_relative_prim_path_to_absolute(self._scene, self.relative_prim_path)}/prototype0"
        prototype = lazy.pxr.UsdGeom.Sphere.Define(og.sim.stage, prototype_prim_path)
        prototype.CreateRadiusAttr().Set(self.particle_radius)
        relative_prototype_prim_path = absolute_prim_path_to_scene_relative(self._scene, prototype_prim_path)
        prototype = VisualGeomPrim(relative_prim_path=relative_prototype_prim_path, name=f"{self.name}_prototype0")
        prototype.load(self._scene)
        prototype.visible = False
        lazy.isaacsim.core.utils.semantics.add_update_semantics(
            prim=prototype.prim,
            semantic_label=self.name,
            type_label="class",
        )
        return [prototype]

    def _get_particle_material_template(self):
        # We use a template from OmniPresets if @_material_mtl_name is specified, else the default OmniSurface
        return OmniSurfaceMaterialPrim.get_material(
            scene=self.scene, prim_path=self.mat_path, name=self.mat_name, preset_name=self._material_mtl_name
        )


class GranularSystem(MicroPhysicalParticleSystem):
    """
    Particle system class simulating granular materials. Individual particles are composed of custom USD objects.
    """

    def __init__(
        self,
        name,
        particle_density,
        create_particle_template,
        scale=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name of the system
            particle_density (float): Particle density for the generated system
            create_particle_template (function): Method for generating the visual particle template that will be duplicated
                when generating groups of particles.
                Expected signature:

                create_particle_template(prim_path: str, name: str) --> EntityPrim

                where @prim_path and @name are the parameters to assign to the generated EntityPrim.
                NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
                    visual mesh attached to its root link, since this will be the actual visual mesh used
            scale (None or 3-array): If specified, sets the scaling factor for the particles' relative scale.
                Else, defaults to 1
            **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
                the class attribute to modify and the values represent the functions / value to set
                (Note: These values should have either @property or @classmethod decorators!)
        """
        self._particle_template = None
        self._create_particle_template_fcn = create_particle_template
        return super().__init__(
            name=name,
            particle_density=particle_density,
            # Cached particle contact offset determined from loaded prototype
            particle_contact_offset=None,
            min_scale=scale,
            max_scale=scale,
            **kwargs,
        )

    @property
    def self_collision(self):
        # Don't self-collide to improve physics stability
        # For whatever reason, granular (non-fluid) particles tend to explode when sampling Filled states, and it seems
        # the only way to avoid this unstable behavior is to disable self-collisions. This actually enables the granular
        # particles to converge to zero velocity.
        return False

    def _clear(self):
        self.scene.remove_object(self._particle_template)

        super()._clear()

        self._particle_template = None
        self._particle_contact_offset = None

    @property
    def particle_contact_offset(self):
        return self._particle_contact_offset

    @property
    def is_fluid(self):
        return False

    def _create_particle_prototypes(self):
        # Load the particle template
        particle_template = self._create_particle_template()
        self._scene.add_object(particle_template, register=False)
        self._particle_template = particle_template
        # Make sure there is no ambiguity about which mesh to use as the particle from this template
        assert len(particle_template.links) == 1, "GranularSystem particle template has more than one link"
        assert (
            len(particle_template.root_link.visual_meshes) == 1
        ), "GranularSystem particle template has more than one visual mesh"

        # Make sure template scaling is [1, 1, 1] -- any particle scaling should be done via self.min/max_scale
        assert th.all(particle_template.scale == 1.0)

        # The prototype is assumed to be the first and only visual mesh belonging to the root link
        visual_geom = list(particle_template.root_link.visual_meshes.values())[0]

        # Copy it to the standardized prim path
        prototype_path = f"{self.prim_path}/prototype0"
        lazy.omni.kit.commands.execute("CopyPrim", path_from=visual_geom.prim_path, path_to=prototype_path)

        # Wrap it with VisualGeomPrim with the correct scale
        relative_prototype_path = absolute_prim_path_to_scene_relative(self._scene, prototype_path)
        prototype = VisualGeomPrim(relative_prim_path=relative_prototype_path, name=prototype_path)
        prototype.load(self._scene)
        prototype.scale *= self.max_scale
        prototype.visible = False
        lazy.isaacsim.core.utils.semantics.add_update_semantics(
            prim=prototype.prim,
            semantic_label=self.name,
            type_label="class",
        )

        # Store the contact offset based on a minimum sphere
        # Threshold the lower-bound to avoid super small particles
        vertices = th.tensor(prototype.get_attribute("points")) * prototype.scale
        _, particle_contact_offset = trimesh.nsphere.minimum_nsphere(trimesh.Trimesh(vertices=vertices))
        particle_contact_offset = th.tensor(particle_contact_offset, dtype=th.float32).item()
        if particle_contact_offset < m.MIN_PARTICLE_CONTACT_OFFSET:
            prototype.scale *= m.MIN_PARTICLE_CONTACT_OFFSET / particle_contact_offset
            particle_contact_offset = m.MIN_PARTICLE_CONTACT_OFFSET

        self._particle_contact_offset = particle_contact_offset

        return [prototype]

    def _create_particle_template(self):
        """
        Creates the particle template to be used for this system.

        NOTE: The loaded particle template is expected to be a non-articulated, single-link object with a single
            visual mesh attached to its root link, since this will be the actual visual mesh used

        Returns:
            EntityPrim: Particle template that will be duplicated when generating future particle groups
        """
        return self._create_particle_template_fcn(
            relative_prim_path=f"/{self.name}/template", name=f"{self.name}_template"
        )


class Cloth(MicroParticleSystem):
    """
    Particle system class to simulate cloth.
    """

    def __init__(
        self,
        name,
        **kwargs,
    ):
        """
        Args:
            name (str): Name of the system
        """
        self._particle_contact_offset = m.CLOTH_PARTICLE_CONTACT_OFFSET
        return super().__init__(name=name, **kwargs)

    def remove_all_particles(self):
        # Override base method since there are no particles to be deleted
        pass

    def clothify_mesh_prim(self, mesh_prim):
        """
        Clothifies @mesh_prim by applying the appropriate Cloth API, optionally re-meshing the mesh so that the
        resulting generated particles are roughly @particle_distance apart from each other.

        Args:
            mesh_prim (Usd.Prim): Mesh prim to clothify
        """
        # Convert into particle cloth
        lazy.omni.physx.scripts.particleUtils.add_physx_particle_cloth(
            stage=og.sim.stage,
            path=mesh_prim.GetPath(),
            dynamic_mesh_path=None,
            particle_system_path=self.system_prim_path,
            spring_stretch_stiffness=m.CLOTH_STRETCH_STIFFNESS,
            spring_bend_stiffness=m.CLOTH_BEND_STIFFNESS,
            spring_shear_stiffness=m.CLOTH_SHEAR_STIFFNESS,
            spring_damping=m.CLOTH_DAMPING,
            self_collision=True,
            self_collision_filter=True,
        )

        # Disable welding because it can potentially make thin objects non-manifold
        auto_particle_cloth_api = lazy.pxr.PhysxSchema.PhysxAutoParticleClothAPI(mesh_prim)
        auto_particle_cloth_api.GetDisableMeshWeldingAttr().Set(True)

    @property
    def _pbd_material_kwargs(self):
        return dict(
            friction=m.CLOTH_FRICTION,
            drag=m.CLOTH_DRAG,
            lift=m.CLOTH_LIFT,
        )

    @property
    def _register_system(self):
        # We should register this system since it's an "actual" system (not an intermediate class)
        return True

    @property
    def particle_contact_offset(self):
        return m.CLOTH_PARTICLE_CONTACT_OFFSET

    @property
    def state_size(self):
        # Default is no state
        return 0

    def _dump_state(self):
        # Empty by default
        return dict()

    def _load_state(self, state):
        # Nothing by default
        pass

    def serialize(self, state):
        # Nothing by default
        return th.empty(0, dtype=th.float32)

    def deserialize(self, state):
        # Nothing by default
        return dict(), 0
