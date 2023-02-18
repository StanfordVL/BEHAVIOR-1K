from abc import abstractmethod
from collections import defaultdict
import numpy as np
import omnigibson as og
from omnigibson.macros import create_module_macros, macros, gm
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.contact_particles import ContactParticles
from omnigibson.object_states.covered import Covered
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
from omnigibson.systems.system_base import get_element_name_from_system
from omnigibson.systems.macro_particle_system import VisualParticleSystem
from omnigibson.systems.micro_particle_system import PhysicalParticleSystem
from omnigibson.utils.constants import ParticleModifyMethod, PrimType
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function, get_particle_positions_from_frame
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.deprecated_utils import Core
from omnigibson.utils.usd_utils import create_primitive_mesh, FlatcacheAPI
import omnigibson.utils.transform_utils as T
from omnigibson.utils.sampling_utils import sample_cuboid_on_object
from omni.physx import get_physx_scene_query_interface as psqi
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, move_prim, is_prim_path_valid
from pxr import PhysicsSchemaTools, UsdGeom, Gf, Sdf


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.APPLICATION_LINK_NAME = "particleapplication_link"
m.REMOVAL_LINK_NAME = "particleremover_link"

# How many samples within the application area to generate per update step
m.MAX_VISUAL_PARTICLES_APPLIED_PER_STEP = 2
m.MAX_PHYSICAL_PARTICLES_APPLIED_PER_STEP = 10

# How many steps between generating particle samples
m.N_STEPS_PER_APPLICATION = 5
m.N_STEPS_PER_REMOVAL = 1

# Saturation thresholds -- maximum number of particles that can be applied by a ParticleApplier
m.VISUAL_PARTICLES_APPLICATION_LIMIT = 1000000
m.PHYSICAL_PARTICLES_APPLICATION_LIMIT = 1000000

# Saturation thresholds -- maximum number of particles that can be removed ("absorbed") by a ParticleRemover
m.VISUAL_PARTICLES_REMOVAL_LIMIT = 40
m.PHYSICAL_PARTICLES_REMOVAL_LIMIT = 400

# Fallback particle visualization radius for visualizing projected visual particles
m.VISUAL_PARTICLE_PROJECTION_PARTICLE_RADIUS = 0.01

# The margin (> 0) to add to the remover area's AABB when detecting overlaps with other objects
m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN = 0.05

# Settings for determining how the projection particles are visualized as they're projected
m.PROJECTION_VISUALIZATION_CONE_TIP_RADIUS = 0.001
m.PROJECTION_VISUALIZATION_RATE = 200
m.PROJECTION_VISUALIZATION_SPEED = 3.0
m.PROJECTION_VISUALIZATION_ORIENTATION_BIAS = 1e6
m.PROJECTION_VISUALIZATION_SPREAD_FACTOR = 0.8


def create_projection_visualization(
        prim_path,
        shape,
        projection_name,
        projection_radius,
        projection_height,
        particle_radius,
        material=None,
):
    """
    Helper function to generate a projection visualization using Omniverse's particle visualization system

    NOTE: Due to limitations with omniverse's generation scheme, the generated projection must have its origin at
    the origin of its parent frame, with its cone / cylinder facing in the local x-axis direction. The parent frame
    should also be aligned to its own parent frame to one of its own parent frame's axes - ie: any orientation such
    that its axes are exactly parallel / orthogonal to its parent axes.

    Args:
        prim_path (str): Stage location for where to generate the projection visualization
        shape (str): Shape of the projection to generate. Valid options are: {Sphere, Cone}
        projection_name (str): Name associated with this projection visualization. Should be unique!
        projection_radius (float): Radius of the generated projection visualization overall volume
        projection_height (float): Height of the generated projection visualization overall volume
        particle_radius (float): Radius of the particles composing the projection visualization
        material (None or MaterialPrim): If specified, specifies the material to associate with the generated
            particles within the projection visualization

    Returns:
        2-tuple:
            - UsdPrim: Generated ParticleSystem (ComputeGraph) prim generated
            - UsdPrim: Generated Emitter (ComputeGraph) prim generated
    """
    # Create the desired shape which will be used as the source input prim into the generated projection visualization
    source = UsdGeom.Sphere.Define(og.sim.stage, Sdf.Path(prim_path))
    # Modify the radius according to the desired @shape (and also infer the desired spread values)
    if shape == "Cylinder":
        source_radius = projection_radius
        spread = np.zeros(3)
    elif shape == "Cone":
        # Default to close to singular point otherwise
        source_radius = m.PROJECTION_VISUALIZATION_CONE_TIP_RADIUS
        spread_ratio = projection_radius * 2.0 / projection_height
        spread = np.ones(3) * spread_ratio * m.PROJECTION_VISUALIZATION_SPREAD_FACTOR
    else:
        raise ValueError(f"Invalid shape specified for projection visualization! Valid options are: [Sphere, Cylinder], got: {shape}")
    # Set the radius
    # Note that we divide the expected value in half since the native Sphere geom has native extents [2, 2, 2]
    source.GetRadiusAttr().Set(source_radius / 2.0)
    # Also make the prim invisible
    UsdGeom.Imageable(source.GetPrim()).MakeInvisible()
    # Generate the ComputeGraph nodes to render the projection
    core = Core(lambda val: None, particle_system_name=projection_name)
    system_path, _, emitter_path, vis_path, instancer_path, sprite_path, mat_path, output_path = core.create_particle_system(display="point_instancer", paths=[prim_path])
    # Override the prototype with our own sphere with optional material
    prototype_path = "/".join(sprite_path.split("/")[:-1]) + "/prototype"
    create_primitive_mesh(prototype_path, primitive_type="Sphere")
    prototype = VisualGeomPrim(prim_path=prototype_path, name=f"{projection_name}_prototype")
    prototype.initialize()
    # Set the scale (native scaling --> radius 0.5) and possibly update the material
    prototype.scale = particle_radius * 2.0
    if material is not None:
        prototype.material = material
    # Override the prototype used by the instancer
    instancer_prim = get_prim_at_path(instancer_path)
    instancer_prim.GetProperty("inputs:prototypes").SetTargets([prototype_path])

    # Destroy the old mat path since we don't use the sprites
    delete_prim(mat_path)

    # Modify the settings of the emitter to match the desired shape from inputs
    emitter_prim = get_prim_at_path(emitter_path)
    emitter_prim.GetProperty("inputs:rate").Set(m.PROJECTION_VISUALIZATION_RATE)
    emitter_prim.GetProperty("inputs:lifespan").Set(projection_height / m.PROJECTION_VISUALIZATION_SPEED)
    emitter_prim.GetProperty("inputs:speed").Set(m.PROJECTION_VISUALIZATION_SPEED)
    emitter_prim.GetProperty("inputs:alongAxis").Set(m.PROJECTION_VISUALIZATION_ORIENTATION_BIAS)
    emitter_prim.GetProperty("inputs:scale").Set(Gf.Vec3f(1.0, 1.0, 1.0))
    emitter_prim.GetProperty("inputs:directionRandom").Set(Gf.Vec3f(*spread))
    emitter_prim.GetProperty("inputs:addSourceVelocity").Set(1.0)

    # Move the output path so it moves with the particle system prim
    og.sim.render()
    output_name = output_path.split("/")[-1]
    move_prim(output_path, f"{system_path}/{output_name}")

    # Return the particle system prim which "owns" everything
    return get_prim_at_path(system_path), emitter_prim


class ParticleModifier(AbsoluteObjectState, LinkBasedStateMixin, UpdateStateMixin):
    """
    Object state representing an object that has the ability to modify visual and / or physical particles within the
    active simulation.

    Args:
        obj (StatefulObject): Object to which this state will be applied
        method (ParticleModifyMethod): Method to modify particles. Current options supported are ADJACENCY (i.e.:
            "touching" particles) or PROJECTION (i.e.: "spraying" particles)
        conditions (dict): Dictionary mapping ParticleSystem to None or corresponding condition / list of conditions
            (where None represents no conditions) necessary in order for this particle modifier to be able to
            modify particles belonging to @ParticleSystem. Each condition should be a function, whose signature
            is as follows:

                def condition(obj) --> bool

            Where @obj is the specific object that this ParticleModifier state belongs to.
            For a given ParticleSystem, if all of its conditions evaluate to True and particles are detected within
            this particle modifier area, then we potentially modify those particles
        projection_mesh_params (None or dict): If specified and @method is ParticleModifyMethod.PROJECTION,
            manually overrides any metadata found from @obj.metadata to infer what projection volume to generate
            for this particle modifier. Expected entries are as follows:

                "type": (str), one of {"Cylinder", "Cone"}
                "extents": (3-array), the (x,y,z) extents of the generated volume

            If None, information found from @obj.metadata will be used instead.
    """
    def __init__(self, obj, method, conditions, projection_mesh_params=None):

        # Store internal variables
        self.method = method
        self.conditions = conditions
        self.projection_source_sphere = None
        self.projection_mesh = None
        self.projection_system = None
        self.projection_emitter = None
        self._check_in_mesh = None
        self._check_overlap = None
        self._link_prim_paths = None
        self._current_step = None
        self._projection_mesh_params = projection_mesh_params

        # Map of system to number of modified particles for this object corresponding to the specific system
        self.modified_particle_count = dict([(system, 0) for system in self.supported_systems])

        # Standardize the conditions (make sure every system has at least one condition, which to make sure
        # the particle modifier isn't already limited with the specific number of particles)
        for system, conds in conditions.items():
            # Make sure the system is supported
            assert_valid_key(key=system, valid_keys=self.supported_systems, name="particle system")
            # Make sure conds isn't empty and is a list
            conds = [] if conds is None else list(conds)
            # Add the condition to avoid limits
            conds.append(self._generate_limit_condition(system=system))
            conditions[system] = conds

        # Run super method
        super().__init__(obj)

    @staticmethod
    def get_state_link_name():
        raise NotImplementedError()

    def _initialize(self):
        # Run link initialization
        self.initialize_link_mixin()

        # Initialize internal variables
        self._current_step = 0

        # Grab link prim paths and potentially update projection mesh params
        self._link_prim_paths = set(self.obj.link_prim_paths)

        # Define callback used during overlap method
        # We want to ignore any hits that are with this object itself
        valid_hit = False
        def overlap_callback(hit):
            nonlocal valid_hit
            valid_hit = hit.rigid_body not in self._link_prim_paths
            # Continue traversal only if we don't have a valid hit yet
            return not valid_hit

        # Possibly create a projection volume if we're using the projection method
        if self.method == ParticleModifyMethod.PROJECTION:
            # Make sure link is defined
            assert self.link is not None, f"Cannot use particle projection method without a metalink specified!"
            # Make sure projection mesh params are specified
            # Import here to avoid circular imports
            from omnigibson.objects.dataset_object import DatasetObject
            if self._projection_mesh_params is None and isinstance(self.obj, DatasetObject):
                # We try to grab metadata for this object
                self._projection_mesh_params = self.obj.metadata.get("meta_links", dict()).get(m.LINK_NAME, None)
            # Sanity check to make sure projection mesh params is not None
            assert self._projection_mesh_params is not None, \
                f"Projection mesh params must be specified for {self.obj.name}'s {self.__class__.__name__} state " \
                f"when method=ParticleModifyMethod.PROJECTION!"

            mesh_prim_path = f"{self.link.prim_path}/projection_mesh"
            # Create a primitive shape if it doesn't already exist
            radius, height = self._projection_mesh_params["extents"][0] / 2.0, self._projection_mesh_params["extents"][2]
            if not get_prim_at_path(mesh_prim_path):
                mesh = UsdGeom.__dict__[self._projection_mesh_params["type"]].Define(og.sim.stage, mesh_prim_path).GetPrim()
                # Set the height and radius (scaled by half since the native objects have extents [2, 2, 2]
                # TODO: Generalize to objects other than cylinder and radius
                mesh.GetAttribute("height").Set(height / 2.0)
                mesh.GetAttribute("radius").Set(radius / 2.0)

            # Create the visual geom instance referencing the generated mesh prim, and then hide it
            self.projection_mesh = VisualGeomPrim(prim_path=mesh_prim_path, name=f"{self.obj.name}_projection_mesh")
            self.projection_mesh.initialize()
            self.projection_mesh.visible = False

            # Make sure the object updates its meshes
            self.link.update_meshes()

            # Make sure the mesh is translated so that its tip lies at the metalink origin, and rotated so the vector
            # from tip to tail faces the positive x axis
            self.projection_mesh.set_local_pose(
                translation=np.array([self._projection_mesh_params["extents"][2] / (2 * self.link.scale[2]), 0, 0]),
                orientation=T.euler2quat([0, -np.pi / 2, 0]),
            )

            # Generate the projection visualization
            system = list(self.conditions.keys())[0]    # Only one system should be included for a ParticleApplier!
            particle_radius = m.VISUAL_PARTICLE_PROJECTION_PARTICLE_RADIUS if issubclass(system, VisualParticleSystem) else system.particle_radius
            particle_material = system.particle_object.material if issubclass(system, VisualParticleSystem) else system.material

            # Create the projection visualization if it doesn't already exist, otherwise we reference it directly
            projection_name = f"{self.obj.name}_projection_visualization"
            projection_path = f"/OmniGraph/{projection_name}"
            projection_visualization_path = f"{self.link.prim_path}/projection_visualization"
            if is_prim_path_valid(projection_path):
                self.projection_system = get_prim_at_path(projection_path)
                self.projection_emitter = get_prim_at_path(f"{projection_path}/emitter")
            else:
                self.projection_system, self.projection_emitter = create_projection_visualization(
                    prim_path=projection_visualization_path,
                    shape=self._projection_mesh_params["type"],
                    projection_name=projection_name,
                    projection_radius=radius,
                    projection_height=height,
                    particle_radius=particle_radius,
                    material=particle_material,
                )

            # Create the visual geom instance referencing the generated source mesh prim, and then hide it
            self.projection_source_sphere = VisualGeomPrim(prim_path=projection_visualization_path, name=f"{self.obj.name}_projection_source_sphere")
            self.projection_source_sphere.initialize()
            self.projection_source_sphere.visible = False

            # Generate the function for checking whether points are within the projection mesh
            self._check_in_mesh, _ = generate_points_in_volume_checker_function(
                obj=self.obj,
                volume_link=self.link,
                mesh_name_prefixes="projection",
            )

            # Store the projection mesh's IDs
            projection_mesh_ids = PhysicsSchemaTools.encodeSdfPath(self.projection_mesh.prim_path)

            # We also generate the function for checking overlaps at runtime
            def check_overlap():
                nonlocal valid_hit
                valid_hit = False
                psqi().overlap_shape(*projection_mesh_ids, reportFn=overlap_callback)
                return valid_hit

        elif self.method == ParticleModifyMethod.ADJACENCY:
            # Define the function for checking whether points are within the adjacency mesh
            def check_in_adjacency_mesh(particle_positions):
                # Define the AABB bounds
                lower, upper = self.obj.states[AABB].get_value() if self.link is None else self.link.aabb
                # Add the margin
                lower -= m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
                upper += m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
                return ((lower < particle_positions) & (particle_positions < upper)).all(axis=-1)
            self._check_in_mesh = check_in_adjacency_mesh

            # Define the function for checking overlaps at runtime
            def check_overlap():
                nonlocal valid_hit
                valid_hit = False
                aabb = self.obj.states[AABB].get_value() if self.link is None else self.link.aabb
                psqi().overlap_box(
                    halfExtent=(aabb[1] - aabb[0]) / 2.0 + m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN,
                    pos=(aabb[1] + aabb[0]) / 2.0,
                    rot=np.array([0, 0, 0, 1.0]),
                    reportFn=overlap_callback,
                )
                return valid_hit

        else:
            raise ValueError(f"Unsupported ParticleModifyMethod: {self.method}!")

        # Store check overlap function
        self._check_overlap = check_overlap

    @abstractmethod
    def _modify_particles(self, system):
        """
        Helper function to modify any particles belonging to @system.

        NOTE: This should handle both cases for @self.method:

            ParticleModifyMethod.ADJACENCY: modify any particles that are overlapping within the relaxed AABB
                defining adjacency to this object's modification link.
            ParticleModifyMethod.PROJECTION: modify any particles that are overlapping within the projection mesh.

        Must be implemented by subclass.

        Args:
            system (ParticleSystem): Particle system whose corresponding particles will be checked for modification
        """
        raise NotImplementedError()

    def _generate_limit_condition(self, system):
        """
        Generates a limit function condition for specific system @system

        Args:
             system (ParticleSystem): Particle system for which to generate a limit checker function

        Returns:
            function: Limit checker function, with signature condition(obj) --> bool, where @obj is the specific object
                that this ParticleModifier state belongs to
        """
        if issubclass(system, VisualParticleSystem):
            def condition(obj):
                return self.modified_particle_count[system] < self.visual_particle_modification_limit
        elif issubclass(system, PhysicalParticleSystem):
            def condition(obj):
                return self.modified_particle_count[system] < self.physical_particle_modification_limit
        else:
            self.unsupported_system_error(system=system)

        return condition

    def _update(self):
        # If we're using projection method and flatcache, we need to manually update this object's transforms on the USD
        # so the corresponding visualization and overlap meshes are updated properly
        if self.method == ParticleModifyMethod.PROJECTION and gm.ENABLE_FLATCACHE:
            FlatcacheAPI.sync_raw_object_transforms_in_usd(prim=self.obj)

        # Check if there's any overlap and if we're at the correct step
        if self._current_step == 0 and self._check_overlap():
            # Iterate over all owned systems for this particle modifier
            for system, conditions in self.conditions.items():
                # Check if all conditions are met
                if np.all([condition(self.obj) for condition in conditions]):
                    # Sanity check for oversaturation
                    self.check_at_limit(system=system, verify_not_over_limit=True)
                    # Potentially modify particles within the volume
                    self._modify_particles(system=system)

        # Update the current step
        self._current_step = (self._current_step + 1) % self.n_steps_per_modification

    def _set_value(self, new_value):
        raise ValueError(f"Cannot set valueless state {self.__class__.__name__}.")

    def _get_value(self):
        pass

    def remove(self):
        # We need to remove the generated particle system if we've created one
        if self.method == ParticleModifyMethod.PROJECTION:
            delete_prim(self.projection_system.GetPrimPath().pathString)

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [Covered, ToggledOn, ContactBodies, ContactParticles]

    def check_at_limit(self, system, verify_not_over_limit=False):
        """
        Checks whether this object is fully limited with particles modified from particle system @system. Also,
        potentially sanity checks whether the object is over the limit, if @verify_not_over_limit is True

        Args:
            system (ParticleSystem): System to check for particle limitations within this object
            verify_not_over_limit (bool): Whether to sanity check whether this object is over the limit with particles
                from @system

        Returns:
            bool: True if the object has reached its limit with objects from @system, otherwise False
        """
        if issubclass(system, VisualParticleSystem):
            limit = self.visual_particle_modification_limit
        elif issubclass(system, PhysicalParticleSystem):
            limit = self.physical_particle_modification_limit
        else:
            self.unsupported_system_error(system=system)

        # If requested, run sanity check to make sure we're not over the limit with this system's particles
        if verify_not_over_limit:
            assert self.modified_particle_count[system] <= limit, \
                f"{self.__class__.__name__} should not be over the limit! " \
                f"Max: {limit}, got: {self.modified_particle_count[system]}"

        return self.modified_particle_count[system] == limit

    def set_at_limit(self, system, value):
        """
        Sets whether this particle modifier is at its limit for system @system

        Args:
            system (ParticleSystem): System to set corresponding absorbed particle count limit level for
            value (bool): Whether to set the particle limit level to be at its limit or not
        """
        n_particles = 0
        if value:
            if issubclass(system, VisualParticleSystem):
                n_particles = self.visual_particle_modification_limit
            elif issubclass(system, PhysicalParticleSystem):
                n_particles = self.physical_particle_modification_limit
            else:
                self.unsupported_system_error(system=system)
        self.modified_particle_count[system] = n_particles

    @classmethod
    def unsupported_system_error(cls, system):
        """
        Raises a ValueError given unsupported system @system

        Args:
            system (ParticleSystem): Any unsupported system (any system that does not exist in @self.supported_systems)
        """
        raise ValueError(f"Invalid system for {cls.__name__}! Supported systems: "
                         f"{[sys.name for sys in cls.supported_systems]}, got: {system.name}")

    @classproperty
    def supported_systems(self):
        """
        Returns:
            list: All systems used in this state, ordered deterministically
        """
        return list(VisualParticleSystem.get_systems().values()) + list(PhysicalParticleSystem.get_systems().values())

    @property
    def n_steps_per_modification(self):
        """
        Returns:
            int: How many steps to take in between potentially modifying particles within the simulation
        """
        raise NotImplementedError()

    @property
    def visual_particle_modification_limit(self):
        """
        Returns:
            int: Maximum number of visual particles from a specific system that can be modified by this object
        """
        raise NotImplementedError()

    @property
    def physical_particle_modification_limit(self):
        """
        Returns:
            int: Maximum number of physical particles from a specific system that can be modified by this object
        """
        raise NotImplementedError()

    @property
    def state_size(self):
        # One entry per system plus the current_step
        return len(self.modified_particle_count) + 1

    def _dump_state(self):
        state = dict()
        for system, val in self.modified_particle_count.items():
            state[get_element_name_from_system(system)] = val
        # Add current step
        state["current_step"] = self._current_step
        return state

    def _load_state(self, state):
        for system in self.supported_systems:
            self.modified_particle_count[system] = state[get_element_name_from_system(system)]
        # Set current step
        self._current_step = state["current_step"]

    def _serialize(self, state):
        return np.array(list(state.values()), dtype=float)

    def _deserialize(self, state):
        state_dict = dict()
        for i, system in enumerate(self.modified_particle_count.keys()):
            state_dict[get_element_name_from_system(system)] = int(state[i])
        state_dict["current_step"] = int(state[len(self.modified_particle_count)])

        return state_dict, len(self.modified_particle_count) + 1


class ParticleRemover(ParticleModifier):
    """
    ParticleModifier where the modification results in potentially removing particles from the simulation.
    """

    def _modify_particles(self, system):
        # If at the limit, don't modify anything
        if self.check_at_limit(system=system):
            return
        # Check the system
        if issubclass(system, VisualParticleSystem):
            # Only modify particles if there are any that exist
            if system.n_particles > 0:
                # Iterate over all particles and remove any that are within the relaxed AABB of the remover volume
                particle_names = list(system.particles.keys())
                particle_positions = np.array([system.get_particle_position_orientation(name=name)[0] for name in system.particles.keys()])
                inbound_idxs = self._check_in_mesh(particle_positions).nonzero()[0]
                max_particle_absorbed = self.visual_particle_modification_limit - self.modified_particle_count[system]
                for idx in inbound_idxs[:max_particle_absorbed]:
                    system.remove_particle(particle_names[idx])
                self.modified_particle_count[system] += min(len(inbound_idxs), max_particle_absorbed)

        elif issubclass(system, PhysicalParticleSystem):
            instancer_to_particle_idxs = {}
            # If we're a cloth and using adjacency, we have to use check_in_mesh with the relaxed AABB since we
            # can't detect collisions via scene query interface. Alternatively, if we're using the projection method,
            # we also need to use check_in_mesh to check for overlap with the projection mesh
            # We'll check for if the physical particles are within this relaxed AABB
            if self.obj.prim_type == PrimType.CLOTH or self.method == ParticleModifyMethod.PROJECTION:
                for inst in system.particle_instancers.values():
                    inbound_idxs = self._check_in_mesh(inst.particle_positions).nonzero()[0]
                    instancer_to_particle_idxs[inst] = inbound_idxs
            # Otherwise, we can simply use the ContactParticle state to infer contacts
            else:
                instancer_to_particle_idxs = self.obj.states[ContactParticles].get_value(system, self.link)

            # Iterate over all particles and hide any that are detected to be removed
            for inst, particle_idxs in instancer_to_particle_idxs.items():
                # If at the limit, stop absorbing
                if self.check_at_limit(system=system):
                    break
                max_particle_absorbed = self.physical_particle_modification_limit - self.modified_particle_count[
                    system]
                particles_to_absorb = min(len(particle_idxs), max_particle_absorbed)
                particle_idxs_to_absorb = list(particle_idxs)[:particles_to_absorb]

                # Remove these particles from the instancer
                inst.remove_particles(idxs=particle_idxs_to_absorb)

                # Keep track of the particles that have been absorbed
                self.modified_particle_count[system] += particles_to_absorb

        else:
            # Invalid system queried
            self.unsupported_system_error(system=system)

    @staticmethod
    def get_state_link_name():
        return m.REMOVAL_LINK_NAME

    @property
    def n_steps_per_modification(self):
        return m.N_STEPS_PER_REMOVAL

    @property
    def visual_particle_modification_limit(self):
        return m.VISUAL_PARTICLES_REMOVAL_LIMIT

    @property
    def physical_particle_modification_limit(self):
        return m.PHYSICAL_PARTICLES_REMOVAL_LIMIT


class ParticleApplier(ParticleModifier):
    """
    ParticleModifier where the modification results in potentially adding particles into the simulation.
    """
    def __init__(self, obj, method, conditions, projection_mesh_params=None):
        # Store internal value
        self._sample_particle_locations = None

        # Run super
        super().__init__(obj=obj, method=method, conditions=conditions, projection_mesh_params=projection_mesh_params)

    def _initialize(self):
        # First, sanity check to make sure only one system is being applied, since unlike a ParticleRemover, which
        # can potentially remove multiple types of particles, a ParticleApplier should only apply one type of particle
        assert len(self.conditions) == 1, f"A ParticleApplier can only have a single ParticleSystem associated " \
                                          f"with it! Got: {[system.name for system in self.conditions.keys()]}"

        # Run super
        super()._initialize()

        # Store which method to use for sampling particle locations
        if self.method == ParticleModifyMethod.PROJECTION:
            self._sample_particle_locations = self._sample_particle_locations_from_projection_volume
        elif self.method == ParticleModifyMethod.ADJACENCY:
            self._sample_particle_locations = self._sample_particle_locations_from_adjacency_area
        else:
            raise ValueError(f"Unsupported ParticleModifyMethod: {self.method}!")

    def _modify_particles(self, system):
        # If at the limit, don't modify anything
        if self.check_at_limit(system=system):
            return

        # Sample potential locations to apply particles, and then apply them
        start_points, end_points = self._sample_particle_locations(system=system)
        n_samples = len(start_points)

        # Sample the rays to see where particle can be generated
        hits = [result for result in sample_cuboid_on_object(
            obj=None,
            start_points=start_points.reshape(n_samples, 1, 3),
            end_points=end_points.reshape(n_samples, 1, 3),
            cuboid_dimensions=system.sample_scales(
                group=system.get_group_name(obj=self.obj), n=len(start_points)) * system.particle_object.aabb_extent.reshape(1, 3)
            if issubclass(system, VisualParticleSystem) else np.zeros(3),
            ignore_objs=[self.obj],
            hit_proportion=0.0,             # We want all hits
            undo_cuboid_bottom_padding=issubclass(system, VisualParticleSystem),      # micro particles have zero cuboid dimensions so we need to maintain padding
            cuboid_bottom_padding=system.particle_radius if issubclass(system, PhysicalParticleSystem) else
            macros.utils.sampling_utils.DEFAULT_CUBOID_BOTTOM_PADDING,
        ) if result[0] is not None]

        self._apply_particles_at_raycast_hits(system=system, hits=hits)

    def _apply_particles_at_raycast_hits(self, system, hits):
        """
        Helper function to apply particles from system @system given raycast hits @hits,
        which are the filtered results from omnigibson.utils.sampling_utils.raytest_batch that include only
        the results with a valid hit

        Args:
            system (ParticleSystem): System to apply particles from
            hits (list of dict): Valid hit results from a batched raycast representing locations for sampling particles
        """
        # Check the system
        if issubclass(system, VisualParticleSystem):
            # Sample potential application points
            z_up = np.zeros(3)
            z_up[-1] = 1.0
            n_particles = min(len(hits), m.VISUAL_PARTICLES_APPLICATION_LIMIT - self.modified_particle_count[system])
            # Generate particle info -- maps group name to particle info for that group,
            # i.e.: positions, orientations, and link_prim_paths
            particles_info = defaultdict(lambda: defaultdict(lambda: []))
            for hit in hits[:n_particles]:
                # Infer which object was hit
                hit_obj = og.sim.scene.object_registry("prim_path", "/".join(hit[3].split("/")[:-1]), None)
                if hit_obj is not None:
                    # Create an attachment group if necessary
                    group = system.get_group_name(obj=hit_obj)
                    if group not in system.groups:
                        system.create_attachment_group(obj=hit_obj)
                    # Add to info
                    particles_info[group]["positions"].append(hit[0])
                    particles_info[group]["orientations"].append(hit[2])
                    particles_info[group]["link_prim_paths"].append(hit[3])
            # Generate all the particles for each group
            for group, particle_info in particles_info.items():
                # Generate particles for this group
                system.generate_group_particles(
                    group=group,
                    positions=np.array(particle_info["positions"]),
                    orientations=np.array(particle_info["orientations"]),
                    link_prim_paths=particle_info["link_prim_paths"],
                )
                # Update our particle count
                self.modified_particle_count[system] += len(particle_info["link_prim_paths"])

        elif issubclass(system, PhysicalParticleSystem):
            # Compile the particle poses to generate and sample the particles
            n_particles = min(len(hits), m.PHYSICAL_PARTICLES_APPLICATION_LIMIT - self.modified_particle_count[system])
            # Generate particles
            if n_particles > 0:
                system.default_particle_instancer.add_particles(positions=np.array([hit[0] for hit in hits[:n_particles]]))
                # Update our particle count
                self.modified_particle_count[system] += n_particles

        else:
            # Invalid system queried
            self.unsupported_system_error(system=system)

    def _sample_particle_locations_from_projection_volume(self, system):
        """
        Helper function for generating potential particle locations from projection volume

        Args:
            system (ParticleSystem): System to sample potential particle positions for

        Returns:
            2-tuple:
                - (n, 3) array: Ray start points to sample
                - (n, 3) array: Ray end points to sample
        """
        # Randomly sample end points from the base of the cone / cylinder
        n_samples = self._get_max_particles_limit_per_step(system=system)
        r, h = self._projection_mesh_params["extents"][0] / 2, self._projection_mesh_params["extents"][2]
        sampled_r_theta = np.random.rand(n_samples, 2)
        sampled_r_theta = sampled_r_theta * np.array([r, np.pi * 2]).reshape(1, 2)
        # Get start, end points in local link frame
        end_points = np.stack([
            h * np.ones(n_samples),
            sampled_r_theta[:, 0] * np.cos(sampled_r_theta[:, 1]),
            sampled_r_theta[:, 0] * np.sin(sampled_r_theta[:, 1]),
        ], axis=1)
        if self._projection_mesh_params["type"] == "Cone":
            # All start points are the cone tip, which is the local link origin
            start_points = np.zeros((n_samples, 3))
        elif self._projection_mesh_params["type"] == "Cylinder":
            # All start points are the parallel point for their corresponding end point
            # i.e.: (x, y, 0)
            start_points = end_points + np.array([-h, 0, 0]).reshape(1, 3)
        else:
            raise ValueError(f"Unsupported projection mesh type: {self._projection_mesh_params['type']}!")

        # Convert sampled normalized radius and angle into 3D points
        # We convert r, theta --> 3D point in local link frame --> 3D point in global world frame
        # We also combine start and end points for efficiency when doing the transform, then split them up again
        points = np.concatenate([start_points, end_points], axis=0)
        pos, quat = self.link.get_position_orientation()
        points = get_particle_positions_from_frame(
            pos=pos,
            quat=quat,
            scale=np.ones(3),
            particle_positions=points,
        )

        return points[:n_samples, :], points[n_samples:, :]

    def _sample_particle_locations_from_adjacency_area(self, system):
        """
        Helper function for generating potential particle locations from adjacency area

        Args:
            system (ParticleSystem): System to sample potential particle positions for

        Returns:
            2-tuple:
                - (n, 3) array: Ray start points to sample
                - (n, 3) array: Ray end points to sample
        """
        # Randomly sample end points from within the object's AABB
        n_samples = self._get_max_particles_limit_per_step(system=system)
        lower, upper = self.obj.states[AABB].get_value() if self.link is None else self.link.aabb
        lower = lower.reshape(1, 3) - m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
        upper = upper.reshape(1, 3) + m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
        lower_upper = np.concatenate([lower, upper], axis=0)

        # Sample in all directions, shooting from the center of the link / object frame
        pos = self.obj.get_position() if self.link is None else self.link.get_position()
        start_points = np.ones((n_samples, 3)) * pos.reshape(1, 3)
        end_points = np.random.uniform(low=lower, high=upper, size=(n_samples, 3))
        sides, axes = np.random.randint(2, size=(n_samples,)), np.random.randint(3, size=(n_samples,))
        end_points[np.arange(n_samples), axes] = lower_upper[sides, axes]

        return start_points, end_points

    def _get_max_particles_limit_per_step(self, system):
        """
        Helper function for grabbing the maximum particle limit per step

        Args:
            system (ParticleSystem): System for which to get max particle limit per step

        Returns:
            int: Maximum particles to apply per step for the given system @system
        """
        # Check the system
        if issubclass(system, VisualParticleSystem):
            val = m.MAX_VISUAL_PARTICLES_APPLIED_PER_STEP
        elif issubclass(system, PhysicalParticleSystem):
            val = m.MAX_PHYSICAL_PARTICLES_APPLIED_PER_STEP
        else:
            # Invalid system queried
            self.unsupported_system_error(system=system)
        return val

    @staticmethod
    def get_state_link_name():
        return m.APPLICATION_LINK_NAME

    @property
    def n_steps_per_modification(self):
        return m.N_STEPS_PER_APPLICATION

    @property
    def visual_particle_modification_limit(self):
        return m.VISUAL_PARTICLES_APPLICATION_LIMIT

    @property
    def physical_particle_modification_limit(self):
        return m.PHYSICAL_PARTICLES_APPLICATION_LIMIT
