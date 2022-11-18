from collections import namedtuple, Sized, OrderedDict
import numpy as np
from omnigibson.macros import gm, create_module_macros
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.covered import Covered
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.saturated import Saturated
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.systems.macro_particle_system import VisualParticleSystem, get_visual_particle_systems
from omnigibson.systems.micro_particle_system import FluidSystem, get_fluid_systems
from omnigibson.utils.constants import ParticleModifyMethod, PrimType
from omnigibson.utils.usd_utils import create_primitive_mesh
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omni.physx import get_physx_scene_query_interface as psqi
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysicsSchemaTools


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.LINK_NAME = "particle_remover_area"

# The margin (> 0) to add to the remover area's AABB when detecting overlaps with other objects
m.VISUAL_PARTICLE_REMOVER_AREA_MARGIN = 0.05


# Saturation thresholds -- maximum number of particles that can be removed ("absorbed") by this object
m.FLUID_PARTICLES_REMOVAL_LIMIT = 40
m.VISUAL_PARTICLES_REMOVAL_LIMIT = 40


class ParticleRemover(AbsoluteObjectState, LinkBasedStateMixin):
    """
    Object state representing an object that has the ability to remove visual and / or fluid particles from the active
    simulation.

    Args:
        obj (StatefulObject): Object to which this state will be applied
        method (ParticleModifyMethod): Method to modify particles. Current options supported are ADJACENCY (i.e.:
            "touching" particles) or PROJECTION (i.e.: "spraying" particles)
        conditions (dict): Dictionary mapping ParticleSystem to None or corresponding condition / list of conditions
            (where None represents no conditions) necessary in order for this particle remover to be able to
            remove particles belonging to @ParticleSystem. Each condition should be a function, whose signature
            is as follows:

                def condition() --> bool

            For a given ParticleSystem, if all of its conditions evaluate to True and particles are detected within
            this particle remover's remover area, then we remove those particles
        projection_mesh_params (None or dict): If specified and @method is ParticleModifyMethod.PROJECTION,
            manually overrides any metadata found from @obj.metadata to infer what projection volume to generate
            for this particle remover. Expected entries are as follows:

                "type": (str), one of {"Cylinder", "Cone", "Cube"}
                "extents": (3-array), the (x,y,z) extents of the generated volume

            If None, information found from @obj.metdata will be used instead.
    """
    def __init__(self, obj, method, conditions, projection_mesh_params=None):

        # Store internal variables
        self.method = method
        self.conditions = conditions
        self.projection_mesh = None
        self._check_in_projection_mesh = None
        self._check_overlap = None
        self._remove_particles = None
        self._link_prim_paths = set([link.prim_path for link in obj.links.values()])
        self._projection_mesh_params = obj.metadata["meta_links"][m.LINK_NAME] if \
            projection_mesh_params is None else projection_mesh_params

        # Map of system to number of absorbed particles for this object corresponding to the specific system
        self.absorbed_particle_count = OrderedDict([(system, 0) for system in self._supported_systems])

        # Standardize the conditions (make sure every system has at least one condition, which to make sure
        # the object isn't already saturated with the specific number of particles)
        for system, conds in conditions.items():
            # Make sure the system is supported
            assert system in self._supported_systems, \
                f"Invalid system for particle remover! Supported systems: " \
                f"{[sys.name for sys in self._supported_systems]}, got: {system.name}"
            # Make sure conds isn't empty
            if conds is None:
                conds = []
            # Make sure conds is a list
            if not isinstance(conds, Sized):
                conds = [conds]
            # Add the condition to avoid saturation
            conds.append(self._generate_saturation_condition(system=system))
            conditions[system] = conds

        # Run super method
        super().__init__(obj)

    @staticmethod
    def get_state_link_name():
        return m.LINK_NAME

    def _initialize(self):
        # Run link initialization
        self.initialize_link_mixin()

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
            mesh_prim_path = f"{self.link.prim_path}/projection_mesh"
            # Create a primitive mesh if it doesn't already exist
            if not get_prim_at_path(mesh_prim_path):
                mesh = create_primitive_mesh(
                    prim_path=mesh_prim_path,
                    primitive_type=self._projection_mesh_params["type"],
                    extents=self._projection_mesh_params["extents"],
                )

            # Create the visual geom instance referencing the generated mesh prim
            self.projection_mesh = VisualGeomPrim(prim_path=mesh_prim_path, name=f"{self.obj.name}_projection_mesh")
            self.projection_mesh.initialize()

            # Make sure the mesh isn't translated at all
            self.projection_mesh.set_local_pose(translation=np.zeros(3), orientation=np.array([0, 0, 0, 1.0]))

            # Set the removal method used
            self._remove_particles = self._remove_overlap_particles_in_projection_mesh

            # Generate the function for checking whether points are within the projection mesh
            self._check_in_projection_mesh, _ = generate_points_in_volume_checker_function(
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
                psqi().overlap_mesh(*projection_mesh_ids, reportFn=overlap_callback)
                return valid_hit

        elif self.method == ParticleModifyMethod.ADJACENCY:
            # Set the removal method used
            self._remove_particles = self._remove_overlap_particles_in_adjacency

            # Define the function for checking overlaps at runtime
            def check_overlap():
                nonlocal valid_hit
                valid_hit = False
                aabb = self.obj.states[AABB].get_value() if self.link is None else self.link.aabb
                psqi().overlap_box(
                    halfExtent=(aabb[1] - aabb[0]) / 2.0 + m.VISUAL_PARTICLE_REMOVER_AREA_MARGIN,
                    pos=(aabb[1] + aabb[0]) / 2.0,
                    rot=np.array([0, 0, 0, 1.0]),
                    reportFn=overlap_callback,
                )
                return valid_hit

        else:
            raise ValueError(f"Unsupported ParticleModifyMethod: {self.method}!")

        # Store check overlap function
        self._check_overlap = check_overlap

    def _remove_overlap_particles_in_adjacency(self, system):
        """
        Helper function to remove any particles belonging to @system that are overlapping within the relaxed AABB
        defining adajacency to this object's remover link.
        NOTE: This should only be used if @self.method = ParticleModifyMethod.ADJACENCY

        Args:
            system (ParticleSystem): Particle system whose corresponding particles will be checked for removal
        """
        # Define the AABB bounds
        lower, upper = self.obj.states[AABB].get_value() if self.link is None else self.link.aabb
        # Check the system
        if issubclass(system, VisualParticleSystem):
            # Margin is defined manually for visual particles
            lower -= m.VISUAL_PARTICLE_REMOVER_AREA_MARGIN
            upper += m.VISUAL_PARTICLE_REMOVER_AREA_MARGIN
            # Iterate over all particles and remove any that are within the relaxed AABB of the remover volume
            for particle_name in list(system.particles.keys()):
                # If saturated, stop absorbing
                if not self.conditions[system][-1]():
                    break
                particle = system.particles[particle_name]
                pos = particle.get_position()
                if BoundingBoxAPI.aabb_contains_point(pos, (lower, upper)):
                    system.remove_particle(particle_name)
                    self.absorbed_particle_count[system] += 1

        elif issubclass(system, FluidSystem):
            instancer_to_particle_idxs = {}
            # If we're a cloth, we have to use relaxed AABB since we can't detect collisions via scene query interface
            # We'll check for if the fluid particles are within this relaxed AABB
            if self.obj.prim_type == PrimType.CLOTH:
                # Margin is defined by particle radius
                lower -= system.particle_radius
                upper += system.particle_radius
                for inst in system.particle_instancers.values():
                    inbound = ((lower < inst.particle_positions) & (inst.particle_positions < upper))
                    inbound_idxs = inbound.all(axis=1).nonzero()[0]
                    instancer_to_particle_idxs[inst] = inbound_idxs
            # Otherwise, we can simply use the contact cached information for each particle
            else:
                instancer_to_particle_idxs = system.state_cache["obj_particle_contacts"][self.obj] if \
                    self.link is None else system.state_cache["link_particle_contacts"][self.link.prim_path]

            # Iterate over all particles and hide any that are detected to be removed
            for instancer, particle_idxs in instancer_to_particle_idxs.items():
                # If saturated, stop absorbing
                if not self.conditions[system][-1]():
                    break
                max_particle_absorbed = m.FLUID_PARTICLES_REMOVAL_LIMIT - self.absorbed_particle_count[system]
                particles_to_absorb = min(len(particle_idxs), max_particle_absorbed)
                particle_idxs_to_absorb = list(particle_idxs)[:particles_to_absorb]

                # Hide particles that have been absorbed
                inst.particle_visibilities[particle_idxs_to_absorb] = 0

                # Keep track of the particles that have been absorbed
                self.absorbed_particle_count[system] += particles_to_absorb

        else:
            # Invalid system queried
            self._unsupported_system_error(system=system)

    def _remove_overlap_particles_in_projection_mesh(self, system):
        """
        Helper function to remove any particles belonging to @system that are overlapping within the projection mesh.
        NOTE: This should only be used if @self.method = ParticleModifyMethod.PROJECTION

        Args:
            system (ParticleSystem): Particle system whose corresponding particles will be checked for removal
        """
        # Check the system
        if issubclass(system, VisualParticleSystem):
            # Iterate over all particles and remove any that are within the volume
            for particle_name in list(system.particles.keys()):
                # If saturated, stop absorbing
                if not self.conditions[system][-1]():
                    break
                particle = system.particles[particle_name]
                pos = particle.get_position()
                if self._check_in_projection_mesh(pos.reshape(1, 3))[0]:
                    system.remove_particle(particle_name)
                    self.absorbed_particle_count[system] += 1

        elif issubclass(system, FluidSystem):
            # Iterate over all particles and remove any that are within the volume
            for inst in system.particle_instancers.values():
                # If saturated, stop absorbing
                if not self.conditions[system][-1]():
                    break
                particle_idxs = np.where(self._check_in_projection_mesh(inst.particle_positions))
                max_particle_absorbed = m.FLUID_PARTICLES_REMOVAL_LIMIT - self.absorbed_particle_count[system]
                particles_to_absorb = min(len(particle_idxs), max_particle_absorbed)
                particle_idxs_to_absorb = particle_idxs[:particles_to_absorb]
                inst.particle_visibilities[particle_idxs_to_absorb] = 0

        else:
            # Invalid system queried
            self._unsupported_system_error(system=system)

    def _generate_saturation_condition(self, system):
        """
        Generates a saturation function condition for specific system @system

        Args:
             system (ParticleSystem): Particle system for which to generate a saturation checker function

        Returns:
            function: Saturation checker function, with signature condition() --> bool
        """
        if issubclass(system, VisualParticleSystem):
            def condition():
                return self.absorbed_particle_count[system] < m.VISUAL_PARTICLES_REMOVAL_LIMIT
        elif issubclass(system, FluidSystem):
            def condition():
                return self.absorbed_particle_count[system] < m.FLUID_PARTICLES_REMOVAL_LIMIT
        else:
            self._unsupported_system_error(system=system)

        return condition

    def _update(self):
        # Check if there's any overlap
        if self._check_overlap():
            # Iterate over all owned systems for this particle remover
            for system, conditions in self.conditions.items():
                # Check if all conditions are met
                if np.all([condition() for condition in conditions]):
                    # Potentially remove particles within the volume
                    self._remove_particles(system=system)

    def _set_value(self, new_value):
        raise ValueError(f"Cannot set valueless state ParticleRemover.")

    def _get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [Covered, Saturated, ToggledOn, ContactBodies]

    def _unsupported_system_error(self, system):
        """
        Raises a ValueError given unsupported system @system

        Args:
            system (ParticleSystem): Any unsupported system (any system that does not exist in @self._supported_systems)
        """
        raise ValueError(f"Invalid system for particle remover! Supported systems: "
                         f"{[sys.name for sys in self._supported_systems]}, got: {system.name}")

    @property
    def _supported_systems(self):
        """
        Returns:
            list: All systems used in this state, ordered deterministically
        """
        return list(get_visual_particle_systems().values()) + list(get_fluid_systems().values())
