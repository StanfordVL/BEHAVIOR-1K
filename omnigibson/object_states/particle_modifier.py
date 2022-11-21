from abc import abstractmethod
from collections import namedtuple, Sized, OrderedDict, defaultdict
import numpy as np
import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.covered import Covered
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.systems.system_base import get_system_from_element_name, get_element_name_from_system
from omnigibson.systems.macro_particle_system import VisualParticleSystem, get_visual_particle_systems
from omnigibson.systems.micro_particle_system import FluidSystem, get_fluid_systems
from omnigibson.utils.constants import ParticleModifyMethod, PrimType
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function, get_particle_positions_in_frame
from omnigibson.utils.python_utils import assert_valid_key, classproperty
import omnigibson.utils.transform_utils as T
from omnigibson.utils.sampling_utils import raytest_batch
from omni.physx import get_physx_scene_query_interface as psqi
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysicsSchemaTools, UsdGeom


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.APPLICATION_LINK_NAME = "particle_application_area"
m.REMOVAL_LINK_NAME = "particle_remover_area"

# How many samples within the application area to generate per update step
m.MAX_VISUAL_PARTICLES_APPLIED_PER_STEP = 2
m.MAX_FLUID_PARTICLES_APPLIED_PER_STEP = 10

# How many steps between generating particle samples
m.N_STEPS_PER_APPLICATION = 5
m.N_STEPS_PER_REMOVAL = 1

# Saturation thresholds -- maximum number of particles that can be applied by a ParticleApplier
m.VISUAL_PARTICLES_APPLICATION_LIMIT = 1000000
m.FLUID_PARTICLES_APPLICATION_LIMIT = 1000000

# Saturation thresholds -- maximum number of particles that can be removed ("absorbed") by a ParticleRemover
m.VISUAL_PARTICLES_REMOVAL_LIMIT = 40
m.FLUID_PARTICLES_REMOVAL_LIMIT = 400

# The margin (> 0) to add to the remover area's AABB when detecting overlaps with other objects
m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN = 0.05


class ParticleModifier(AbsoluteObjectState, LinkBasedStateMixin):
    """
    Object state representing an object that has the ability to modify visual and / or fluid particles within the active
    simulation.

    Args:
        obj (StatefulObject): Object to which this state will be applied
        method (ParticleModifyMethod): Method to modify particles. Current options supported are ADJACENCY (i.e.:
            "touching" particles) or PROJECTION (i.e.: "spraying" particles)
        conditions (dict): Dictionary mapping ParticleSystem to None or corresponding condition / list of conditions
            (where None represents no conditions) necessary in order for this particle modifier to be able to
            modify particles belonging to @ParticleSystem. Each condition should be a function, whose signature
            is as follows:

                def condition() --> bool

            For a given ParticleSystem, if all of its conditions evaluate to True and particles are detected within
            this particle modifier area, then we potentially modify those particles
        projection_mesh_params (None or dict): If specified and @method is ParticleModifyMethod.PROJECTION,
            manually overrides any metadata found from @obj.metadata to infer what projection volume to generate
            for this particle modifier. Expected entries are as follows:

                "type": (str), one of {"Cylinder", "Cone", "Cube"}
                "extents": (3-array), the (x,y,z) extents of the generated volume

            If None, information found from @obj.metadata will be used instead.
    """
    def __init__(self, obj, method, conditions, projection_mesh_params=None):

        # Store internal variables
        self.method = method
        self.conditions = conditions
        self.projection_mesh = None
        self._check_in_projection_mesh = None
        self._check_overlap = None
        self._modify_particles = None
        self._link_prim_paths = None
        self._current_hit = None
        self._current_step = None
        self._projection_mesh_params = projection_mesh_params

        # Map of system to number of modified particles for this object corresponding to the specific system
        self.modified_particle_count = OrderedDict([(system, 0) for system in self.supported_systems])

        # Standardize the conditions (make sure every system has at least one condition, which to make sure
        # the particle modifier isn't already limited with the specific number of particles)
        for system, conds in conditions.items():
            # Make sure the system is supported
            assert_valid_key(key=system, valid_keys=self.supported_systems, name="particle system")
            # Make sure conds isn't empty
            if conds is None:
                conds = []
            # Make sure conds is a list
            if not isinstance(conds, Sized):
                conds = [conds]
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
        self._link_prim_paths = set([link.prim_path for link in self.obj.links.values()])

        # Define callback used during overlap method
        # We want to ignore any hits that are with this object itself
        valid_hit = False
        def overlap_callback(hit):
            nonlocal valid_hit
            valid_hit = hit.rigid_body not in self._link_prim_paths
            # Update current hit if we have a valid hit
            if valid_hit:
                self._current_hit = hit
            # Continue traversal only if we don't have a valid hit yet
            return not valid_hit

        # Possibly create a projection volume if we're using the projection method
        if self.method == ParticleModifyMethod.PROJECTION:
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
            # Create a primitive mesh if it doesn't already exist
            if not get_prim_at_path(mesh_prim_path):
                mesh = UsdGeom.__dict__[self._projection_mesh_params["type"]].Define(og.sim.stage, mesh_prim_path).GetPrim()
                # Set the height and radius
                # TODO: Generalize to objects other than cylinder and radius
                mesh.GetAttribute("height").Set(self._projection_mesh_params["extents"][2] / 2.0)
                mesh.GetAttribute("radius").Set(self._projection_mesh_params["extents"][0] / 4.0)

            # Create the visual geom instance referencing the generated mesh prim
            self.projection_mesh = VisualGeomPrim(prim_path=mesh_prim_path, name=f"{self.obj.name}_projection_mesh")
            self.projection_mesh.initialize()

            # Make sure the object updates its meshes
            self.link.update_meshes()

            # Make sure the mesh is translated so that its tip lies at the metalink origin
            self.projection_mesh.set_local_pose(
                translation=np.array([0, 0, -self._projection_mesh_params["extents"][2] / (2 * self.link.scale[2])]),
                orientation=np.array([0, 0, 0, 1.0]),
            )

            # Set the modification method used
            self._modify_particles = self._modify_overlap_particles_in_projection_mesh

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
                psqi().overlap_shape(*projection_mesh_ids, reportFn=overlap_callback)
                return valid_hit

        elif self.method == ParticleModifyMethod.ADJACENCY:
            # Set the modification method used
            self._modify_particles = self._modify_overlap_particles_in_adjacency

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
    def _modify_overlap_particles_in_adjacency(self, system):
        """
        Helper function to modify any particles belonging to @system that are overlapping within the relaxed AABB
        defining adjacency to this object's modification link.
        NOTE: This should only be used if @self.method = ParticleModifyMethod.ADJACENCY

        Must be implemented by subclass.

        Args:
            system (ParticleSystem): Particle system whose corresponding particles will be checked for modification
        """
        raise NotImplementedError()

    @abstractmethod
    def _modify_overlap_particles_in_projection_mesh(self, system):
        """
        Helper function to modify any particles belonging to @system that are overlapping within the projection mesh.
        NOTE: This should only be used if @self.method = ParticleModifyMethod.PROJECTION

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
            function: Limit checker function, with signature condition() --> bool
        """
        if issubclass(system, VisualParticleSystem):
            def condition():
                return self.modified_particle_count[system] < self.visual_particle_modification_limit
        elif issubclass(system, FluidSystem):
            def condition():
                return self.modified_particle_count[system] < self.fluid_particle_modification_limit
        else:
            self.unsupported_system_error(system=system)

        return condition

    def _update(self):
        # Check if there's any overlap and if we're at the correct step
        if self._current_step == 0 and self._check_overlap():
            # Iterate over all owned systems for this particle modifier
            for system, conditions in self.conditions.items():
                # Check if all conditions are met
                if issubclass(system, VisualParticleSystem):
                    print(f"{system.name} limited: {self.check_at_limit(system=system)}")
                if np.all([condition() for condition in conditions]):
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

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [Covered, ToggledOn, ContactBodies]

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
        elif issubclass(system, FluidSystem):
            limit = self.fluid_particle_modification_limit
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
            elif issubclass(system, FluidSystem):
                n_particles = self.fluid_particle_modification_limit
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
        return list(get_visual_particle_systems().values()) + list(get_fluid_systems().values())

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
    def fluid_particle_modification_limit(self):
        """
        Returns:
            int: Maximum number of fluid particles from a specific system that can be modified by this object
        """
        raise NotImplementedError()

    @property
    def stateful(self):
        return True

    @property
    def state_size(self):
        return len(self.modified_particle_count)

    def _dump_state(self):
        state = OrderedDict()
        for system, val in self.modified_particle_count.items():
            state[get_element_name_from_system(system)] = val
        return state

    def _load_state(self, state):
        for system_name, val in state.items():
            self.modified_particle_count[get_system_from_element_name(system_name)] = val

    def _serialize(self, state):
        return np.array(list(state.values()), dtype=float)

    def _deserialize(self, state):
        state_dict = OrderedDict()
        for i, system in enumerate(self.modified_particle_count.keys()):
            state_dict[get_element_name_from_system(system)] = int(state[i])

        return state_dict, len(self.modified_particle_count)


class ParticleRemover(ParticleModifier):
    """
    ParticleModifier where the modification results in potentially removing particles from the simulation.
    """

    def _modify_overlap_particles_in_adjacency(self, system):
        # Define the AABB bounds
        lower, upper = self.obj.states[AABB].get_value() if self.link is None else self.link.aabb
        # Check the system
        if issubclass(system, VisualParticleSystem):
            # Margin is defined manually for visual particles
            lower -= m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
            upper += m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
            # Iterate over all particles and remove any that are within the relaxed AABB of the remover volume
            for particle_name in list(system.particles.keys()):
                # If at the limit, stop absorbing
                if self.check_at_limit(system=system):
                    break
                particle = system.particles[particle_name]
                pos = particle.get_position()
                if BoundingBoxAPI.aabb_contains_point(pos, (lower, upper)):
                    system.remove_particle(particle_name)
                    self.modified_particle_count[system] += 1

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
            for inst, particle_idxs in instancer_to_particle_idxs.items():
                # If at the limit, stop absorbing
                if self.check_at_limit(system=system):
                    break
                max_particle_absorbed = m.FLUID_PARTICLES_REMOVAL_LIMIT - self.modified_particle_count[system]
                particles_to_absorb = min(len(particle_idxs), max_particle_absorbed)
                particle_idxs_to_absorb = list(particle_idxs)[:particles_to_absorb]

                # Hide particles that have been absorbed
                visibilities = inst.particle_visibilities
                visibilities[particle_idxs_to_absorb] = 0
                inst.particle_visibilities = visibilities

                # Keep track of the particles that have been absorbed
                self.modified_particle_count[system] += particles_to_absorb

        else:
            # Invalid system queried
            self.unsupported_system_error(system=system)

    def _modify_overlap_particles_in_projection_mesh(self, system):
        # Check the system
        if issubclass(system, VisualParticleSystem):
            # Iterate over all particles and remove any that are within the volume
            for particle_name in list(system.particles.keys()):
                # If at the limit, stop absorbing
                if self.check_at_limit(system=system):
                    break
                particle = system.particles[particle_name]
                pos = particle.get_position()
                if self._check_in_projection_mesh(pos.reshape(1, 3))[0]:
                    system.remove_particle(particle_name)
                    self.modified_particle_count[system] += 1

        elif issubclass(system, FluidSystem):
            # Iterate over all particles and remove any that are within the volume
            for inst in system.particle_instancers.values():
                # If at the limit, stop absorbing
                if self.check_at_limit(system=system):
                    break
                particle_idxs = np.where(self._check_in_projection_mesh(inst.particle_positions))
                max_particle_absorbed = m.FLUID_PARTICLES_REMOVAL_LIMIT - self.modified_particle_count[system]
                particles_to_absorb = min(len(particle_idxs), max_particle_absorbed)
                particle_idxs_to_absorb = particle_idxs[:particles_to_absorb]
                visibilities = inst.particle_visibilities
                visibilities[particle_idxs_to_absorb] = 0
                inst.particle_visibilities = visibilities

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
    def fluid_particle_modification_limit(self):
        return m.FLUID_PARTICLES_REMOVAL_LIMIT


class ParticleApplier(ParticleModifier):
    """
    ParticleModifier where the modification results in potentially adding particles into the simulation.
    """

    def _modify_overlap_particles_in_adjacency(self, system):
        # Sample potential locations to apply particles
        hits = self._sample_particle_locations_from_adjacency_area(system=system)
        self._apply_particles_at_raycast_hits(system=system, hits=hits)

    def _modify_overlap_particles_in_projection_mesh(self, system):
        # Sample potential locations to apply particles
        hits = self._sample_particle_locations_from_projection_volume(system=system)
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
        # Check how many particles we can sample
        print(f"n hits: {len(hits)}")
        # Check the system
        if issubclass(system, VisualParticleSystem):
            # Sample potential application points
            z_up = np.zeros(3)
            z_up[-1] = 1.0
            n_particles = min(len(hits), m.VISUAL_PARTICLES_APPLICATION_LIMIT - self.modified_particle_count[system])
            for hit in hits[:n_particles]:
                # Infer which object was hit
                hit_obj = og.sim.scene.object_registry("prim_path", "/".join(hit["rigidBody"].split("/")[:-1]), None)
                print(f"hit obj: {hit_obj}")
                if hit_obj is not None:
                    # Create an attachment group if necessary
                    group = system.get_group_name(obj=hit_obj)
                    if group not in system.groups:
                        system.create_attachment_group(obj=hit_obj)
                    # Generate a particle for this group
                    system.create_group_particles(
                        group=group,
                        positions=hit["position"].reshape(1, 3),
                        orientations=T.axisangle2quat(T.vecs2axisangle(z_up, hit["normal"])).reshape(1, 4),
                        link_prim_paths=[hit["rigidBody"]],
                    )
                    # Update our particle count
                    self.modified_particle_count[system] += 1

        elif issubclass(system, FluidSystem):
            # Compile the particle poses to generate
            positions = []
            n_particles = min(len(hits), m.FLUID_PARTICLES_APPLICATION_LIMIT - self.modified_particle_count[system])
            # Only generate particles if we have a positive number of requested particles
            if n_particles > 0:
                for hit in hits[:n_particles]:
                    # Calculate sampled particle position in world space, which is the hit position offset in the normal
                    # direction by particle radius distance
                    positions.append(hit["position"] + hit["normal"] * system.particle_radius)
                # Generate particle instancer
                system.generate_particle_instancer(
                    n_particles=n_particles,
                    positions=np.array(positions),
                )
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
            list of dict: Successful particle hit information resulting from
                omnigibson.utils.sampling_utils.raytest_batch
        """
        # Randomly sample end points from the base of the cone / cylinder
        n_samples = self._get_max_particles_limit_per_step(system=system)
        r, h = self._projection_mesh_params["extents"][0] / 2, self._projection_mesh_params["extents"][2]
        sampled_r_theta = np.random.rand(n_samples, 2)
        sampled_r_theta = sampled_r_theta * np.array([r, np.pi * 2]).reshape(1, 2)
        # Get start, end points in local link frame
        end_points = np.stack([
            sampled_r_theta[:, 0] * np.cos(sampled_r_theta[:, 1]),
            sampled_r_theta[:, 0] * np.cos(sampled_r_theta[:, 1]),
            -h * np.ones(n_samples),
        ], axis=1)
        if self._projection_mesh_params["type"] == "Cone":
            # All start points are the cone tip, which is the local link origin
            start_points = np.zeros((n_samples, 3))
        elif self._projection_mesh_params["type"] == "Cylinder":
            # All start points are the parallel point for their corresponding end point
            # i.e.: (x, y, 0)
            start_points = end_points + np.array([0, 0, h]).reshape(1, 3)
        else:
            raise ValueError(f"Unsupported projection mesh type: {self._projection_mesh_params['type']}!")

        # Convert sampled normalized radius and angle into 3D points
        # We convert r, theta --> 3D point in local link frame --> 3D point in global world frame
        # We also combine start and end points for efficiency when doing the transform, then split them up again
        points = np.concatenate([start_points, end_points], axis=0)
        pos, quat = self.link.get_position_orientation()
        points = get_particle_positions_in_frame(
            pos=-pos,
            quat=T.quat_inverse(quat),
            scale=np.ones(3),
            particle_positions=points,
        )

        # Run the batched raytest and return results
        return [result for result in raytest_batch(
            start_points=points[:n_samples, :],
            end_points=points[n_samples:, :],
            ignore_bodies=self._link_prim_paths,
        ) if result["hit"]]

    def _sample_particle_locations_from_adjacency_area(self, system):
        """
        Helper function for generating potential particle locations from adjacency area

        Args:
            system (ParticleSystem): System to sample potential particle positions for

        Returns:
            list of dict: Successful particle hit information resulting from
                omnigibson.utils.sampling_utils.raytest_batch
        """
        # Randomly sample end points from within the object's AABB
        n_samples = self._get_max_particles_limit_per_step(system=system)
        lower, upper = self.obj.states[AABB].get_value() if self.link is None else self.link.aabb
        lower = lower.reshape(1, 3) - m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
        upper = upper.reshape(1, 3) + m.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
        start_points = lower + (upper - lower) * np.random.rand(n_samples, 3)
        # Get the direction of the contact, by inferring which object was hit and then getting the vector pointing
        # in that object link's relative direction
        pos = self.obj.get_position() if self.link is None else self.link.get_position()
        # Infer which object was hit
        obj_prim_path = "/".join(self._current_hit.rigid_body.split("/")[:-1])
        obj_link_name = self._current_hit.rigid_body.split("/")[-1]
        hit_obj = og.sim.scene.object_registry("prim_path", obj_prim_path, None)
        # Break early if we have an invalid hit obj
        if hit_obj is None:
            return []
        direction = hit_obj.links[obj_link_name].get_position() - pos
        # All the end points will the be the sampled start points pointing in @direction, and then clipped to be within
        # the relaxed bounding box
        end_points = (start_points + direction.reshape(1, 3)).clip(lower, upper)

        # Run the batched raytest and return results

        return [result for result in raytest_batch(
            start_points=start_points,
            end_points=end_points,
            ignore_bodies=self._link_prim_paths,
        ) if result["hit"]]

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
        elif issubclass(system, FluidSystem):
            val = m.MAX_FLUID_PARTICLES_APPLIED_PER_STEP
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
    def fluid_particle_modification_limit(self):
        return m.FLUID_PARTICLES_APPLICATION_LIMIT
