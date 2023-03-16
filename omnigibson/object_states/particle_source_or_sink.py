import numpy as np

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.particle_modifier import ParticleApplier, ParticleRemover
from omnigibson.systems.micro_particle_system import PhysicalParticleSystem
from omnigibson.utils.constants import ParticleModifyMethod
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Metalink naming prefixes
# TODO: Update to particlesource / sink when the assets are updated
m.SOURCE_LINK_PREFIX = "fluidsource"
m.SINK_LINK_PREFIX = "fluidsink"

# Maximum number of particles that can be sourced / sunk per step
m.MAX_SOURCE_PARTICLES_PER_STEP = 1000
m.MAX_SINK_PARTICLES_PER_STEP = 1000

# How many steps between sinking particles
m.N_STEPS_PER_SINK = 5

# Upper limit to number of particles that can be sourced / sunk globally by a single object
m.SOURCE_PARTICLES_LIMIT = 1e6
m.SINK_PARTICLES_LIMIT = 1e6


class ParticleSource(ParticleApplier):
    """
        ParticleApplier where physical particles are spawned continuously in a cylindrical fashion from the
        metalink pose.

        Args:
            obj (StatefulObject): Object to which this state will be applied
            conditions (dict): Dictionary mapping ParticleSystem to None or corresponding condition / list of conditions
                (where None represents no conditions) necessary in order for this particle modifier to be able to
                modify particles belonging to @ParticleSystem. Each condition should be a function, whose signature
                is as follows:

                    def condition(obj) --> bool

                Where @obj is the specific object that this ParticleModifier state belongs to.
                For a given ParticleSystem, if all of its conditions evaluate to True and particles are detected within
                this particle modifier area, then we potentially modify those particles
            source_radius (float): Radius of the cylinder representing particles' spawning volume
            source_height (float): Height of the cylinder representing particles' spawning volume
            initial_speed (float): The initial speed for generated particles. Note that the
                direction of the velocity is inferred from the particle sampling process
            """
    def __init__(self, obj, conditions, source_radius, source_height, initial_speed=0.0):
        # Initialize variables that will be filled in at runtime
        self._n_steps_per_modification = None

        # Convert inputs into arguments to pass to particle applier class
        super().__init__(
            obj=obj,
            method=ParticleModifyMethod.PROJECTION,
            conditions=conditions,
            projection_mesh_params={
                "type": "Cylinder",
                "extents": [source_radius * 2, source_radius * 2, source_height],
                "visualize": False,
            },
            sample_with_raycast=False,
            initial_speed=initial_speed,
        )

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Calculate how many steps we need in between particle cluster spawnings
        # This is equivalent to the time it takes for a generated particle to travel @source_height distance
        # Note that object state steps are discretized by og.sim.render_step
        # Note: t derived from quadratic formula: height = 0.5 g t^2 + v0 t
        t = (-self._initial_speed + np.sqrt(self._initial_speed ** 2 + 2 * og.sim.gravity * self._projection_mesh_params["extents"][2])) / og.sim.gravity
        self._n_steps_per_modification = np.ceil(1 + t / og.sim.get_rendering_dt()).astype(int)

    def _get_max_particles_limit_per_step(self, system):
        # Check the system
        if issubclass(system, PhysicalParticleSystem):
            val = m.MAX_SOURCE_PARTICLES_PER_STEP
        else:
            # Invalid system queried
            self.unsupported_system_error(system=system)
        return val

    @classproperty
    def metalink_prefix(cls):
        return m.SOURCE_LINK_PREFIX

    @classproperty
    def supported_systems(self):
        return list(PhysicalParticleSystem.get_systems().values())

    @property
    def n_steps_per_modification(self):
        return self._n_steps_per_modification

    @property
    def physical_particle_modification_limit(self):
        return m.SOURCE_PARTICLES_LIMIT


class ParticleSink(ParticleRemover):
    """
        ParticleRemover where physical particles are removed continuously within a cylindrical volume located
        at the metalink pose.

        Args:
            obj (StatefulObject): Object to which this state will be applied
            conditions (dict): Dictionary mapping ParticleSystem to None or corresponding condition / list of conditions
                (where None represents no conditions) necessary in order for this particle modifier to be able to
                modify particles belonging to @ParticleSystem. Each condition should be a function, whose signature
                is as follows:

                    def condition(obj) --> bool

                Where @obj is the specific object that this ParticleModifier state belongs to.
                For a given ParticleSystem, if all of its conditions evaluate to True and particles are detected within
                this particle modifier area, then we potentially modify those particles
            sink_radius (float): Radius of the cylinder representing particles' sinking volume
            sink_height (float): Height of the cylinder representing particles' sinking volume
            """
    def __init__(self, obj, conditions, sink_radius, sink_height):
        # Initialize variables that will be filled in at runtime
        self._n_steps_per_modification = None

        # Convert inputs into arguments to pass to particle applier class
        super().__init__(
            obj=obj,
            method=ParticleModifyMethod.PROJECTION,
            conditions=conditions,
            # TODO: Discuss how this will sync with new asset metalinks
            projection_mesh_params={
                "type": "Cylinder",
                "extents": [sink_radius * 2, sink_radius * 2, sink_height],
                "visualize": False,
            },
        )

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Override check overlap such that it always returns True (since we are ignoring overlaps and directly
        # removing particles
        self._check_overlap = lambda: True

    def _get_max_particles_limit_per_step(self, system):
        # Check the system
        if issubclass(system, PhysicalParticleSystem):
            val = m.MAX_PHYSICAL_PARTICLES_SOURCED_PER_STEP
        else:
            # Invalid system queried
            self.unsupported_system_error(system=system)
        return val

    @classproperty
    def metalink_prefix(cls):
        return m.SINK_LINK_PREFIX

    @classproperty
    def supported_systems(self):
        return list(PhysicalParticleSystem.get_systems().values())

    @property
    def n_steps_per_modification(self):
        return m.N_STEPS_PER_SINK

    @property
    def physical_particle_modification_limit(self):
        return m.SINK_PARTICLES_LIMIT
