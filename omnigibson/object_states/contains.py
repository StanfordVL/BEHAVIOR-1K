import numpy as np
from collections import namedtuple
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.systems.micro_particle_system import PhysicalParticleSystem
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.CONTAINER_LINK_PREFIX = "container"


"""
ContainedParticlesData contains the following fields:
    n_in_volume (int): number of particles in the container volume
    positions (np.array): (N, 3) array representing the raw global particle positions
    in_volume (np.array): (N,) boolean array representing whether each particle is inside the container volume or not
"""
ContainedParticlesData = namedtuple("ContainedParticlesData", ("n_in_volume", "positions", "in_volume"))


class ContainedParticles(RelativeObjectState, LinkBasedStateMixin):
    """
    Object state for computing the number of particles of a given system contained in this object's container volume
    """
    def __init__(self, obj):
        super().__init__(obj)
        self.check_in_volume = None         # Function to check whether particles are in volume for this container
        self._volume = None                 # Volume of this container
        self._compute_info = None           # Intermediate computation information to store

    @classproperty
    def metalink_prefix(cls):
        return m.CONTAINER_LINK_PREFIX

    def _get_value(self, system):
        """
        Args:
            system (PhysicalParticleSystem): System whose number of particles will be checked inside this object's
                container volume

        Returns:
            ContainedParticlesData: namedtuple with the following keys:
                - n_in_volume (int): Number of @system's particles inside this object's container volume
                - positions (np.array): (N, 3) Particle positions of all @system's particles
                - in_volume (np.array): (N,) boolean array, True if the corresponding particle is inside this
                    object's container volume, else False
        """
        # Sanity check to make sure system is valid
        assert issubclass(system, PhysicalParticleSystem), "Can only get Contains state with a valid PhysicalParticleSystem!"
        # Check how many particles are included
        n_particles_in_volume, particle_positions, particles_in_volume = 0, np.array([]), np.array([])
        if len(system.particle_instancers) > 0:
            particle_positions = np.concatenate([inst.particle_positions for inst in system.particle_instancers.values()], axis=0)
            particles_in_volume = self.check_in_volume(particle_positions)
            n_particles_in_volume = particles_in_volume.sum()

        return ContainedParticlesData(n_particles_in_volume, particle_positions, particles_in_volume)

    def _set_value(self, system, new_value):
        # Cannot set this value
        raise ValueError("set_value not supported for ContainedParticles state.")

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

        # Generate volume checker function for this object
        self.check_in_volume, calculate_volume = \
            generate_points_in_volume_checker_function(obj=self.obj, volume_link=self.link, mesh_name_prefixes="container")

        # Calculate volume
        self._volume = calculate_volume()

    @property
    def volume(self):
        """
        Returns:
            float: Total volume for this container
        """
        return self._volume

    @staticmethod
    def get_optional_dependencies():
        return []


class Contains(RelativeObjectState, BooleanState):
    def _get_value(self, system):
        # Grab value from Contains state; True if value is greater than 0
        return self.obj.states[ContainedParticles].get_value(system=system).n_in_volume > 0

    def _set_value(self, system, new_value):
        # Cannot set this value
        raise ValueError("set_value not supported for Contains state.")

    @staticmethod
    def get_dependencies():
        return [ContainedParticles]
