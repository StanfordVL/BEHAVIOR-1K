import numpy as np
from collections import namedtuple
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.systems.macro_particle_system import VisualParticleSystem
from omnigibson.systems.micro_particle_system import MicroPhysicalParticleSystem
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty
import omnigibson.utils.transform_utils as T

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.CONTAINER_LINK_PREFIX = "container"
m.VISUAL_PARTICLE_OFFSET = 0.01    # Offset to visual particles' poses when checking overlaps with container volume


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
        self._visual_particle_group = None  # Name corresponding to this object's set of visual particles

    @classproperty
    def metalink_prefix(cls):
        return m.CONTAINER_LINK_PREFIX

    def _get_value(self, system):
        """
        Args:
            system (MicroPhysicalParticleSystem): System whose number of particles will be checked inside this object's
                container volume

        Returns:
            ContainedParticlesData: namedtuple with the following keys:
                - n_in_volume (int): Number of @system's particles inside this object's container volume
                - positions (np.array): (N, 3) Particle positions of all @system's particles
                - in_volume (np.array): (N,) boolean array, True if the corresponding particle is inside this
                    object's container volume, else False
        """
        # Value is false by default
        n_particles_in_volume, raw_positions, checked_positions, particles_in_volume = 0, np.array([]), np.array([]), np.array([])

        # First, we check what type of system
        # Currently, we support VisualParticleSystems and MicroPhysicalParticleSystems
        if issubclass(system, VisualParticleSystem):
            if self._visual_particle_group in system.groups:
                # Grab global particle poses and offset them in the direction of their orientation
                raw_positions, quats = system.get_particles_position_orientation(group=self._visual_particle_group)
                unit_z = np.zeros((len(raw_positions), 3, 1))
                unit_z[:, -1, :] = m.VISUAL_PARTICLE_OFFSET
                checked_positions = (T.quat2mat(quats) @ unit_z).reshape(-1, 3) + raw_positions
        elif issubclass(system, MicroPhysicalParticleSystem):
            # We only check if we have particle instancers currently
            if len(system.particle_instancers) > 0:
                raw_positions = np.concatenate([inst.particle_positions for inst in system.particle_instancers.values()], axis=0)
                checked_positions = raw_positions
        else:
            raise ValueError(f"Invalid system {system} received for getting Covered state!"
                             f"Currently, only VisualParticleSystems and MicroPhysicalParticleSystems are supported.")

        # Only calculate if we have valid positions
        if len(checked_positions) > 0:
            particles_in_volume = self.check_in_volume(checked_positions)
            n_particles_in_volume = particles_in_volume.sum()

        return ContainedParticlesData(n_particles_in_volume, raw_positions, particles_in_volume)

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

        # Grab group name
        self._visual_particle_group = VisualParticleSystem.get_group_name(obj=self.obj)

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
