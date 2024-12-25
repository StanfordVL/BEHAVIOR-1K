from collections import namedtuple

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.CONTAINER_LINK_PREFIX = "container"
m.VISUAL_PARTICLE_OFFSET = 0.01  # Offset to visual particles' poses when checking overlaps with container volume


"""
ContainedParticlesData contains the following fields:
    n_in_volume (int): number of particles in the container volume
    positions (th.tensor): (N, 3) array representing the raw global particle positions
    in_volume (th.tensor): (N,) boolean array representing whether each particle is inside the container volume or not
"""
ContainedParticlesData = namedtuple("ContainedParticlesData", ("n_in_volume", "positions", "in_volume"))


class ContainedParticles(RelativeObjectState, LinkBasedStateMixin):
    """
    Object state for computing the number of particles of a given system contained in this object's container volume
    """

    def __init__(self, obj):
        super().__init__(obj)
        self.check_in_volume = None  # Function to check whether particles are in volume for this container
        self._volume = None  # Volume of this container
        self._compute_info = None  # Intermediate computation information to store

    @classproperty
    def metalink_prefix(cls):
        return m.CONTAINER_LINK_PREFIX

    def _get_value(self, system):
        """
        Args:
            system (VisualParticleSystem or PhysicalParticleSystem): System whose number of particles will be checked inside this object's
                container volume

        Returns:
            ContainedParticlesData: namedtuple with the following keys:
                - n_in_volume (int): Number of @system's particles inside this object's container volume
                - positions (th.tensor): (N, 3) Particle positions of all @system's particles
                - in_volume (th.tensor): (N,) boolean array, True if the corresponding particle is inside this
                    object's container volume, else False
        """
        # Value is false by default
        n_particles_in_volume, raw_positions, checked_positions, particles_in_volume = (
            0,
            th.empty(0),
            th.empty(0),
            th.empty(0),
        )

        # Only run additional computations if there are any particles
        if system.n_particles > 0:
            # First, we check what type of system
            # Currently, we support VisualParticleSystems and PhysicalParticleSystems
            if self.obj.scene.is_visual_particle_system(system_name=system.name):
                # Grab global particle poses and offset them in the direction of their orientation
                raw_positions, quats = system.get_particles_position_orientation()
                unit_z = th.zeros((len(raw_positions), 3, 1))
                unit_z[:, -1, :] = m.VISUAL_PARTICLE_OFFSET
                checked_positions = (T.quat2mat(quats) @ unit_z).reshape(-1, 3) + raw_positions
            elif self.obj.scene.is_physical_particle_system(system_name=system.name):
                raw_positions = system.get_particles_position_orientation()[0]
                checked_positions = raw_positions
            else:
                raise ValueError(
                    f"Invalid system {system} received for getting ContainedParticles state!"
                    f"Currently, only VisualParticleSystems and PhysicalParticleSystems are supported."
                )

        # Only calculate if we have valid positions
        if len(checked_positions) > 0:
            particles_in_volume = self.check_in_volume(checked_positions)
            n_particles_in_volume = particles_in_volume.sum()

        return ContainedParticlesData(n_particles_in_volume, raw_positions, particles_in_volume)

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

        # Generate volume checker function for this object
        self.check_in_volume, calculate_volume = generate_points_in_volume_checker_function(
            obj=self.obj, volume_link=self.link
        )

        # Calculate volume
        self._volume = calculate_volume()

    @property
    def volume(self):
        """
        Returns:
            float: Total volume for this container
        """
        return self._volume


class Contains(RelativeObjectState, BooleanStateMixin):
    def _get_value(self, system):
        # Grab value from Contains state; True if value is greater than 0
        return self.obj.states[ContainedParticles].get_value(system=system).n_in_volume > 0

    def _set_value(self, system, new_value):
        if new_value:
            # Cannot set contains = True, only False
            raise NotImplementedError(f"{self.__class__.__name__} does not support set_value(system, True)")
        else:
            # Remove all particles from inside the volume
            system.remove_particles(idxs=self.obj.states[ContainedParticles].get_value(system).in_volume.nonzero())

        return True

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ContainedParticles)
        return deps
