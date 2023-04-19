import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.systems.micro_particle_system import PhysicalParticleSystem
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.CONTAINER_LINK_PREFIX = "container"


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
            int: Number of @system's particles inside this object's container volume
        """
        # Sanity check to make sure system is valid
        assert issubclass(system, PhysicalParticleSystem), "Can only get Contains state with a valid PhysicalParticleSystem!"
        # Check how many particles are included
        value = 0
        self._compute_info = dict(positions=np.array([]), in_volume=np.array([]))
        if len(system.particle_instancers) > 0:
            particle_positions = np.concatenate([inst.particle_positions for inst in system.particle_instancers.values()], axis=0)
            particles_in_volume = self.check_in_volume(particle_positions)
            value = particles_in_volume.sum()

            # Also store compute info
            self._compute_info["positions"] = particle_positions
            self._compute_info["in_volume"] = particles_in_volume

        return value

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

    def cache_info(self, get_value_args):
        # Call super first
        info = super().cache_info(get_value_args=get_value_args)
        info.update(self._compute_info)

        return info

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
        return self.obj.states[ContainedParticles].get_value(system=system) > 0

    def _set_value(self, system, new_value):
        # Cannot set this value
        raise ValueError("set_value not supported for Contains state.")

    @staticmethod
    def get_dependencies():
        return [ContainedParticles]
