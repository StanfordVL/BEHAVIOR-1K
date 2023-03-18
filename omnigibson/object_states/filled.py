import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.systems.micro_particle_system import PhysicalParticleSystem
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Proportion of object's volume that must be filled for object to be considered filled
m.VOLUME_FILL_PROPORTION = 0.3

m.FILLED_LINK_PREFIX = "container"


class Filled(RelativeObjectState, BooleanState, LinkBasedStateMixin):
    def __init__(self, obj):
        super().__init__(obj)
        self.check_in_volume = None        # Function to check whether particles are in volume for this container
        self.calculate_volume = None       # Function to calculate the real-world volume for this container

    @classproperty
    def metalink_prefix(cls):
        return m.FILLED_LINK_PREFIX

    def _get_value(self, system):
        # Sanity check to make sure system is valid
        assert issubclass(system, PhysicalParticleSystem), "Can only get Filled state with a valid PhysicalParticleSystem!"
        # Check what volume is filled
        if len(system.particle_instancers) > 0:
            particle_positions = np.concatenate([inst.particle_positions for inst in system.particle_instancers.values()], axis=0)
            particles_in_volume = self.check_in_volume(particle_positions)
            particle_volume = 4 / 3 * np.pi * (system.particle_radius ** 3)
            prop_filled = particle_volume * particles_in_volume.sum() / self.calculate_volume()
            # If greater than threshold, then the volume is filled
            # Explicit bool cast needed here because the type is bool_ instead of bool which is not JSON-Serializable
            # This has to do with numpy, see https://stackoverflow.com/questions/58408054/typeerror-object-of-type-bool-is-not-json-serializable
            value = bool(prop_filled > m.VOLUME_FILL_PROPORTION)
        else:
            # No particles exists, so we're obviously empty
            value = False

        return value

    def _set_value(self, system, new_value):
        # Sanity check to manke sure system is valid
        assert issubclass(system, PhysicalParticleSystem), \
            "Can only set Filled state with a valid PhysicalParticleSystem!"

        # First, check our current state
        current_state = self.get_value(system)

        # Only do something if we're changing state
        if current_state != new_value:
            if new_value:
                # Going from False --> True, sample volume with particles
                system.generate_particles_from_link(
                    obj=self.obj,
                    link=self.link,
                    mesh_name_prefixes="container",
                )
            else:
                # Going from True --> False, delete all particles inside the volume
                for inst in system.particle_instancers.values():
                    inst.remove_particles(self.check_in_volume(inst.particle_positions).nonzero()[0])

        return True

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

        # Generate volume checker function for this object
        self.check_in_volume, self.calculate_volume = \
            generate_points_in_volume_checker_function(obj=self.obj, volume_link=self.link, mesh_name_prefixes="container")

    @staticmethod
    def get_optional_dependencies():
        return []
