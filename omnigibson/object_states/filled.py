import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.contains import ContainedParticles
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.systems.system_base import PhysicalParticleSystem, is_physical_particle_system

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Proportion of object's volume that must be filled for object to be considered filled
m.VOLUME_FILL_PROPORTION = 0.3


class Filled(RelativeObjectState, BooleanState):

    def _get_value(self, system):
        # Sanity check to make sure system is valid
        assert is_physical_particle_system(system_name=system.name), \
            "Can only get Filled state with a valid PhysicalParticleSystem!"

        # Check what volume is filled
        if len(system.particle_instancers) > 0:
            particle_volume = 4 / 3 * np.pi * (system.particle_radius ** 3)
            n_particles = self.obj.states[ContainedParticles].get_value(system).n_in_volume
            prop_filled = particle_volume * n_particles / self.obj.states[ContainedParticles].volume
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
        assert is_physical_particle_system(system_name=system.name), \
            "Can only set Filled state with a valid PhysicalParticleSystem!"

        # First, check our current state
        current_state = self.get_value(system)

        # Only do something if we're changing state
        if current_state != new_value:
            contained_particles_state = self.obj.states[ContainedParticles]
            if new_value:
                # Going from False --> True, sample volume with particles
                system.generate_particles_from_link(
                    obj=self.obj,
                    link=contained_particles_state.link,
                    mesh_name_prefixes="container",
                    check_contact=True,
                )
            else:
                # Going from True --> False, remove all particles inside the volume
                system.remove_particles(idxs=contained_particles_state.get_value().in_volume.nonzero()[0])

        return True

    @staticmethod
    def get_dependencies():
        return [ContainedParticles]

    @staticmethod
    def get_optional_dependencies():
        return []
