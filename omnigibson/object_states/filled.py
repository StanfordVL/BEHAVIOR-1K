import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.contains import ContainedParticles
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanStateMixin
from omnigibson.systems.system_base import PhysicalParticleSystem, is_physical_particle_system

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Proportion of object's volume that must be filled for object to be considered filled
m.VOLUME_FILL_PROPORTION = 0.2


class Filled(RelativeObjectState, BooleanStateMixin):

    def _get_value(self, system):
        # Sanity check to make sure system is valid
        assert is_physical_particle_system(system_name=system.name), \
            "Can only get Filled state with a valid PhysicalParticleSystem!"

        # Check what volume is filled
        if system.n_particles > 0:
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
        # Sanity check to make sure system is valid
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
                    check_contact=True,
                )
            else:
                # Cannot set False
                raise NotImplementedError(f"{self.__class__.__name__} does not support set_value(system, False)")

        return True

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ContainedParticles)
        return deps
