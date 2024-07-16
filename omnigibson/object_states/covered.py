from omnigibson.macros import create_module_macros
from omnigibson.object_states import AABB
from omnigibson.object_states.contact_particles import ContactParticles
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.systems.system_base import VisualParticleSystem
from omnigibson.utils.constants import PrimType

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Number of visual particles needed in order for Covered --> True
m.VISUAL_PARTICLE_THRESHOLD = 1

# Maximum number of visual particles to sample when setting an object to be covered = True
m.MAX_VISUAL_PARTICLES = 20

# Number of physical particles needed in order for Covered --> True
m.PHYSICAL_PARTICLE_THRESHOLD = 1

# Maximum number of physical particles to sample when setting an object to be covered = True
m.MAX_PHYSICAL_PARTICLES = 5000


class Covered(RelativeObjectState, BooleanStateMixin):
    def __init__(self, obj):
        # Run super first
        super().__init__(obj)

        # Set internal values
        self._visual_particle_group = None

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB, ContactParticles})
        return deps

    def remove(self):
        if self._initialized:
            self._clear_attachment_groups()

    def _clear_attachment_groups(self):
        """
        Utility function to destroy all corresponding attachment groups for this object
        """
        for system in self.obj.scene.active_systems.values():
            if isinstance(system, VisualParticleSystem) and self._visual_particle_group in system.groups:
                system.remove_attachment_group(self._visual_particle_group)

    def _initialize(self):
        super()._initialize()
        # Grab group name
        self._visual_particle_group = VisualParticleSystem.get_group_name(obj=self.obj)

    def _get_value(self, system):
        # Value is false by default
        value = False
        # First, we check what type of system
        # Currently, we support VisualParticleSystems and PhysicalParticleSystems
        if system.n_particles > 0:
            if self.obj.scene.is_visual_particle_system(system_name=system.name):
                if self._visual_particle_group in system.groups:
                    # check whether the current number of particles assigned to the group is greater than the threshold
                    value = system.num_group_particles(group=self._visual_particle_group) >= m.VISUAL_PARTICLE_THRESHOLD
            elif self.obj.scene.is_physical_particle_system(system_name=system.name):
                # Make sure we're not cloth -- not supported yet
                assert (
                    self.obj.prim_type != PrimType.CLOTH
                ), "Cloth objects currently cannot be Covered by physical particles!"
                # We've already cached particle contacts, so we merely search through them to see if any particles are
                # touching the object and are visible (the non-visible ones are considered already "removed")
                n_near_particles = len(self.obj.states[ContactParticles].get_value(system))
                # Heuristic: If the number of near particles is above the threshold, we consdier this covered
                value = n_near_particles >= m.PHYSICAL_PARTICLE_THRESHOLD
            else:
                raise ValueError(
                    f"Invalid system {system} received for getting Covered state!"
                    f"Currently, only VisualParticleSystems and PhysicalParticleSystems are supported."
                )

        return value

    def _set_value(self, system, new_value):
        # Default success value is True
        success = True
        # First, we check what type of system
        # Currently, we support VisualParticleSystems and PhysicalParticleSystems
        if self.obj.scene.is_visual_particle_system(system_name=system.name):
            # Create the group if it doesn't exist already
            if self._visual_particle_group not in system.groups:
                system.create_attachment_group(obj=self.obj)

            # Check current state and only do something if we're changing state
            if self.get_value(system) != new_value:
                if new_value:
                    # Generate particles
                    success = system.generate_group_particles_on_object(
                        group=self._visual_particle_group,
                        max_samples=m.MAX_VISUAL_PARTICLES,
                        min_samples_for_success=m.VISUAL_PARTICLE_THRESHOLD,
                    )
                else:
                    # We remove all of this group's particles
                    system.remove_all_group_particles(group=self._visual_particle_group)

        elif self.obj.scene.is_physical_particle_system(system_name=system.name):
            # Make sure we're not cloth -- not supported yet
            assert (
                self.obj.prim_type != PrimType.CLOTH
            ), "Cloth objects currently cannot be Covered by physical particles!"
            # Check current state and only do something if we're changing state
            if self.get_value(system) != new_value:
                if new_value:
                    # Sample particles on top of the object
                    success = system.generate_particles_on_object(
                        obj=self.obj,
                        max_samples=m.MAX_PHYSICAL_PARTICLES,
                        min_samples_for_success=m.PHYSICAL_PARTICLE_THRESHOLD,
                    )
                else:
                    # We remove all particles touching this object
                    system.remove_particles(idxs=list(self.obj.states[ContactParticles].get_value(system)))

        else:
            raise ValueError(
                f"Invalid system {system} received for setting Covered state!"
                f"Currently, only VisualParticleSystems and PhysicalParticleSystems are supported."
            )

        return success
