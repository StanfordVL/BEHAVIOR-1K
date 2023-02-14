from omnigibson.macros import create_module_macros
from omnigibson.object_states import AABB
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.object_states.contact_fluids import ContactFluids
from omnigibson.systems.system_base import get_element_name_from_system, get_system_from_element_name
from omnigibson.systems.macro_particle_system import VisualParticleSystem, get_visual_particle_systems
from omnigibson.systems.micro_particle_system import FluidSystem, get_fluid_systems
from omnigibson.utils.python_utils import classproperty
import numpy as np

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Value in [0, 1] determining the minimum proportion of particles needed in order for Covered --> True
m.VISUAL_PARTICLE_THRESHOLD = 0.75

# Number of fluid particles needed in order for Covered --> True
m.FLUID_THRESHOLD = 50

# Maximum number of fluid particles to sample when setting an object to be covered = True
m.MAX_FLUID_PARTICLES = 5000


class Covered(RelativeObjectState, BooleanState):
    def __init__(self, obj):
        # Run super first
        super().__init__(obj)

        # Set internal values
        self._visual_particle_groups = None
        self._n_initial_visual_particles = None

    @staticmethod
    def get_dependencies():
        # AABB needed for sampling visual particles on an object
        return RelativeObjectState.get_dependencies() + [AABB, ContactFluids]

    def reset(self):
        # Run super first
        super().reset()

        # Make sure all groups are cleared if initialized, and then re-initialize
        if self._initialized:
            self._clear_attachment_groups()

            # Re-initialize system
            self._initialize()

    def remove(self):
        if self._initialized:
            self._clear_attachment_groups()

    def _clear_attachment_groups(self):
        """
        Utility function to destroy all corresponding attachment groups for this object
        """
        for system_name, group in self._visual_particle_groups.items():
            system = get_system_from_element_name(system_name)
            if group in system.groups:
                system.remove_attachment_group(group)

    def _initialize(self):
        # Create the visual particle groups
        self._visual_particle_groups = dict((get_element_name_from_system(system), system.create_attachment_group(obj=self.obj))
                                                   for system in get_visual_particle_systems().values())

        # Default initial particles is 0
        self._n_initial_visual_particles = dict((get_element_name_from_system(system), 0)
                                                       for system in get_visual_particle_systems().values())

    def _get_value(self, system):
        # Value is false by default
        value = False
        # First, we check what type of system
        # Currently, we support VisualParticleSystems and FluidSystems
        if issubclass(system, VisualParticleSystem):
            # Create the group if it doesn't exist already
            name = get_element_name_from_system(system)
            if self._visual_particle_groups[name] not in system.groups:
                system.create_attachment_group(obj=self.obj)
            # We check whether the current number of particles assigned to the group is greater than the threshold
            value = system.num_group_particles(group=self._visual_particle_groups[name]) \
                   > m.VISUAL_PARTICLE_THRESHOLD * self._n_initial_visual_particles[name]
        elif issubclass(system, FluidSystem):
            # We only check if we have particle instancers currently
            if len(system.particle_instancers) > 0:
                # We've already cached particle contacts, so we merely search through them to see if any particles are
                # touching the object and are visible (the non-visible ones are considered already "removed")
                n_near_particles = np.sum([len(idxs) for idxs in self.obj.states[ContactFluids].get_value(system).values()])
                # Heuristic: If the number of near particles is above the threshold, we consdier this covered
                value = n_near_particles >= m.FLUID_THRESHOLD
        else:
            raise ValueError(f"Invalid system {system} received for getting Covered state!"
                             f"Currently, only VisualParticleSystems and FluidSystems are supported.")

        return value

    def _set_value(self, system, new_value):
        # Default success value is True
        success = True
        # First, we check what type of system
        # Currently, we support VisualParticleSystems and FluidSystems
        if issubclass(system, VisualParticleSystem):
            # Create the group if it doesn't exist already
            name = get_element_name_from_system(system)
            if self._visual_particle_groups[name] not in system.groups:
                system.create_attachment_group(obj=self.obj)

            # Check current state and only do something if we're changing state
            if self.get_value(system) != new_value:
                group = self._visual_particle_groups[name]
                if new_value:
                    # Generate particles
                    success = system.generate_group_particles_on_object(group=group)
                    # If we succeeded with generating particles (new_value = True), store additional info
                    if success:
                        # Store how many particles there are now -- this is the "maximum" number possible
                        self._n_initial_visual_particles[name] = system.num_group_particles(group=group)
                else:
                    # We remove all of this group's particles
                    system.remove_all_group_particles(group=group)

        elif issubclass(system, FluidSystem):
            # Check current state and only do something if we're changing state
            if self.get_value(system) != new_value:
                if new_value:
                    # Sample particles on top of the object
                    system.generate_particles_on_object(obj=self.obj, max_samples=m.MAX_FLUID_PARTICLES)
                else:
                    # We delete all particles touching this object
                    for inst, particle_idxs in self.obj.states[ContactFluids].get_value(system).items():
                        inst.remove_particles(idxs=list(particle_idxs))

        else:
            raise ValueError(f"Invalid system {system} received for setting Covered state!"
                             f"Currently, only VisualParticleSystems and FluidSystems are supported.")

        return success

    @property
    def state_size(self):
        # We have a single value for every visual particle system
        return len(get_visual_particle_systems())

    @classproperty
    def supported_systems(self):
        """
        Returns:
            list: All systems used in this state, ordered deterministically
        """
        return list(get_visual_particle_systems().values()) + list(get_fluid_systems().values())

    def _dump_state(self):
        # For fluid systems, we don't need to dump state, because the fluid systems themselves handle all state dumping
        # related to fluids
        # For every visual particle system, add the initial number of particles
        state = dict()
        for system in get_visual_particle_systems().values():
            name = get_element_name_from_system(system)
            state[f"{name}_initial_visual_particles"] = self._n_initial_visual_particles[name]

        return state

    def _load_state(self, state):
        # For fluid systems, we don't need to load state, because the fluid systems themselves handle all state loading
        # related to fluids
        # For every visual particle system, set the initial number of particles
        for system in get_visual_particle_systems().values():
            name = get_element_name_from_system(system)
            self._n_initial_visual_particles[name] = state[f"{name}_initial_visual_particles"]

    def _serialize(self, state):
        return np.array([val for val in state.values()], dtype=float)

    def _deserialize(self, state):
        state_dict = dict()
        for i, system in enumerate(get_visual_particle_systems().values()):
            name = get_element_name_from_system(system)
            state_dict[f"{name}_initial_visual_particles"] = int(state[i])

        return state_dict, len(state_dict)
