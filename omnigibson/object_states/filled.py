import numpy as np
from collections import OrderedDict
from omnigibson.macros import gm, create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.systems.micro_particle_system import FluidSystem
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.systems import get_fluid_systems, get_system_from_element_name, get_element_name_from_system

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Proportion of object's volume that must be filled for object to be considered filled
m.VOLUME_FILL_PROPORTION = 0.3


class Filled(RelativeObjectState, BooleanState, LinkBasedStateMixin):
    def __init__(self, obj):
        super().__init__(obj)
        self.check_in_volume = None        # Function to check whether particles are in volume for this container
        self.calculate_volume = None       # Function to calculate the real-world volume for this container

    def _get_value(self, fluid_system):
        # Sanity check to manke sure fluid system is valid
        assert issubclass(fluid_system, FluidSystem), "Can only get Filled state with a valid FluidSystem!"
        # Check what volume is filled
        if len(fluid_system.particle_instancers) > 0:
            particle_positions = np.concatenate([inst.particle_positions[inst.particle_visibilities.nonzero()[0]] for inst in fluid_system.particle_instancers.values()], axis=0)
            particles_in_volume = self.check_in_volume(particle_positions)
            particle_volume = 4 / 3 * np.pi * (fluid_system.particle_radius ** 3)
            prop_filled = particle_volume * particles_in_volume.sum() / self.calculate_volume()
            # If greater than threshold, then the volume is filled
            # Explicit bool cast needed here because the type is bool_ instead of bool which is not JSON-Serializable
            # This has to do with numpy, see https://stackoverflow.com/questions/58408054/typeerror-object-of-type-bool-is-not-json-serializable
            value = bool(prop_filled > m.VOLUME_FILL_PROPORTION)
        else:
            # No fluid exists, so we're obviously empty
            value = False

        return value

    def _set_value(self, fluid_system, new_value):
        # Sanity check to manke sure fluid system is valid
        assert issubclass(fluid_system, FluidSystem), "Can only set Filled state with a valid FluidSystem!"

        # If we found no link, directly return
        if self.link is None:
            return False

        # First, check our current state
        current_state = self.get_value(fluid_system)

        # Only do something if we're changing state
        if current_state != new_value:
            if new_value:
                # Going from False --> True, sample volume with particles
                fluid_system.generate_particle_instancer_from_link(
                    obj=self.obj,
                    link=self.link,
                    mesh_name_prefixes="container",
                )
            else:
                # Going from True --> False, hide all particles within the current volume to be garbage collected
                # by fluid system
                for inst in fluid_system.particle_instancers.values():
                    indices = self.check_in_volume(inst.particle_positions).nonzero()[0]
                    current_visibilities = inst.particle_visibilities
                    current_visibilities[indices] = 0
                    inst.particle_visibilities = current_visibilities

        return True

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

        # If we found no link, directly return
        if self.link is None:
            return

        # Generate volume checker function for this object
        self.check_in_volume, self.calculate_volume = \
            generate_points_in_volume_checker_function(obj=self.obj, volume_link=self.link, mesh_name_prefixes="container")

    @staticmethod
    def get_state_link_name():
        # Should be implemented by subclass
        return "container_link"

    @staticmethod
    def get_optional_dependencies():
        return []

    @property
    def state_size(self):
        return len(get_fluid_systems())

    def _dump_state(self):
        # Store whether we're filled for each volume or not
        state = OrderedDict()
        for system in get_fluid_systems().values():
            fluid_name = get_element_name_from_system(system)
            state[fluid_name] = self.get_value(system)

        return state

    def _load_state(self, state):
        # Check to see if the value is different from what we currently have
        # This should always be the same, because our get_value() reads from the particle system, which should
        # hav already updated / synchronized its state
        for fluid_name, val in state.items():
            assert val == self.get_value(get_system_from_element_name(fluid_name)), \
            f"Expected state {self.__class__.__name__} to have synchronized values, but got current value: {self.get_value(get_system_from_element_name(fluid_name))} with desired value: {val}"

    def _serialize(cls, state):
        return np.array(list(state.values()), dtype=float)

    def _deserialize(self, state):
        state_dict = OrderedDict()
        for i, fluid_system in enumerate(get_fluid_systems().values()):
            fluid_name = get_element_name_from_system(fluid_system)
            state_dict[fluid_name] = bool(state[i])

        return state_dict, len(state_dict)
