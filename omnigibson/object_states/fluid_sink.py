import numpy as np

from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.update_state_mixin import UpdateStateMixin


# Create settings for this module
m = create_module_macros(module_path=__file__)

# Fraction of a single particle group required to be within @m.MAX_SINK_DISTANCE in order to be deleted
m.MIN_GROUP_FRACTION = 0.6
# Maximum distance (m) away from a sink a particle can be in order to be considered ready for deletion
m.MAX_SINK_DISTANCE = 0.05


class FluidSink(AbsoluteObjectState, LinkBasedStateMixin, UpdateStateMixin):
    def __init__(self, obj, max_distance=m.MAX_SINK_DISTANCE):
        super().__init__(obj)

        # Store internal values
        self.max_sink_distance = m.MAX_SINK_DISTANCE

    @property
    def fluid_system(self):
        """
        Returns:
            FluidSystem: Fluid system to use to generate / handle fluid particles
        """
        raise NotImplementedError

    @staticmethod
    def get_state_link_name():
        # Should be implemented by subclass
        raise NotImplementedError

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

    def _update(self):
        fluid_sink_position = self.get_link_position()
        if fluid_sink_position is None:
            # Terminate early, this is a "dead" fluid sink
            return

        # We iterate over all active fluid groups in this sink's corresponding fluid system,
        # and check to see if each owned particle matches distance criteria in order to "sink" (delete) it
        for name, inst in self.fluid_system.particle_instancers.items():
            # Grab particle positions, shape (N, 3)
            particle_pos = inst.particle_positions
            # Get distances
            idxs_to_remove = np.where(np.linalg.norm(particle_pos - fluid_sink_position.reshape(1, 3),
                                           axis=-1) < self.max_sink_distance)[0]
            inst.remove_particles(idxs=idxs_to_remove)

    def _set_value(self, new_value):
        raise ValueError("set_value not supported for FluidSink.")

    def _get_value(self):
        pass

    @staticmethod
    def get_optional_dependencies():
        return []

    @staticmethod
    def get_dependencies():
        return []
