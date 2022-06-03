import numpy as np

from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState

# Fraction of a single particle group required to be within @MAX_SINK_DISTANCE in order to be deleted
MIN_GROUP_FRACTION = 0.6
# Maximum distance (m) away from a sink a particle can be in order to be considered ready for deletion
MAX_SINK_DISTANCE = 0.05


class FluidSink(AbsoluteObjectState, LinkBasedStateMixin):
    def __init__(self, obj, max_distance=MAX_SINK_DISTANCE):
        super().__init__(obj)

        # Store internal values
        self.max_sink_distance = MAX_SINK_DISTANCE

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
        # and check to see if the group matches both the (a) distance and (b) fraction criteria in
        # order to "sink" (delete) it
        names_to_remove = []
        for name, inst in self.fluid_system.particle_instancers.items():
            # Grab particle positions, shape (N, 3)
            particle_pos = inst.particle_positions
            # Get distances and check fraction simultaneously
            frac_in_sink = (np.linalg.norm(particle_pos - fluid_sink_position.reshape(1, 3), axis=-1) < self.max_sink_distance).mean()
            if frac_in_sink >= MIN_GROUP_FRACTION:
                names_to_remove.append(name)

        # Delete all recorded groups
        for name in names_to_remove:
            inst = self.fluid_system.particle_instancers.pop(name)
            self._simulator.stage.RemovePrim(inst.prim_path)

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
