from igibson.settings import settings
from igibson.object_states import AABB
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.utils.constants import SemanticClass
from igibson.systems.macro_particle_system import DustSystem, StainSystem
from collections import OrderedDict
import numpy as np

s = settings.object_states.dirty


class _Dirty(AbsoluteObjectState, BooleanState):
    """
    This class represents common logic between particle-based dirtyness states like
    dusty and stained. It should not be directly instantiated - use subclasses instead.
    """
    # This must be filled by subclass!
    DIRT_CLASS = None

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @property
    def settable(self):
        return True

    def __init__(self, obj):
        super(_Dirty, self).__init__(obj)
        self.dirt_group = None
        self._max_particles_for_clean = None

    def _initialize(self):
        # Make sure dirt class is filled!
        assert self.DIRT_CLASS is not None, "Dirt class must be specified to initialize Dirty state!"

        # Create the dirt group
        self.dirt_group = self.DIRT_CLASS.create_attachment_group(obj=self.obj)

        # Default max particles for clean is 0
        self._max_particles_for_clean = 0

    def _get_value(self):
        return self.DIRT_CLASS.num_group_particles(group=self.dirt_group) > self._max_particles_for_clean

    def _set_value(self, new_value):
        sampling_success = True

        # First, check our current state
        current_state = self.get_value()

        # Only do something if we're changing state
        if current_state != new_value:
            if not new_value:
                # We remove all of this group's particles
                self.DIRT_CLASS.remove_all_group_particles(group=self.dirt_group)
            else:
                # Generate dirt particles
                sampling_success = self.DIRT_CLASS.generate_group_particles(group=self.dirt_group)
                # If we succeeded with generating particles (new_value = True), we are dirty, let's store additional info
                if sampling_success:
                    # Store how many particles there are now -- this is the "maximum" number possible
                    clean_threshold = s.FLOOR_CLEAN_THRESHOLD if self.obj.category == "floors" else s.CLEAN_THRESHOLD
                    self._max_particles_for_clean = \
                        self.DIRT_CLASS.num_group_particles(group=self.dirt_group) * clean_threshold

        return sampling_success

    @property
    def state_size(self):
        return 2

    def _dump_state(self):
        return OrderedDict(value=self.get_value(), max_particles_for_clean=self._max_particles_for_clean)

    def _load_state(self, state):
        # Check to see if the value is different from what we currently have
        # This should always be the same, because our get_value() reads from the particle system, which should
        # hav already updated / synchronized its state
        assert state["value"] == self.get_value(), \
            f"Expected state {self.__class__.__name__} to have synchronized values, but got current value: {self.get_value()} with desired value: {state['value']}"

        # also set the max particles for clean
        self._max_particles_for_clean = state["max_particles_for_clean"]

    def _serialize(cls, state):
        return np.array([state["value"], state["max_particles_for_clean"]])

    def _deserialize(cls, state):
        return OrderedDict(value=state[0], max_particles_for_clean=state[1]), 2


class Dusty(_Dirty):
    DIRT_CLASS = DustSystem


class Stained(_Dirty):
    DIRT_CLASS = StainSystem
