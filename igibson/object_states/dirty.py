from igibson.settings import settings
from igibson.object_states import AABB
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.utils.constants import SemanticClass
from igibson.systems.particle_system import DustSystem, StainSystem
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

    def __init__(self, obj):
        super(_Dirty, self).__init__(obj)
        self.dirt_group = None
        self._max_particles_for_clean = None

    def _initialize(self):
        # Make sure dirt class is filled!
        assert self.DIRT_CLASS is not None, "Dirt class must be specified to initialize Dirty state!"

        # Create the dirt group
        self.dirt_group = self.DIRT_CLASS.create_attachment_group(obj=self.obj)

    def _get_value(self):
        return self.DIRT_CLASS.num_group_particles(group=self.dirt_group) > self._max_particles_for_clean

    def _set_value(self, new_value):
        if not new_value:
            # We remove all of this group's particles
            self.DIRT_CLASS.remove_all_group_particles(group=self.dirt_group)
        else:
            # Generate dirt particles
            new_value = self.DIRT_CLASS.generate_particles_on_object(obj=self.obj)
            # If we succeeded with generating particles (new_value = True), we are dirty, let's store additional info
            if new_value:
                # Store how many particles there are now -- this is the "maximum" number possible
                clean_threshold = s.FLOOR_CLEAN_THRESHOLD if self.obj.category == "floors" else s.CLEAN_THRESHOLD
                self._max_particles_for_clean = \
                    self.DIRT_CLASS.num_group_particles(group=self.dirt_group) * clean_threshold

        return new_value

    # TODO!
    @classmethod
    def serialize(cls, data):
        raise NotImplementedError()
        # return np.array([NONE])

    @classmethod
    def deserialize(cls, data):
        raise NotImplementedError()


class Dusty(_Dirty):
    DIRT_CLASS = DustSystem


class Stained(_Dirty):
    DIRT_CLASS = StainSystem
