import numpy as np
from collections import OrderedDict
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.water_source import WaterSource


# Proportion of fluid particles per group required to intersect with this object for them all to be absorbed
# by a soakable object
PARTICLE_GROUP_PROPORTION = 0.7

# Soak threshold -- number of fluid particle required to be "absorbed" in order for the object to be
# considered soaked
SOAK_PARTICLE_THRESHOLD = 40


class Soaked(AbsoluteObjectState, BooleanState):
    def __init__(self, obj):
        super(Soaked, self).__init__(obj)
        self.value = False

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        # TODO (mjlbach)
        return

    @staticmethod
    def get_texture_change_params():
        # Increase all channels by 0.1
        albedo_add = 0.1
        # Then scale up "blue" color and scale down others
        diffuse_tint = (0.5, 0.5, 1.5)
        return albedo_add, diffuse_tint

    @property
    def settable(self):
        return True

    def _dump_state(self):
        return OrderedDict(soaked=self.value)

    def _load_state(self, state):
        self.value = state["soaked"]

    def _serialize(self, state):
        return np.array([float(state["soaked"])])

    def _deserialize(self, state):
        return OrderedDict(soaked=(state[0] == 1.0)), 1

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]