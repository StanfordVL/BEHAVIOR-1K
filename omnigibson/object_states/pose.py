import numpy as np

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.POSITIONAL_VALIDATION_EPSILON = 1e-10


class Pose(AbsoluteObjectState):

    def _get_value(self):
        pos = self.obj.get_position()
        orn = self.obj.get_orientation()
        return np.array(pos), np.array(orn)

    def _set_value(self, new_value):
        raise NotImplementedError("Pose state currently does not support setting.")

    def _has_changed(self, get_value_args, t):
        # Only changed if the squared distance between old position and current position has
        # changed above some threshold
        old_pos = self._history[(t, *get_value_args)][0]
        current_pos = self.get_value()[0]
        dist_squared = np.sum(np.square(current_pos - old_pos))
        return dist_squared > m.POSITIONAL_VALIDATION_EPSILON
