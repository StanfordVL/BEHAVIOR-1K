import numpy as np

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.POSITIONAL_VALIDATION_EPSILON = 1e-10
m.ORIENTATION_VALIDATION_EPSILON = 0.003        # ~5 degrees error tolerance


class Pose(AbsoluteObjectState):

    def _get_value(self):
        pos = self.obj.get_position()
        orn = self.obj.get_orientation()
        return np.array(pos), np.array(orn)

    def _has_changed(self, get_value_args, value, info):
        # Only changed if the squared distance between old position and current position has
        # changed above some threshold
        old_pos, old_quat = value
        # Get current pose
        current_pos, current_quat = self.get_value()
        # Check position and orientation -- either changing means we've changed poses
        dist_squared = np.sum(np.square(current_pos - old_pos))
        if dist_squared > m.POSITIONAL_VALIDATION_EPSILON:
            return True
        # Calculate quat distance simply as the dot product
        # A * B = |A||B|cos(theta)
        quat_cos_angle = np.abs(np.dot(old_quat, current_quat))
        if (1 - quat_cos_angle) > m.ORIENTATION_VALIDATION_EPSILON:
            return True

        return False
