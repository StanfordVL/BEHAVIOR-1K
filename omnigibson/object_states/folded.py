import numpy as np

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanState, AbsoluteObjectState
from omnigibson.utils.object_state_utils import (
    calculate_projection_area_and_diagonal,
    calculate_projection_area_and_diagonal_maximum,
    calculate_smoothness,
)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Criterion #1: the threshold on the area ratio of the convex hull of the projection on the XY plane
m.AREA_THRESHOLD = 0.75

# Criterion #2: the threshold on the diagonal ratio of the convex hull of the projection on the XY plane
m.DIAGONAL_THRESHOLD = 0.9

# Criterion #3: the percentage of face normals that need to be close to the z-axis.
m.NORMAL_Z_PERCENTAGE = 0.5

class Folded(AbsoluteObjectState, BooleanState):
    def _initialize(self):
        # Assume the initial state is unfolded
        self.area_unfolded, self.diagonal_unfolded = calculate_projection_area_and_diagonal_maximum(self.obj)

    def _get_value(self):
        # Check the smoothness of the cloth
        percent_smooth = calculate_smoothness(self.obj)

        # Early stopping
        if percent_smooth < m.NORMAL_Z_PERCENTAGE:
            return False

        # Calculate the area and the diagonal of the current state
        area, diagonal = calculate_projection_area_and_diagonal(self.obj, [0, 1])

        # Check area ratio
        flag_area = (area / self.area_unfolded) < m.AREA_THRESHOLD

        # Check the diagonal ratio
        flag_diagonal = (diagonal / self.diagonal_unfolded) < m.DIAGONAL_THRESHOLD

        return flag_area and flag_diagonal

    def _set_value(self, new_value):
        if not new_value:
            raise NotImplementedError("Folded does not support set_value(False)")

        # TODO (eric): add this support
        raise NotImplementedError("Folded does not support set_value(True)")

    # We don't need to dump / load anything since the cloth objects should handle it themselves
