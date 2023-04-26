from omnigibson.object_states.kinematics import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanState, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import sample_cloth_on_rigid
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros

import omnigibson as og

from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.qhull import QhullError
import numpy as np
import trimesh
import itertools

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Maximum velocity allowed for the draped object
m.DRAPED_MAX_VELOCITY = 0.05

class Draped(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + RelativeObjectState.get_dependencies() + [Touching]

    def _set_value(self, other, new_value):
        if not new_value:
            raise NotImplementedError("DrapedOver does not support set_value(False)")

        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("DrapedOver state requires obj1 is cloth and obj2 is rigid.")

        state = og.sim.dump_state(serialized=False)

        for _ in range(10):
            if sample_cloth_on_rigid(self.obj, other, randomize_xy=True) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        """
        Check whether the (cloth) object is draped on the other (rigid) object.
        The cloth object needs to touch the rigid object and its velocity needs to be small.
        """
        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("Draped state requires obj1 is cloth and obj2 is rigid.")

        return self.obj.states[Touching].get_value(other)

        # return self.obj.states[Touching].get_value(other) and \
        #     np.linalg.norm(self.obj.get_linear_velocity()) < m.DRAPED_MAX_VELOCITY
