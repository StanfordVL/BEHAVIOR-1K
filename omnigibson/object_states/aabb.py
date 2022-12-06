import numpy as np
from collections import OrderedDict

from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.object_states.object_state_base import AbsoluteObjectState


class AABB(AbsoluteObjectState):
    def _get_value(self):
        aabb_low, aabb_hi = BoundingBoxAPI.union(self.obj.link_prim_paths)
        return np.array(aabb_low), np.array(aabb_hi)

    def _set_value(self, new_value):
        raise NotImplementedError("AABB state currently does not support setting.")

    # Nothing needs to be done to save/load AABB since it will happen due to pose caching.
