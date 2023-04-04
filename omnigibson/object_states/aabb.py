import numpy as np

from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.constants import PrimType


class AABB(AbsoluteObjectState):
    def _get_value(self):
        if self.obj.prim_type == PrimType.RIGID:
            aabb_low, aabb_hi = BoundingBoxAPI.compute_aabb(self.obj)
            aabb_low, aabb_hi = np.array(aabb_low), np.array(aabb_hi)
        elif self.obj.prim_type == PrimType.CLOTH:
            particle_positions = self.obj.root_link.particle_positions
            aabb_low, aabb_hi = np.min(particle_positions, axis=0), np.max(particle_positions, axis=0)
        else:
            raise ValueError(f"unknown prim type {self.obj.prim_type}")

        return aabb_low, aabb_hi

    def _set_value(self, new_value):
        raise NotImplementedError("AABB state currently does not support setting.")

    # Nothing needs to be done to save/load AABB since it will happen due to pose caching.
