import numpy as np
from collections import OrderedDict

from igibson.utils.usd_utils import BoundingBoxAPI
from igibson.object_states.object_state_base import CachingEnabledObjectState, NONE


class AABB(CachingEnabledObjectState):
    def _compute_value(self):
        aabb_low, aabb_hi = BoundingBoxAPI.union(self.obj.get_body_ids())

        if not hasattr(self.obj, "category") or self.obj.category != "floors" or self.obj.room_floor is None:
            return np.array(aabb_low), np.array(aabb_hi)

        # TODO: remove after split floors
        # room_floor will be set to the correct RoomFloor beforehand
        room_instance = self.obj.room_floor.room_instance

        # Get the x-y values from the room segmentation map
        room_aabb_low, room_aabb_hi = self.obj.room_floor.scene.get_aabb_by_room_instance(room_instance)

        if room_aabb_low is None:
            return np.array(aabb_low), np.array(aabb_hi)

        # Use the z values from pybullet
        room_aabb_low[2] = aabb_low[2]
        room_aabb_hi[2] = aabb_hi[2]

        return np.array(room_aabb_low), np.array(room_aabb_hi)

    def _set_value(self, new_value):
        raise NotImplementedError("AABB state currently does not support setting.")

    # Nothing needs to be done to save/load AABB since it will happen due to pose caching.
