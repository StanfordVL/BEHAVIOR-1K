import numpy as np
from collections import OrderedDict

from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.object_states.object_state_base import AbsoluteObjectState


class AABB(AbsoluteObjectState):
    def _get_value(self):
        aabb_low, aabb_hi = BoundingBoxAPI.union(self.obj.link_prim_paths)

        if hasattr(self.obj, "category") and self.obj.category == "floors" and self.obj.room_floor is not None:
            # TODO: remove after split floors
            # room_floor will be set to the correct RoomFloor beforehand
            room_instance = self.obj.room_floor.room_instance

            # Get the x-y values from the room segmentation map
            room_aabb_low, room_aabb_hi = self.obj.room_floor.scene.get_aabb_by_room_instance(room_instance)

            if room_aabb_low is None:
                low, high = np.array(aabb_low), np.array(aabb_hi)

            # Otherwise use the z values from omni
            else:
                room_aabb_low[2] = aabb_low[2]
                room_aabb_hi[2] = aabb_hi[2]
                low, high = np.array(room_aabb_low), np.array(room_aabb_hi)

        else:
            low, high = np.array(aabb_low), np.array(aabb_hi)

        return low, high

    def _set_value(self, new_value):
        raise NotImplementedError("AABB state currently does not support setting.")

    # Nothing needs to be done to save/load AABB since it will happen due to pose caching.
