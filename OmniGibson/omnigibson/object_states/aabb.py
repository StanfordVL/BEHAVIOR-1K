from omnigibson.object_states.object_state_base import AbsoluteObjectState


class AABB(AbsoluteObjectState):
    def _get_value(self):
        return self.obj.aabb

    # Nothing needs to be done to save/load AABB since it will happen due to pose caching.
