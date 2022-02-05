

# from igibson.external.pybullet_tools.utils import ContactResult
from igibson.object_states.object_state_base import CachingEnabledObjectState


class ContactBodies(CachingEnabledObjectState):
    def _compute_value(self):
        return [
            ContactResult(*item[:10])
            for body_id in self.obj.get_body_ids()
            for item in p.getContactPoints(bodyA=body_id)
        ]

    def _set_value(self, new_value):
        raise NotImplementedError("ContactBodies state currently does not support setting.")

    # Nothing needs to be done to save/load ContactBodies since it will happen due to pose caching.
    def _dump(self):
        return None

    def load(self, data):
        return
