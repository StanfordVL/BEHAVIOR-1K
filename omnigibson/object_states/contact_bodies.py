from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.sim_utils import get_collisions


class ContactBodies(AbsoluteObjectState):
    def _get_value(self, include_objs=None, ignore_objs=None):
        # Ignore_objs should either be None or tuple (CANNOT be list because we need to hash these inputs)
        assert ignore_objs is None or isinstance(ignore_objs, tuple), \
            "ignore_objs must either be None or a tuple of objects to ignore!"
        return get_collisions(prims=self.obj, prims_check=include_objs, prims_exclude=ignore_objs, step_physics=False)

    def _set_value(self, new_value):
        raise NotImplementedError("ContactBodies state currently does not support setting.")
