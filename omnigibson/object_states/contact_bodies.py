import omnigibson as og
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.sim_utils import prims_to_rigid_prim_set


class ContactBodies(AbsoluteObjectState):

    def _get_value(self, ignore_objs=None):
        # Compute bodies in contact, minus the self-owned bodies
        bodies = set()
        for contact in self.obj.contact_list():
            bodies.update({contact.body0, contact.body1})
        bodies -= set(self.obj.link_prim_paths)
        rigid_prims = set()
        for body in bodies:
            tokens = body.split("/")
            obj_prim_path = "/".join(tokens[:-1])
            link_name = tokens[-1]
            obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
            if obj is not None:
                rigid_prims.add(obj.links[link_name])
        # Ignore_objs should either be None or tuple (CANNOT be list because we need to hash these inputs)
        assert ignore_objs is None or isinstance(ignore_objs, tuple), \
            "ignore_objs must either be None or a tuple of objects to ignore!"
        return rigid_prims if ignore_objs is None else rigid_prims - prims_to_rigid_prim_set(ignore_objs)
