from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.sim_utils import prim_paths_to_rigid_prims

import numpy as np


class ContactBodies(AbsoluteObjectState):
    """
    NOTE: This is slow and uncached, but it works even for sleeping objects.
    For frequent contact checks, consider using RigidContactAPI for performance.
    """

    def _get_value(self, ignore_objs=None, non_zero_impulse=False):
        # Compute bodies in contact, minus the self-owned bodies
        bodies = set()
        for contact in self.obj.contact_list():
            if not non_zero_impulse or np.linalg.norm(tuple(contact.impulse)) > 0:
                bodies.update({contact.body0, contact.body1})
        bodies -= set(self.obj.link_prim_paths)
        rigid_prims = prim_paths_to_rigid_prims(bodies, self.obj.scene)
        # Ignore_objs should either be None or tuple (CANNOT be list because we need to hash these inputs)
        assert ignore_objs is None or isinstance(
            ignore_objs, tuple
        ), "ignore_objs must either be None or a tuple of objects to ignore!"
        return {p for o, p in rigid_prims if ignore_objs is None or o not in ignore_objs}
