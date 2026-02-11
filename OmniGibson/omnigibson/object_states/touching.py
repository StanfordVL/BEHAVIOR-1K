from omnigibson.utils.sim_utils import get_rigid_contact_bodies
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType


class Touching(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @staticmethod
    def _check_contact(obj_a, obj_b):
        return len(set(obj_a.links.values()) & get_rigid_contact_bodies(obj_b)) > 0

    def _get_value(self, other):
        if self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect contact between two cloth objects.")
        # If one of the objects is the cloth object, the contact will be asymmetrical.
        # The rigid object will appear in rigid-contact checks from cloth, but not necessarily the other way around.
        elif self.obj.prim_type == PrimType.CLOTH:
            return self._check_contact(other, self.obj)
        elif other.prim_type == PrimType.CLOTH:
            return self._check_contact(self.obj, other)
        else:
            return self._check_contact(other, self.obj) and self._check_contact(self.obj, other)
