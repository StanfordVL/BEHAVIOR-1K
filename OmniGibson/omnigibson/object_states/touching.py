from omnigibson.utils.sim_utils import get_rigid_contact_bodies
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType


class Touching(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @staticmethod
    def _check_rigid_contact(obj_a, obj_b):
        return len(set(obj_a.links.values()) & get_rigid_contact_bodies(obj_b)) > 0

    @staticmethod
    def _check_cloth_contact(cloth_obj, other_obj):
        other_link_paths = set(other_obj.link_prim_paths)
        return any(len({contact.body0, contact.body1} & other_link_paths) > 0 for contact in cloth_obj.contact_list())

    def _get_value(self, other):
        if self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect contact between two cloth objects.")
        # If one of the objects is cloth, rely on cloth contact_list (RigidContactAPI does not include cloth).
        elif self.obj.prim_type == PrimType.CLOTH:
            return self._check_cloth_contact(self.obj, other)
        elif other.prim_type == PrimType.CLOTH:
            return self._check_cloth_contact(other, self.obj)
        else:
            return self._check_rigid_contact(other, self.obj) and self._check_rigid_contact(self.obj, other)
