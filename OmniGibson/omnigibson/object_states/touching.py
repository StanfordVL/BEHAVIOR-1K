from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType


class Touching(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @staticmethod
    def _check_rigid_contact(obj_a, obj_b):
        # If both objects are kinematic, we can't check for contact.
        if obj_a.kinematic_only and obj_b.kinematic_only:
            return False
        # Find a kinematic object and check for contact with the other object.
        kinematic_obj = obj_a if obj_a.kinematic_only else obj_b
        non_kinematic_obj = obj_b if obj_a.kinematic_only else obj_a
        return RigidContactAPI.is_in_contact(
            scene_idx=kinematic_obj.scene.idx, query_set=[non_kinematic_obj], with_set=[kinematic_obj]
        )

    @staticmethod
    def _check_cloth_contact(cloth_obj, other_obj):
        other_link_paths = set(other_obj.link_prim_paths)
        return any(contact_prim_path in other_link_paths for contact_prim_path, _ in cloth_obj.root_link.get_contacts())

    def _get_value(self, other):
        if self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect contact between two cloth objects.")
        # If one of the objects is cloth, rely on cloth's get_contacts() method (RigidContactAPI does not include cloth).
        elif self.obj.prim_type == PrimType.CLOTH:
            return self._check_cloth_contact(self.obj, other)
        elif other.prim_type == PrimType.CLOTH:
            return self._check_cloth_contact(other, self.obj)
        else:
            return self._check_rigid_contact(other, self.obj) and self._check_rigid_contact(self.obj, other)
