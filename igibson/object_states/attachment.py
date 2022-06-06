from abc import abstractmethod
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState
from igibson.utils.usd_utils import create_joint
import igibson.utils.transform_utils as T
from pxr import Gf

# After detachment, an object cannot be attached again within this number of steps.
_DEFAULT_ATTACH_PROTECTION_COUNTDOWN = 10


class Attached(RelativeObjectState, BooleanState):
    """
        Child (e.g. painting) attaches to parent (e.g. wall) and holds the joint attaching them
        Whenever a new parent is set, the old joint is deleted. A child can have at most one parent.
    """

    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies()

    def __init__(self, obj):
        super(Attached, self).__init__(obj)

        self.attached_obj = None
        self.attached_joints = [None, None]
        self.attached_joint_paths = [None, None]
        self.enable_attach_this_step = False
        self.attach_protection_countdown = -1

    def _update(self):
        # The object cannot be attached to anything when self.attach_protection_countdown is positive.
        if self.attach_protection_countdown > 0:
            self.attach_protection_countdown -= 1
            return

        # The objects were attached in the last simulation step.
        if self.enable_attach_this_step:
            print(f"{self.obj.name} is attached to {self.attached_obj.name}.")
            self.attached_joints[0].GetAttribute("physics:jointEnabled").Set(True)
            self.attached_joints[1].GetAttribute("physics:jointEnabled").Set(True)
            self.enable_attach_this_step = False
            return

        contact_list = self.obj.states[ContactBodies].get_value()
        # No need to update if this object does not touch anything or is already attached to an object.
        if not contact_list or self.attached_obj is not None:
            return
        # Exclude links from this object.
        link_paths = {link.prim_path for link in self.obj.links.values()}
        for c in contact_list:
            # Extract the prim path of the other body.
            if c.body0 in link_paths and c.body1 in link_paths:
                continue
            path = c.body0 if c.body0 not in link_paths else c.body1
            path_split_arr = path.split("/")
            # Get object's prim path from /World/<obj_name>/.../<obj_link>.
            for i in range(3, len(path_split_arr)):
                path = "/".join(path_split_arr[:i])
                contact_obj = self._simulator.scene.object_registry("prim_path", path)
                if contact_obj is None:
                    continue
                self._set_value(contact_obj, True)
                return

    def _set_value(self, other, new_value):
        # When _set_value is first called, trigger both sides when applicable.
        self._set_value_helper(other, new_value, reverse_trigger=True)

    def _set_value_helper(self, other, new_value, reverse_trigger):
        # Unattach. Remove any existing joint.
        if not new_value or self.attached_obj not in [None, other]:
            print(f"{self.obj.name} is unattached from {self.attached_obj.name}.")
            self._simulator.stage.RemovePrim(self.attached_joint_paths[0])
            self._simulator.stage.RemovePrim(self.attached_joint_paths[1])
            self.attached_obj = None
            self.attached_joints = [None, None]
            self.attached_joint_paths = [None, None]
            self.enable_attach_this_step = False
            # The object should not be able to attach again within some steps after unattaching.
            # Otherwise the object may be attached right away.
            self.attach_protection_countdown = _DEFAULT_ATTACH_PROTECTION_COUNTDOWN
            # Wake up objects so that passive forces like gravity can be applied.
            self.obj.wake()
            other.wake()

        # Already attached. Do nothing.
        elif self.attached_obj is other:
            return False

        # Attach.
        elif new_value and self._can_attach(other):
            self.attached_obj = other

            # Need two fixed joints to make the attachment stable.
            for idx, (obj1, obj2) in enumerate(((self.obj, other), (other, self.obj))):
                self.attached_joint_paths[idx] = f"{obj1.prim_path}/attachment_joint"
                self.attached_joints[idx] = create_joint(
                    prim_path=self.attached_joint_paths[idx],
                    joint_type="FixedJoint",
                    body0=f"{obj1.prim_path}/base_link",
                    body1=f"{obj2.prim_path}/base_link",
                    enabled=False,
                )

                # Set the local pose of the attachment joint.
                pos0, quat0 = obj1.get_position_orientation()
                pos1, quat1 = obj2.get_position_orientation()

                rel_pos, rel_quat = T.relative_pose_transform(pos1, quat1, pos0, quat0)
                rel_pos /= obj1.scale
                rel_quat = rel_quat[[3, 0, 1, 2]]

                self.attached_joints[idx].GetAttribute("physics:localPos0").Set(Gf.Vec3f(*rel_pos))
                self.attached_joints[idx].GetAttribute("physics:localRot0").Set(Gf.Quatf(*rel_quat))

            # We have to toggle the joint from off to on after a physics step because of an omni quirk
            # Otherwise the joint transform is very weird.
            self.enable_attach_this_step = True

        return True

    def _get_value(self, other):
        return other == self.attached_obj

    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies() + [ContactBodies]

    @abstractmethod
    def _can_attach(self, other):
        raise NotImplementedError()


class StickyAttachment(Attached):
    def _can_attach(self, other):
        """ Returns true if touching and at least one object has sticky state. """
        return True


class MagneticAttachment(Attached):
    def _set_value_helper(self, other, new_value, reverse_trigger):
        if MagneticAttachment not in other.states:
            return
        # Both objects need to be set to the same new_value.
        Attached._set_value_helper(self, other, new_value, reverse_trigger=False)
        if reverse_trigger:
            other.states[MagneticAttachment]._set_value_helper(self.obj, new_value, reverse_trigger=False)

    def _can_attach(self, other):
        """ Returns true if touching and both objects have magnetic state. """
        return MagneticAttachment in other.states


class MaleAttachment(MagneticAttachment):
    def _set_value_helper(self, other, new_value, reverse_trigger):
        if FemaleAttachment not in other.states:
            return
        # Both objects need to be set to the same new_value.
        Attached._set_value_helper(self, other, new_value, reverse_trigger=False)
        if reverse_trigger:
            other.states[FemaleAttachment]._set_value_helper(self.obj, new_value, reverse_trigger=False)

    def _can_attach(self, other):
        """ Returns true if touching, self is male and the other is female. """
        return FemaleAttachment in other.states


class FemaleAttachment(MagneticAttachment):
    def _set_value_helper(self, other, new_value, reverse_trigger):
        if MaleAttachment not in other.states:
            return
        # Both objects need to be set to the same new_value.
        Attached._set_value_helper(self, other, new_value, reverse_trigger=False)
        if reverse_trigger:
            other.states[MaleAttachment]._set_value_helper(self.obj, new_value, reverse_trigger=False)

    def _can_attach(self, other):
        """ Returns true if touching, the other is male and self is female. """
        return MaleAttachment in other.states


class HungMaleAttachment(MaleAttachment):
    def _set_value_helper(self, other, new_value, reverse_trigger):
        if HungFemaleAttachment not in other.states:
            return
        # Both objects need to be set to the same new_value.
        Attached._set_value_helper(self, other, new_value, reverse_trigger=False)
        if reverse_trigger:
            other.states[HungFemaleAttachment]._set_value_helper(self.obj, new_value, reverse_trigger=False)

    def _can_attach(self, other):
        """ 
        Returns true if touching, self is male, the other is female,
        and the male hanging object is "below" the female mounting object (center of bbox).
        """
        male_center_z = self.obj.get_position()[2]
        female_center_z = other.get_position()[2]
        return male_center_z < female_center_z and HungFemaleAttachment in other.states


class HungFemaleAttachment(FemaleAttachment):
    def _set_value_helper(self, other, new_value, reverse_trigger):
        if HungMaleAttachment not in other.states:
            return
        # Both objects need to be set to the same new_value.
        Attached._set_value_helper(self, other, new_value, reverse_trigger=False)
        if reverse_trigger:
            other.states[HungMaleAttachment]._set_value_helper(self.obj, new_value, reverse_trigger=False)

    def _can_attach(self, other):
        """ 
        Returns true if touching, the other is male, self is female
        and the male hanging object is "below" the female mounting object (center of bbox).
        """
        male_center_z = other.get_position()[2]
        female_center_z = self.obj.get_position()[2]
        return male_center_z < female_center_z and HungMaleAttachment in other.states
