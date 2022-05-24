from abc import abstractmethod
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState
import igibson.utils.transform_utils as T
from igibson.utils.usd_utils import create_joint
from pxr import Gf

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
        self.attached_joint = None
        self.attached_joint_path = None

    def _update(self):
        contact_list = self.obj.states[ContactBodies].get_value()
        # exclude links from our own object
        link_paths = {link.prim_path for link in self.obj.links.values()}
        for c in contact_list:
            # extract the prim path of the other body
            if c.body0 in link_paths and c.body1 in link_paths:
                continue
            path = c.body0 if c.body0 not in link_paths else c.body1
            path = path.replace("/base_link", "")
            contact_obj = self._simulator.scene.object_registry("prim_path", path)
            if contact_obj is None:
                continue
            self._set_value(contact_obj, True)
    
    def _set_value(self, other, new_value):
        if not new_value or self.attached_obj not in [None, other]:
            # delete old joint
            self._simulator.stage.RemovePrim(self.attached_joint_path)
            self.attached_obj = None
            self.attached_joint = None
            self.attached_joint_path = None
        
        if self.attached_obj is other:
            # do nothing; already attached
            return False
        
        if new_value and self._can_attach(other):
            self.attached_obj = other
            self.attached_joint_path = f"{self.obj.prim_path}/AttachmentJoint"
            self.attached_joint = create_joint(
                prim_path=self.attached_joint_path,
                joint_type="FixedJoint",
                body0=f"{self.obj.prim_path}/base_link",
                body1=f"{other.prim_path}/base_link",
            )

            pos0, quat0 = self.obj.get_position_orientation()
            pos1, quat1 = other.get_position_orientation()
            
            rel_pos, rel_quat = T.relative_pose_transform(pos1, quat1, pos0, quat0)
            rel_pos /= self.obj.scale
            rel_quat = rel_quat[[3,0,1,2]]

            self.attached_joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*rel_pos))
            self.attached_joint.GetAttribute("physics:localRot0").Set(Gf.Quatf(*rel_quat))

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
        return StickyAttachment in self.obj.states or StickyAttachment in other.states 
        # if touching and at least one has sticky state

class MagneticAttachment(Attached):
    def _can_attach(self, other):
        return MagneticAttachment in self.obj.states and MagneticAttachment in other.states
        # if touching and both have magnetic state

class MaleAttachment(MagneticAttachment):
    def _can_attach(self, other):
        return True
        # if touching and self is male and the other is female

class FemaleAttachment(MagneticAttachment):
    def _can_attach(self, other):
        return True
        # if touching and other is male and self is female

class HungMaleAttachment(MaleAttachment):
    def _can_attach(self, other):
        return True
        # if touching and self is male and the other is female
        # and hanging object is "below" mounting object (center of mass)

class HungFemaleAttachment(FemaleAttachment):
    def _can_attach(self, other):
        return True
        # if touching and other is male and self is female
        # and hanging object is "below" mounting object (center of mass)