from omni.physx.bindings._physx import ContactEventType

import numpy as np
from collections import defaultdict

import omnigibson as og
from omnigibson.macros import create_module_macros
import omnigibson.utils.transform_utils as T
from omnigibson.object_states.contact_subscribed_state_mixin import ContactSubscribedStateMixin
from omnigibson.object_states.joint_break_subscribed_state_mixin import JointBreakSubscribedStateMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.utils.constants import JointType
from omnigibson.utils.usd_utils import create_joint
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import classproperty

# Create module logger
log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.ATTACHMENT_LINK_PREFIX = "attachment"

m.DEFAULT_POSITION_THRESHOLD = 0.05  # 5cm
m.DEFAULT_ORIENTATION_THRESHOLD = np.deg2rad(5.0)  # 5 degrees
m.DEFAULT_JOINT_TYPE = JointType.JOINT_FIXED
m.DEFAULT_BREAK_FORCE = 10000  # Newton
m.DEFAULT_BREAK_TORQUE = 10000  # Newton-Meter


class AttachedTo(RelativeObjectState, BooleanStateMixin, ContactSubscribedStateMixin, JointBreakSubscribedStateMixin, LinkBasedStateMixin):
    """
        Handles attachment between two rigid objects, by creating a fixed/spherical joint between self.obj (child) and
        other (parent). At any given moment, an object can only be attached to at most one other object, i.e.
        a parent can have multiple children, but a child can only have one parent.
        Note that generally speaking only child.states[AttachedTo].get_value(parent) will return True.
        One of the child's male meta links will be attached to one of the parent's female meta links.

        Subclasses ContactSubscribedStateMixin, JointBreakSubscribedStateMixin
        on_contact function attempts to attach self.obj to other when a CONTACT_FOUND event happens
        on_joint_break function breaks the current attachment
    """

    @classproperty
    def metalink_prefix(cls):
        """
        Returns:
            str: Unique keyword that defines the metalink associated with this object state
        """
        return m.ATTACHMENT_LINK_PREFIX

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ContactBodies)
        return deps

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

        # Reference to the parent object (DatasetObject)
        self.parent = None

        # Reference to the female meta link of the parent object (RigidPrim)
        self.parent_link = None

        # Mapping from the female meta link names of self.obj to their children (Dict[str, Optional[DatasetObject] = None])
        self.children = {link_name: None for link_name in self.links if link_name.split("_")[1].endswith("F")}

        # Cache of parent link candidates for other objects (Dict[DatasetObject, Dict[str, str]])
        # @other -> (the male meta link names of @self.obj -> the correspounding female meta link names of @other))
        self.parent_link_candidates = dict()

    def on_joint_break(self, joint_prim_path):
        # Note that when this function is invoked when a joint break event happens, @self.obj is the parent of the
        # attachment joint, not the child. We access the child of the broken joint, and call the setter with False
        child = self.children[joint_prim_path.split("/")[-2]]
        child.states[AttachedTo].set_value(self.obj, False)

    # Attempts to attach two objects when a CONTACT_FOUND event happens
    def on_contact(self, other, contact_headers, contact_data):
        for contact_header in contact_headers:
            if contact_header.type == ContactEventType.CONTACT_FOUND:
                # If it has successfully attached to something, break.
                if self.set_value(other, True):
                    break

    def _set_value(self, other, new_value, bypass_alignment_checking=False):
        # Attempt to attach
        if new_value:
            if self.parent == other:
                # Already attached to this object. Do nothing.
                return True
            elif self.parent is None:
                # Find attachment links that satisfy the proximity requirements
                child_link, parent_link = self._find_attachment_links(other, bypass_alignment_checking)
                if child_link is not None:
                    self._attach(other, child_link, parent_link)
                    return True
                else:
                    return False
            else:
                log.debug(f"Trying to attach object {self.obj.name} to object {other.name},"
                          f"but it is already attached to object {self.parent.name}. Try detaching first.")
                return False

        # Attempt to detach
        else:
            if self.parent == other:
                self._detach()

                # Wake up objects so that passive forces like gravity can be applied.
                self.obj.wake()
                other.wake()
            return True

    def _get_value(self, other):
        # Simply return if the current parent matches other
        return other == self.parent

    def _find_attachment_links(self,
                               other,
                               bypass_alignment_checking=False,
                               pos_thresh=m.DEFAULT_POSITION_THRESHOLD,
                               orn_thresh=m.DEFAULT_ORIENTATION_THRESHOLD):
        """
        Args:
            other (DatasetObject): parent object to find potential attachment links.
            bypass_alignment_checking (bool): whether to bypass alignment checking when finding attachment links.
                Normally when finding attachment links, we check if the child and parent links have aligned positions
                or poses. This flag allows users to bypass this check and find attachment links solely based on the
                attachment meta link types. Default is False.
            pos_thresh (float): position difference threshold to activate attachment, in meters.
            orn_thresh (float): orientation difference threshold to activate attachment, in radians.

        Returns:
            2-tuple:
                - RigidPrim or None: link belonging to @self.obj that should be aligned to that corresponding link of @other
                - RigidPrim or None: the corresponding link of @other
        """
        parent_candidates = self._get_parent_candidates(other)
        if not parent_candidates:
            return None, None

        for child_link_name, parent_link_names in parent_candidates.items():
            child_link = self.links[child_link_name]
            for parent_link_name in parent_link_names:
                parent_link = other.states[AttachedTo].links[parent_link_name]
                if other.states[AttachedTo].children[parent_link_name] is None:
                    if bypass_alignment_checking:
                        return child_link, parent_link
                    pos_diff = np.linalg.norm(child_link.get_position() - parent_link.get_position())
                    orn_diff = T.get_orientation_diff_in_radian(child_link.get_orientation(), parent_link.get_orientation())
                    if pos_diff < pos_thresh and orn_diff < orn_thresh:
                        return child_link, parent_link

        return None, None

    def _get_parent_candidates(self, other):
        """
        Helper function to return the parent link candidates for @other

        Returns:
            Dict[str, str] or None: mapping from the male meta link names of self.obj to the correspounding female meta
            link names of @other. Returns None if @other does not have the AttachedTo state.
        """
        if AttachedTo not in other.states:
            return None

        if other not in self.parent_link_candidates:
            parent_link_names = defaultdict(set)
            for child_link_name, child_link in self.links.items():
                child_category = child_link_name.split("_")[1]
                if child_category.endswith("F"):
                    continue
                assert child_category.endswith("M")
                parent_category = child_category[:-1] + "F"
                for parent_link_name, parent_link in other.states[AttachedTo].links.items():
                    if parent_category in parent_link_name:
                        parent_link_names[child_link_name].add(parent_link_name)
            self.parent_link_candidates[other] = parent_link_names

        return self.parent_link_candidates[other]

    @property
    def attachment_joint_prim_path(self):
        return f"{self.parent_link.prim_path}/{self.obj.name}_attachment_joint" if self.parent_link is not None else None

    def _attach(self, other, child_link, parent_link, joint_type=m.DEFAULT_JOINT_TYPE, break_force=m.DEFAULT_BREAK_FORCE, break_torque=m.DEFAULT_BREAK_TORQUE):
        """
        Creates a fixed or spherical joint between a male meta link of self.obj (@child_link) and a female meta link of
         @other (@parent_link) with a given @joint_type, @break_force and @break_torque

         Args:
            other (DatasetObject): parent object to attach to.
            child_link (RigidPrim): male meta link of @self.obj.
            parent_link (RigidPrim): female meta link of @other.
            joint_type (JointType): joint type of the attachment, {JointType.JOINT_FIXED, JointType.JOINT_SPHERICAL}
            break_force (float or None): break force for linear dofs, unit is Newton.
            break_torque (float or None): break torque for angular dofs, unit is Newton-meter.
        """
        assert joint_type in {JointType.JOINT_FIXED, JointType.JOINT_SPHERICAL}, f"Unsupported joint type {joint_type}"
        # Set the parent references
        self.parent = other
        self.parent_link = parent_link

        # Set the child reference for @other
        other.states[AttachedTo].children[parent_link.body_name] = self.obj

        # Set pose for self.obj so that child_link and parent_link align (6dof alignment for FixedJoint and 3dof alignment for SphericalJoint)
        parent_pos, parent_quat = parent_link.get_position_orientation()
        child_pos, child_quat = child_link.get_position_orientation()

        child_root_pos, child_root_quat = self.obj.get_position_orientation()

        if joint_type == JointType.JOINT_FIXED:
            # For FixedJoint: find the relation transformation of the two frames and apply it to self.obj.
            rel_pos, rel_quat = T.mat2pose(T.pose2mat((parent_pos, parent_quat)) @ T.pose_inv(T.pose2mat((child_pos, child_quat))))
            new_child_root_pos, new_child_root_quat = T.pose_transform(rel_pos, rel_quat, child_root_pos, child_root_quat)
        else:
            # For SphericalJoint: move the position of self.obj to align the two frames and keep the rotation unchanged.
            new_child_root_pos = child_root_pos + (parent_pos - child_pos)
            new_child_root_quat = child_root_quat

        # Actually move the object and also keep it still for stability purposes.
        self.obj.set_position_orientation(new_child_root_pos, new_child_root_quat)
        self.obj.keep_still()

        if joint_type == JointType.JOINT_FIXED:
            # FixedJoint: the parent link, the child link and the joint frame all align.
            parent_local_quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # SphericalJoint: the same except that the rotation of the parent link doesn't align with the joint frame.
            # The child link and the joint frame still align.
            _, parent_local_quat = T.relative_pose_transform([0, 0, 0], child_quat, [0, 0, 0], parent_quat)

        # Create the joint
        create_joint(
            prim_path=self.attachment_joint_prim_path,
            joint_type=joint_type,
            body0=f"{parent_link.prim_path}",
            body1=f"{child_link.prim_path}",
            joint_frame_in_parent_frame_pos=np.zeros(3),
            joint_frame_in_parent_frame_quat=parent_local_quat,
            joint_frame_in_child_frame_pos=np.zeros(3),
            joint_frame_in_child_frame_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            break_force=break_force,
            break_torque=break_torque,
        )

    def _detach(self):
        """
        Removes the current attachment joint
        """
        # Remove the attachment joint prim from the stage
        og.sim.stage.RemovePrim(self.attachment_joint_prim_path)

        # Remove child reference from the parent object
        self.parent.states[AttachedTo].children[self.parent_link.body_name] = None

        # Remove reference to the parent object and link
        self.parent = None
        self.parent_link = None

    @property
    def settable(self):
        return True

    @property
    def state_size(self):
        return 1

    def _dump_state(self):
        return dict(attached_obj_uuid=-1 if self.parent is None else self.parent.uuid)

    def _load_state(self, state):
        uuid = state["attached_obj_uuid"]
        if uuid == -1:
            attached_obj = None
        else:
            attached_obj = og.sim.scene.object_registry("uuid", uuid)
            assert attached_obj is not None, "attached_obj_uuid does not match any object in the scene."

        # If it's currently attached to something, detach.
        if self.parent is not None:
            self.set_value(self.parent, False)
            assert self.parent is None, "parent reference is not cleared after detachment"

        # If the loaded state requires attachment, attach.
        if attached_obj is not None:
            self.set_value(attached_obj, True)
            assert self.parent == attached_obj, "parent reference is not updated after attachment"

    def _serialize(self, state):
        return np.array([state["attached_obj_uuid"]], dtype=float)

    def _deserialize(self, state):
        return dict(attached_obj_uuid=int(state[0])), 1
