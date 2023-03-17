from pxr import Gf
from omni.physx.bindings._physx import ContactEventType

from enum import IntEnum
import numpy as np

import omnigibson as og
from omnigibson.macros import create_module_macros, gm
import omnigibson.utils.transform_utils as T
from omnigibson.object_states.contact_subscribed_state_mixin import ContactSubscribedStateMixin
from omnigibson.object_states.object_state_base import BooleanState, RelativeObjectState
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.utils.usd_utils import create_joint
from omnigibson.utils.sim_utils import check_collision
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log

# Create module logger
log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)


class AttachmentType(IntEnum):
    STICKY = 0
    SYMMETRIC = 1
    MALE = 2
    FEMALE = 3


class Attached(RelativeObjectState, BooleanState, ContactSubscribedStateMixin):
    """
        Handles attachment between two rigid objects, by creating a fixed joint between self (child) and other (parent)
        At any given moment, an object can only be attached to at most one other object.
        There are three types of attachment:
            STICKY: unidirectional, attach if in contact.
            SYMMETRIC: bidirectional, attach if in contact AND the other object has SYMMETRIC type
                with the same attachment_category (e.g. "magnet").
            MALE/FEMALE: bidirectional, attach if in contact AND the other object has the opposite end (male / female)
                with the same attachment_category (e.g. "usb")

        Subclasses ContactSubscribedStateMixin
        on_contact_found function attempts to attach two objects when a CONTACT_FOUND event happens
    """

    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies() + [ContactBodies]

    def __init__(self, obj, attachment_type=AttachmentType.STICKY, attachment_category=None):
        super(Attached, self).__init__(obj)
        self.attachment_type = attachment_type
        self.attachment_category = attachment_category

        self.attached_obj = None

    # Attempts to attach two objects when a CONTACT_FOUND event happens
    def on_contact(self, other, contact_headers, contact_data):
        for contact_header in contact_headers:
            if contact_header.type == ContactEventType.CONTACT_FOUND:
                self.set_value(other, True, check_contact=False)
                break

    def _set_value(self, other, new_value, check_contact=True):
        # Attempt to attach
        if new_value:
            if self.attached_obj == other:
                # Already attached to this object. Do nothing.
                return True
            elif self.attached_obj is None:
                # If the attachment type and category match, and they are in contact, they should attach
                if self._can_attach(other) and (not check_contact or check_collision(self.obj, other)):
                    self._attach(other)

                    # Non-sticky attachment is bidirectional
                    if self.attachment_type != AttachmentType.STICKY:
                        other.states[Attached].attached_obj = self.obj

                    return True
                else:
                    log.debug(f"Trying to attach object {self.obj.name} to object {other.name}, "
                                    f"but they have attachment type/category mismatch or they are not in contact.")
                    return False
            else:
                log.debug(f"Trying to attach object {self.obj.name} to object {other.name}, "
                                f"but it is already attached to object {self.attached_obj.name}. Try detaching first.")
                return False

        # Attempt to detach
        else:
            if self.attached_obj == other:
                self._detach()

                # Non-sticky attachment is bidirectional
                if self.attachment_type != AttachmentType.STICKY:
                    other.states[Attached].attached_obj = None

                # Wake up objects so that passive forces like gravity can be applied.
                self.obj.wake()
                other.wake()

            return True

    def _get_value(self, other):
        return other == self.attached_obj

    def _can_attach(self, other):
        """
        Returns:
            bool: True if self is sticky or self and other have matching attachment type and category
        """
        if self.attachment_type == AttachmentType.STICKY:
            return True
        else:
            if Attached not in other.states or other.states[Attached].attachment_category != self.attachment_category:
                return False
            elif self.attachment_type == AttachmentType.SYMMETRIC:
                return other.states[Attached].attachment_type == AttachmentType.SYMMETRIC
            elif self.attachment_type == AttachmentType.MALE:
                return other.states[Attached].attachment_type == AttachmentType.FEMALE
            else:
                return other.states[Attached].attachment_type == AttachmentType.MALE

    def _attach(self, other):
        """
        Creates a fixed joint between self.obj and other (where other is the parent and self.obj is the child)
        """
        self.attached_obj = other
        attached_joint = create_joint(
            prim_path=f"{other.prim_path}/attachment_joint",
            joint_type="FixedJoint",
            body0=f"{other.prim_path}/base_link",
            body1=f"{self.obj.prim_path}/base_link",
        )

        # Set the local pose of the attachment joint.
        parent_pos, parent_quat = other.get_position_orientation()
        child_pos, child_quat = self.obj.get_position_orientation()

        # The child frame aligns with the joint frame.
        # Compute the transformation of the child frame in the parent frame
        rel_pos, rel_quat = T.relative_pose_transform(child_pos, child_quat, parent_pos, parent_quat)
        # The child frame position in the parent frame needs to be divided by the parent's scale
        rel_pos /= other.scale
        rel_quat = rel_quat[[3, 0, 1, 2]]

        attached_joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*rel_pos))
        attached_joint.GetAttribute("physics:localRot0").Set(Gf.Quatf(*rel_quat))
        attached_joint.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0.0, 0.0, 0.0))
        attached_joint.GetAttribute("physics:localRot1").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        # We update the simulation now without actually stepping physics so we can bypass the snapping warning from
        # PhysicsUSD
        with suppress_omni_log(channels=["omni.physx.plugin"]):
            og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)

    def _detach(self):
        """
        Removes the attachment joint
        """
        attached_joint_path = f"{self.attached_obj.prim_path}/attachment_joint"
        self._simulator.stage.RemovePrim(attached_joint_path)
        self.attached_obj = None

    @property
    def settable(self):
        return True

    @property
    def state_size(self):
        return 1

    def _dump_state(self):
        return dict(attached_obj_uuid=-1 if self.attached_obj is None else self.attached_obj.uuid)

    def _load_state(self, state):
        uuid = state["attached_obj_uuid"]
        attached_obj = None if uuid == -1 else og.sim.scene.object_registry("uuid", uuid)

        if self.attached_obj != attached_obj:
            # If it's currently attached to something, detach.
            if self.attached_obj is not None:
                self._detach()

            # If the loaded state requires new attachment, attach.
            if attached_obj is not None:
                self._attach(attached_obj)

    def _serialize(self, state):
        return np.array([state["attached_obj_uuid"]], dtype=float)

    def _deserialize(self, state):
        return dict(attached_obj_uuid=int(state[0])), 1
