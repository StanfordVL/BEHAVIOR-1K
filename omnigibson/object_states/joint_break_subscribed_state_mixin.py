from abc import ABC, abstractmethod


class JointBreakSubscribedStateMixin(ABC):
    """
    Handles JOINT_BREAK event.
    The subclass should implement its own on_joint_break method
    """

    @abstractmethod
    def on_joint_break(self, joint_prim_path):
        raise NotImplementedError("Subclasses of JointBreakSubscribedStateMixin should implement the on_joint_break method.")
