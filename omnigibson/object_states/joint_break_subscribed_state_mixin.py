from abc import abstractmethod

from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.utils.python_utils import classproperty


class JointBreakSubscribedStateMixin(BaseObjectState):
    """
    Handles JOINT_BREAK event.
    The subclass should implement its own on_joint_break method
    """

    @abstractmethod
    def on_joint_break(self, joint_prim_path):
        raise NotImplementedError(
            "Subclasses of JointBreakSubscribedStateMixin should implement the on_joint_break method."
        )

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("JointBreakSubscribedStateMixin")
        return classes
