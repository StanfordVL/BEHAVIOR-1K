from abc import abstractmethod

from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.utils.python_utils import classproperty


class ContactSubscribedStateMixin(BaseObjectState):
    """
    Handles contact events (including CONTACT_FOUND, CONTACT_PERSIST, and CONTACT_LOST).
    The subclass should implement its own on_contact method
    """

    @abstractmethod
    def on_contact(self, other, contact_headers, contact_data):
        raise NotImplementedError("Subclasses of ContactSubscribedStateMixin should implement the on_contact method.")

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ContactSubscribedStateMixin")
        return classes
