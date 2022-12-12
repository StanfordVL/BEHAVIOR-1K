from abc import ABC, abstractmethod


class ContactSubscribedStateMixin(ABC):
    """
    Handles contact events (including CONTACT_FOUND, CONTACT_PERSIST, and CONTACT_LOST).
    The subclass should implement its own on_contact method
    """
    @abstractmethod
    def on_contact(self, other, contact_header, contact_data):
        raise NotImplementedError("Subclasses of ContactSubscribedStateMixin should implement the on_contact method.")
