import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states import ContactBodies
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.SLICER_LINK_NAME = "slicer_link"


class Slicer(AbsoluteObjectState, LinkBasedStateMixin):
    def __init__(self, obj):
        super(Slicer, self).__init__(obj)

    @staticmethod
    def get_state_link_name():
        return m.SLICER_LINK_NAME

    def _initialize(self):
        self.initialize_link_mixin()

    def _set_value(self, new_value):
        raise ValueError("set_value not supported for valueless states like Slicer.")

    def _get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [ContactBodies]
