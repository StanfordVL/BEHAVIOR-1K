from omnigibson.macros import create_module_macros
from omnigibson.object_states import ContactBodies
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.python_utils import classproperty


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.SLICER_LINK_PREFIX = "slicer"


class Slicer(AbsoluteObjectState, LinkBasedStateMixin):
    def __init__(self, obj):
        super(Slicer, self).__init__(obj)

    @classproperty
    def metalink_prefix(cls):
        return m.SLICER_LINK_PREFIX

    @property
    def _default_link(self):
        return self.obj.root_link

    def _initialize(self):
        self.initialize_link_mixin()

    def _set_value(self, new_value):
        raise ValueError("set_value not supported for valueless states like Slicer.")

    def _get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [ContactBodies]
