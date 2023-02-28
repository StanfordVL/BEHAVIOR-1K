import numpy as np
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class LinkBasedStateMixin:
    def __init__(self):
        super().__init__()

        self.link = None

    @staticmethod
    def get_state_link_name():
        raise ValueError("LinkBasedState child should specify link name by overriding get_state_link_name.")

    def initialize_link_mixin(self):
        assert not self._initialized

        self.link = None

        try:
            self.link = self.obj.links[self.get_state_link_name()]
        except KeyError:
            # Metalink not found, so we failed to initialize, so we assume this is a "dead" linkbasedstatemixin
            log.warning(f"Warning: failed to initialize LinkBasedStateMixin {self.__class__.__name__} for object {self.obj.name}, no metalink"
                            f"with name {self.get_state_link_name()} found!")
            return False

        return True

    def get_link_position(self):
        # The necessary link is not found
        if self.link is None:
            return None

        pos, _ = self.link.get_position_orientation()
        return np.array(pos)