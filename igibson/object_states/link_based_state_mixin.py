import numpy as np
import logging

class LinkBasedStateMixin(object):
    def __init__(self):
        super(LinkBasedStateMixin, self).__init__()

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
            logging.warning(f"Warning: failed to initialize LinkBasedStateMixin {self.__class__.__name__} for object {self.obj.name}, no metalink"
                            f"with name {self.get_state_link_name()} found!")
            return False

        return True

    def get_link_position(self):
        # The necessary link is not found
        if self.link is None:
            return

        pos, orn = self.link.get_position_orientation()
        return np.array(pos)