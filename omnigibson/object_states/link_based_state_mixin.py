import numpy as np
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import classproperty

# Create module logger
log = create_module_logger(module_name=__name__)


class LinkBasedStateMixin:
    def __init__(self):
        super().__init__()

        self._link = None

    @classproperty
    def metalink_prefix(cls):
        """
        Returns:
            str: Unique keyword that defines the metalink associated with this object state
        """
        NotImplementedError()

    @property
    def link(self):
        """
        Returns:
            None or RigidPrim: The link associated with this link-based state, if it exists
        """
        return self._default_link if self._link is None else self._link

    @property
    def _default_link(self):
        """
        Returns:
            None or RigidPrim: If supported, the fallback link associated with this link-based state if
                no valid metalink is found
        """
        # No default link by default
        # TODO: Make this None to enforce that downstream object states have properly annotated objects
        return self.obj.root_link

    def initialize_link_mixin(self):
        assert not self._initialized

        self._link = None
        # TODO: Extend logic to account for multiple instances of the same metalink? e.g: _0, _1, ... suffixes
        for name, link in self.obj.links.items():
            if self.metalink_prefix in name:
                self._link = link
                break

        # Raise an error if we did not find a valid link
        # Note that we check the public accessor for self.link because a subclass might implement a fallback
        # default_link to use
        if self.link is None:
            raise ValueError(f"Error: failed to initialize LinkBasedStateMixin {self.__class__.__name__} for object "
                             f"{self.obj.name}; no metalink with prefix {self.metalink_prefix} found!")
        return True
