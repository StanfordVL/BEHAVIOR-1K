import numpy as np
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import classproperty

# Create module logger
log = create_module_logger(module_name=__name__)


class LinkBasedStateMixin:
    def __init__(self):
        super().__init__()

        self._links = dict()

    @classproperty
    def metalink_prefix(cls):
        """
        Returns:
            str: Unique keyword that defines the metalink associated with this object state
        """
        NotImplementedError()

    @classmethod
    def requires_metalink(cls, *args, **kwargs):
        """
        Returns:
            Whether an object state instantiated with constructor arguments *args and **kwargs will require a metalink
                or not
        """
        # True by default
        return True

    @property
    def link(self):
        """
        Returns:
            None or RigidPrim: The link associated with this link-based state, if it exists
        """
        return next(iter(self.links.values()))

    @property
    def links(self):
        """
        Returns:
            dict: mapping from link names to links that match the metalink_prefix
        """
        # Raise an error if we did not find a valid link
        if len(self._links) == 0:
            raise ValueError(f"Error: failed to query LinkBasedStateMixin {self.__class__.__name__} for object "
                             f"{self.obj.name}; no metalink with prefix {self.metalink_prefix} found! "
                             f"Please use get_all_object_category_models_with_abilities(...) from "
                             f"omnigibson.utils.asset_utils to grab models with properly annotated metalinks.")
        return self._links

    @property
    def _default_link(self):
        """
        Returns:
            None or RigidPrim: If supported, the fallback link associated with this link-based state if
                no valid metalink is found
        """
        # No default link by default
        # TODO: Make NotImplementedError() and force downstream Object states to implement, once
        # assets are fully updated
        return self.obj.root_link

    def initialize_link_mixin(self):
        assert not self._initialized

        # TODO: Extend logic to account for multiple instances of the same metalink? e.g: _0, _1, ... suffixes
        for name, link in self.obj.links.items():
            if self.metalink_prefix in name or (self._default_link is not None and link.name == self._default_link.name):
                self._links[name] = link
                assert np.allclose(link.scale, self.obj.scale), \
                    f"the meta link {name} has a inconsistent scale with the object {self.obj.name}"
