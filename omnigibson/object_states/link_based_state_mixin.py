import torch as th

from omnigibson.object_states.object_state_base import BaseObjectState
from omnigibson.prims.cloth_prim import ClothPrim
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


class LinkBasedStateMixin(BaseObjectState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._links = dict()

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state requires metalink
        if not cls.requires_metalink(**kwargs):
            return True, None
        metalink_prefix = cls.metalink_prefix
        for link in obj.links.values():
            if metalink_prefix in link.name:
                return True, None

        return (
            False,
            f"LinkBasedStateMixin {cls.__name__} requires metalink with prefix {cls.metalink_prefix} "
            f"for obj {obj.name} but none was found! To get valid compatible object models, please use "
            f"omnigibson.utils.asset_utils.get_all_object_category_models_with_abilities(...)",
        )

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible_asset(prim, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state requires metalink
        if not cls.requires_metalink(**kwargs):
            return True, None
        metalink_prefix = cls.metalink_prefix
        for child in prim.GetChildren():
            if child.GetTypeName() == "Xform":
                if metalink_prefix in child.GetName():
                    return True, None

        return (
            False,
            f"LinkBasedStateMixin {cls.__name__} requires metalink with prefix {cls.metalink_prefix} "
            f"for asset prim {prim.GetName()} but none was found! To get valid compatible object models, "
            f"please use omnigibson.utils.asset_utils.get_all_object_category_models_with_abilities(...)",
        )

    @classproperty
    def metalink_prefix(cls):
        """
        Returns:
            str: Unique keyword that defines the metalink associated with this object state
        """
        NotImplementedError()

    @classmethod
    def requires_metalink(cls, **kwargs):
        """
        Returns:
            Whether an object state instantiated with constructor arguments **kwargs will require a metalink
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
        assert self.links, f"LinkBasedStateMixin link not found for {self.obj.name}"
        return next(iter(self.links.values()))

    @property
    def links(self):
        """
        Returns:
            dict: mapping from link names to links that match the metalink_prefix
        """
        return self._links

    @property
    def _default_link(self):
        """
        Returns:
            None or RigidPrim: If supported, the fallback link associated with this link-based state if
                no valid metalink is found
        """
        # No default link by default
        return None

    def initialize_link_mixin(self):
        assert not self._initialized

        # TODO: Extend logic to account for multiple instances of the same metalink? e.g: _0, _1, ... suffixes
        for name, link in self.obj.links.items():
            if self.metalink_prefix in name or (
                self._default_link is not None and link.name == self._default_link.name
            ):
                self._links[name] = link
                # Make sure the scale is similar if the link is not a cloth prim
                if not isinstance(link, ClothPrim):
                    assert th.allclose(
                        link.scale, self.obj.scale
                    ), f"the meta link {name} has a inconsistent scale with the object {self.obj.name}"

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("LinkBasedStateMixin")
        return classes
