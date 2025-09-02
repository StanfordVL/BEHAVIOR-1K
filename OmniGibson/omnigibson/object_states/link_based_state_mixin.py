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

        # Check whether this state requires meta link
        if not cls.requires_meta_link(**kwargs):
            return True, None
        meta_link_types = cls.meta_link_types
        for link in obj.links.values():
            if link.is_meta_link and link.meta_link_type in meta_link_types:
                return True, None

        return (
            False,
            f"LinkBasedStateMixin {cls.__name__} requires meta link with type {cls.meta_link_types} "
            f"for obj {obj.name} but none was found! To get valid compatible object models, please use "
            f"omnigibson.utils.asset_utils.get_all_object_category_models_with_abilities(...)",
        )

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible_asset(prim, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state requires meta link
        if not cls.requires_meta_link(**kwargs):
            return True, None
        meta_link_types = cls.meta_link_types
        for child in prim.GetChildren():
            if child.GetTypeName() == "Xform":
                # With the new format, we can know for sure by checking the meta link type
                if (
                    child.HasAttribute("ig:metaLinkType")
                    and child.GetAttribute("ig:metaLinkType").Get() in meta_link_types
                ):
                    return True, None

                # Until the next dataset release, also accept the old format
                # TODO: Remove this block after the next dataset release
                if any(meta_link_type in child.GetName() for meta_link_type in meta_link_types):
                    return True, None
        return (
            False,
            f"LinkBasedStateMixin {cls.__name__} requires meta link with prefix {cls.meta_link_types} "
            f"for asset prim {prim.GetName()} but none was found! To get valid compatible object models, "
            f"please use omnigibson.utils.asset_utils.get_all_object_category_models_with_abilities(...)",
        )

    @classproperty
    def meta_link_types(cls):
        """
        Returns:
            str: Unique keyword that defines the meta link associated with this object state
        """
        NotImplementedError()

    @classmethod
    def requires_meta_link(cls, **kwargs):
        """
        Returns:
            bool: Whether an object state instantiated with constructor arguments **kwargs will require a meta link
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
            dict: mapping from link names to links that match the meta_link_type
        """
        return self._links

    @property
    def _default_link(self):
        """
        Returns:
            None or RigidPrim: If supported, the fallback link associated with this link-based state if
                no valid meta link is found
        """
        # No default link by default
        return None

    def initialize_link_mixin(self):
        assert not self._initialized

        # TODO: Extend logic to account for multiple instances of the same meta link? e.g: _0, _1, ... suffixes
        for name, link in self.obj.links.items():
            is_appropriate_meta_link = link.is_meta_link and link.meta_link_type in self.meta_link_types
            # TODO: Remove support for this default meta link logic after the next dataset release
            is_default_link = self._default_link is not None and link.name == self._default_link.name
            if is_appropriate_meta_link or is_default_link:
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
