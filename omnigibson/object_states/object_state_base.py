from abc import ABC, abstractmethod
from collections import OrderedDict
from omnigibson.utils.python_utils import classproperty, Serializable, Registerable, Recreatable


# Global dicts that will contain mappings
REGISTERED_OBJECT_STATES = OrderedDict()


class BaseObjectState(Serializable, Registerable, Recreatable, ABC):
    """
    Base ObjectState class. Do NOT inherit from this class directly - use either AbsoluteObjectState or
    RelativeObjectState.
    """

    @staticmethod
    def get_dependencies():
        """
        Get the dependency states for this state, e.g. states that need to be explicitly enabled on the current object
        before the current state is usable. States listed here will be enabled for all objects that have this current
        state, and all dependency states will be processed on *all* objects prior to this state being processed on
        *any* object.

        :return: List of strings corresponding to state keys.
        """
        return []

    @staticmethod
    def get_optional_dependencies():
        """
        Get states that should be processed prior to this state if they are already enabled. These states will not be
        enabled because of this state's dependency on them, but if they are already enabled for another reason (e.g.
        because of an ability or another state's dependency etc.), they will be processed on *all* objects prior to this
        state being processed on *any* object.

        :return: List of strings corresponding to state keys.
        """
        return []

    def __init__(self, obj):
        super(BaseObjectState, self).__init__()
        self.obj = obj
        self._initialized = False
        self._cache = None
        self._simulator = None

    @classproperty
    def stateful(cls):
        """
        Returns:
            bool: True if this object has a state that can be directly dumped / loaded via dump_state() and
                load_state(), otherwise, returns False. Note that any sub object states that are NOT stateful do
                not need to implement any of _dump_state(), _load_state(), _serialize(), or _deserialize()!
        """
        # False by default
        return False

    @property
    def cache(self):
        """
        Returns:
            OrdereDict: Dictionary mapping specific argument combinations from @self.get_value() to cached values and
                information stored for that specific combination
        """
        return self._cache

    def _update(self):
        """This function will be called once for every simulator step."""
        pass

    def _initialize(self):
        """This function will be called once; should be used for any object state-related objects have been loaded."""
        pass

    def initialize(self, simulator):
        assert not self._initialized, "State is already initialized."

        # Store simulator reference and create cache
        self._simulator = simulator
        self._cache = OrderedDict()

        self._initialize()
        self._initialized = True

    def update(self):
        assert self._initialized, "Cannot update uninitialized state."
        # Potentially clear cache
        self.clear_cache(force=False)
        return self._update()

    def clear_cache(self, force=True):
        """
        Clears the internal cache, either softly (checking under certain conditions under which the cache will not
        be cleared), or forcefully (if @force=True)

        Args:
            force (bool): Whether to force a clearing of cached values or to potentially check whether they should
                be cleared or not
        """
        if force:
            # Clear all entries
            self._cache = OrderedDict()
        else:
            for args in list(self._cache.keys()):
                # Check whether we should clear the cache
                if self._should_clear_cache(get_value_args=args, cache_info=self._cache[args]["info"]):
                    self._cache.pop(args)

    def _cache_info(self, get_value_args):
        """
        Helper function to cache relevant information at the current timestep.
        Stores it under @self._cache[<KEY>]["value"]

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value whose caching information should be computed

        Returns:
            OrderedDict: Any caching information to include at the current timestep when this state's value is computed
        """
        # Default is an empty dictionary
        return OrderedDict()

    def _should_clear_cache(self, get_value_args, cache_info):
        """
        Checks whether the cache should be cleared based on information from @cache_info

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value whose caching condition should be checked
            cache_info (OrderedDict): Cache information associated the argument tuple @key

        Returns:
            bool: Whether the cache should be cleared for specific combination @get_value_args
        """
        # Default is always True
        return True

    def get_value(self, *args, **kwargs):
        assert self._initialized

        # Compile args and kwargs deterministically, and if it already exists in our cache, we return that value,
        # otherwise we calculate the value and store it in our cache
        key = (*args, *tuple(kwargs.values()))
        if key in self._cache:
            val = self._cache[key]["value"]
        else:
            val = self._get_value(*args, **kwargs)
            self._cache[key] = OrderedDict(value=val, info=self._cache_info(get_value_args=key))

        return val

    def _get_value(self, *args, **kwargs):
        raise NotImplementedError

    def set_value(self, *args, **kwargs):
        assert self._initialized
        return self._set_value(*args, **kwargs)

    def _set_value(self, *args, **kwargs):
        raise NotImplementedError

    def remove(self):
        """
        Any cleanup functionality to deploy when @self.obj is removed from the simulator
        """
        pass

    def dump_state(self, serialized=False):
        assert self._initialized
        assert self.stateful
        return super().dump_state(serialized=serialized)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseObjectState")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_OBJECT_STATES
        return REGISTERED_OBJECT_STATES


class AbsoluteObjectState(BaseObjectState):
    """
    This class is used to track object states that are absolute, e.g. do not require a second object to compute
    the value.
    """

    @abstractmethod
    def _get_value(self):
        raise NotImplementedError()

    @abstractmethod
    def _set_value(self, new_value):
        raise NotImplementedError()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("AbsoluteObjectState")
        return classes


class RelativeObjectState(BaseObjectState):
    """
    This class is used to track object states that are relative, e.g. require two objects to compute a value.
    Note that subclasses will typically compute values on-the-fly.
    """

    @abstractmethod
    def _get_value(self, other):
        raise NotImplementedError()

    @abstractmethod
    def _set_value(self, other, new_value):
        raise NotImplementedError()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("RelativeObjectState")
        return classes


class BooleanState:
    """
    This class is a mixin used to indicate that a state has a boolean value.
    """

    pass
