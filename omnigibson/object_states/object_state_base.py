import inspect
from abc import ABC

import omnigibson as og
from omnigibson.utils.python_utils import Recreatable, Registerable, Serializable, classproperty

# Global dicts that will contain mappings
REGISTERED_OBJECT_STATES = dict()


class BaseObjectRequirement:
    """
    Base ObjectRequirement class. This allows for sanity checking a given asset / BaseObject to check whether a set
    of conditions are met or not. This can be useful for sanity checking dependencies for properties such as requested
    abilities or object states.
    """

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        """
        Determines whether this requirement is compatible with object @obj or not (i.e.: whether this requirement is
        satisfied by @obj given other constructor arguments **kwargs).

        NOTE: Must be implemented by subclass.

        Args:
            obj (StatefulObject): Object whose compatibility with this state should be checked

        Returns:
            2-tuple:
                - bool: Whether the given object is compatible with this requirement or not
                - None or str: If not compatible, the reason why it is not compatible. Otherwise, None
        """
        raise NotImplementedError

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        """
        Determines whether this requirement is compatible with prim @prim or not (i.e.: whether this requirement is
        satisfied by @prim given other constructor arguments **kwargs).
        This is a useful check to evaluate an object's USD that hasn't been explicitly imported into OmniGibson yet.

        NOTE: Must be implemented by subclass

        Args:
            prim (Usd.Prim): Object prim whose compatibility with this requirement should be checked

        Returns:
            2-tuple:
                - bool: Whether the given prim is compatible with this requirement or not
                - None or str: If not compatible, the reason why it is not compatible. Otherwise, None
        """
        raise NotImplementedError


class BaseObjectState(BaseObjectRequirement, Serializable, Registerable, Recreatable, ABC):
    """
    Base ObjectState class. Do NOT inherit from this class directly - use either AbsoluteObjectState or
    RelativeObjectState.
    """

    @classmethod
    def get_dependencies(cls):
        """
        Get the dependency states for this state, e.g. states that need to be explicitly enabled on the current object
        before the current state is usable. States listed here will be enabled for all objects that have this current
        state, and all dependency states will be processed on *all* objects prior to this state being processed on
        *any* object.

        Returns:
            set of str: Set of strings corresponding to state keys.
        """
        return set()

    @classmethod
    def get_optional_dependencies(cls):
        """
        Get states that should be processed prior to this state if they are already enabled. These states will not be
        enabled because of this state's dependency on them, but if they are already enabled for another reason (e.g.
        because of an ability or another state's dependency etc.), they will be processed on *all* objects prior to this
        state being processed on *any* object.

        Returns:
            set of str: Set of strings corresponding to state keys.
        """
        return set()

    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self._initialized = False
        self._cache = None
        self._changed = None
        self._last_t_updated = -1  # Last timestep when this state was updated

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Make sure all required dependencies are included in this object's state dictionary
        for dep in cls.get_dependencies():
            if dep not in obj.states:
                return False, f"Missing required dependency state {dep.__name__}"
        # Make sure all required kwargs are specified
        default_kwargs = inspect.signature(cls.__init__).parameters
        for kwarg, val in default_kwargs.items():
            if val.default == inspect._empty and kwarg not in kwargs and kwarg not in {"obj", "self", "args", "kwargs"}:
                return False, f"Missing required kwarg '{kwarg}'"
        # Default is True if all kwargs are met
        return True, None

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Make sure all required kwargs are specified
        default_kwargs = inspect.signature(cls.__init__).parameters
        for kwarg, val in default_kwargs.items():
            if val.default == inspect._empty and kwarg not in kwargs and kwarg not in {"obj", "self"}:
                return False, f"Missing required kwarg '{kwarg}'"
        # Default is True if all kwargs are met
        return True, None

    @classmethod
    def postprocess_ability_params(cls, params, scene):
        """
        Post-processes ability parameters if needed. The default implementation is a simple passthrough.
        """
        return params

    @property
    def stateful(self):
        """
        Returns:
            bool: True if this object has a state that can be directly dumped / loaded via dump_state() and
                load_state(), otherwise, returns False. Note that any sub object states that are NOT stateful do
                not need to implement any of _dump_state(), _load_state(), _serialize(), or _deserialize()!
        """
        # Default is whether state size > 0
        return self.state_size > 0

    @property
    def state_size(self):
        return 0

    @property
    def cache(self):
        """
        Returns:
            dict: Dictionary mapping specific argument combinations from @self.get_value() to cached values and
                information stored for that specific combination
        """
        return self._cache

    def _initialize(self):
        """
        This function will be called once; should be used for any object state-related objects have been loaded.
        """
        pass

    def initialize(self):
        """
        Initialize this object state
        """
        assert not self._initialized, "State is already initialized."

        # Validate compatibility with the created object
        init_args = {k: v for k, v in self.get_init_info()["args"].items() if k != "obj"}
        assert self.is_compatible(
            obj=self.obj, **init_args
        ), f"ObjectState {self.__class__.__name__} is not compatible with object {self.obj.name}."

        # Clear cache
        self.clear_cache()

        self._initialize()

        self._initialized = True

    def clear_cache(self):
        """
        Clears the internal cache
        """
        # Clear all entries
        self._cache = dict()
        self._changed = dict()
        self._last_t_updated = -1

    def update_cache(self, get_value_args):
        """
        Updates the internal cached value based on the evaluation of @self._get_value(*get_value_args)

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value / @self._get_value
        """
        t = og.sim.current_time_step_index
        # Compute value and update cache
        val = self._get_value(*get_value_args)
        self._cache[get_value_args] = dict(value=val, info=self.cache_info(get_value_args=get_value_args), t=t)

    def cache_info(self, get_value_args):
        """
        Helper function to cache relevant information at the current timestep.
        Stores it under @self._cache [<KEY>]["info"]

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value whose caching information should be computed

        Returns:
            dict: Any caching information to include at the current timestep when this state's value is computed
        """
        # Default is an empty dictionary
        return dict()

    def cache_is_valid(self, get_value_args):
        """
        Helper function to check whether the current cached value is valid or not at the current timestep.
        Default is False unless we're at the current timestep.

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value whose cached values should be validated

        Returns:
            bool: True if the cache is valid, else False
        """
        # If t == the current timestep, then our cache is obviously valid otherwise we assume it isn't
        return (
            True
            if self._cache[get_value_args]["t"] == og.sim.current_time_step_index
            else self._cache_is_valid(get_value_args=get_value_args)
        )

    def _cache_is_valid(self, get_value_args):
        """
        Helper function to check whether the current cached value is valid or not at the current timestep.
        Default is False. Subclasses should implement special logic otherwise.

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value whose cached values should be validated

        Returns:
            bool: True if the cache is valid, else False
        """
        return False

    def has_changed(self, get_value_args, value, info, t):
        """
        A helper function to query whether this object state has changed between the current timestep and an arbitrary
        previous timestep @t with the corresponding cached value @value and cache information @info

        Note that this may require some non-trivial compute, so we leverage @t, in addition to @get_value_args,
        as a unique key into an internal dictionary, such that specific @t will result in a computation conducted
        exactly once.
        This is done for performance reasons; so that multiple states relying on the same state dependency can all
        query whether that state has changed between the same timesteps with only a single computation.

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value
            value (any): Cached value computed at timestep @t for this object state
            info (dict): Information calculated at timestep @t when computing this state's value
            t (int): Initial timestep to compare against. This should be an index of the steps taken,
                i.e. a value queried from og.sim.current_time_step_index at some point in time. It is assumed @value
                and @info were computed at this timestep

        Returns:
            bool: Whether this object state has changed between @t and the current timestep index for the specific
                @get_value_args
        """
        # Check current sim step index; if it doesn't match the internal value, we need to clear the changed history
        current_t = og.sim.current_time_step_index
        if self._last_t_updated != current_t:
            self._changed = dict()
            self._last_t_updated = current_t
        # Compile t, args, and kwargs deterministically
        history_key = (t, *get_value_args)
        # If t == the current timestep, then we obviously haven't changed so our value is False
        if t == current_t:
            val = False
        # Otherwise, check if it already exists in our has changed dictionary; we return that value if so
        elif history_key in self._changed:
            val = self._changed[history_key]
        # Otherwise, we calculate the value and store it in our changed dictionary
        else:
            val = self._has_changed(get_value_args=get_value_args, value=value, info=info)
            self._changed[history_key] = val

        return val

    def _has_changed(self, get_value_args, value, info):
        """
        Checks whether the previous value evaluated at time @t has changed with the current timestep.
        By default, it returns True.

        Any custom checks should be overridden by subclass.

        Args:
            get_value_args (tuple): Specific argument combinations (usually tuple of objects) passed into
                @self.get_value
            value (any): Cached value computed at timestep @t for this object state
            info (dict): Information calculated at timestep @t when computing this state's value

        Returns:
            bool: Whether the value has changed between @value and @info and the coresponding value and info computed
                at the current timestep
        """
        return True

    def get_value(self, *args, **kwargs):
        """
        Get this state's value

        Returns:
            any: Object state value given input @args and @kwargs
        """
        assert self._initialized

        # Compile args and kwargs deterministically
        key = (*args, *tuple(kwargs.values()))
        # We need to see if we need to update our cache -- we do so if and only if one of the following conditions are met:
        # (a) key is NOT in the cache
        # (b) Our cache is not valid
        if key not in self._cache or not self.cache_is_valid(get_value_args=key):
            # Update the cache
            self.update_cache(get_value_args=key)

        # Value is the cached value
        val = self._cache[key]["value"]

        return val

    def _get_value(self, *args, **kwargs):
        raise NotImplementedError(f"_get_value not implemented for {self.__class__.__name__} state.")

    def set_value(self, *args, **kwargs):
        """
        Set this state's value

        Returns:
            bool: True if setting the value was successful, otherwise False
        """
        assert self._initialized
        # Clear cache because the state may be changed
        self.clear_cache()
        # Set the value
        val = self._set_value(*args, **kwargs)
        # Add this object to the current state update set in its scene
        self.obj.state_updated()
        return val

    def _set_value(self, *args, **kwargs):
        raise NotImplementedError(f"_set_value not implemented for {self.__class__.__name__} state.")

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

    def _get_value(self):
        raise NotImplementedError(f"_get_value not implemented for {self.__class__.__name__} state.")

    def _set_value(self, new_value):
        raise NotImplementedError(f"_set_value not implemented for {self.__class__.__name__} state.")

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

    def _get_value(self, other):
        raise NotImplementedError(f"_get_value not implemented for {self.__class__.__name__} state.")

    def _set_value(self, other, new_value):
        raise NotImplementedError(f"_set_value not implemented for {self.__class__.__name__} state.")

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("RelativeObjectState")
        return classes


class IntrinsicObjectState(BaseObjectState):
    """
    This class is used to track object states that should NOT have getters / setters implemented, since the associated
    ability / state is intrinsic to the state
    """

    def _get_value(self):
        raise NotImplementedError(
            f"_get_value not implemented for IntrinsicObjectState {self.__class__.__name__} state."
        )

    def _set_value(self, new_value):
        raise NotImplementedError(
            f"_set_value not implemented for IntrinsicObjectState {self.__class__.__name__} state."
        )

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("IntrinsicObjectState")
        return classes


class BooleanStateMixin(BaseObjectState):
    """
    This class is a mixin used to indicate that a state has a boolean value.
    """

    pass
