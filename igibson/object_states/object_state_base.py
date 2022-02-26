from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
from igibson.utils.python_utils import Serializable


# Hacky method to serialize "None" values as a number -- we choose magic number 400 since:
# sum([ord(c) for c in "None"]) = 400!
NONE = 400.0

# Global dicts that will contain mappings
REGISTERED_OBJECT_STATES = OrderedDict()
REGISTERED_ABSOLUTE_OBJECT_STATES = OrderedDict()
OBJECT_STATES_TO_ID = OrderedDict()

# Counter for keeping track of unique object state classes that are created
_STATE_COUNTER = 0


def register_object_state(cls):
    global _STATE_COUNTER
    if cls.__name__ not in REGISTERED_OBJECT_STATES:
        REGISTERED_OBJECT_STATES[cls.__name__] = cls
        OBJECT_STATES_TO_ID[cls.__name__] = _STATE_COUNTER
        _STATE_COUNTER += 1


def register_absolute_object_state(cls):
    if cls.__name__ not in REGISTERED_OBJECT_STATES:
        REGISTERED_ABSOLUTE_OBJECT_STATES[cls.__name__] = cls


class BaseObjectState(Serializable, metaclass=ABCMeta):
    """
    Base ObjectState class. Do NOT inherit from this class directly - use either AbsoluteObjectState or
    RelativeObjectState.
    """
    def __init_subclass__(cls, **kwargs):
        """
        Registers all subclasses as part of this registry. This is useful to decouple internal codebase from external
        user additions. This way, users can add their custom object state by simply extending this BaseObjectState class,
        and it will automatically be registered internally. This allows users to then specify their object state
        directly in string-from in e.g., their config files, without having to manually set the str-to-class mapping
        in our code.
        """
        register_object_state(cls)

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
        self.simulator = None

    @property
    def settable(self):
        """
        Returns:
            bool: True if this object has a state that can be directly dumped / loaded via dump_state() and
                load_state(), otherwise, returns False. Note that any sub object states that are NOT settable do
                not need to implement any of _dump_state(), _load_state(), _serialize(), or _deserialize()!
        """
        # False by default
        return False

    def _update(self):
        """This function will be called once for every simulator step."""
        pass

    def _initialize(self):
        """This function will be called once, after the object has been loaded."""
        pass

    def initialize(self, simulator):
        assert not self._initialized, "State is already initialized."

        self.simulator = simulator
        self._initialize()
        self._initialized = True

    def update(self):
        assert self._initialized, "Cannot update uninitalized state."
        return self._update()

    def get_value(self, *args, **kwargs):
        assert self._initialized
        return self._get_value(*args, **kwargs)

    def _get_value(self, *args, **kwargs):
        raise NotImplementedError

    def set_value(self, *args, **kwargs):
        assert self._initialized
        return self._set_value(*args, **kwargs)

    def _set_value(self, *args, **kwargs):
        raise NotImplementedError

    def dump_state(self, serialized=False):
        assert self._initialized
        assert self.settable
        return super().dump_state(serialized=serialized)


class AbsoluteObjectState(BaseObjectState):
    """
    This class is used to track object states that are absolute, e.g. do not require a second object to compute
    the value.
    """
    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)
        register_absolute_object_state(cls)

    @abstractmethod
    def _get_value(self):
        raise NotImplementedError()

    @abstractmethod
    def _set_value(self, new_value):
        raise NotImplementedError()


class CachingEnabledObjectState(AbsoluteObjectState):
    """
    This class is used to track absolute states that are expensive to compute. It adds out-of-the-box support for
    caching the results for each simulator step.
    """

    def __init__(self, obj):
        super(CachingEnabledObjectState, self).__init__(obj)
        self.value = None

    @abstractmethod
    def _compute_value(self):
        """
        This function should compute the value of the state and return it. It should not set self.value.

        :return: The computed value.
        """
        raise NotImplementedError()

    def _get_value(self):
        # If we don't have a value cached, compute it now.
        if self.value is None:
            self.value = self._compute_value()

        return self.value

    def clear_cached_value(self):
        self.value = None

    def _update(self):
        # Reset the cached state value on Simulator step.
        super(CachingEnabledObjectState, self)._update()
        self.clear_cached_value()


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


class BooleanState(object):
    """
    This class is a mixin used to indicate that a state has a boolean value.
    """

    pass
