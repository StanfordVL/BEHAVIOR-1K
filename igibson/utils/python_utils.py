"""
A set of utility functions for general python usage
"""
import inspect
from abc import ABCMeta
from copy import deepcopy
from functools import wraps
from importlib import import_module

import numpy as np

# Global dictionary storing all unique names
NAMES = set()
CLASS_NAMES = set()


class classproperty:

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def save_init_info(func):
    """
    Decorator to save the init info of an object to object._init_info.

    _init_info contains class name and class constructor's input args.
    """
    sig = inspect.signature(func)

    @wraps(func) # preserve func name, docstring, arguments list, etc.
    def wrapper(self, *args, **kwargs):
        values = sig.bind(self, *args, **kwargs)

        # Prevent args of super init from being saved.
        if hasattr(self, "_init_info"):
            func(*values.args, **values.kwargs)
            return

        # Initialize class's self._init_info.
        self._init_info = {}
        self._init_info["class_module"] = self.__class__.__module__
        self._init_info["class_name"] = self.__class__.__name__
        self._init_info["args"] = {}

        # Populate class's self._init_info.
        for k, p in sig.parameters.items():
            if k == 'self':
                continue
            if k in values.arguments:
                val = values.arguments[k]
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    self._init_info["args"][k] = val
                elif p.kind == inspect.Parameter.VAR_KEYWORD:
                    for kwarg_k, kwarg_val in values.arguments[k].items():
                        self._init_info["args"][kwarg_k] = kwarg_val

        # Call the original function.
        func(*values.args, **values.kwargs)

    return wrapper


class RecreatableMeta(type):
    """
    Simple metaclass that automatically saves __init__ args of the instances it creates.
    """

    def __new__(cls, clsname, bases, clsdict):
        if "__init__" in clsdict:
            clsdict["__init__"] = save_init_info(clsdict["__init__"])
        return super().__new__(cls, clsname, bases, clsdict)


class RecreatableAbcMeta(RecreatableMeta, ABCMeta):
    """
    A composite metaclass of both RecreatableMeta and ABCMeta.

    Adding in ABCMeta to resolve metadata conflicts.
    """

    pass


class Recreatable(metaclass=RecreatableAbcMeta):
    """
    Simple class that provides an abstract interface automatically saving __init__ args of
    the classes inheriting it.
    """

    def __init__(self):
        pass

    def get_init_info(self):
        """
        Grabs relevant initialization information for this class instance. Useful for directly
        reloading an object from this information, using @create_object_from_init_info.

        Returns:
            dict: Nested dictionary that contains this object's initialization information
        """
        # Note: self._init_info is procedurally generated via @save_init_info called in metaclass
        return self._init_info


def create_object_from_init_info(init_info):
    """
    Create a new object based on an given init info.

    Args:
        init_info (dict): Nested dictionary that contains an object's init information.
    Returns:
        any: Newly created object.
    """
    module = import_module(init_info["class_module"])
    cls = getattr(module, init_info["class_name"])
    return cls(**init_info["args"], **init_info.get("kwargs", {}))


def merge_nested_dicts(base_dict, extra_dict, inplace=False, verbose=False):
    """
    Iteratively updates @base_dict with values from @extra_dict. Note: This generates a new dictionary!
    Args:
        base_dict (dict): Nested base dictionary, which should be updated with all values from @extra_dict
        extra_dict (dict): Nested extra dictionary, whose values will overwrite corresponding ones in @base_dict
        inplace (bool): Whether to modify @base_dict in place or not
        verbose (bool): If True, will print when keys are mismatched
    Returns:
        dict: Updated dictionary
    """
    # Loop through all keys in @extra_dict and update the corresponding values in @base_dict
    base_dict = base_dict if inplace else deepcopy(base_dict)
    for k, v in extra_dict.items():
        if k not in base_dict:
            base_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(base_dict[k], dict):
                base_dict[k] = merge_nested_dicts(base_dict[k], v)
            else:
                not_equal = base_dict[k] != v
                if isinstance(not_equal, np.ndarray):
                    not_equal = not_equal.any()
                if not_equal and verbose:
                    print(f"Different values for key {k}: {base_dict[k]}, {v}\n")
                base_dict[k] = np.array(v) if isinstance(v, list) else v

    # Return new dict
    return base_dict


def get_class_init_kwargs(cls):
    """
    Helper function to return a list of all valid keyword arguments (excluding "self") for the given @cls class.
    Args:
        cls (object): Class from which to grab __init__ kwargs
    Returns:
        list: All keyword arguments (excluding "self") specified by @cls __init__ constructor method
    """
    return list(inspect.signature(cls.__init__).parameters.keys())[1:]


def extract_subset_dict(dic, keys, copy=False):
    """
    Helper function to extract a subset of dictionary key-values from a current dictionary. Optionally (deep)copies
    the values extracted from the original @dic if @copy is True.
    Args:
        dic (dict): Dictionary containing multiple key-values
        keys (Iterable): Specific keys to extract from @dic. If the key doesn't exist in @dic, then the key is skipped
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys
    Returns:
        dict: Extracted subset dictionary containing only the specified @keys and their corresponding values
    """
    subset = {k: dic[k] for k in keys if k in dic}
    return deepcopy(subset) if copy else subset


def extract_class_init_kwargs_from_dict(cls, dic, copy=False):
    """
    Helper function to return a dictionary of key-values that specifically correspond to @cls class's __init__
    constructor method, from @dic which may or may not contain additional, irrelevant kwargs.
    Note that @dic may possibly be missing certain kwargs as specified by cls.__init__. No error will be raised.
    Args:
        cls (object): Class from which to grab __init__ kwargs that will be be used as filtering keys for @dic
        dic (dict): Dictionary containing multiple key-values
        copy (bool): If True, will deepcopy all values corresponding to the specified @keys
    Returns:
        dict: Extracted subset dictionary possibly containing only the specified keys from cls.__init__ and their
            corresponding values
    """
    # extract only relevant kwargs for this specific backbone
    return extract_subset_dict(
        dic=dic,
        keys=get_class_init_kwargs(cls),
        copy=copy,
    )


def assert_valid_key(key, valid_keys, name=None):
    """
    Helper function that asserts that @key is in dictionary @valid_keys keys. If not, it will raise an error.

    :param key: Any, key to check for in dictionary @dic's keys
    :param valid_keys: Iterable, contains keys should be checked with @key
    :param name: str or None, if specified, is the name associated with the key that will be printed out if the
        key is not found. If None, default is "value"
    """
    if name is None:
        name = "value"
    assert key in valid_keys, "Invalid {} received! Valid options are: {}, got: {}".format(
        name, valid_keys.keys() if isinstance(valid_keys, dict) else valid_keys, key)


def create_class_from_registry_and_config(cls_name, cls_registry, cfg, cls_type_descriptor):
    """
    Helper function to create a class with str type @cls_name, which should be a valid entry in @cls_registry, using
    kwargs in dictionary form @cfg to pass to the constructor, with @cls_type_name specified for debugging

    Args:
        cls_name (str): Name of the class to create. This should correspond to the actual class type, in string form
        cls_registry (dict): Class registry. This should map string names of valid classes to create to the
            actual class type itself
        cfg (dict): Any keyword arguments to pass to the class constructor
        cls_type_descriptor (str): Description of the class type being created. This can be any string and is used
            solely for debugging purposes

    Returns:
        any: Created class instance
    """
    # Make sure the requested class type is valid
    assert_valid_key(key=cls_name, valid_keys=cls_registry, name=f"{cls_type_descriptor} type")

    # Grab the kwargs relevant for the specific class
    cls = cls_registry[cls_name]
    cls_kwargs = extract_class_init_kwargs_from_dict(cls=cls, dic=cfg, copy=False)

    # Create the class
    return cls(**cls_kwargs)


class UniquelyNamed:
    """
    Simple class that implements a name property, that must be implemented by a subclass. Note that any @Named
    entity must be UNIQUE!
    """
    def __init__(self, *args, **kwargs):
        global NAMES
        # Register this object, making sure it's name is unique
        assert self.name not in NAMES, \
            f"UniquelyNamed object with name {self.name} already exists!"
        NAMES.add(self.name)

    def __del__(self):
        # Remove this object name from the registry if it's still there
        if self.name in NAMES:
            NAMES.remove(self.name)

    @property
    def name(self):
        """
        Returns:
            str: Name of this instance. Must be unique!
        """
        raise NotImplementedError


class UniquelyNamedNonInstance:
    """
    Identical to UniquelyNamed, but intended for non-instanceable classes
    """

    def __init_subclass__(cls, **kwargs):
        global CLASS_NAMES
        # Register this object, making sure it's name is unique
        assert cls.name not in CLASS_NAMES, \
            f"UniquelyNamed class with name {cls.name} already exists!"
        CLASS_NAMES.add(cls.name)

    @classproperty
    def name(cls):
        """
        Returns:
            str: Name of this instance. Must be unique!
        """
        raise NotImplementedError


class Registerable:
    """
    Simple class template that provides an abstract interface for registering classes.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Registers all subclasses as part of this registry. This is useful to decouple internal codebase from external
        user additions. This way, users can add their custom subclasses by simply extending this class,
        and it will automatically be registered internally. This allows users to then specify their classes
        directly in string-form in e.g., their config files, without having to manually set the str-to-class mapping
        in our code.
        """
        cls._register_cls()

    @classmethod
    def _register_cls(cls):
        """
        Register this class. Can be extended by subclass.
        """
        # print(f"registering: {cls.__name__}")
        # print(f"registry: {cls._cls_registry}", cls.__name__ not in cls._cls_registry)
        # print(f"do not register: {cls._do_not_register_classes}", cls.__name__ not in cls._do_not_register_classes)
        # input()
        if cls.__name__ not in cls._cls_registry and cls.__name__ not in cls._do_not_register_classes:
            cls._cls_registry[cls.__name__] = cls

    @classproperty
    def _do_not_register_classes(cls):
        """
        Returns:
            set of str: Name(s) of classes that should not be registered. Default is empty set.
                Subclasses that shouldn't be added should call super() and then add their own class name to the set
        """
        return set()

    @classproperty
    def _cls_registry(cls):
        """
        Returns:
            OrderedDict: Mapping from all registered class names to their classes. This should be a REFERENCE
                to some external, global dictionary that will be filled-in at runtime.
        """
        raise NotImplementedError()


class Serializable:
    """
    Simple class that provides an abstract interface to dump / load states, optionally with serialized functionality
    as well.
    """
    @property
    def state_size(self):
        """
        Returns:
            int: Size of this object's serialized state
        """
        raise NotImplementedError()

    def _dump_state(self):
        """
        Dumps the state of this object in dictionary form (can be empty). Should be implemented by subclass.

        Returns:
            OrderedDict: Keyword-mapped states of this object
        """
        raise NotImplementedError()

    def dump_state(self, serialized=False):
        """
        Dumps the state of this object in either dictionary of flattened numerical form.

        Args:
            serialized (bool): If True, will return the state of this object as a 1D numpy array. Otherewise, will return
                a (potentially nested) dictionary of states for this object

        Returns:
            OrderedDict or n-array: Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical np.array capturing this object's state, where n is @self.state_size
        """
        state = self._dump_state()
        return self.serialize(state=state) if serialized else state

    def _load_state(self, state):
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (OrderedDict): Keyword-mapped states of this object to set
        """
        raise NotImplementedError()

    def load_state(self, state, serialized=False):
        """
        Deserializes and loads this object's state based on @state

        Args:
            state (OrderedDict or n-array): Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical np.array capturing this object's state, where n is @self.state_size
            serialized (bool): If True, will interpret @state as a 1D numpy array. Otherewise, will assume the input is
                a (potentially nested) dictionary of states for this object
        """
        state = self.deserialize(state=state) if serialized else state
        self._load_state(state=state)

    def _serialize(self, state):
        """
        Serializes nested dictionary state @state into a flattened 1D numpy array for encoding efficiency.
        Should be implemented by subclass.

        Args:
            state (OrderedDict): Keyword-mapped states of this object to encode. Should match structure of output from
                self._dump_state()

        Returns:
            n-array: encoded + serialized, 1D numerical np.array capturing this object's state
        """
        raise NotImplementedError()

    def serialize(self, state):
        """
        Serializes nested dictionary state @state into a flattened 1D numpy array for encoding efficiency.
        Should be implemented by subclass.

        Args:
            state (OrderedDict): Keyword-mapped states of this object to encode. Should match structure of output from
                self._dump_state()

        Returns:
            n-array: encoded + serialized, 1D numerical np.array capturing this object's state
        """
        # Simply returns self._serialize() for now. this is for future proofing
        return self._serialize(state=state)

    def _deserialize(self, state):
        """
        De-serializes flattened 1D numpy array @state into nested dictionary state.
        Should be implemented by subclass.

        Args:
            state (n-array): encoded + serialized, 1D numerical np.array capturing this object's state

        Returns:
            2-tuple:
                - OrderedDict: Keyword-mapped states of this object. Should match structure of output from
                    self._dump_state()
                - int: current index of the flattened state vector that is left off. This is helpful for subclasses
                    that inherit partial deserializations from parent classes, and need to know where the
                    deserialization left off before continuing.
        """
        raise NotImplementedError

    def deserialize(self, state):
        """
        De-serializes flattened 1D numpy array @state into nested dictionary state.
        Should be implemented by subclass.

        Args:
            state (n-array): encoded + serialized, 1D numerical np.array capturing this object's state

        Returns:
            OrderedDict: Keyword-mapped states of this object. Should match structure of output from
                self._dump_state()
        """
        # Sanity check the idx with the expected state size
        state_dict, idx = self._deserialize(state=state)
        assert idx == self.state_size, f"Invalid state deserialization occurred! Expected {self.state_size} total " \
                                           f"values to be deserialized, only {idx} were."

        return state_dict


class SerializableNonInstance:
    """
    Identical to Serializable, but intended for non-instanceable classes
    """
    @classproperty
    def state_size(cls):
        """
        Returns:
            int: Size of this object's serialized state
        """
        raise NotImplementedError()

    @classmethod
    def _dump_state(cls):
        """
        Dumps the state of this object in dictionary form (can be empty). Should be implemented by subclass.

        Returns:
            OrderedDict: Keyword-mapped states of this object
        """
        raise NotImplementedError()

    @classmethod
    def dump_state(cls, serialized=False):
        """
        Dumps the state of this object in either dictionary of flattened numerical form.

        Args:
            serialized (bool): If True, will return the state of this object as a 1D numpy array. Otherewise, will return
                a (potentially nested) dictionary of states for this object

        Returns:
            OrderedDict or n-array: Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical np.array capturing this object's state, where n is @self.state_size
        """
        state = cls._dump_state()
        return cls.serialize(state=state) if serialized else state

    @classmethod
    def _load_state(cls, state):
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (OrderedDict): Keyword-mapped states of this object to set
        """
        raise NotImplementedError()

    @classmethod
    def load_state(cls, state, serialized=False):
        """
        Deserializes and loads this object's state based on @state

        Args:
            state (OrderedDict or n-array): Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical np.array capturing this object's state, where n is @self.state_size
            serialized (bool): If True, will interpret @state as a 1D numpy array. Otherewise, will assume the input is
                a (potentially nested) dictionary of states for this object
        """
        state = cls.deserialize(state=state) if serialized else state
        cls._load_state(state=state)

    @classmethod
    def _serialize(cls, state):
        """
        Serializes nested dictionary state @state into a flattened 1D numpy array for encoding efficiency.
        Should be implemented by subclass.

        Args:
            state (OrderedDict): Keyword-mapped states of this object to encode. Should match structure of output from
                self._dump_state()

        Returns:
            n-array: encoded + serialized, 1D numerical np.array capturing this object's state
        """
        raise NotImplementedError()

    @classmethod
    def serialize(cls, state):
        """
        Serializes nested dictionary state @state into a flattened 1D numpy array for encoding efficiency.
        Should be implemented by subclass.

        Args:
            state (OrderedDict): Keyword-mapped states of this object to encode. Should match structure of output from
                self._dump_state()

        Returns:
            n-array: encoded + serialized, 1D numerical np.array capturing this object's state
        """
        # Simply returns self._serialize() for now. this is for future proofing
        return cls._serialize(state=state)

    @classmethod
    def _deserialize(cls, state):
        """
        De-serializes flattened 1D numpy array @state into nested dictionary state.
        Should be implemented by subclass.

        Args:
            state (n-array): encoded + serialized, 1D numerical np.array capturing this object's state

        Returns:
            2-tuple:
                - OrderedDict: Keyword-mapped states of this object. Should match structure of output from
                    self._dump_state()
                - int: current index of the flattened state vector that is left off. This is helpful for subclasses
                    that inherit partial deserializations from parent classes, and need to know where the
                    deserialization left off before continuing.
        """
        raise NotImplementedError

    @classmethod
    def deserialize(cls, state):
        """
        De-serializes flattened 1D numpy array @state into nested dictionary state.
        Should be implemented by subclass.

        Args:
            state (n-array): encoded + serialized, 1D numerical np.array capturing this object's state

        Returns:
            OrderedDict: Keyword-mapped states of this object. Should match structure of output from
                self._dump_state()
        """
        # Sanity check the idx with the expected state size
        state_dict, idx = cls._deserialize(state=state)
        assert cls.state_size is not None, "State size must be specified by subclass!"
        assert idx == cls.state_size, f"Invalid state deserialization occurred! Expected {cls.state_size} total " \
                                      f"values to be deserialized, only {idx} were."

        return state_dict


def clear():
    """
    Clear state tied to singleton classes
    """
    NAMES.clear()
