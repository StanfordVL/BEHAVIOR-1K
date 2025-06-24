"""
A set of utility functions for general python usage
"""

import inspect
from abc import ABCMeta
from collections.abc import Iterable
from copy import deepcopy
from functools import wraps
from hashlib import md5
from importlib import import_module
import sys

import h5py
import torch as th

# Global dictionary storing all unique names
NAMES = set()
CLASS_NAMES = set()


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def subclass_factory(name, base_classes, __init__=None, **kwargs):
    """
    Programmatically generates a new class type with name @name, subclassing from base classes @base_classes, with
    corresponding __init__ call @__init__.

    NOTE: If __init__ is None (default), the __init__ call from @base_classes will be used instead.

    cf. https://stackoverflow.com/questions/15247075/how-can-i-dynamically-create-derived-classes-from-a-base-class

    Args:
        name (str): Generated class name
        base_classes (type, or list of type): Base class(es) to use for generating the subclass
        __init__ (None or function): Init call to use for the base class when it is instantiated. If None if specified,
            the newly generated class will automatically inherit the __init__ call from @base_classes
        **kwargs (any): keyword-mapped parameters to override / set in the child class, where the keys represent
            the class / instance attribute to modify and the values represent the functions / value to set
    """
    # Standardize base_classes
    base_classes = tuple(base_classes if isinstance(base_classes, Iterable) else [base_classes])

    # Generate the new class
    if __init__ is not None:
        kwargs["__init__"] = __init__
    return type(name, base_classes, kwargs)


def save_init_info(func):
    """
    Decorator to save the init info of an object to object._init_info.

    _init_info contains class name and class constructor's input args.
    """
    sig = inspect.signature(func)

    @wraps(func)  # preserve func name, docstring, arguments list, etc.
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
            if k == "self":
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
    Create a new object based on given init info.

    Args:
        init_info (dict): Nested dictionary that contains an object's init information.

    Returns:
        any: Newly created object.
    """
    module = import_module(init_info["class_module"])
    cls = getattr(module, init_info["class_name"])
    return cls(**init_info["args"], **init_info.get("kwargs", {}))


def safe_equal(a, b):
    if isinstance(a, th.Tensor) and isinstance(b, th.Tensor):
        return a.shape == b.shape and (a == b).all().item()
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(safe_equal(a_item, b_item) for a_item, b_item in zip(a, b))
    else:
        return a == b


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
                equal = safe_equal(base_dict[k], v)
                if not equal and verbose:
                    print(f"Different values for key {k}: {base_dict[k]}, {v}\n")
                base_dict[k] = v

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

    Args:
        key (any): key to check for in dictionary @dic's keys
        valid_keys (Iterable): contains keys should be checked with @key
        name (str or None): if specified, is the name associated with the key that will be printed out if the
            key is not found. If None, default is "value"
    """
    if name is None:
        name = "value"
    assert key in valid_keys, "Invalid {} received! Valid options are: {}, got: {}".format(
        name, valid_keys.keys() if isinstance(valid_keys, dict) else valid_keys, key
    )


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


def get_uuid(name, n_digits=8, deterministic=True):
    """
    Helper function to create a unique @n_digits uuid given a unique @name

    Args:
        name (str): Name of the object or class
        n_digits (int): Number of digits of the uuid, default is 8
        deterministic (bool): Whether the outputted UUID should be deterministic or not

    Returns:
        int: uuid
    """
    # Make sure the number is float32 compatible
    val = int(md5(name.encode()).hexdigest(), 16) if deterministic else abs(hash(name))
    return int(th.tensor(val % (10**n_digits), dtype=th.float32).item())


def meets_minimum_version(test_version, minimum_version):
    """
    Verify that @test_version meets the @minimum_version

    Args:
        test_version (str): Python package version. Should be, e.g., 0.26.1
        minimum_version (str): Python package version to test against. Should be, e.g., 0.27.2

    Returns:
        bool: Whether @test_version meets @minimum_version
    """
    test_nums = [int(num) for num in test_version.split(".")]
    minimum_nums = [int(num) for num in minimum_version.split(".")]
    assert len(test_nums) == 3
    assert len(minimum_nums) == 3

    for test_num, minimum_num in zip(test_nums, minimum_nums):
        if test_num > minimum_num:
            return True
        elif test_num < minimum_num:
            return False
        # Otherwise, we continue through all sub-versions

    # If we get here, that means test_version == threshold_version, so this is a success
    return True


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
            dict: Mapping from all registered class names to their classes. This should be a REFERENCE
                to some external, global dictionary that will be filled-in at runtime.
        """
        raise NotImplementedError()


class Serializable:
    """
    Simple class that provides an abstract interface to dump / load states, optionally with serialized functionality
    as well.
    """

    def _dump_state(self):
        """
        Dumps the state of this object in dictionary form (can be empty). Should be implemented by subclass.

        Returns:
            dict: Keyword-mapped states of this object
        """
        raise NotImplementedError()

    def dump_state(self, serialized=False):
        """
        Dumps the state of this object in either dictionary of flattened numerical form.

        Args:
            serialized (bool): If True, will return the state of this object as a 1D numpy array. Otherewise, will return
                a (potentially nested) dictionary of states for this object

        Returns:
            dict or n-array: Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical th.tensor capturing this object's state
        """
        state = self._dump_state()
        return self.serialize(state=state) if serialized else state

    def _load_state(self, state):
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (dict): Keyword-mapped states of this object to set
        """
        raise NotImplementedError()

    def load_state(self, state, serialized=False):
        """
        Deserializes and loads this object's state based on @state

        Args:
            state (dict or n-array): Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical th.tensor capturing this object's state.
            serialized (bool): If True, will interpret @state as a 1D numpy array. Otherewise, will assume the input is
                a (potentially nested) dictionary of states for this object
        """
        if serialized:
            orig_state_len = len(state)
            state, deserialized_items = self.deserialize(state=state)
            assert deserialized_items == orig_state_len, (
                f"Invalid state deserialization occurred! Expected {orig_state_len} total "
                f"values to be deserialized, only {deserialized_items} were."
            )
        self._load_state(state=state)

    def serialize(self, state):
        """
        Serializes nested dictionary state @state into a flattened 1D numpy array for encoding efficiency.
        Should be implemented by subclass.

        Args:
            state (dict): Keyword-mapped states of this object to encode. Should match structure of output from
                self._dump_state()

        Returns:
            n-array: encoded + serialized, 1D numerical th.tensor capturing this object's state
        """
        raise NotImplementedError()

    def deserialize(self, state):
        """
        De-serializes flattened 1D numpy array @state into nested dictionary state.
        Should be implemented by subclass.

        Args:
            state (n-array): encoded + serialized, 1D numerical th.tensor capturing this object's state

        Returns:
            2-tuple:
                - dict: Keyword-mapped states of this object. Should match structure of output from
                    self._dump_state()
                - int: current index of the flattened state vector that is left off. This is helpful for subclasses
                    that inherit partial deserializations from parent classes, and need to know where the
                    deserialization left off before continuing.
        """
        raise NotImplementedError


class SerializableNonInstance:
    """
    Identical to Serializable, but intended for non-instanceable classes
    """

    @classmethod
    def _dump_state(cls):
        """
        Dumps the state of this object in dictionary form (can be empty). Should be implemented by subclass.

        Returns:
            dict: Keyword-mapped states of this object
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
            dict or n-array: Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical th.tensor capturing this object's state.
        """
        state = cls._dump_state()
        return cls.serialize(state=state) if serialized else state

    @classmethod
    def _load_state(cls, state):
        """
        Load the internal state to this object as specified by @state. Should be implemented by subclass.

        Args:
            state (dict): Keyword-mapped states of this object to set
        """
        raise NotImplementedError()

    @classmethod
    def load_state(cls, state, serialized=False):
        """
        Deserializes and loads this object's state based on @state

        Args:
            state (dict or n-array): Either:
                - Keyword-mapped states of this object, or
                - encoded + serialized, 1D numerical th.tensor capturing this object's state.
            serialized (bool): If True, will interpret @state as a 1D numpy array. Otherewise, will assume the input is
                a (potentially nested) dictionary of states for this object
        """
        if serialized:
            orig_state_len = len(state)
            state, deserialized_items = cls.deserialize(state=state)
            assert deserialized_items == orig_state_len, (
                f"Invalid state deserialization occurred! Expected {orig_state_len} total "
                f"values to be deserialized, only {deserialized_items} were."
            )
        cls._load_state(state=state)

    @classmethod
    def serialize(cls, state):
        """
        Serializes nested dictionary state @state into a flattened 1D numpy array for encoding efficiency.
        Should be implemented by subclass.

        Args:
            state (dict): Keyword-mapped states of this object to encode. Should match structure of output from
                self._dump_state()

        Returns:
            n-array: encoded + serialized, 1D numerical th.tensor capturing this object's state
        """
        # Simply returns self.serialize() for now. this is for future proofing
        return NotImplementedError()

    @classmethod
    def deserialize(cls, state):
        """
        De-serializes flattened 1D numpy array @state into nested dictionary state.
        Should be implemented by subclass.

        Args:
            state (n-array): encoded + serialized, 1D numerical th.tensor capturing this object's state

        Returns:
            2-tuple:
                - dict: Keyword-mapped states of this object. Should match structure of output from
                    self._dump_state()
                - int: current index of the flattened state vector that is left off. This is helpful for subclasses
                    that inherit partial deserializations from parent classes, and need to know where the
                    deserialization left off before continuing.
        """
        raise NotImplementedError


class CachedFunctions:
    """
    Thin object which owns a dictionary in which each entry should be a function -- when a key is queried via get()
    and it exists, it will call the function exactly once, and cache the value so that subsequent calls will refer
    to the cached value.

    This allows the dictionary to be created with potentially expensive operations, but only queried up to exaclty once
    as needed.
    """

    def __init__(self, **kwargs):
        # Create internal dict to store functions
        self._fcns = dict()
        self._cache = dict()
        for kwarg in kwargs:
            self._fcns[kwarg] = kwargs[kwarg]

    def __getitem__(self, item):
        return self.get(name=item)

    def __setitem__(self, key, value):
        self.add_fcn(name=key, fcn=value)

    def get(self, name):
        """
        Computes the function referenced by @name with the corresponding @args and @kwargs. Note that for a unique
        set of arguments, this value will be internally cached

        Args:
            name (str): The name of the function to call

        Returns:
            any: Output of the function referenced by @name
        """
        if name not in self._cache:
            self._cache[name] = self._fcns[name]()
        return self._cache[name]

    def get_fcn(self, name):
        """
        Gets the raw stored function referenced by @name

        Args:
            name (str): The name of the function to grab

        Returns:
            function: The stored function
        """
        return self._fcns[name]

    def get_fcn_names(self):
        """
        Get all stored function names

        Returns:
            tuple of str: Names of stored functions
        """
        return tuple(self._fcns.keys())

    def add_fcn(self, name, fcn):
        """
        Adds a function to the internal registry.

        Args:
            name (str): Name of the function. This is the name that should be queried with self.get()
            fcn (function): Function to add. Can be an arbitrary signature
        """
        assert callable(fcn), "Only functions can be added via add_fcn!"
        self._fcns[name] = fcn


class Wrapper:
    """
    Base class for all wrappers in OmniGibson

    Args:
        obj (any): Arbitrary python object instance to wrap
    """

    def __init__(self, obj):
        # Set the internal attributes -- store wrapped obj
        self.wrapped_obj = obj

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        obj = self.wrapped_obj
        while True:
            if isinstance(obj, Wrapper):
                if obj.class_name() == self.class_name():
                    raise Exception("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                obj = obj.wrapped_obj
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped object

        Returns:
            any: The unwrapped object instance
        """
        return self.wrapped_obj.unwrapped if hasattr(self.wrapped_obj, "unwrapped") else self.wrapped_obj

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # If we're querying wrapped_obj, raise an error
        if attr == "wrapped_obj":
            raise AttributeError("wrapped_obj attribute not initialized yet!")

        # Sanity check to make sure wrapped obj is not None -- if so, raise error
        assert self.wrapped_obj is not None, f"Cannot access attribute {attr} since wrapped_obj is None!"

        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.wrapped_obj, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.wrapped_obj):
                    return self
                return result

            return hooked
        else:
            return orig_attr

    def __setattr__(self, key, value):
        # Call setattr on wrapped obj if it has the attribute, otherwise, operate on this object
        if hasattr(self, "wrapped_obj") and self.wrapped_obj is not None and hasattr(self.wrapped_obj, key):
            setattr(self.wrapped_obj, key, value)
        else:
            super().__setattr__(key, value)


def torch_compile(func):
    """
    Decorator to compile a function with torch.compile on Linux and torch.jit.script on Windows. This is because of poor support for torch.compile on Windows.

    Args:
        func (function): Function to compile

    Returns:
        function: Compiled function
    """
    # If we're on Windows, return a jitscript option
    if sys.platform == "win32":
        return th.jit.script(func)
    # Otherwise, return a torch.compile option
    else:
        return th.compile(func)


def nums2array(nums, dim, dtype=th.float32):
    """
    Converts input @nums into numpy array of length @dim. If @nums is a single number, broadcasts input to
    corresponding dimension size @dim before converting into numpy array

    Args:
        nums (float or array): Numbers to map to numpy array
        dim (int): Size of array to broadcast input to

    Returns:
        torch.Tensor: Mapped input numbers
    """
    # Make sure the inputted nums isn't a string
    assert not isinstance(nums, str), "Only numeric types are supported for this operation!"

    out = th.tensor(nums, dtype=dtype) if isinstance(nums, Iterable) else th.ones(dim, dtype=dtype) * nums

    return out


def clear():
    """
    Clear state tied to singleton classes
    """
    NAMES.clear()
    CLASS_NAMES.clear()


def torch_delete(tensor: th.Tensor, indices: th.Tensor | int, dim: int | None = None) -> th.Tensor:
    """
    Delete elements from a tensor along a specified dimension.

    Parameters:
    tensor (torch.Tensor): Input tensor.
    indices (int or torch.Tensor): Indices of elements to remove.
    dim (int, optional): The dimension along which to delete the elements.
                         If None, the tensor is flattened before deletion.

    Returns:
    torch.Tensor: Tensor with specified elements removed.
    """
    assert tensor.dim() > 0, "Input tensor must have at least one dimension"

    if dim is None:
        # Flatten the tensor if no dim is specified
        tensor = tensor.flatten()
        dim = 0

    if not isinstance(indices, th.Tensor):
        indices = th.tensor(indices, dtype=th.long, device=tensor.device)

    assert th.all(indices >= 0) and th.all(indices < tensor.size(dim)), "Indices out of bounds"

    # Create a mask for the indices to keep
    keep_indices = th.ones(tensor.size(dim), dtype=th.bool, device=tensor.device)
    keep_indices[indices] = False

    return th.index_select(tensor, dim, th.nonzero(keep_indices).squeeze(1))


def recursively_convert_to_torch(state):
    # For all the lists in state dict, convert to torch tensor
    for key, value in state.items():
        if isinstance(value, dict):
            state[key] = recursively_convert_to_torch(value)
        elif isinstance(value, list):
            # Convert to torch tensor if all elements are numeric and have consistent shapes
            try:
                state[key] = th.tensor(value, dtype=th.float32)
            except:
                pass

    return state


def recursively_convert_from_torch(state):
    # For all the lists in state dict, convert from torch tensor -> numpy array
    import numpy as np

    for key, value in state.items():
        if isinstance(value, dict):
            state[key] = recursively_convert_from_torch(value)
        elif isinstance(value, th.Tensor):
            state[key] = value.cpu().numpy()
        elif (isinstance(value, list) or isinstance(value, tuple)) and len(value) > 0:
            if isinstance(value[0], dict):
                state[key] = [recursively_convert_from_torch(val) for val in value]
            elif isinstance(value[0], th.Tensor):
                state[key] = [tensor.numpy() for tensor in value]
            elif isinstance(value[0], int) or isinstance(value[0], float):
                state[key] = np.array(value)
    return state


def h5py_group_to_torch(group):
    state = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            state[key] = h5py_group_to_torch(value)
        else:
            state[key] = th.from_numpy(value[()])
    return state


@th.jit.script
def multi_dim_linspace(start: th.Tensor, stop: th.Tensor, num: int, endpoint: bool = True) -> th.Tensor:
    """
    Generate a tensor with evenly spaced values along multiple dimensions.
    This function creates a tensor where each slice along the first dimension
    contains values linearly interpolated between the corresponding elements
    of 'start' and 'stop'. It's similar to numpy.linspace but works with
    multi-dimensional inputs in PyTorch.
    Args:
        start (th.Tensor): Starting values for each dimension.
        stop (th.Tensor): Ending values for each dimension.
        num (int): Number of samples to generate along the interpolated dimension.
        endpoint (bool, optional): If True, stop is the last sample. Otherwise, it is not included.
    Returns:
        th.Tensor: A tensor of shape (num, *start.shape) containing the interpolated values.
    Example:
        >>> start = th.tensor([0.0, 10.0, 100.0])
        >>> stop = th.tensor([1.0, 20.0, 200.0])
        >>> result = multi_dim_linspace(start, stop, num=5, endpoint=True)
        >>> print(result.shape)
        torch.Size([5, 3])
        >>> print(result)
        tensor([[  0.0000,  10.0000, 100.0000],
                [  0.2500,  12.5000, 125.0000],
                [  0.5000,  15.0000, 150.0000],
                [  0.7500,  17.5000, 175.0000],
                [  1.0000,  20.0000, 200.0000]])
        >>> result = multi_dim_linspace(start, stop, num=5, endpoint=False)
        >>> print(result.shape)
        torch.Size([5, 3])
        >>> print(result)
        tensor([[  0.0000,  10.0000, 100.0000],
                [  0.2000,  12.0000, 120.0000],
                [  0.4000,  14.0000, 140.0000],
                [  0.6000,  16.0000, 160.0000],
                [  0.8000,  18.0000, 180.0000]])
    """
    if endpoint:
        steps = th.linspace(0, 1, num, dtype=start.dtype, device=start.device)
    else:
        steps = th.linspace(0, 1, num + 1, dtype=start.dtype, device=start.device)[:-1]

    # Create a new shape for broadcasting
    new_shape = [num] + [1] * start.dim()
    steps = steps.reshape(new_shape)

    return start + steps * (stop - start)
