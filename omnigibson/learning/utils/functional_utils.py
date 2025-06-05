import functools
import tree
import warnings
from collections import abc
from typing import Any, List, Literal


def is_sequence(obj: Any) -> bool:
    """
    Returns:
        bool: True if the sequence is a collections.Sequence and not a string.
    """
    return isinstance(obj, abc.Sequence) and not isinstance(obj, str)


def is_mapping(obj: Any) -> bool:
    """
    Returns:
        bool: True if the sequence is a collections.Mapping
    """
    return isinstance(obj, abc.Mapping)


def unstack_sequence_fields(struct: Any, batch_size: int) -> List[Any]:
    """Converts a struct of batched arrays to a list of structs.

    Args:
      struct: An (arbitrarily nested) structure of arrays.
      batch_size: The length of the leading dimension of each array in the struct.
        This is assumed to be static and known.

    Returns:
      A list of structs with the same structure as `struct`, where each leaf node
       is an unbatched element of the original leaf node.
    """

    return [tree.map_structure(lambda s, i=i: s[i], struct) for i in range(batch_size)]


def meta_decorator(decor):
    """
    a decorator decorator, allowing the wrapped decorator to be used as:
    @decorator(*args, **kwargs)
    def callable()
      -- or --
    @decorator  # without parenthesis, args and kwargs will use default
    def callable()

    Args:
      decor: a decorator whose first argument is a callable (function or class
        to be decorated), and the rest of the arguments can be omitted as default.
        decor(f, ... the other arguments must have default values)

    Warning:
      decor can NOT be a function that receives a single, callable argument.
      See stackoverflow: http://goo.gl/UEYbDB
    """
    single_callable = lambda args, kwargs: len(args) == 1 and len(kwargs) == 0 and callable(args[0])

    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f.
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_decorator
def call_once(func, on_second_call: Literal["noop", "raise", "warn"] = "noop"):
    """
    Decorator to ensure that a function is only called once.

    Args:
      on_second_call (str): what happens when the function is called a second time.
    """
    assert on_second_call in [
        "noop",
        "raise",
        "warn",
    ], "mode must be one of 'noop', 'raise', 'warn'"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if wrapper._called:
            if on_second_call == "raise":
                raise RuntimeError(f"{func.__name__} has already been called. Can only call once.")
            elif on_second_call == "warn":
                warnings.warn(f"{func.__name__} has already been called. Should only call once.")
        else:
            wrapper._called = True
            return func(*args, **kwargs)

    wrapper._called = False
    return wrapper


def pack_varargs(args):
    """
    Pack *args or a single list arg as list

    def f(*args):
        arg_list = pack_varargs(args)
        # arg_list is now packed as a list
    """
    assert isinstance(args, tuple), "please input the tuple `args` as in *args"
    if len(args) == 1 and is_sequence(args[0]):
        return args[0]
    else:
        return args


def accumulate(iterable, fn=lambda x, y: x + y):
    """
    Return running totals
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total
