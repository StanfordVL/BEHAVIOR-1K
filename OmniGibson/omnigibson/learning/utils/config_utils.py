import functools
import warnings
from copy import deepcopy
from omegaconf import OmegaConf
from typing import Literal


_NO_INSTANTIATE = "__no_instantiate__"  # return config as-is


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


@call_once(on_second_call="noop")
def register_omegaconf_resolvers():
    import numpy as np

    OmegaConf.register_new_resolver("_optional", lambda v: f"_{v}" if v else "")
    OmegaConf.register_new_resolver("optional_", lambda v: f"{v}_" if v else "")
    OmegaConf.register_new_resolver("_optional_", lambda v: f"_{v}_" if v else "")
    OmegaConf.register_new_resolver("__optional", lambda v: f"__{v}" if v else "")
    OmegaConf.register_new_resolver("optional__", lambda v: f"{v}__" if v else "")
    OmegaConf.register_new_resolver("__optional__", lambda v: f"__{v}__" if v else "")
    OmegaConf.register_new_resolver("iftrue", lambda cond, v_default: cond if cond else v_default)
    OmegaConf.register_new_resolver("ifelse", lambda cond, v1, v2="": v1 if cond else v2)
    OmegaConf.register_new_resolver("ifequal", lambda query, key, v1, v2: v1 if query == key else v2)
    OmegaConf.register_new_resolver("intbool", lambda cond: 1 if cond else 0)
    OmegaConf.register_new_resolver("mult", lambda *x: np.prod(x).tolist())
    OmegaConf.register_new_resolver("add", lambda *x: sum(x))
    OmegaConf.register_new_resolver("div", lambda x, y: x / y)
    OmegaConf.register_new_resolver("intdiv", lambda x, y: x // y)

    # try each key until the key exists. Useful for multiple classes that have different
    # names for the same key
    def _try_key(cfg, *keys):
        for k in keys:
            if k in cfg:
                return cfg[k]
        raise KeyError(f"no key in {keys} is valid")

    OmegaConf.register_new_resolver("trykey", _try_key)
    # replace `resnet.gn.ws` -> `resnet_gn_ws`, because omegaconf doesn't support
    # keys with dots. Useful for generating run name with dots
    OmegaConf.register_new_resolver("underscore_to_dots", lambda s: s.replace("_", "."))

    def _no_instantiate(cfg):
        cfg = deepcopy(cfg)
        cfg[_NO_INSTANTIATE] = True
        return cfg

    OmegaConf.register_new_resolver("no_instantiate", _no_instantiate)
