import numpy as np
import torch as th
import tree
import functools
from omnigibson.learning.utils.config_utils import meta_decorator
from typing import List


@meta_decorator
def make_recursive_func(fn, *, with_path=False):
    """
    Decorator that turns a function that works on a single array/tensor to working on
    arbitrary nested structures.
    """

    @functools.wraps(fn)
    def _wrapper(tensor_struct, *args, **kwargs):
        if with_path:
            return tree.map_structure_with_path(lambda paths, x: fn(paths, x, *args, **kwargs), tensor_struct)
        else:
            return tree.map_structure(lambda x: fn(x, *args, **kwargs), tensor_struct)

    return _wrapper


def any_concat(xs: List, *, dim: int = 0):
    """
    Works for both th Tensor and numpy array
    """

    def _any_concat_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif th.is_tensor(x):
            return th.cat(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_concat_helper, *xs)


@make_recursive_func
def any_to_torch(tensor_struct, device: str = "cuda"):
    """
    Converts all arrays/tensors in a nested structure to PyTorch tensors.
    """
    return tree.map_structure(lambda x: th.tensor(x, dtype=th.float32).to(device), tensor_struct)