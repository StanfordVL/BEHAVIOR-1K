import numpy as np
import torch as th
import tree
import functools
from omnigibson.learning.utils.config_utils import meta_decorator
from typing import List, Union, Dict


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
def torch_to_numpy(tensor_struct):
    def safe_convert(x):
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, (int, float, bool)):
            return np.array(x)
        else:
            # leave strings, np array or other objects untouched
            return x

    return tree.map_structure(safe_convert, tensor_struct)


@make_recursive_func
def any_ones_like(x: Union[Dict, np.ndarray, th.Tensor, int, float, np.number]):
    """Returns a one-filled object of the same (d)type and shape as the input.
    The difference between this and `np.ones_like()` is that this works well
    with `np.number`, `int`, `float`, and `jax.numpy.DeviceArray` objects without
    converting them to `np.ndarray`s.
    Args:
      x: The object to replace with 1s.
    Returns:
      A one-filed object of the same (d)type and shape as the input.
    """
    if isinstance(x, (int, float, np.number)):
        return type(x)(1)
    elif th.is_tensor(x):
        return th.ones_like(x)
    elif isinstance(x, np.ndarray):
        return np.ones_like(x)
    else:
        raise ValueError(f"Input ({type(x)}) must be either a numpy array, a tensor, an int, or a float.")


def any_stack(xs: List, *, dim: int = 0):
    """
    Works for both th Tensor and numpy array
    """

    def _any_stack_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.stack(xs, axis=dim)
        elif th.is_tensor(x):
            return th.stack(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_stack_helper, *xs)


def get_batch_size(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and th tensor
        strict: True to check all batch sizes are the same
    """

    def _get_batch_size(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        elif th.is_tensor(x):
            return x.size(0)
        else:
            return len(x)

    xs = tree.flatten(x)

    if strict:
        batch_sizes = [_get_batch_size(x) for x in xs]
        assert all(
            b == batch_sizes[0] for b in batch_sizes
        ), f"batch sizes must all be the same in nested structure: {batch_sizes}"
        return batch_sizes[0]
    else:
        return _get_batch_size(xs[0])


@make_recursive_func
def any_slice(x, slice):
    """
    Args:
        slice: you can use np.s_[...] to return the slice object
    """
    if isinstance(x, (np.ndarray, th.Tensor)):
        return x[slice]
    else:
        return x


def sequential_sum_balanced_partitioning(nums, M, i):
    """
    Split a list of numbers into M partitions, where the i-th partition is returned.
    The i-th partition is balanced such that the sum of the numbers in each partition
    is as equal as possible.
    NOTE: if sum not divisible by M, the first `sum % M` partitions will have one more element.
    Args:
        nums: list of numbers to be partitioned
        M: number of partitions
        i: index of the partition to be returned (0-indexed)
    Returns:
        start_idx: starting index of the i-th partition
        start_offset: offset of the first element in the i-th partition
        end_idx: ending index of the i-th partition (not inclusive)
        end_offset: offset of the last element in the i-th partition
    Example:
        nums = [1, 2, 3, 4, 5, 6]
        M = 3
        i = 1
        sequential_sum_balanced_partitioning(nums, M, i)
        Returns: (3, 1, 4, 4)
    """
    total = sum(nums)
    target = total // M
    num_offsets = total % M

    acc = 0
    start_idx = end_idx = -1
    start_offset = end_offset = -1

    # actual start / end indices of the i-th chunk
    chunk_start_idx = target * i + min(num_offsets, i)
    chunk_end_idx = target * (i + 1) + min(num_offsets, i + 1)
    # find which number chunk_start_idx and chunk_end_idx fall into
    for idx, num in enumerate(nums):
        if start_idx == -1 and acc + num > chunk_start_idx:
            start_idx = idx
            start_offset = chunk_start_idx - acc
        if end_idx == -1 and acc + num >= chunk_end_idx:
            end_idx = idx
            end_offset = chunk_end_idx - acc
            break
        acc += num
    return start_idx, start_offset, end_idx, end_offset
