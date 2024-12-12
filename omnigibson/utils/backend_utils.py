import torch as th
import numpy as np
import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.utils.python_utils import recursively_convert_from_torch


class _ComputeBackend:
    array = None
    int_array = None
    prod = None
    cat = None
    zeros = None
    ones = None
    to_numpy = None
    from_numpy = None
    to_torch = None
    from_torch = None
    from_torch_recursive = None
    allclose = None
    arr_type = None
    as_int = None
    as_float32 = None
    pinv = None
    meshgrid = None
    full = None
    logical_or = None
    all = None
    abs = None
    sqrt = None
    mean = None
    copy = None
    eye = None
    view = None
    arange = None
    where = None
    squeeze = None
    T = None

    @classmethod
    def set_methods(cls, backend):
        for attr, fcn in backend.__dict__.items():
            # Do not override reserved functions
            if attr.startswith("__"):
                continue
            # Set function to this backend
            setattr(cls, attr, fcn)


class _ComputeTorchBackend(_ComputeBackend):
    array = lambda *args: th.tensor(*args, dtype=th.float32)
    int_array = lambda *args: th.tensor(*args, dtype=th.int32)
    prod = th.prod
    cat = th.cat
    zeros = lambda *args: th.zeros(*args, dtype=th.float32)
    ones = lambda *args: th.ones(*args, dtype=th.float32)
    to_numpy = lambda x: x.numpy()
    from_numpy = lambda x: th.from_numpy()
    to_torch = lambda x: x
    from_torch = lambda x: x
    from_torch_recursive = lambda dic: dic
    allclose = th.allclose
    arr_type = th.Tensor
    as_int = lambda arr: arr.int()
    as_float32 = lambda arr: arr.float()
    pinv = th.linalg.pinv
    meshgrid = lambda idx_a, idx_b: th.meshgrid(idx_a, idx_b, indexing="xy")
    full = lambda shape, fill_value: th.full(shape, fill_value, dtype=th.float32)
    logical_or = th.logical_or
    all = th.all
    abs = th.abs
    sqrt = th.sqrt
    mean = lambda val, dim=None, keepdim=False: th.mean(val, dim=dim, keepdim=keepdim)
    copy = lambda arr: arr.clone()
    eye = th.eye
    view = lambda arr, shape: arr.view(shape)
    arange = th.arange
    where = th.where
    squeeze = lambda arr, dim=None: arr.squeeze(dim=dim)
    T = TT


class _ComputeNumpyBackend(_ComputeBackend):
    array = lambda *args: np.array(*args, dtype=np.float32)
    int_array = lambda *args: np.array(*args, dtype=np.int32)
    prod = np.prod
    cat = np.concatenate
    zeros = lambda *args: np.zeros(*args, dtype=np.float32)
    ones = lambda *args: np.ones(*args, dtype=np.float32)
    to_numpy = lambda x: x
    from_numpy = lambda x: x
    to_torch = lambda x: th.from_numpy(x)
    from_torch = lambda x: x.numpy()
    from_torch_recursive = recursively_convert_from_torch
    allclose = np.allclose
    arr_type = np.ndarray
    as_int = lambda arr: arr.astype(int)
    as_float32 = lambda arr: arr.astype(np.float32)
    pinv = np.linalg.pinv
    meshgrid = lambda idx_a, idx_b: np.ix_(idx_a, idx_b)
    full = lambda shape, fill_value: np.full(shape, fill_value, dtype=np.float32)
    logical_or = np.logical_or
    all = np.all
    abs = np.abs
    sqrt = np.sqrt
    mean = lambda val, dim=None, keepdim=False: np.mean(val, axis=dim, keepdims=keepdim)
    copy = lambda arr: np.array(arr)
    eye = np.eye
    view = lambda arr, shape: arr.reshape(shape)
    arange = np.arange
    where = np.where
    squeeze = lambda arr, dim=None: arr.squeeze(axis=dim)
    T = NT


_compute_backend = _ComputeBackend
