import numpy as np
import os.path
import torch
from PIL import Image
from typing import Union


__all__ = [
    "np_dtype_size",
    "torch_dtype",
    "torch_device",
    "torch_dtype_size",
    "any_to_torch_tensor",
    "any_to_numpy",
    "any_to_float",
    "img_to_tensor",
]


def np_dtype_size(dtype: Union[str, np.dtype]) -> int:
    return np.dtype(dtype).itemsize


_TORCH_DTYPE_TABLE = {
    torch.bool: 1,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.short: 2,
    torch.int32: 4,
    torch.int: 4,
    torch.int64: 8,
    torch.long: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.half: 2,
    torch.float32: 4,
    torch.float: 4,
    torch.float64: 8,
    torch.double: 8,
}


def torch_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f'"{dtype}" is not a valid torch dtype')
        assert isinstance(dtype, torch.dtype), f"dtype {dtype} is not a valid torch tensor type"
        return dtype
    else:
        raise NotImplementedError(f"{dtype} not supported")


def torch_device(device: Union[str, int, None]) -> torch.device:
    """
    Args:
        device:
            - "auto": use current torch context device, same as `.to('cuda')`
            - int: negative for CPU, otherwise GPU index
    """
    if device is None:
        return None
    elif device == "auto":
        return torch.device("cuda")
    elif isinstance(device, int) and device < 0:
        return torch.device("cpu")
    else:
        return torch.device(device)


def torch_dtype_size(dtype: Union[str, torch.dtype]) -> int:
    return _TORCH_DTYPE_TABLE[torch_dtype(dtype)]


def _convert_then_transfer(x, dtype, device, copy, non_blocking):
    x = x.to(dtype=dtype, copy=copy, non_blocking=non_blocking)
    return x.to(device=device, copy=False, non_blocking=non_blocking)


def _transfer_then_convert(x, dtype, device, copy, non_blocking):
    x = x.to(device=device, copy=copy, non_blocking=non_blocking)
    return x.to(dtype=dtype, copy=False, non_blocking=non_blocking)


def any_to_torch_tensor(
    x,
    dtype: Union[str, torch.dtype, None] = None,
    device: Union[str, int, torch.device, None] = None,
    copy=False,
    non_blocking=False,
    smart_optimize: bool = True,
):
    dtype = torch_dtype(dtype)
    device = torch_device(device)

    if not isinstance(x, (torch.Tensor, np.ndarray)):
        # x is a primitive python sequence
        x = torch.tensor(x, dtype=dtype)
        copy = False

    # This step does not create any copy.
    # If x is a numpy array, simply wraps it in Tensor. If it's already a Tensor, do nothing.
    x = torch.as_tensor(x)
    # avoid passing None to .to(), PyTorch 1.4 bug
    dtype = dtype or x.dtype
    device = device or x.device

    if not smart_optimize:
        # do a single stage type conversion and transfer
        return x.to(dtype=dtype, device=device, copy=copy, non_blocking=non_blocking)

    # we have two choices: (1) convert dtype and then transfer to GPU
    # (2) transfer to GPU and then convert dtype
    # because CPU-to-GPU memory transfer is the bottleneck, we will reduce it as
    # much as possible by sending the smaller dtype

    src_dtype_size = torch_dtype_size(x.dtype)

    # destination dtype size
    if dtype is None:
        dest_dtype_size = src_dtype_size
    else:
        dest_dtype_size = torch_dtype_size(dtype)

    if x.dtype != dtype or x.device != device:
        # a copy will always be performed, no need to force copy again
        copy = False

    if src_dtype_size > dest_dtype_size:
        # better to do conversion on one device (e.g. CPU) and then transfer to another
        return _convert_then_transfer(x, dtype, device, copy, non_blocking)
    elif src_dtype_size == dest_dtype_size:
        # when equal, we prefer to do the conversion on whichever device that's GPU
        if x.device.type == "cuda":
            return _convert_then_transfer(x, dtype, device, copy, non_blocking)
        else:
            return _transfer_then_convert(x, dtype, device, copy, non_blocking)
    else:
        # better to transfer data across device first, and then do conversion
        return _transfer_then_convert(x, dtype, device, copy, non_blocking)


def any_to_numpy(
    x,
    dtype: Union[str, np.dtype, None] = None,
    copy: bool = False,
    non_blocking: bool = False,
    smart_optimize: bool = True,
):
    if isinstance(x, torch.Tensor):
        x = any_to_torch_tensor(
            x,
            dtype=dtype,
            device="cpu",
            copy=copy,
            non_blocking=non_blocking,
            smart_optimize=smart_optimize,
        )
        return x.detach().numpy()
    else:
        # primitive python sequence or ndarray
        return np.array(x, dtype=dtype, copy=copy)


def any_to_float(x, strict: bool = False):
    """
    Convert a singleton torch tensor or ndarray to float

    Args:
        strict: True to check if the input is a singleton and raise Exception if not.
            False to return the original value if not a singleton
    """

    if torch.is_tensor(x) and x.numel() == 1:
        return float(x)
    elif isinstance(x, np.ndarray) and x.size == 1:
        return float(x)
    else:
        if strict:
            raise ValueError(f"{x} cannot be converted to a single float.")
        else:
            return x


def img_to_tensor(file_path: str, dtype=None, device=None, add_batch_dim: bool = False):
    """
    Args:
        scale_255: if True, scale to [0, 255]
        add_batch_dim: if 3D, add a leading batch dim

    Returns:
        tensor between [0, 255]

    """
    # image path
    pic = Image.open(os.path.expanduser(file_path)).convert("RGB")
    # code referenced from torchvision.transforms.functional.to_tensor
    # handle PIL Image
    assert pic.mode == "RGB"
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()

    img = any_to_torch_tensor(img, dtype=dtype, device=device)
    if add_batch_dim:
        img.unsqueeze_(dim=0)
    return img
