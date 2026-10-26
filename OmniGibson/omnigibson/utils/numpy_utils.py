import numpy as np
import torch as th


class NumpyTypes:
    FLOAT32 = np.float32
    INT32 = np.int32
    INT64 = np.int64
    UINT8 = np.uint8
    UINT32 = np.uint32


def numpy_to_torch(np_array, dtype=th.float32, device="cpu"):
    if device == "cpu":
        return th.from_numpy(np_array).to(dtype)
    else:
        assert device.startswith("cuda")
        return th.tensor(np_array, dtype=dtype, device=device)


def vtarray_to_torch(vtarray, dtype=th.float32, device="cpu"):
    np_array = np.array(vtarray)
    return numpy_to_torch(np_array, dtype=dtype, device=device)


def gf_quat_to_torch(gf_quat, dtype=th.float32, device="cpu"):
    # Read the Gf quat directly (xyzw) rather than via isaacsim, which is unavailable without Kit
    np_array = np.array([*gf_quat.GetImaginary(), gf_quat.GetReal()])
    return numpy_to_torch(np_array, dtype=dtype, device=device)


def pil_to_tensor(pil_image):
    return th.tensor(np.array(pil_image), dtype=th.uint8)


def list_to_np_array(list):
    return np.array(list)
