import numpy as np


class NumpyTypes:
    FLOAT32 = np.float32
    INT32 = np.int32
    UINT8 = np.uint8
    UINT32 = np.uint32


def to_numpy(arr, dtype=None):
    return np.array(arr, dtype=dtype)
