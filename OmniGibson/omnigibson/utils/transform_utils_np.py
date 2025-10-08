"""
Utility functions of matrix and vector transformations.

NOTE: This file has a 1-to-1 correspondence to transform_utils.py. By default, we use scipy for most transform-related
    operations, but we optionally implement numba versions for functions that are often called in "batch" mode

NOTE: convention for quaternions is (x, y, z, w)
"""

import math

import numpy as np
from numba import jit, prange
from scipy.spatial.transform import Rotation as R

PI = np.pi
EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def copysign(a, b):
    a = np.array(a).repeat(b.shape[0])
    return np.abs(a) * np.sign(b)


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms alogn specified axes."""
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    norm = anorm(v, axis=axis, keepdims=True)
    return v / np.where(norm < eps, eps, norm)


def dot(v1, v2, dim=-1, keepdim=False):
    return np.sum(v1 * v2, axis=dim, keepdims=keepdim)


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0 * v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0 * v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quat_apply(quat, vec):
    """
    Apply a quaternion rotation to a vector (equivalent to R.from_quat(x).apply(y))
    Args:
        quat (np.array): (4,) or (N, 4) or (N, 1, 4) quaternion in (x, y, z, w) format
        vec (np.array): (3,) or (M, 3) or (1, M, 3) vector to rotate

    Returns:
        np.array: (M, 3) or (N, M, 3) rotated vector
    """
    return R.from_quat(quat).apply(vec)


def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q (np.array): a 4-dim array corresponding to a quaternion
        to (str): either 'xyzw' or 'wxyz', determining which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat_multiply(quaternion1, quaternion0):
    if quaternion1.dtype != np.float32:
        quaternion1 = quaternion1.astype(np.float32)
    if quaternion0.dtype != np.float32:
        quaternion0 = quaternion0.astype(np.float32)
    return _quat_multiply(quaternion1, quaternion0)


@jit(nopython=True)
def _quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0).

    E.g.:
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = np.split(quaternion0, 4, axis=-1)
    x1, y1, z1, w1 = np.split(quaternion1, 4, axis=-1)
    return np.concatenate(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        axis=-1,
    )


def quat_conjugate(quaternion):
    """
    Return conjugate of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion conjugate
    """
    n_dims = len(quaternion.shape)
    # Reshape to explicitly handle batched calls
    return quaternion * np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32).reshape([1] * (n_dims - 1) + [4])


def quat_inverse(quaternion):
    """
    Return inverse of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion inverse
    """
    return quat_conjugate(quaternion) / np.sum(quaternion * quaternion, axis=-1, keepdims=True)


def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1
    Always returns the shorter rotation path.

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion or (..., 4) batched quaternions
        quaternion0 (np.array): (x,y,z,w) quaternion or (..., 4) batched quaternions

    Returns:
        np.array: (x,y,z,w) quaternion distance or (..., 4) batched quaternion distances
    """
    # Compute dot product along the last axis (quaternion components)
    d = np.sum(quaternion0 * quaternion1, axis=-1, keepdims=True)
    # If dot product is negative, negate one quaternion to get shorter path
    quaternion1 = np.where(d < 0.0, -quaternion1, quaternion1)

    return quat_multiply(quaternion1, quat_inverse(quaternion0))


def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    E.g.:
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True

    >>> q = quat_slerp(q0, q1, 1.0)
    >>> np.allclose(q, q1)
    True

    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True

    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


@jit(nopython=True)
def random_quaternion(num_quaternions=1):
    """
    Generate random rotation quaternions, uniformly distributed over SO(3).

    Arguments:
        num_quaternions (int): number of quaternions to generate (default: 1)

    Returns:
        np.array: A tensor of shape (num_quaternions, 4) containing random unit quaternions.
    """
    # Generate four random numbers between 0 and 1
    rand = np.random.rand(num_quaternions, 4)

    # Use the formula from Ken Shoemake's "Uniform Random Rotations"
    r1 = np.sqrt(1.0 - rand[:, 0])
    r2 = np.sqrt(rand[:, 0])
    t1 = 2 * np.pi * rand[:, 1]
    t2 = 2 * np.pi * rand[:, 2]

    quaternions = np.stack((r1 * np.sin(t1), r1 * np.cos(t1), r2 * np.sin(t2), r2 * np.cos(t2)), axis=1)

    return quaternions


def random_axis_angle(angle_limit=None, random_state=None):
    """
    Samples an axis-angle rotation by first sampling a random axis
    and then sampling an angle. If @angle_limit is provided, the size
    of the rotation angle is constrained.

    If @random_state is provided (instance of np.random.RandomState), it
    will be used to generate random numbers.

    Args:
        angle_limit (None or float): If set, determines magnitude limit of angles to generate
        random_state (None or RandomState): RNG to use if specified

    Raises:
        AssertionError: [Invalid RNG]
    """
    if angle_limit is None:
        angle_limit = 2.0 * np.pi

    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState)
        npr = random_state
    else:
        npr = np.random

    # sample random axis using a normalized sample from spherical Gaussian.
    # see (http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/)
    # for why it works.
    random_axis = npr.randn(3)
    random_axis /= np.linalg.norm(random_axis)
    random_angle = npr.uniform(low=0.0, high=angle_limit)
    return random_axis, random_angle


def quat2mat(quaternion):
    if quaternion.dtype != np.float32:
        quaternion = quaternion.astype(np.float32)
    return _quat2mat(quaternion)


@jit(nopython=True)
def _quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (x, y, z, w).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    # broadcast array is necessary to use numba parallel mode
    q1, q2 = np.broadcast_arrays(quaternion[..., np.newaxis], quaternion[..., np.newaxis, :])
    outer = q1 * q2

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rotation_matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float32)

    rotation_matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrix[..., 0, 1] = 2 * (xy - zw)
    rotation_matrix[..., 0, 2] = 2 * (xz + yw)

    rotation_matrix[..., 1, 0] = 2 * (xy + zw)
    rotation_matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrix[..., 1, 2] = 2 * (yz - xw)

    rotation_matrix[..., 2, 0] = 2 * (xz - yw)
    rotation_matrix[..., 2, 1] = 2 * (yz + xw)
    rotation_matrix[..., 2, 2] = 1 - 2 * (xx + yy)

    return rotation_matrix


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): (..., 3, 3) rotation matrix

    Returns:
        np.array: (..., 4) (x,y,z,w) float quaternion angles
    """
    return R.from_matrix(rmat).as_quat()


@jit(nopython=True, fastmath=True)
def _norm_2d_final_dim(mat):
    n_elements = mat.shape[0]
    out = np.zeros(n_elements, dtype=np.float32)
    for i in prange(n_elements):
        vec = mat[i]
        out[i] = np.sqrt(np.sum(vec * vec))
    return out


@jit(nopython=True)
def mat2quat_batch(rmat):
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat (torch.Tensor): (3, 3) or (..., 3, 3) rotation matrix
    Returns:
        torch.Tensor: (4,) or (..., 4) (x,y,z,w) float quaternion angles
    """
    batch_shape = rmat.shape[:-2]
    mat_flat = rmat.reshape(-1, 3, 3)

    m00, m01, m02 = mat_flat[:, 0, 0], mat_flat[:, 0, 1], mat_flat[:, 0, 2]
    m10, m11, m12 = mat_flat[:, 1, 0], mat_flat[:, 1, 1], mat_flat[:, 1, 2]
    m20, m21, m22 = mat_flat[:, 2, 0], mat_flat[:, 2, 1], mat_flat[:, 2, 2]

    trace = m00 + m11 + m22

    trace_positive = trace > 0
    cond1 = (m00 > m11) & (m00 > m22) & ~trace_positive
    cond2 = (m11 > m22) & ~(trace_positive | cond1)
    cond3 = ~(trace_positive | cond1 | cond2)

    # Trace positive condition
    sq = np.where(trace_positive, np.sqrt(trace + 1.0) * 2.0, np.zeros_like(trace))
    qw = np.where(trace_positive, 0.25 * sq, np.zeros_like(trace))
    qx = np.where(trace_positive, (m21 - m12) / sq, np.zeros_like(trace))
    qy = np.where(trace_positive, (m02 - m20) / sq, np.zeros_like(trace))
    qz = np.where(trace_positive, (m10 - m01) / sq, np.zeros_like(trace))

    # Condition 1
    sq = np.where(cond1, np.sqrt(1.0 + m00 - m11 - m22) * 2.0, sq)
    qw = np.where(cond1, (m21 - m12) / sq, qw)
    qx = np.where(cond1, 0.25 * sq, qx)
    qy = np.where(cond1, (m01 + m10) / sq, qy)
    qz = np.where(cond1, (m02 + m20) / sq, qz)

    # Condition 2
    sq = np.where(cond2, np.sqrt(1.0 + m11 - m00 - m22) * 2.0, sq)
    qw = np.where(cond2, (m02 - m20) / sq, qw)
    qx = np.where(cond2, (m01 + m10) / sq, qx)
    qy = np.where(cond2, 0.25 * sq, qy)
    qz = np.where(cond2, (m12 + m21) / sq, qz)

    # Condition 3
    sq = np.where(cond3, np.sqrt(1.0 + m22 - m00 - m11) * 2.0, sq)
    qw = np.where(cond3, (m10 - m01) / sq, qw)
    qx = np.where(cond3, (m02 + m20) / sq, qx)
    qy = np.where(cond3, (m12 + m21) / sq, qy)
    qz = np.where(cond3, 0.25 * sq, qz)

    quat = np.stack((qx, qy, qz, qw), axis=-1)

    # Normalize the quaternion
    quat = quat / _norm_2d_final_dim(quat)[..., np.newaxis]

    # Reshape to match input batch shape
    quat = quat.reshape(batch_shape + (4,))

    return quat


def decompose_mat(hmat):
    """Batched decompose_mat function - assumes input is already batched

    Args:
        hmat (np.ndarray): (B, 4, 4) batch of homogeneous matrices

    Returns:
        scale: (B, 3) scale factors
        shear: (B, 3) shear factors
        quat: (B, 4) quaternions
        translate: (B, 3) translations
    """
    batch_size = hmat.shape[0]
    M = np.array(hmat, dtype=np.float32).transpose(0, 2, 1)  # (B, 4, 4) transposed

    # Check M[3, 3] for all batch items
    diag_vals = M[:, 3, 3]  # (B,)
    if np.any(np.abs(diag_vals) < EPS):
        raise ValueError("Some M[3, 3] values are zero")

    M = M / diag_vals[:, np.newaxis, np.newaxis]  # (B, 4, 4)
    P = M.copy()
    P[:, :, 3] = np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :].repeat(batch_size, axis=0)

    det_P = np.linalg.det(P[:, :3, :3])  # (B,)
    if np.any(np.abs(det_P) < EPS):
        raise ValueError("Some matrices are singular and cannot be decomposed")

    if not np.allclose(M[:, :3, 3], 0.0):
        raise ValueError("Some matrices have perspective components")

    scale = np.zeros((batch_size, 3))
    shear = np.zeros((batch_size, 3))

    translate = M[:, 3, :3].copy()  # (B, 3)
    M[:, 3, :3] = 0.0

    row = M[:, :3, :3].copy()  # (B, 3, 3)

    # Scale and orthogonalize rows
    scale[:, 0] = np.linalg.norm(row[:, 0], axis=-1)  # (B,)
    row[:, 0] = row[:, 0] / scale[:, 0, np.newaxis]  # (B, 3)

    shear[:, 0] = np.sum(row[:, 0] * row[:, 1], axis=-1)  # (B,)
    row[:, 1] = row[:, 1] - row[:, 0] * shear[:, 0, np.newaxis]  # (B, 3)

    scale[:, 1] = np.linalg.norm(row[:, 1], axis=-1)  # (B,)
    row[:, 1] = row[:, 1] / scale[:, 1, np.newaxis]  # (B, 3)
    shear[:, 0] = shear[:, 0] / scale[:, 1]

    shear[:, 1] = np.sum(row[:, 0] * row[:, 2], axis=-1)  # (B,)
    row[:, 2] = row[:, 2] - row[:, 0] * shear[:, 1, np.newaxis]  # (B, 3)

    shear[:, 2] = np.sum(row[:, 1] * row[:, 2], axis=-1)  # (B,)
    row[:, 2] = row[:, 2] - row[:, 1] * shear[:, 2, np.newaxis]  # (B, 3)

    scale[:, 2] = np.linalg.norm(row[:, 2], axis=-1)  # (B,)
    row[:, 2] = row[:, 2] / scale[:, 2, np.newaxis]  # (B, 3)
    shear[:, 1:] = shear[:, 1:] / scale[:, 2, np.newaxis]

    # Check orientation
    cross_product = np.cross(row[:, 1], row[:, 2], axis=-1)  # (B, 3)
    dot_product = np.sum(row[:, 0] * cross_product, axis=-1)  # (B,)
    neg_mask = dot_product < 0  # (B,)

    scale = np.where(neg_mask[:, np.newaxis], -scale, scale)  # (B, 3)
    row = np.where(neg_mask[:, np.newaxis, np.newaxis], -row, row)  # (B, 3, 3)

    # Convert to quaternions - assuming mat2quat can handle batched input
    quat = mat2quat(row.transpose(0, 2, 1))  # (B, 4)

    return scale, shear, quat, translate


def mat2pose(hmat):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat (np.array): a 4x4 homogeneous matrix

    Returns:
        2-tuple:
            - (np.array) (3,) position array in cartesian coordinates
            - (np.array) (4,) orientation array in quaternion form
    """
    # Add batch dimension, process, then squeeze
    hmat_batched = hmat[np.newaxis, ...]  # (1, 4, 4)
    _, _, quat, translate = decompose_mat(hmat_batched)
    return translate.squeeze(0), quat.squeeze(0)


def mat2pose_batched(hmat):
    """
    Converts batched homogeneous 4x4 matrices into poses.

    Args:
        hmat (np.array): (B, 4, 4) batch of homogeneous matrices

    Returns:
        2-tuple:
            - (np.array) (B, 3) position arrays in cartesian coordinates
            - (np.array) (B, 4) orientation arrays in quaternion form
    """
    _, _, quat, translate = decompose_mat(hmat)
    return translate, quat


def vec2quat(vec, up=(0, 0, 1.0)):
    """
    Converts given 3d-direction vector @vec to quaternion orientation with respect to another direction vector @up

    Args:
        vec (3-array): (x,y,z) direction vector (possible non-normalized)
        up (3-array): (x,y,z) direction vector representing the canonical up direction (possible non-normalized)
    """
    # See https://stackoverflow.com/questions/15873996/converting-a-direction-vector-to-a-quaternion-rotation
    # Take cross product of @up and @vec to get @s_n, and then cross @vec and @s_n to get @u_n
    # Then compose 3x3 rotation matrix and convert into quaternion
    vec_n = vec / np.linalg.norm(vec)  # x
    up_n = up / np.linalg.norm(up)
    s_n = np.cross(up_n, vec_n)  # y
    u_n = np.cross(vec_n, s_n)  # z
    return mat2quat(np.array([vec_n, s_n, u_n]).T)


def euler2quat(euler):
    """
    Converts extrinsic euler angles into quaternion form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: (x,y,z,w) float quaternion angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    return R.from_euler("xyz", euler).as_quat()


def quat2euler(quat):
    """
    Converts extrinsic euler angles into quaternion form

    Args:
        quat (np.array): (x,y,z,w) float quaternion angles

    Returns:
        np.array: (r,p,y) angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    return R.from_quat(quat).as_euler("xyz")


def euler2mat(euler):
    """
    Converts extrinsic euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    return R.from_euler("xyz", euler).as_matrix()


def mat2euler(rmat):
    """
    Converts given rotation matrix to extrinsic euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (r,p,y) converted extrinsic euler angles in radian vec3 float
    """
    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    return R.from_matrix(M).as_euler("xyz")


@jit(nopython=True)
def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = _quat2mat(pose[1])
    homo_pose_mat[:3, 3] = pose[0]
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    return R.from_quat(quat).as_rotvec()


def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    return R.from_rotvec(vec).as_quat()


def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A
    to a homogenous matrix corresponding to the same point C in frame B.

    Args:
        pose_A (np.array): 4x4 matrix corresponding to the pose of C in frame A
        pose_A_in_B (np.array): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        np.array: 4x4 matrix corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return pose_A_in_B.dot(pose_A)


def pose_inv(pose_mat):
    if pose_mat.dtype != np.float32:
        pose_mat = pose_mat.astype(np.float32)
    return _pose_inv(pose_mat)


@jit(nopython=True)
def _pose_inv(pose_mat):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose_mat (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4), dtype=np.float32)
    pose_inv[:3, :3] = pose_mat[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose_mat[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def pose_transform(pos1, quat1, pos0, quat0):
    """
    Conducts forward transform from pose (pos0, quat0) to pose (pos1, quat1):

    pose1 @ pose0, NOT pose0 @ pose1

    Args:
        pos1 (np.ndarray): (x,y,z) position to transform
        quat1 (np.ndarray): (x,y,z,w) orientation to transform
        pos0 (np.ndarray): (x,y,z) initial position
        quat0 (np.ndarray): (x,y,z,w) initial orientation

    Returns:
        2-tuple:
            - (np.array) (x,y,z) position array in cartesian coordinates
            - (np.array) (x,y,z,w) orientation array in quaternion form
    """
    # Get poses
    mat0 = pose2mat((pos0, quat0))
    mat1 = pose2mat((pos1, quat1))

    # Multiply and convert back to pos, quat
    return mat2pose(mat1 @ mat0)


def invert_pose_transform(pos, quat):
    """
    Inverts a pose transform

    Args:
        pos (np.ndarray): (x,y,z) position to transform
        quat (np.ndarray): (x,y,z,w) orientation to transform

    Returns:
        2-tuple:
            - (np.array) (x,y,z) position array in cartesian coordinates
            - (np.array) (x,y,z,w) orientation array in quaternion form
    """
    # Get pose
    mat = pose2mat((pos, quat))

    # Invert pose and convert back to pos, quat
    return mat2pose(pose_inv(mat))


def relative_pose_transform(pos1, quat1, pos0, quat0):
    """
    Computes relative forward transform from pose (pos0, quat0) to pose (pos1, quat1), i.e.: solves:

    pose1 = pose0 @ transform

    Args:
        pos1 (np.ndarray): (x,y,z) position to transform
        quat1 (np.ndarray): (x,y,z,w) orientation to transform
        pos0 (np.ndarray): (x,y,z) initial position
        quat0 (np.ndarray): (x,y,z,w) initial orientation

    Returns:
        2-tuple:
            - (np.array) (x,y,z) position array in cartesian coordinates
            - (np.array) (x,y,z,w) orientation array in quaternion form
    """
    # Get poses
    mat0 = pose2mat((pos0, quat0))
    mat1 = pose2mat((pos1, quat1))

    # Invert pose0 and calculate transform
    return mat2pose(pose_inv(mat0) @ mat1)


def _skew_symmetric_translation(pos_A_in_B):
    """
    Helper function to get a skew symmetric translation matrix for converting quantities
    between frames.

    Args:
        pos_A_in_B (np.array): (x,y,z) position of A in frame B

    Returns:
        np.array: 3x3 skew symmetric translation matrix
    """
    return np.array(
        [
            0.0,
            -pos_A_in_B[2],
            pos_A_in_B[1],
            pos_A_in_B[2],
            0.0,
            -pos_A_in_B[0],
            -pos_A_in_B[1],
            pos_A_in_B[0],
            0.0,
        ]
    ).reshape((3, 3))


def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Converts linear and angular velocity of a point in frame A to the equivalent in frame B.

    Args:
        vel_A (np.array): (vx,vy,vz) linear velocity in A
        ang_vel_A (np.array): (wx,wy,wz) angular velocity in A
        pose_A_in_B (np.array): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (np.array) (vx,vy,vz) linear velocities in frame B
            - (np.array) (wx,wy,wz) angular velocities in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = rot_A_in_B.dot(vel_A) + skew_symm.dot(rot_A_in_B.dot(ang_vel_A))
    ang_vel_B = rot_A_in_B.dot(ang_vel_A)
    return vel_B, ang_vel_B


def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Converts linear and rotational force at a point in frame A to the equivalent in frame B.

    Args:
        force_A (np.array): (fx,fy,fz) linear force in A
        torque_A (np.array): (tx,ty,tz) rotational force (moment) in A
        pose_A_in_B (np.array): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (np.array) (fx,fy,fz) linear forces in frame B
            - (np.array) (tx,ty,tz) moments in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = rot_A_in_B.T.dot(force_A)
    torque_B = -rot_A_in_B.T.dot(skew_symm.dot(force_A)) + rot_A_in_B.T.dot(torque_A)
    return force_B, torque_B


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2 * math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle - 2 * math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi * 2, direc))
        True

        >>> numpy.allclose(2.0, numpy.trace(rotation_matrix(math.pi / 2, direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def transformation_matrix(angle, direction, point=None):
    """
    Returns a 4x4 homogeneous transformation matrix to rotate about axis defined by point and direction.
    Args:
        angle (float): Magnitude of rotation in radians
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (bool): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous transformation matrix
    """
    R = rotation_matrix(angle, direction)

    M = np.eye(4)
    M[:3, :3] = R

    if point is not None:
        # Rotation not about origin
        M[:3, 3] = point - R @ point
    return M


def clip_translation(dpos, limit):
    """
    Limits a translation (delta position) to a specified limit

    Scales down the norm of the dpos to 'limit' if norm(dpos) > limit, else returns immediately

    Args:
        dpos (n-array): n-dim Translation being clipped (e,g.: (x, y, z)) -- numpy array
        limit (float): Value to limit translation by -- magnitude (scalar, in same units as input)

    Returns:
        2-tuple:

            - (np.array) Clipped translation (same dimension as inputs)
            - (bool) whether the value was clipped or not
    """
    input_norm = np.linalg.norm(dpos)
    return (dpos * limit / input_norm, True) if input_norm > limit else (dpos, False)


def clip_rotation(quat, limit):
    """
    Limits a (delta) rotation to a specified limit

    Converts rotation to axis-angle, clips, then re-converts back into quaternion

    Args:
        quat (np.array): (x,y,z,w) rotation being clipped
        limit (float): Value to limit rotation by -- magnitude (scalar, in radians)

    Returns:
        2-tuple:

            - (np.array) Clipped rotation quaternion (x, y, z, w)
            - (bool) whether the value was clipped or not
    """
    clipped = False

    # First, normalize the quaternion
    quat = quat / np.linalg.norm(quat)

    den = np.sqrt(max(1 - quat[3] * quat[3], 0))
    if den == 0:
        # This is a zero degree rotation, immediately return
        return quat, clipped
    else:
        # This is all other cases
        x = quat[0] / den
        y = quat[1] / den
        z = quat[2] / den
        a = 2 * math.acos(quat[3])

    # Clip rotation if necessary and return clipped quat
    if abs(a) > limit:
        a = limit * np.sign(a) / 2
        sa = math.sin(a)
        ca = math.cos(a)
        quat = np.array([x * sa, y * sa, z * sa, ca])
        clipped = True

    return quat, clipped


def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation

    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def get_orientation_error(target_orn, current_orn):
    """
    Returns the difference between two quaternion orientations as a 3 DOF numpy array.
    For use in an impedance controller / task-space PD controller.

    Args:
        target_orn (np.array): (x, y, z, w) desired quaternion orientation
        current_orn (np.array): (x, y, z, w) current quaternion orientation

    Returns:
        orn_error (np.array): (ax,ay,az) current orientation error, corresponds to
            (target_orn - current_orn)
    """
    current_orn = np.array([current_orn[3], current_orn[0], current_orn[1], current_orn[2]])
    target_orn = np.array([target_orn[3], target_orn[0], target_orn[1], target_orn[2]])

    pinv = np.zeros((3, 4))
    pinv[0, :] = [-current_orn[1], current_orn[0], -current_orn[3], current_orn[2]]
    pinv[1, :] = [-current_orn[2], current_orn[3], current_orn[0], -current_orn[1]]
    pinv[2, :] = [-current_orn[3], -current_orn[2], current_orn[1], current_orn[0]]
    orn_error = 2.0 * pinv.dot(np.array(target_orn))
    return orn_error


def get_orientation_diff_in_radian(orn0, orn1):
    """
    Returns the difference between two quaternion orientations in radian

    Args:
        orn0 (np.array): (x, y, z, w)
        orn1 (np.array): (x, y, z, w)

    Returns:
        orn_diff (float): orientation difference in radian
    """
    vec0 = quat2axisangle(orn0)
    vec0 /= np.linalg.norm(vec0)
    vec1 = quat2axisangle(orn1)
    vec1 /= np.linalg.norm(vec1)
    return np.arccos(np.dot(vec0, vec1))


def get_pose_error(target_pose, current_pose):
    """
    Computes the error corresponding to target pose - current pose as a 6-dim vector.
    The first 3 components correspond to translational error while the last 3 components
    correspond to the rotational error.

    Args:
        target_pose (np.array): a 4x4 homogenous matrix for the target pose
        current_pose (np.array): a 4x4 homogenous matrix for the current pose

    Returns:
        np.array: 6-dim pose error.
    """
    error = np.zeros(6)

    # compute translational error
    target_pos = target_pose[:3, 3]
    current_pos = current_pose[:3, 3]
    pos_err = target_pos - current_pos

    # compute rotational error
    r1 = current_pose[:3, 0]
    r2 = current_pose[:3, 1]
    r3 = current_pose[:3, 2]
    r1d = target_pose[:3, 0]
    r2d = target_pose[:3, 1]
    r3d = target_pose[:3, 2]
    rot_err = 0.5 * (np.cross(r1, r1d) + np.cross(r2, r2d) + np.cross(r3, r3d))

    error[:3] = pos_err
    error[3:] = rot_err
    return error


def matrix_inverse(matrix):
    """
    Helper function to have an efficient matrix inversion function.

    Args:
        matrix (np.array): 2d-array representing a matrix

    Returns:
        np.array: 2d-array representing the matrix inverse
    """
    return np.linalg.inv(matrix)


def vecs2axisangle(vec0, vec1):
    """
    Converts the angle from unnormalized 3D vectors @vec0 to @vec1 into an axis-angle representation of the angle

    Args:
        vec0 (np.array): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        vec1 (np.array): (..., 3) (x,y,z) 3D vector, possibly unnormalized
    """
    # Normalize vectors
    vec0 = normalize(vec0, axis=-1)
    vec1 = normalize(vec1, axis=-1)

    # Get cross product for direction of angle, and multiply by arcos of the dot product which is the angle
    return np.cross(vec0, vec1) * np.arccos((vec0 * vec1).sum(-1, keepdims=True))


def vecs2quat(vec0, vec1, normalized=False):
    """
    Converts the angle from unnormalized 3D vectors @vec0 to @vec1 into a quaternion representation of the angle

    Args:
        vec0 (np.array): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        vec1 (np.array): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        normalized (bool): If True, @vec0 and @vec1 are assumed to already be normalized and we will skip the
            normalization step (more efficient)
    """
    # Normalize vectors if requested
    if not normalized:
        vec0 = normalize(vec0, axis=-1)
        vec1 = normalize(vec1, axis=-1)

    # Half-way Quaternion Solution -- see https://stackoverflow.com/a/11741520
    cos_theta = np.sum(vec0 * vec1, axis=-1, keepdims=True)
    quat_unnormalized = np.where(
        cos_theta == -1, np.array([1.0, 0, 0, 0]), np.concatenate([np.cross(vec0, vec1), 1 + cos_theta], axis=-1)
    )
    return quat_unnormalized / np.linalg.norm(quat_unnormalized, axis=-1, keepdims=True)


def align_vector_sets(vec_set1, vec_set2):
    """
    Computes a single quaternion representing the rotation that best aligns vec_set1 to vec_set2.

    Args:
        vec_set1 (np.array): (N, 3) tensor of N 3D vectors
        vec_set2 (np.array): (N, 3) tensor of N 3D vectors

    Returns:
        np.array: (4,) Normalized quaternion representing the overall rotation
    """
    rot, _ = R.align_vectors(vec_set1, vec_set2)
    return rot.as_quat()


def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def deg2rad(deg):
    return deg * np.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / np.pi


def check_quat_right_angle(quat, atol=5e-2):
    """
    Check by making sure the quaternion is some permutation of +/- (1, 0, 0, 0),
    +/- (0.707, 0.707, 0, 0), or +/- (0.5, 0.5, 0.5, 0.5)
    Because orientations are all normalized (same L2-norm), every orientation should have a unique L1-norm
    So we check the L1-norm of the absolute value of the orientation as a proxy for verifying these values

    Args:
        quat (4-array): (x,y,z,w) quaternion orientation to check
        atol (float): Absolute tolerance permitted

    Returns:
        bool: Whether the quaternion is a right angle or not
    """
    return np.any(np.isclose(np.abs(quat).sum(), np.array([1.0, 1.414, 2.0]), atol=atol))


def z_angle_from_quat(quat):
    """Get the angle around the Z axis produced by the quaternion."""
    rotated_X_axis = R.from_quat(quat).apply([1, 0, 0])
    return np.arctan2(rotated_X_axis[1], rotated_X_axis[0])


def integer_spiral_coordinates(n):
    """A function to map integers to 2D coordinates in a spiral pattern around the origin."""
    # Map integers from Z to Z^2 in a spiral pattern around the origin.
    # Sources:
    # https://www.reddit.com/r/askmath/comments/18vqorf/find_the_nth_coordinate_of_a_square_spiral/
    # https://oeis.org/A174344
    m = np.floor(np.sqrt(n))
    x = ((-1) ** m) * ((n - m * (m + 1)) * (np.floor(2 * np.sqrt(n)) % 2) - np.ceil(m / 2))
    y = ((-1) ** (m + 1)) * ((n - m * (m + 1)) * (np.floor(2 * np.sqrt(n) + 1) % 2) + np.ceil(m / 2))
    return int(x), int(y)


@jit(nopython=True)
def transform_points(points, matrix, translate=True):
    """
    Returns points rotated by a homogeneous
    transformation matrix.
    If points are (n, 2) matrix must be (3, 3)
    If points are (n, 3) matrix must be (4, 4)

    Arguments:
        points (np.array): (n, dim) where `dim` is 2 or 3.
        matrix (np.array): (3, 3) or (4, 4) homogeneous rotation matrix.
        translate (bool): whether to apply translation from matrix or not.

    Returns:
        np.array: (n, dim) transformed points.
    """
    if len(points) == 0 or matrix is None:
        return points.copy()

    count, dim = points.shape
    # Check if the matrix is close to an identity matrix
    identity = np.eye(dim + 1)
    if np.abs(matrix - identity[: dim + 1, : dim + 1]).max() < 1e-8:
        return points.copy()

    if translate:
        stack = np.ascontiguousarray(np.concatenate((points, np.ones((count, 1))), axis=1))
        return (matrix @ stack.T).T[:, :dim]
    else:
        return (matrix[:dim, :dim] @ points.T).T


def quaternions_close(q1, q2, atol=1e-3):
    """
    Whether two quaternions represent the same rotation,
    allowing for the possibility that one is the negative of the other.

    Arguments:
        q1 (np.array): First quaternion
        q2 (np.array): Second quaternion
        atol (float): Absolute tolerance for comparison

    Returns:
        bool: Whether the quaternions are close
    """
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)


@jit(nopython=True)
def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (tensor): (..., 3, 3) where final two dims are 2d array representing target orientation matrix
        current (tensor): (..., 3, 3) where final two dims are 2d array representing current orientation matrix
    Returns:
        tensor: (..., 3) where final dim is (ax, ay, az) axis-angle representing orientation error
    """
    # convert input shapes
    input_shape = desired.shape[:-2]
    desired = desired.reshape(-1, 3, 3)
    current = current.reshape(-1, 3, 3)

    # grab relevant info
    rc1 = current[:, :, 0]
    rc2 = current[:, :, 1]
    rc3 = current[:, :, 2]
    rd1 = desired[:, :, 0]
    rd2 = desired[:, :, 1]
    rd3 = desired[:, :, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    # Reshape
    error = error.reshape(*input_shape, 3)

    return error


def delta_rotation_matrix(omega, delta_t):
    """
    Compute the delta rotation matrix given angular velocity and time elapsed.

    Arguments:
        omega (np.array): Angular velocity vector [omega_x, omega_y, omega_z].
        delta_t (float): Time elapsed.

    Returns:
        np.array: 3x3 Delta rotation matrix.
    """
    # Magnitude of angular velocity (angular speed)
    omega_magnitude = np.linalg.norm(omega)

    # If angular speed is zero, return identity matrix
    if omega_magnitude == 0:
        return np.eye(3)

    # Rotation angle
    theta = omega_magnitude * delta_t

    # Normalized axis of rotation
    axis = omega / omega_magnitude

    # Skew-symmetric matrix K
    u_x, u_y, u_z = axis
    K = np.array([[0, -u_z, u_y], [u_z, 0, -u_x], [-u_y, u_x, 0]])

    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R


def mat2euler_intrinsic(rmat):
    """
    Converts given rotation matrix to intrinsic euler angles in radian.

    Parameters:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (r,p,y) converted intrinsic euler angles in radian vec3 float
    """
    return R.from_matrix(rmat).as_euler("XYZ")


def euler_intrinsic2mat(euler):
    """
    Converts intrinsic euler angles into rotation matrix form

    Parameters:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    return R.from_euler("XYZ", euler).as_matrix()
