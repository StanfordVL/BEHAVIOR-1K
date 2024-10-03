"""
Utility functions of matrix and vector transformations.

NOTE: convention for quaternions is (x, y, z, w)
"""

import math
from typing import List, Optional, Tuple

import torch as th

PI = math.pi
EPS = th.finfo(th.float32).eps * 4.0

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


@th.jit.script
def copysign(a, b):
    # type: (float, th.Tensor) -> th.Tensor
    a = th.tensor(a, device=b.device, dtype=th.float).repeat(b.shape[0])
    return th.abs(a) * th.sign(b)


@th.jit.script
def anorm(x: th.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> th.Tensor:
    """Compute L2 norms along specified axes."""
    return th.norm(x, dim=dim, keepdim=keepdim)


@th.jit.script
def normalize(v: th.Tensor, dim: Optional[int] = None, eps: float = 1e-10) -> th.Tensor:
    """L2 Normalize along specified axes."""
    norm = anorm(v, dim=dim, keepdim=True)
    return v / th.where(norm < eps, th.full_like(norm, eps), norm)


@th.jit.script
def dot(v1, v2, dim=-1, keepdim=False):
    """
    Computes dot product between two vectors along the provided dim, optionally keeping the dimension

    Args:
        v1 (tensor): (..., N, ...) arbitrary vector
        v2 (tensor): (..., N, ...) arbitrary vector
        dim (int): Dimension to sum over for dot product
        keepdim (bool): Whether to keep dimension over which dot product is calculated

    Returns:
        tensor: (..., [1,] ...) dot product of vectors, with optional dimension kept if @keepdim is True
    """
    # type: (Tensor, Tensor, int, bool) -> Tensor
    return th.sum(v1 * v2, dim=dim, keepdim=keepdim)


@th.jit.script
def unit_vector(data: th.Tensor, dim: Optional[int] = None, out: Optional[th.Tensor] = None) -> th.Tensor:
    """
    Returns tensor normalized by length, i.e. Euclidean norm, along axis.

    Args:
        data (th.Tensor): data to normalize
        dim (Optional[int]): If specified, determines specific dimension along data to normalize
        out (Optional[th.Tensor]): If specified, will store computation in this variable

    Returns:
        th.Tensor: Normalized vector
    """
    if out is None:
        if not isinstance(data, th.Tensor):
            data = th.tensor(data, dtype=th.float32)
        else:
            data = data.clone().to(th.float32)

        if data.ndim == 1:
            return data / th.sqrt(th.dot(data, data))
    else:
        if out is not data:
            out.copy_(data)
        data = out

    if dim is None:
        dim = -1

    length = th.sum(data * data, dim=dim, keepdim=True).sqrt()
    data = data / (length + 1e-8)  # Add small epsilon to avoid division by zero

    return data


@th.jit.script
def quat_apply(quat: th.Tensor, vec: th.Tensor) -> th.Tensor:
    """
    Apply a quaternion rotation to a vector (equivalent to R.from_quat(x).apply(y))
    Args:
        quat (th.Tensor): (4,) or (N, 4) or (N, 1, 4) quaternion in (x, y, z, w) format
        vec (th.Tensor): (3,) or (M, 3) or (1, M, 3) vector to rotate
    Returns:
        th.Tensor: (M, 3) or (N, M, 3) rotated vector
    """
    assert quat.shape[-1] == 4, "Quaternion must have 4 components in last dimension"
    assert vec.shape[-1] == 3, "Vector must have 3 components in last dimension"

    # Ensure quat is at least 2D and vec is at least 2D
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)

    # Ensure quat is (N, 1, 4) and vec is (1, M, 3)
    if quat.dim() == 2:
        quat = quat.unsqueeze(1)
    if vec.dim() == 2:
        vec = vec.unsqueeze(0)

    # Extract quaternion components
    qx, qy, qz, qw = quat.unbind(-1)

    # Compute the quaternion multiplication
    t = th.stack(
        [
            2 * (qy * vec[..., 2] - qz * vec[..., 1]),
            2 * (qz * vec[..., 0] - qx * vec[..., 2]),
            2 * (qx * vec[..., 1] - qy * vec[..., 0]),
        ],
        dim=-1,
    )

    # Compute the final rotated vector
    result = vec + qw.unsqueeze(-1) * t + th.cross(quat[..., :3], t, dim=-1)

    # Remove any extra dimensions
    return result.squeeze()


@th.jit.script
def convert_quat(q: th.Tensor, to: str = "xyzw") -> th.Tensor:
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q (th.Tensor): a 4-dim array corresponding to a quaternion
        to (str): either 'xyzw' or 'wxyz', determining which convention to convert to.

    Returns:
        th.Tensor: The converted quaternion
    """
    if to == "xyzw":
        return th.stack([q[1], q[2], q[3], q[0]], dim=0)
    elif to == "wxyz":
        return th.stack([q[3], q[0], q[1], q[2]], dim=0)
    else:
        raise ValueError("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


@th.jit.script
def quat_multiply(quaternion1: th.Tensor, quaternion0: th.Tensor) -> th.Tensor:
    """
    Return multiplication of two quaternions (q1 * q0).

    Args:
        quaternion1 (th.Tensor): (x,y,z,w) quaternion
        quaternion0 (th.Tensor): (x,y,z,w) quaternion

    Returns:
        th.Tensor: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]

    return th.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dim=0,
    )


@th.jit.script
def quat_conjugate(quaternion: th.Tensor) -> th.Tensor:
    """
    Return conjugate of quaternion.

    Args:
        quaternion (th.Tensor): (x,y,z,w) quaternion

    Returns:
        th.Tensor: (x,y,z,w) quaternion conjugate
    """
    return th.cat([-quaternion[:3], quaternion[3:]])


@th.jit.script
def quat_inverse(quaternion: th.Tensor) -> th.Tensor:
    """
    Return inverse of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> th.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True

    Args:
        quaternion (th.tensor): (x,y,z,w) quaternion

    Returns:
        th.tensor: (x,y,z,w) quaternion inverse
    """
    return quat_conjugate(quaternion) / th.dot(quaternion, quaternion)


@th.jit.script
def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1

    Args:
        quaternion1 (th.tensor): (x,y,z,w) quaternion
        quaternion0 (th.tensor): (x,y,z,w) quaternion

    Returns:
        th.tensor: (x,y,z,w) quaternion distance
    """
    return quat_multiply(quaternion1, quat_inverse(quaternion0))


@th.jit.script
def quat_slerp(quat0, quat1, frac, shortestpath=True, eps=1.0e-15):
    """
    Return spherical linear interpolation between two quaternions.

    Args:
        quat0 (tensor): (..., 4) tensor where the final dim is (x,y,z,w) initial quaternion
        quat1 (tensor): (..., 4) tensor where the final dim is (x,y,z,w) final quaternion
        frac (tensor): Values in [0.0, 1.0] representing fraction of interpolation
        shortestpath (bool): If True, will calculate shortest path
        eps (float): Value to check for singularities
    Returns:
        tensor: (..., 4) Interpolated
    """
    # type: (Tensor, Tensor, Tensor, bool, float) -> Tensor
    # reshape quaternion
    quat_shape = quat0.shape
    quat0 = unit_vector(quat0.reshape(-1, 4), dim=-1)
    quat1 = unit_vector(quat1.reshape(-1, 4), dim=-1)

    # Check for endpoint cases
    where_start = frac <= 0.0
    where_end = frac >= 1.0

    d = dot(quat0, quat1, dim=-1, keepdim=True)
    if shortestpath:
        quat1 = th.where(d < 0.0, -quat1, quat1)
        d = th.abs(d)
    angle = th.acos(th.clip(d, -1.0, 1.0))

    # Check for small quantities (i.e.: q0 = q1)
    where_small_diff = th.abs(th.abs(d) - 1.0) < eps
    where_small_angle = abs(angle) < eps

    isin = 1.0 / th.sin(angle)
    val = quat0 * th.sin((1.0 - frac) * angle) * isin + quat1 * th.sin(frac * angle) * isin

    # Filter edge cases
    val = th.where(
        where_small_diff | where_small_angle | where_start,
        quat0,
        th.where(
            where_end,
            quat1,
            val,
        ),
    )

    # Reshape and return values
    return val.reshape(list(quat_shape))


@th.jit.script
def random_axis_angle(angle_limit: float = 2.0 * math.pi):
    """
    Samples an axis-angle rotation by first sampling a random axis
    and then sampling an angle. If @angle_limit is provided, the size
    of the rotation angle is constrained.

    Args:
        angle_limit (float): Determines magnitude limit of angles to generate

    Raises:
        AssertionError: [Invalid RNG]
    """
    # sample random axis using a normalized sample from spherical Gaussian.
    # see (http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/)
    # for why it works.
    random_axis = th.randn(3)
    random_axis /= th.norm(random_axis)
    random_angle = th.rand(1) * angle_limit
    return random_axis, random_angle.item()


@th.jit.script
def quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (w, x, y, z).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    quaternion = quaternion / th.norm(quaternion, dim=-1, keepdim=True)

    x = quaternion[..., 0]
    y = quaternion[..., 1]
    z = quaternion[..., 2]
    w = quaternion[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    rotation_matrix = th.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)

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


@th.jit.script
def mat2quat(rmat: th.Tensor) -> th.Tensor:
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat (th.Tensor): (3, 3) or (..., 3, 3) rotation matrix
    Returns:
        th.Tensor: (4,) or (..., 4) (x,y,z,w) float quaternion angles
    """
    # Check if input is a single matrix or a batch
    is_single = rmat.dim() == 2
    if is_single:
        rmat = rmat.unsqueeze(0)

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
    sq = th.where(trace_positive, th.sqrt(trace + 1.0) * 2.0, th.zeros_like(trace))
    qw = th.where(trace_positive, 0.25 * sq, th.zeros_like(trace))
    qx = th.where(trace_positive, (m21 - m12) / sq, th.zeros_like(trace))
    qy = th.where(trace_positive, (m02 - m20) / sq, th.zeros_like(trace))
    qz = th.where(trace_positive, (m10 - m01) / sq, th.zeros_like(trace))

    # Condition 1
    sq = th.where(cond1, th.sqrt(1.0 + m00 - m11 - m22) * 2.0, sq)
    qw = th.where(cond1, (m21 - m12) / sq, qw)
    qx = th.where(cond1, 0.25 * sq, qx)
    qy = th.where(cond1, (m01 + m10) / sq, qy)
    qz = th.where(cond1, (m02 + m20) / sq, qz)

    # Condition 2
    sq = th.where(cond2, th.sqrt(1.0 + m11 - m00 - m22) * 2.0, sq)
    qw = th.where(cond2, (m02 - m20) / sq, qw)
    qx = th.where(cond2, (m01 + m10) / sq, qx)
    qy = th.where(cond2, 0.25 * sq, qy)
    qz = th.where(cond2, (m12 + m21) / sq, qz)

    # Condition 3
    sq = th.where(cond3, th.sqrt(1.0 + m22 - m00 - m11) * 2.0, sq)
    qw = th.where(cond3, (m10 - m01) / sq, qw)
    qx = th.where(cond3, (m02 + m20) / sq, qx)
    qy = th.where(cond3, (m12 + m21) / sq, qy)
    qz = th.where(cond3, 0.25 * sq, qz)

    quat = th.stack([qx, qy, qz, qw], dim=-1)

    # Normalize the quaternion
    quat = quat / th.norm(quat, dim=-1, keepdim=True)

    # Reshape to match input batch shape
    quat = quat.reshape(batch_shape + (4,))

    # If input was a single matrix, remove the batch dimension
    if is_single:
        quat = quat.squeeze(0)

    return quat


@th.jit.script
def mat2pose(hmat):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat (th.tensor): a 4x4 homogeneous matrix

    Returns:
        2-tuple:
            - (th.tensor) (x,y,z) position array in cartesian coordinates
            - (th.tensor) (x,y,z,w) orientation array in quaternion form
    """
    assert th.allclose(hmat[:3, :3].det(), th.tensor(1.0)), "Rotation matrix must not be scaled"
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


@th.jit.script
def vec2quat(vec: th.Tensor, up: th.Tensor = th.tensor([0.0, 0.0, 1.0])) -> th.Tensor:
    """
    Converts given 3d-direction vector @vec to quaternion orientation with respect to another direction vector @up

    Args:
        vec (th.Tensor): (x,y,z) direction vector (possibly non-normalized)
        up (th.Tensor): (x,y,z) direction vector representing the canonical up direction (possibly non-normalized)

    Returns:
        th.Tensor: (x,y,z,w) quaternion
    """
    # Ensure inputs are 2D
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    if up.dim() == 1:
        up = up.unsqueeze(0)

    vec_n = th.nn.functional.normalize(vec, dim=-1)
    up_n = th.nn.functional.normalize(up, dim=-1)

    s_n = th.cross(up_n, vec_n, dim=-1)
    u_n = th.cross(vec_n, s_n, dim=-1)

    rotation_matrix = th.stack([vec_n, s_n, u_n], dim=-1)

    return mat2quat(rotation_matrix)


@th.jit.script
def euler2quat(euler: th.Tensor) -> th.Tensor:
    """
    Converts euler angles into quaternion form

    Args:
        euler (th.Tensor): (..., 3) (r,p,y) angles

    Returns:
        th.Tensor: (..., 4) (x,y,z,w) float quaternion angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    assert euler.shape[-1] == 3, "Invalid input shape"

    # Unpack roll, pitch, yaw
    roll, pitch, yaw = euler.unbind(-1)

    # Compute sines and cosines of half angles
    cy = th.cos(yaw * 0.5)
    sy = th.sin(yaw * 0.5)
    cr = th.cos(roll * 0.5)
    sr = th.sin(roll * 0.5)
    cp = th.cos(pitch * 0.5)
    sp = th.sin(pitch * 0.5)

    # Compute quaternion components
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    # Stack and return
    return th.stack([qx, qy, qz, qw], dim=-1)


@th.jit.script
def quat2euler(q):

    single_dim = q.dim() == 1

    if single_dim:
        q = q.unsqueeze(0)

    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = th.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = th.where(th.abs(sinp) >= 1, copysign(math.pi / 2.0, sinp), th.asin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = th.atan2(siny_cosp, cosy_cosp)

    euler = th.stack([roll, pitch, yaw], dim=-1) % (2 * math.pi)
    euler[euler > math.pi] -= 2 * math.pi

    if single_dim:
        euler = euler.squeeze(0)

    return euler


@th.jit.script
def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (th.tensor): (r,p,y) angles

    Returns:
        th.tensor: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """
    euler = th.as_tensor(euler, dtype=th.float32)
    assert euler.shape[-1] == 3, f"Invalid shaped euler {euler}"

    # Convert Euler angles to quaternion
    quat = euler2quat(euler)

    # Convert quaternion to rotation matrix
    return quat2mat(quat)


@th.jit.script
def mat2euler(rmat):
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (th.tensor): 3x3 rotation matrix

    Returns:
        th.tensor: (r,p,y) converted euler angles in radian vec3 float
    """
    M = th.as_tensor(rmat, dtype=th.float32)[:3, :3]

    # Convert rotation matrix to quaternion
    # Note: You'll need to implement mat2quat function
    quat = mat2quat(M)

    # Convert quaternion to Euler angles
    euler = quat2euler(quat)
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    return th.stack([roll, pitch, yaw], dim=-1)


@th.jit.script
def pose2mat(pose: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
    pos, orn = pose

    # Ensure pos and orn are the expected shape and dtype
    pos = pos.to(dtype=th.float32).reshape(3)
    orn = orn.to(dtype=th.float32).reshape(4)

    homo_pose_mat = th.eye(4, dtype=th.float32)
    homo_pose_mat[:3, :3] = quat2mat(orn)
    homo_pose_mat[:3, 3] = pos

    return homo_pose_mat


@th.jit.script
def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (tensor): (..., 4) tensor where the final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) axis-angle exponential coordinates
    """
    # reshape quaternion
    quat_shape = quat.shape[:-1]  # ignore last dim
    quat = quat.reshape(-1, 4)
    # clip quaternion
    quat[:, 3] = th.clip(quat[:, 3], -1.0, 1.0)
    # Calculate denominator
    den = th.sqrt(1.0 - quat[:, 3] * quat[:, 3])
    # Map this into a mask

    # Create return array
    ret = th.zeros_like(quat)[:, :3]
    idx = th.nonzero(den).reshape(-1)
    ret[idx, :] = (quat[idx, :3] * 2.0 * th.acos(quat[idx, 3]).unsqueeze(-1)) / den[idx].unsqueeze(-1)

    # Reshape and return output
    ret = ret.reshape(
        list(quat_shape)
        + [
            3,
        ]
    )
    return ret


@th.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = th.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = th.zeros(th.prod(th.tensor(input_shape, dtype=th.int)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps  # th.nonzero(angle).reshape(-1)
    quat[idx, :] = th.cat(
        [vec[idx, :] * th.sin(angle[idx, :] / 2.0) / angle[idx, :], th.cos(angle[idx, :] / 2.0)], dim=-1
    )

    # Reshape and return output
    quat = quat.reshape(
        list(input_shape)
        + [
            4,
        ]
    )
    return quat


@th.jit.script
def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A
    to a homogenous matrix corresponding to the same point C in frame B.

    Args:
        pose_A (th.tensor): 4x4 matrix corresponding to the pose of C in frame A
        pose_A_in_B (th.tensor): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        th.tensor: 4x4 matrix corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return th.matmul(pose_A_in_B, pose_A)


@th.jit.script
def pose_inv(pose_mat):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose_mat (th.tensor): 4x4 matrix for the pose to inverse

    Returns:
        th.tensor: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = th.zeros((4, 4))
    pose_inv[:3, :3] = pose_mat[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3] @ pose_mat[:3, 3]
    pose_inv[3, 3] = 1.0
    return pose_inv


@th.jit.script
def pose_transform(pos1, quat1, pos0, quat0):
    """
    Conducts forward transform from pose (pos0, quat0) to pose (pos1, quat1):

    pose1 @ pose0, NOT pose0 @ pose1

    Args:
        pos1: (x,y,z) position to transform
        quat1: (x,y,z,w) orientation to transform
        pos0: (x,y,z) initial position
        quat0: (x,y,z,w) initial orientation

    Returns:
        2-tuple:
            - (th.tensor) (x,y,z) position array in cartesian coordinates
            - (th.tensor) (x,y,z,w) orientation array in quaternion form
    """
    # Get poses
    mat0 = pose2mat((pos0, quat0))
    mat1 = pose2mat((pos1, quat1))

    # Multiply and convert back to pos, quat
    return mat2pose(mat1 @ mat0)


@th.jit.script
def invert_pose_transform(pos, quat):
    """
    Inverts a pose transform

    Args:
        pos: (x,y,z) position to transform
        quat: (x,y,z,w) orientation to transform

    Returns:
        2-tuple:
            - (th.tensor) (x,y,z) position array in cartesian coordinates
            - (th.tensor) (x,y,z,w) orientation array in quaternion form
    """
    # Get pose
    mat = pose2mat((pos, quat))

    # Invert pose and convert back to pos, quat
    return mat2pose(pose_inv(mat))


@th.jit.script
def relative_pose_transform(pos1, quat1, pos0, quat0):
    """
    Computes relative forward transform from pose (pos0, quat0) to pose (pos1, quat1), i.e.: solves:

    pose1 = pose0 @ transform

    Args:
        pos1: (x,y,z) position to transform
        quat1: (x,y,z,w) orientation to transform
        pos0: (x,y,z) initial position
        quat0: (x,y,z,w) initial orientation

    Returns:
        2-tuple:
            - (th.tensor) (x,y,z) position array in cartesian coordinates
            - (th.tensor) (x,y,z,w) orientation array in quaternion form
    """
    # Get poses
    mat0 = pose2mat((pos0, quat0))
    mat1 = pose2mat((pos1, quat1))

    # Invert pose0 and calculate transform
    return mat2pose(pose_inv(mat0) @ mat1)


@th.jit.script
def _skew_symmetric_translation(pos_A_in_B):
    """
    Helper function to get a skew symmetric translation matrix for converting quantities
    between frames.

    Args:
        pos_A_in_B (th.tensor): (x,y,z) position of A in frame B

    Returns:
        th.tensor: 3x3 skew symmetric translation matrix
    """
    return th.tensor(
        [
            [0.0, -pos_A_in_B[2].item(), pos_A_in_B[1].item()],
            [pos_A_in_B[2].item(), 0.0, -pos_A_in_B[0].item()],
            [-pos_A_in_B[1].item(), pos_A_in_B[0].item(), 0.0],
        ],
        dtype=th.float32,
        device=pos_A_in_B.device,
    )


@th.jit.script
def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Converts linear and angular velocity of a point in frame A to the equivalent in frame B.

    Args:
        vel_A (th.tensor): (vx,vy,vz) linear velocity in A
        ang_vel_A (th.tensor): (wx,wy,wz) angular velocity in A
        pose_A_in_B (th.tensor): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (th.tensor) (vx,vy,vz) linear velocities in frame B
            - (th.tensor) (wx,wy,wz) angular velocities in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = rot_A_in_B @ vel_A + skew_symm @ (rot_A_in_B @ ang_vel_A)
    ang_vel_B = rot_A_in_B @ ang_vel_A
    return vel_B, ang_vel_B


@th.jit.script
def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Converts linear and rotational force at a point in frame A to the equivalent in frame B.

    Args:
        force_A (th.tensor): (fx,fy,fz) linear force in A
        torque_A (th.tensor): (tx,ty,tz) rotational force (moment) in A
        pose_A_in_B (th.tensor): 4x4 matrix corresponding to the pose of A in frame B

    Returns:
        2-tuple:

            - (th.tensor) (fx,fy,fz) linear forces in frame B
            - (th.tensor) (tx,ty,tz) moments in frame B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = rot_A_in_B.T @ force_A
    torque_B = -(rot_A_in_B.T @ (skew_symm @ force_A)) + rot_A_in_B.T @ torque_A
    return force_B, torque_B


@th.jit.script
def rotation_matrix(angle: float, direction: th.Tensor) -> th.Tensor:
    """
    Returns a 3x3 rotation matrix to rotate about the given axis.

    Args:
        angle (float): Magnitude of rotation in radians
        direction (th.Tensor): (ax,ay,az) axis about which to rotate

    Returns:
        th.Tensor: 3x3 rotation matrix
    """
    sina = th.sin(th.tensor(angle, dtype=th.float32))
    cosa = th.cos(th.tensor(angle, dtype=th.float32))

    direction = direction / th.norm(direction)  # Normalize direction vector

    # Create rotation matrix
    R = th.eye(3, dtype=th.float32, device=direction.device)
    R *= cosa
    R += th.outer(direction, direction) * (1.0 - cosa)
    direction *= sina

    # Create the skew-symmetric matrix
    skew_matrix = th.zeros(3, 3, dtype=th.float32, device=direction.device)
    skew_matrix[0, 1] = -direction[2]
    skew_matrix[0, 2] = direction[1]
    skew_matrix[1, 0] = direction[2]
    skew_matrix[1, 2] = -direction[0]
    skew_matrix[2, 0] = -direction[1]
    skew_matrix[2, 1] = direction[0]

    R += skew_matrix

    return R


@th.jit.script
def transformation_matrix(angle: float, direction: th.Tensor, point: Optional[th.Tensor] = None) -> th.Tensor:
    """
    Returns a 4x4 homogeneous transformation matrix to rotate about axis defined by point and direction.

    Args:
        angle (float): Magnitude of rotation in radians
        direction (th.Tensor): (ax,ay,az) axis about which to rotate
        point (Optional[th.Tensor]): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        th.Tensor: 4x4 homogeneous transformation matrix
    """
    R = rotation_matrix(angle, direction)

    M = th.eye(4, dtype=th.float32, device=direction.device)
    M[:3, :3] = R

    if point is not None:
        # Rotation not about origin
        point = point.to(dtype=th.float32)
        M[:3, 3] = point - R @ point
    return M


@th.jit.script
def clip_translation(dpos, limit):
    """
    Limits a translation (delta position) to a specified limit

    Scales down the norm of the dpos to 'limit' if norm(dpos) > limit, else returns immediately

    Args:
        dpos (n-array): n-dim Translation being clipped (e,g.: (x, y, z)) -- numpy array
        limit (float): Value to limit translation by -- magnitude (scalar, in same units as input)

    Returns:
        2-tuple:

            - (th.tensor) Clipped translation (same dimension as inputs)
            - (bool) whether the value was clipped or not
    """
    input_norm = th.norm(dpos)
    return (dpos * limit / input_norm, True) if input_norm > limit else (dpos, False)


@th.jit.script
def clip_rotation(quat, limit):
    """
    Limits a (delta) rotation to a specified limit

    Converts rotation to axis-angle, clips, then re-converts back into quaternion

    Args:
        quat (th.tensor): (x,y,z,w) rotation being clipped
        limit (float): Value to limit rotation by -- magnitude (scalar, in radians)

    Returns:
        2-tuple:

            - (th.tensor) Clipped rotation quaternion (x, y, z, w)
            - (bool) whether the value was clipped or not
    """
    clipped = False

    # First, normalize the quaternion
    quat = quat / th.norm(quat)

    den = math.sqrt(max(1 - quat[3] * quat[3], 0))
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
        a = limit * th.sign(a) / 2
        sa = math.sin(a)
        ca = math.cos(a)
        quat = th.tensor([x * sa, y * sa, z * sa, ca])
        clipped = True

    return quat, clipped


@th.jit.script
def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation (th.tensor): (x,y,z) translation value
        rotation (th.tensor): a 3x3 matrix representing rotation

    Returns:
        pose (th.tensor): a 4x4 homogeneous matrix
    """
    pose = th.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


@th.jit.script
def get_orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector, where inputs are quaternions

    Args:
        desired (tensor): (..., 4) where final dim is (x,y,z,w) quaternion
        current (tensor): (..., 4) where final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) where final dim is (ax, ay, az) axis-angle representing orientation error
    """
    # convert input shapes
    input_shape = desired.shape[:-1]
    desired = desired.reshape(-1, 4)
    current = current.reshape(-1, 4)

    cc = quat_conjugate(current)
    q_r = quat_multiply(desired, cc)
    return (q_r[:, 0:3] * th.sign(q_r[:, 3]).unsqueeze(-1)).reshape(list(input_shape) + [3])


@th.jit.script
def get_orientation_diff_in_radian(orn0: th.Tensor, orn1: th.Tensor) -> th.Tensor:
    """
    Returns the difference between two quaternion orientations in radians.

    Args:
        orn0 (th.Tensor): (x, y, z, w) quaternion
        orn1 (th.Tensor): (x, y, z, w) quaternion

    Returns:
        orn_diff (th.Tensor): orientation difference in radians
    """
    # Compute the difference quaternion
    diff_quat = quat_distance(orn0, orn1)

    # Convert to axis-angle representation
    axis_angle = quat2axisangle(diff_quat)

    # The magnitude of the axis-angle vector is the rotation angle
    return th.norm(axis_angle)


@th.jit.script
def get_pose_error(target_pose, current_pose):
    """
    Computes the error corresponding to target pose - current pose as a 6-dim vector.
    The first 3 components correspond to translational error while the last 3 components
    correspond to the rotational error.

    Args:
        target_pose (th.tensor): a 4x4 homogenous matrix for the target pose
        current_pose (th.tensor): a 4x4 homogenous matrix for the current pose

    Returns:
        th.tensor: 6-dim pose error.
    """
    error = th.zeros(6)

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
    rot_err = 0.5 * (th.linalg.cross(r1, r1d) + th.linalg.cross(r2, r2d) + th.linalg.cross(r3, r3d))

    error[:3] = pos_err
    error[3:] = rot_err
    return error


@th.jit.script
def matrix_inverse(matrix):
    """
    Helper function to have an efficient matrix inversion function.

    Args:
        matrix (th.tensor): 2d-array representing a matrix

    Returns:
        th.tensor: 2d-array representing the matrix inverse
    """
    return th.linalg.inv_ex(matrix).inverse


@th.jit.script
def vecs2axisangle(vec0, vec1):
    """
    Converts the angle from unnormalized 3D vectors @vec0 to @vec1 into an axis-angle representation of the angle

    Args:
        vec0 (th.tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        vec1 (th.tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
    """
    # Normalize vectors
    vec0 = normalize(vec0, dim=-1)
    vec1 = normalize(vec1, dim=-1)

    # Get cross product for direction of angle, and multiply by arcos of the dot product which is the angle
    return th.linalg.cross(vec0, vec1) * th.arccos((vec0 * vec1).sum(-1, keepdim=True))


@th.jit.script
def vecs2quat(vec0: th.Tensor, vec1: th.Tensor, normalized: bool = False) -> th.Tensor:
    """
    Converts the angle from unnormalized 3D vectors @vec0 to @vec1 into a quaternion representation of the angle
    Args:
        vec0 (th.Tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        vec1 (th.Tensor): (..., 3) (x,y,z) 3D vector, possibly unnormalized
        normalized (bool): If True, @vec0 and @vec1 are assumed to already be normalized and we will skip the
            normalization step (more efficient)
    Returns:
        th.Tensor: (..., 4) Normalized quaternion representing the rotation from vec0 to vec1
    """
    # Normalize vectors if requested
    if not normalized:
        vec0 = normalize(vec0, dim=-1)
        vec1 = normalize(vec1, dim=-1)

    # Half-way Quaternion Solution -- see https://stackoverflow.com/a/11741520
    cos_theta = th.sum(vec0 * vec1, dim=-1, keepdim=True)

    # Create a tensor for the case where cos_theta == -1
    batch_shape = vec0.shape[:-1]
    fallback = th.zeros(batch_shape + (4,), device=vec0.device, dtype=vec0.dtype)
    fallback[..., 0] = 1.0

    # Compute the quaternion
    quat_unnormalized = th.where(
        cos_theta == -1,
        fallback,
        th.cat([th.linalg.cross(vec0, vec1), 1 + cos_theta], dim=-1),
    )

    return quat_unnormalized / th.norm(quat_unnormalized, dim=-1, keepdim=True)


@th.jit.script
def align_vector_sets(vec_set1: th.Tensor, vec_set2: th.Tensor) -> th.Tensor:
    """
    Computes a single quaternion representing the rotation that best aligns vec_set1 to vec_set2.

    Args:
        vec_set1 (th.Tensor): (N, 3) tensor of N 3D vectors
        vec_set2 (th.Tensor): (N, 3) tensor of N 3D vectors

    Returns:
        th.Tensor: (4,) Normalized quaternion representing the overall rotation
    """
    # Compute the cross-covariance matrix
    H = vec_set2.T @ vec_set1

    # Compute the elements for the quaternion
    trace = H.trace()
    w = trace + 1
    x = H[1, 2] - H[2, 1]
    y = H[2, 0] - H[0, 2]
    z = H[0, 1] - H[1, 0]

    # Construct the quaternion
    quat = th.stack([x, y, z, w])

    # Handle the case where w is close to zero
    if quat[3] < 1e-4:
        quat[3] = 0
        max_idx = th.argmax(quat[:3].abs()) + 1
        quat[max_idx] = 1

    # Normalize the quaternion
    quat = quat / (th.norm(quat) + 1e-8)  # Add epsilon to avoid division by zero

    return quat


@th.jit.script
def l2_distance(v1: th.Tensor, v2: th.Tensor) -> th.Tensor:
    """Returns the L2 distance between vector v1 and v2."""
    return th.norm(v1 - v2)


@th.jit.script
def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = th.sqrt(x**2 + y**2)
    phi = th.arctan2(y, x)
    return rho, phi


def deg2rad(deg):
    return deg * math.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / math.pi


@th.jit.script
def check_quat_right_angle(quat: th.Tensor, atol: float = 5e-2) -> th.Tensor:
    """
    Check by making sure the quaternion is some permutation of +/- (1, 0, 0, 0),
    +/- (0.707, 0.707, 0, 0), or +/- (0.5, 0.5, 0.5, 0.5)
    Because orientations are all normalized (same L2-norm), every orientation should have a unique L1-norm
    So we check the L1-norm of the absolute value of the orientation as a proxy for verifying these values

    Args:
        quat (th.Tensor): (x,y,z,w) quaternion orientation to check
        atol (float): Absolute tolerance permitted

    Returns:
        th.Tensor: Boolean tensor indicating whether the quaternion is a right angle or not
    """
    l1_norm = th.abs(quat).sum(dim=-1)
    reference_norms = th.tensor([1.0, 1.414, 2.0], device=quat.device, dtype=quat.dtype)
    return th.any(th.abs(l1_norm.unsqueeze(-1) - reference_norms) < atol, dim=-1)


@th.jit.script
def z_angle_from_quat(quat):
    """Get the angle around the Z axis produced by the quaternion."""
    rotated_X_axis = quat_apply(quat, th.tensor([1, 0, 0], dtype=th.float32))
    return th.arctan2(rotated_X_axis[1], rotated_X_axis[0])


@th.jit.script
def integer_spiral_coordinates(n: int) -> Tuple[int, int]:
    """A function to map integers to 2D coordinates in a spiral pattern around the origin."""
    # Map integers from Z to Z^2 in a spiral pattern around the origin.
    # Sources:
    # https://www.reddit.com/r/askmath/comments/18vqorf/find_the_nth_coordinate_of_a_square_spiral/
    # https://oeis.org/A174344
    m = math.floor(math.sqrt(n))
    x = ((-1) ** m) * ((n - m * (m + 1)) * (math.floor(2 * math.sqrt(n)) % 2) - math.ceil(m / 2))
    y = ((-1) ** (m + 1)) * ((n - m * (m + 1)) * (math.floor(2 * math.sqrt(n) + 1) % 2) + math.ceil(m / 2))
    return int(x), int(y)


@th.jit.script
def random_quaternion(num_quaternions: int = 1) -> th.Tensor:
    """
    Generate random rotation quaternions, uniformly distributed over SO(3).

    Arguments:
        num_quaternions: int, number of quaternions to generate (default: 1)

    Returns:
        th.Tensor: A tensor of shape (num_quaternions, 4) containing random unit quaternions.
    """
    # Generate four random numbers between 0 and 1
    rand = th.rand(num_quaternions, 4)

    # Use the formula from Ken Shoemake's "Uniform Random Rotations"
    r1 = th.sqrt(1.0 - rand[:, 0])
    r2 = th.sqrt(rand[:, 0])
    t1 = 2 * th.pi * rand[:, 1]
    t2 = 2 * th.pi * rand[:, 2]

    quaternions = th.stack([r1 * th.sin(t1), r1 * th.cos(t1), r2 * th.sin(t2), r2 * th.cos(t2)], dim=1)

    return quaternions


@th.jit.script
def transform_points(points: th.Tensor, matrix: th.Tensor, translate: bool = True) -> th.Tensor:
    """
    Returns points rotated by a homogeneous
    transformation matrix.
    If points are (n, 2) matrix must be (3, 3)
    If points are (n, 3) matrix must be (4, 4)

    Arguments:
        points : (n, dim) torch.Tensor
        Points where `dim` is 2 or 3.
        matrix : (3, 3) or (4, 4) torch.Tensor
        Homogeneous rotation matrix.
        translate : bool
        Apply translation from matrix or not.

    Returns:
        transformed : (n, dim) torch.Tensor
        Transformed points.
    """
    if len(points) == 0 or matrix is None:
        return points.clone()

    count, dim = points.shape
    # Check if the matrix is close to an identity matrix
    identity = th.eye(dim + 1, device=points.device)
    if th.abs(matrix - identity[: dim + 1, : dim + 1]).max() < 1e-8:
        return points.clone().contiguous()

    if translate:
        stack = th.cat((points, th.ones(count, 1, device=points.device)), dim=1)
        return th.mm(matrix, stack.t()).t()[:, :dim]
    else:
        return th.mm(matrix[:dim, :dim], points.t()).t()


@th.jit.script
def quaternions_close(q1: th.Tensor, q2: th.Tensor, atol: float = 1e-3) -> bool:
    """
    Whether two quaternions represent the same rotation,
    allowing for the possibility that one is the negative of the other.

    Arguments:
        q1: th.Tensor
            First quaternion
        q2: th.Tensor
            Second quaternion
        atol: float
            Absolute tolerance for comparison

    Returns:
        bool
            Whether the quaternions are close
    """
    return th.allclose(q1, q2, atol=atol) or th.allclose(q1, -q2, atol=atol)
