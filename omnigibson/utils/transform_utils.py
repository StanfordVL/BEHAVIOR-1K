"""
Utility functions of matrix and vector transformations.

NOTE: convention for quaternions is (x, y, z, w)
"""

import math

import numpy as np
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


def ewma_vectorized(data, alpha, offset=None, dtype=None, order="C", out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.

    Args:
        data (Iterable): Input data
        alpha (float): scalar in range (0,1)
            The alpha parameter for the moving average.
        offset (None or float): If specified, the offset for the moving average. None defaults to data[0].
        dtype (None or type): Data type used for calculations. If None, defaults to float64 unless
            data.dtype is float32, then it will use float32.
        order (None or str): Order to use when flattening the data. Valid options are {'C', 'F', 'A'}.
            None defaults to 'C'.
        out (None or np.array): If specified, the location into which the result is stored. If provided, it must have
            the same shape as the input. If not provided or `None`,
            a freshly-allocated array is returned.

    Returns:
        np.array: Exponential moving average from @data
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1.0 - alpha, np.arange(data.size + 1, dtype=dtype), dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1], dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


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
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
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
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float32,
    )


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
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
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


def random_quat(rand=None):
    """
    Return uniform random unit quaternion.

    E.g.:
    >>> q = random_quat()
    >>> np.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quat(np.random.random(3))
    >>> q.shape
    (4,)

    Args:
        rand (3-array or None): If specified, must be three independent random variables that are uniformly distributed
            between 0 and 1.

    Returns:
        np.array: (x,y,z,w) random quaternion
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
        dtype=np.float32,
    )


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


def vec(values):
    """
    Converts value tuple into a numpy vector.

    Args:
        values (n-array): a tuple of numbers

    Returns:
        np.array: vector of given values
    """
    return np.array(values, dtype=np.float32)


def mat4(array):
    """
    Converts an array to 4x4 matrix.

    Args:
        array (n-array): the array in form of vec, list, or tuple

    Returns:
        np.array: a 4x4 numpy matrix
    """
    return np.array(array, dtype=np.float32).reshape((4, 4))


def mat2pose(hmat):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat (np.array): a 4x4 homogeneous matrix

    Returns:
        2-tuple:

            - (np.array) (x,y,z) position array in cartesian coordinates
            - (np.array) (x,y,z,w) orientation array in quaternion form
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): (..., 3, 3) rotation matrix

    Returns:
        np.array: (..., 4) (x,y,z,w) float quaternion angles
    """
    return R.from_matrix(rmat).as_quat()


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
    vec_n = vec / np.linalg.norm(vec)       # x
    up_n = up / np.linalg.norm(up)
    s_n = np.cross(up_n, vec_n)             # y
    u_n = np.cross(vec_n, s_n)              # z
    return mat2quat(np.array([vec_n, s_n, u_n]).T)


def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

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
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    return R.from_matrix(M).as_euler("xyz")


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
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (..., 4) (x,y,z,w) float quaternion angles

    Returns:
        np.array: (..., 3, 3) rotation matrix
    """
    return R.from_quat(quaternion).as_matrix()


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


def euler2quat(euler):
    """
    Converts euler angles into quaternion form

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
    Converts euler angles into quaternion form

    Args:
        quat (np.array): (x,y,z,w) float quaternion angles

    Returns:
        np.array: (r,p,y) angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    return R.from_quat(quat).as_euler("xyz")


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

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose_mat[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose_mat[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def pose_transform(pos1, quat1, pos0, quat0):
    """
    Conducts forward transform from pose (pos0, quat0) to pose (pos1, quat1):

    pose1 @ pose0, NOT pose0 @ pose1

    Args:
        pos1: (x,y,z) position to transform
        quat1: (x,y,z,w) orientation to transform
        pos0: (x,y,z) initial position
        quat0: (x,y,z,w) initial orientation
    """
    # Get poses
    mat0 = pose2mat((pos0, quat0))
    mat1 = pose2mat((pos1, quat1))

    # Multiply and convert back to pos, quat
    return mat2pose(mat1 @ mat0)


def relative_pose_transform(pos1, quat1, pos0, quat0):
    """
    Computes relative forward transform from pose (pos0, quat0) to pose (pos1, quat1), i.e.: solves:

    pose1 = pose0 @ transform

    Args:
        pos1: (x,y,z) position to transform
        quat1: (x,y,z,w) orientation to transform
        pos0: (x,y,z) initial position
        quat0: (x,y,z,w) initial orientation
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
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
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
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
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
    quat_unnormalized = np.where(cos_theta == -1, np.array([1.0, 0, 0, 0]), np.concatenate([np.cross(vec0, vec1), 1 + cos_theta], axis=-1))
    return quat_unnormalized / np.linalg.norm(quat_unnormalized, axis=-1, keepdims=True)


def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


def frustum(left, right, bottom, top, znear, zfar):
    """Create view frustum matrix."""
    assert right != left
    assert bottom != top
    assert znear != zfar

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    # TODO: Put this back to 3,1
    # M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M


def ortho(left, right, bottom, top, znear, zfar):
    """Create orthonormal projection matrix."""
    assert right != left
    assert bottom != top
    assert znear != zfar

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 0] = -(right + left) / (right - left)
    M[3, 1] = -(top + bottom) / (top - bottom)
    M[3, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 3] = 1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    """Create perspective projection matrix."""
    # fovy is in degree
    assert znear != zfar
    h = np.tan(fovy / 360.0 * np.pi) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms alogn specified axes."""
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    norm = anorm(v, axis=axis, keepdims=True)
    return v / np.where(norm < eps, eps, norm)


def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def deg2rad(deg):
    return deg * np.pi / 180.


def rad2deg(rad):
    return rad * 180. / np.pi


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