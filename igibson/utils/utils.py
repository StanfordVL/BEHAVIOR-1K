import collections
import json
import os
import random

import numpy as np

import scipy
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transforms3d import quaternions

from igibson.utils.constants import CoordinateSystem

# File I/O related


def parse_config(config):

    """
    Parse iGibson config file / object
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data


def parse_str_config(config):
    """
    Parse string config
    """
    return yaml.safe_load(config)


def dump_config(config):
    """
    Converts YML config into a string
    """
    return yaml.dump(config)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Geometry related


def rotate_vector_3d(v, r, p, y, cck=True):
    """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
    local_to_global = R.from_euler("xyz", [r, p, y]).as_matrix()
    if cck:
        global_to_local = local_to_global.T
        return np.dot(global_to_local, v)
    else:
        return np.dot(local_to_global, v)


def get_transform_from_xyz_rpy(xyz, rpy):
    """
    Returns a homogeneous transformation matrix (numpy array 4x4)
    for the given translation and rotation in roll,pitch,yaw
    xyz = Array of the translation
    rpy = Array with roll, pitch, yaw rotations
    """
    rotation = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
    transformation = np.eye(4)
    transformation[0:3, 0:3] = rotation
    transformation[0:3, 3] = xyz
    return transformation


def get_rpy_from_transform(transform):
    """
    Returns the roll, pitch, yaw angles (Euler) for a given rotation or
    homogeneous transformation matrix
    transformation = Array with the rotation (3x3) or full transformation (4x4)
    """
    rpy = R.from_matrix(transform[0:3, 0:3]).as_euler("xyz")
    return rpy


def rotate_vector_2d(v, yaw):
    """Rotates 2d vector by yaw counterclockwise"""
    local_to_global = R.from_euler("z", yaw).as_matrix()
    global_to_local = local_to_global.T
    global_to_local = global_to_local[:2, :2]
    if len(v.shape) == 1:
        return np.dot(global_to_local, v)
    elif len(v.shape) == 2:
        return np.dot(global_to_local, v.T).T
    else:
        print("Incorrect input shape for rotate_vector_2d", v.shape)
        return v


def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def quatFromXYZW(xyzw, seq):
    """Convert quaternion from XYZW (pybullet convention) to arbitrary sequence."""
    assert (
        len(seq) == 4 and "x" in seq and "y" in seq and "z" in seq and "w" in seq
    ), "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = ["xyzw".index(axis) for axis in seq]
    return xyzw[inds]


def quatToXYZW(orn, seq):
    """Convert quaternion from arbitrary sequence to XYZW (pybullet convention)."""
    assert (
        len(seq) == 4 and "x" in seq and "y" in seq and "z" in seq and "w" in seq
    ), "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = [seq.index(axis) for axis in "xyzw"]
    return orn[inds]


def quatXYZWFromRotMat(rot_mat):
    """Convert quaternion from rotation matrix"""
    quatWXYZ = quaternions.mat2quat(rot_mat)
    quatXYZW = quatToXYZW(quatWXYZ, "wxyz")
    return quatXYZW


def convertPointCoordSystem(xyz, from_system, to_system):
    """
    Convert point from from_system convention to to_system convention.

    OpenCV coordinate system is (right, downward, forward).
    OpenGL coordinate system is (right, upward, backward).
    PyBullet coordinate system is (forward, left, upward).
    SunRgbd coordinate system is (right, forward, upward).

    :param xyz: (x, y, z) coordinate in from_system convension, or an ndarray with the
        last dimension being (x, y, z) coordinates in from_system convension
    :param from_system: choose from OpenCV, OpenGL, PyBullet, SunRgbd
    :param to_system: choose from OpenCV, OpenGL, PyBullet, SunRgbd
    :return: (x, y, z) coordinate in to_system convension, or an ndarray with the
        last dimension being (x, y, z) coordinates in to_system convension
    """
    from_system = CoordinateSystem[from_system.upper()]
    to_system = CoordinateSystem[to_system.upper()]

    if isinstance(xyz, list):
        xyz = np.array(xyz)
    if not isinstance(xyz, np.ndarray) or xyz.shape[-1] != 3:
        raise NotImplementedError

    # Convert from from_system to PyBullet
    if from_system == CoordinateSystem.OPENCV:
        xyz = np.stack((xyz[..., 2], -xyz[..., 0], -xyz[..., 1]), axis=-1)
    elif from_system == CoordinateSystem.OPENGL:
        xyz = np.stack((-xyz[..., 2], -xyz[..., 0], xyz[..., 1]), axis=-1)
    elif from_system == CoordinateSystem.SUNRGBD:
        xyz = np.stack((xyz[..., 1], -xyz[..., 0], xyz[..., 2]), axis=-1)
    # Convert from PyBullet to to_system.
    if to_system == CoordinateSystem.OPENCV:
        xyz = np.stack((-xyz[..., 1], -xyz[..., 2], xyz[..., 0]), axis=-1)
    elif to_system == CoordinateSystem.OPENGL:
        xyz = np.stack((-xyz[..., 1], xyz[..., 2], -xyz[..., 0]), axis=-1)
    elif to_system == CoordinateSystem.SUNRGBD:
        xyz = np.stack((-xyz[..., 1], xyz[..., 0], xyz[..., 2]), axis=-1)
    return xyz


# Represents a rotation by q1, followed by q0
def multQuatLists(q0, q1):
    """Multiply two quaternions that are represented as lists."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    return [
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ]


def normalizeListVec(v):
    """Normalizes a vector list."""
    length = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    if length <= 0:
        length = 1
    v = [val / np.sqrt(length) for val in v]
    return v


# Quat(xyzw)
def quat_pos_to_mat(pos, quat):
    """Convert position and quaternion to transformation matrix"""
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, -1] = pos
    return mat


def mat_to_quat_pos(mat):
    """Convert transformation matrix to position and quaternion"""
    quat = R.from_matrix(mat[0:3, 0:3]).as_quat()
    pos = mat[0:3, 3]
    return pos, quat


# Texture related


def transform_texture(input_filename, output_filename, mixture_weight=0, mixture_color=(0, 0, 0)):
    img = np.array(Image.open(input_filename))
    img = img * (1 - mixture_weight) + np.array(list(mixture_color))[None, None, :] * mixture_weight
    img = img.astype(np.uint8)
    Image.fromarray(img).save(output_filename)


def brighten_texture(input_filename, output_filename, brightness=1):
    img = np.array(Image.open(input_filename))
    img = np.clip(img * brightness, 0, 255)
    img = img.astype(np.uint8)
    Image.fromarray(img).save(output_filename)


# Other


def restoreState(*args, **kwargs):
    """Restore to a given pybullet state, with a mitigation for a known sleep state restore bug.

    When the pybullet state is restored, the object's wake zone (the volume around the object where
    if any other object enters, the object should be waken up) does not get reset correctly,
    causing weird bugs around asleep objects. This function mitigates the issue by forcing the
    sleep code to update each object's wake zone.
    """
    p.restoreState(*args, **kwargs)
    for body_id in range(p.getNumBodies()):
        p.resetBasePositionAndOrientation(
            body_id, *p.getBasePositionAndOrientation(body_id), physicsClientId=kwargs.get("physicsClientId", 0)
        )
    return p.restoreState(*args, **kwargs)


def let_user_pick(options, print_intro=True, random_selection=False):
    if print_intro and not random_selection:
        print("Please choose:")
    for idx, element in enumerate(options):
        print("{}) {}".format(idx + 1, element))
    if not random_selection:
        i = input("Enter number: ")
        if i.isdigit():
            i = int(i)
        else:
            raise (ValueError("Input not a valid number"))
    else:
        i = random.choice(range(len(options))) + 1
    if 0 < i <= len(options):
        return i
    else:
        raise (ValueError("Input not in the list"))
