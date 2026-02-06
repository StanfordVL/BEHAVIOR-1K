# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions to project between pointcloud and depth images, borrowed from Isaac Lab: https://github.com/isaac-sim/IsaacLab."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations
from collections.abc import Sequence
import math
from typing import Literal, Union

import numpy as np
import torch
import omnigibson.utils.transform_utils as T

# import warp as wp

TensorData = Union[np.ndarray, torch.Tensor]  # , wp.array]

"""
Math utils
"""

"""
Depth <-> Pointcloud conversions.
"""


def create_pointcloud_from_depth(
    intrinsic_matrix: TensorData,
    depth: TensorData,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (x, y, z, w) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to "cpu".

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # compute pointcloud
    depth_cloud = unproject_depth(depth, intrinsic_matrix)

    # convert 3D points to world frame
    rot_mat = T.quat2mat(orientation)
    # and apply rotation
    depth_cloud = torch.matmul(depth_cloud, rot_mat.mT)
    # apply translation
    depth_cloud += position[None, :]

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(
            torch.logical_and(~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)),
            dim=1,
        )
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    return depth_cloud


def create_pointcloud_from_rgbd(
    intrinsic_matrix: TensorData,
    depth: TensorData,
    rgb: TensorData | tuple[float, float, float] = None,
    normalize_rgb: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
    num_channels: int = 3,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
    """Creates pointcloud from input depth image and camera transformation matrix.

    This function provides the same functionality as :meth:`create_pointcloud_from_depth` but also allows
    to provide the RGB values for each point.

    The ``rgb`` attribute is used to resolve the corresponding point's color:

    - If a ``np.array``/``wp.array``/``torch.tensor`` of shape (H, W, 3), then the corresponding channels encode RGB values.
    - If a tuple, then the point cloud has a single color specified by the values (r, g, b).
    - If None, then default color is white, i.e. (0, 0, 0).

    If the input ``normalize_rgb`` is set to :obj:`True`, then the RGB values are normalized to be in the range [0, 1].

    Args:
        intrinsic_matrix: A (3, 3) array/tensor providing camera's calibration matrix.
        depth: An array/tensor of shape (H, W) with values encoding the depth measurement.
        rgb: Color for generated point cloud. Defaults to None.
        normalize_rgb: Whether to normalize input rgb. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation `(x, y, z, w)` of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed. Defaults to "cpu".

    Returns:
        A tuple of (N, 3) arrays or tensors containing the 3D coordinates of points and their RGB color respectively.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).
    """
    points_xyz = create_pointcloud_from_depth(intrinsic_matrix, depth, True, position, orientation, device=device)

    # convert the matrix to (W, H, 3) from (H, W, 3) since depth processing
    # is done in the order (u, v) where u: (0, W-1) and v: (0 - H-1)
    points_rgb = rgb[:, :, :3].permute(1, 0, 2).reshape(-1, 3)
    return points_xyz, points_rgb


# @torch.jit.script
def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor, is_ortho: bool = True) -> torch.Tensor:
    r"""Un-project depth image into a pointcloud.

    This function converts orthogonal or perspective depth images into points given the calibration matrix
    of the camera. It uses the following transformation based on camera geometry:

    .. math::
        p_{3D} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`p_{3D}` is the 3D point, :math:`d` is the depth value (measured from the image plane),
    :math:`u` and :math:`v` are the pixel coordinates and :math:`K` is the intrinsic matrix.

    The function assumes that the width and height are both greater than 1. This makes the function
    deal with many possible shapes of depth images and intrinsics matrices.

    .. note::
        If :attr:`is_ortho` is False, the input depth images are transformed to orthogonal depth images
        by using the :meth:`orthogonalize_perspective_depth` method.

    Args:
        depth: The depth measurement. Shape is (N, H, W)
        intrinsics: The camera's calibration matrix. If a single matrix is provided, the same
            calibration matrix is used across all the depth images in the batch.
            Shape is (N, 3, 3).
        is_ortho: Whether the input depth image is orthogonal or perspective depth image. If True, the input
            depth image is considered as the *orthogonal* type, where the measurements are from the camera's
            image plane. If False, the depth image is considered as the *perspective* type, where the
            measurements are from the camera's optical center. Defaults to True.

    Returns:
        The 3D coordinates of points. Shape is (N, P, 3).
    """
    # get image height and width
    im_height, im_width = depth.shape[1:]

    # convert depth image to orthogonal if needed
    if not is_ortho:
        # Get the intrinsics parameters
        fx = intrinsics[:, 0, 0].view(-1, 1, 1)
        fy = intrinsics[:, 1, 1].view(-1, 1, 1)
        cx = intrinsics[:, 0, 2].view(-1, 1, 1)
        cy = intrinsics[:, 1, 2].view(-1, 1, 1)

        # Create meshgrid of pixel coordinates
        u_grid = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
        v_grid = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
        u_grid, v_grid = torch.meshgrid(u_grid, v_grid, indexing="xy")

        # Expand the grids for batch processing
        u_grid = u_grid.unsqueeze(0).expand(depth.shape[0], -1, -1)
        v_grid = v_grid.unsqueeze(0).expand(depth.shape[0], -1, -1)

        # Compute the squared terms for efficiency
        x_term = ((u_grid - cx) / fx) ** 2
        y_term = ((v_grid - cy) / fy) ** 2

        # Calculate the orthogonal (normal) depth
        depth = depth / torch.sqrt(1 + x_term + y_term)

    # create image points in homogeneous coordinates (3, H x W)
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)
    pixels = pixels.unsqueeze(0)  # (3, H x W) -> (1, 3, H x W)

    # unproject points into 3D space
    points = torch.matmul(torch.inverse(intrinsics), pixels)  # (N, 3, H x W)
    points = points / points[:, -1, :].unsqueeze(1)  # normalize by last coordinate
    # flatten depth image (N, H, W) -> (N, H x W)
    depth = depth.transpose_(1, 2).reshape(depth.shape[0], -1).unsqueeze(2)
    depth = depth.expand(-1, -1, 3)
    # scale points by depth
    points_xyz = points.transpose_(1, 2) * depth  # (N, H x W, 3)

    return points_xyz


def _axis_angle_rotation(axis: Literal["X", "Y", "Z"], angle: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrices for one of the rotations about an axis of which Euler angles describe,
    for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: Euler angles in radians of any shape.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L164-L191
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_from_euler(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians. Shape is (..., 3).
        convention: Convention string of three uppercase letters from {"X", "Y", and "Z"}.
            For example, "XYZ" means that the rotations should be applied first about x,
            then y, then z.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L194-L220
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def convert_camera_frame_orientation_convention(
    orientation: torch.Tensor,
    origin: Literal["opengl", "ros", "world"] = "opengl",
    target: Literal["opengl", "ros", "world"] = "ros",
) -> torch.Tensor:
    r"""Converts a quaternion representing a rotation from one convention to another.

    In USD, the camera follows the ``"opengl"`` convention. Thus, it is always in **Y up** convention.
    This means that the camera is looking down the -Z axis with the +Y axis pointing up , and +X axis pointing right.
    However, in ROS, the camera is looking down the +Z axis with the +Y axis pointing down, and +X axis pointing right.
    Thus, the camera needs to be rotated by :math:`180^{\circ}` around the X axis to follow the ROS convention.

    .. math::

        T_{ROS} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    On the other hand, the typical world coordinate system is with +X pointing forward, +Y pointing left,
    and +Z pointing up. The camera can also be set in this convention by rotating the camera by :math:`90^{\circ}`
    around the X axis and :math:`-90^{\circ}` around the Y axis.

    .. math::

        T_{WORLD} = \begin{bmatrix} 0 & 0 & -1 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} T_{USD}

    Thus, based on their application, cameras follow different conventions for their orientation. This function
    converts a quaternion from one convention to another.

    Possible conventions are:

    - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
    - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
    - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

    Args:
        orientation: Quaternion of form `(x, y, z, w)` with shape (..., 4) in source convention.
        origin: Convention to convert from. Defaults to "opengl".
        target: Convention to convert to. Defaults to "ros".

    Returns:
        Quaternion of form `(x, y, z, w)` with shape (..., 4) in target convention
    """
    if target == origin:
        return orientation.clone()

    # -- unify input type
    if origin == "ros":
        # convert from ros to opengl convention
        rotm = T.quat2mat(orientation)
        rotm[:, 2] = -rotm[:, 2]
        rotm[:, 1] = -rotm[:, 1]
        # convert to opengl convention
        quat_gl = T.mat2quat(rotm)
    elif origin == "world":
        # convert from world (x forward and z up) to opengl convention
        rotm = T.quat2mat(orientation)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ"),
        )
        # convert to isaac-sim convention
        quat_gl = T.mat2quat(rotm)
    else:
        quat_gl = orientation

    # -- convert to target convention
    if target == "ros":
        # convert from opengl to ros convention
        rotm = T.quat2mat(quat_gl)
        rotm[:, 2] = -rotm[:, 2]
        rotm[:, 1] = -rotm[:, 1]
        return T.mat2quat(rotm)
    elif target == "world":
        # convert from opengl to world (x forward and z up) convention
        rotm = T.quat2mat(quat_gl)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ").T,
        )
        return T.mat2quat(rotm)
    else:
        return quat_gl.clone()
