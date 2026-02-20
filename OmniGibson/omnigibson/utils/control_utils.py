"""
Set of utilities for helping to execute robot control
"""
import torch as th

import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
import numpy as np
from numpy.typing import NDArray
from numba import jit


class FKSolver:
    """
    Class for thinly wrapping Lula Forward Kinematics solver
    """

    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
    ):
        # Create robot description and kinematics
        self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()

    def get_link_poses(
        self,
        joint_positions,
        link_names,
    ):
        """
        Given @joint_positions, get poses of the desired links (specified by @link_names)

        Args:
            joint_positions (n-array): Joint positions in configuration space
            link_names (list): List of robot link names we want to specify (e.g. "gripper_link")

        Returns:
            link_poses (dict): Dictionary mapping each robot link name to its pose
        """
        # TODO: Refactor this to go over all links at once
        link_poses = {}
        for link_name in link_names:
            pose3_lula = self.kinematics.pose(joint_positions, link_name)

            # get position
            link_position = th.tensor(pose3_lula.translation, dtype=th.float32)

            # get orientation
            rotation_lula = pose3_lula.rotation
            link_orientation = th.tensor(
                [rotation_lula.x(), rotation_lula.y(), rotation_lula.z(), rotation_lula.w()], dtype=th.float32
            )
            link_poses[link_name] = (link_position, link_orientation)
        return link_poses


class IKSolver:
    """
    Class for thinly wrapping Lula IK solver
    """

    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
        eef_name,
        reset_joint_pos,
    ):
        # Create robot description, kinematics, and config
        self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.config = lazy.lula.CyclicCoordDescentIkConfig()
        self.eef_name = eef_name
        self.reset_joint_pos = reset_joint_pos

    def solve(
        self,
        target_pos,
        target_quat=None,
        tolerance_pos=0.002,
        tolerance_quat=0.01,
        weight_pos=1.0,
        weight_quat=0.05,
        bfgs_orientation_weight=100.0,
        max_iterations=10,
        initial_joint_pos=None,
    ):
        """
        Backs out joint positions to achieve desired @target_pos and @target_quat

        Args:
            target_pos (3-array): desired (x,y,z) local target cartesian position in robot's base coordinate frame
            target_quat (4-array or None): If specified, desired (x,y,z,w) local target quaternion orientation in
                robot's base coordinate frame. If None, IK will be position-only (will override settings such that
                orientation's tolerance is very high and weight is 0)
            tolerance_pos (float): Maximum position error (L2-norm) for a successful IK solution
            tolerance_quat (float): Maximum orientation error (per-axis L2-norm) for a successful IK solution
            weight_pos (float): Weight for the relative importance of position error during CCD
            weight_quat (float): Weight for the relative importance of position error during CCD
            bfgs_orientation_weight (float): Weight when applying BFGS algorithm during optimization. Only used if
                target_quat is specified
            max_iterations (int): Number of iterations used for each cyclic coordinate descent.
            initial_joint_pos (None or n-array): If specified, will set the initial cspace seed when solving for joint
                positions. Otherwise, will use self.reset_joint_pos

        Returns:
            None or n-array: Joint positions for reaching desired target_pos and target_quat, otherwise None if no
                solution was found
        """
        pos = (
            target_pos.to(th.float64) if isinstance(target_pos, th.Tensor) else th.tensor(target_pos, dtype=th.float64)
        ).reshape(3, 1)

        if target_quat is None:
            rot = T.quat2mat(th.tensor([0, 0, 0, 1.0], dtype=th.float64))
        else:
            rot = T.quat2mat(
                target_quat.to(th.float64)
                if isinstance(target_quat, th.Tensor)
                else th.tensor(target_quat, dtype=th.float64)
            )
        ik_target_pose = lazy.lula.Pose3(lazy.lula.Rotation3(rot), pos)

        # Set the cspace seed and tolerance
        initial_joint_pos = self.reset_joint_pos if initial_joint_pos is None else initial_joint_pos
        self.config.cspace_seeds = [initial_joint_pos]
        self.config.position_tolerance = tolerance_pos
        self.config.orientation_tolerance = 100.0 if target_quat is None else tolerance_quat

        self.config.ccd_position_weight = weight_pos
        self.config.ccd_orientation_weight = 0.0 if target_quat is None else weight_quat
        self.config.bfgs_orientation_weight = 0.0 if target_quat is None else bfgs_orientation_weight
        self.config.max_num_descents = max_iterations

        # Compute target joint positions
        ik_results = lazy.lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        if ik_results.success:
            return th.tensor(ik_results.cspace_position, dtype=th.float32)
        else:
            return None


"""
Core J-PARSE algorithm implementation.

This module contains the pure J-PARSE algorithm that only requires numpy.
It operates directly on Jacobian matrices without any robot kinematics dependencies.

This is code directly taken from the jparse_robotics library: https://github.com/armlabstanford/jparse and modified to be numba-compatible

J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively in Inverse Kinematic Control of Serial Manipulators

Authors:
Shivani Guptasarma Matt Strong Honghao Zhen Monroe Kennedy

Software licensed under MIT License, taken from the original repository: https://github.com/armlabstanford/jparse

MIT License

Copyright (c) 2025 Assistive Robotics and Manipulation Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#### JPARSE LIBRARY ####


@jit(nopython=True)
def _jparse_svd_compose(U, S, Vt):
    """Recompose a matrix from SVD components: U @ diag(S) @ Vt."""
    Sfull = np.zeros((U.shape[1], Vt.shape[0]))
    k = min(len(S), Sfull.shape[0], Sfull.shape[1])
    for i in range(k):
        Sfull[i, i] = S[i]
    return U @ Sfull @ Vt


@jit(nopython=True)
def _jparse_compute_core(jacobian, gamma, sg_gain_pos, sg_gain_ang, pos_dims, ang_dims):
    """
    Core J-PARSE algorithm returning (J_parse, J_safety_pinv, J_safety).

    J-PARSE decomposes the Jacobian via SVD and constructs a modified pseudo-inverse:
      J_parse = J_safety^+ @ J_proj @ J_proj^+ + J_safety^+ @ Phi_singular

    Where J_safety clamps singular values below gamma*sigma_max, J_proj retains only
    non-singular directions, and Phi_singular provides smooth feedback in singular directions.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix to invert.
    gamma : float
        Singularity threshold (0 < gamma < 1).
    sg_gain_pos : float
        Gain for position-related singular directions.
    sg_gain_ang : float
        Gain for orientation-related singular directions.
    pos_dims : int
        Number of position rows (-1 = unspecified).
    ang_dims : int
        Number of angular rows (-1 = unspecified).

    Returns
    -------
    J_parse : ndarray, shape (n, m)
    J_safety_pinv : ndarray, shape (n, m)
    J_safety : ndarray, shape (m, n)
    """
    J = jacobian.astype(np.float64)
    m, n = J.shape

    U, S, Vt = np.linalg.svd(J)
    k = len(S)
    sigma_max = S[0]
    threshold = gamma * sigma_max

    n_nonsing = 0
    for i in range(k):
        if S[i] > threshold:
            n_nonsing += 1
    n_sing = k - n_nonsing

    # Projection Jacobian: only non-singular directions
    if n_nonsing > 0:
        U_proj = np.empty((m, n_nonsing))
        S_proj = np.empty(n_nonsing)
        idx = 0
        for i in range(k):
            if S[i] > threshold:
                U_proj[:, idx] = U[:, i]
                S_proj[idx] = S[i]
                idx += 1
        J_proj = _jparse_svd_compose(U_proj, S_proj, Vt)
    else:
        J_proj = J.copy()

    # Safety Jacobian: clamp singular values to threshold
    S_safety = np.empty(k)
    for i in range(k):
        S_safety[i] = S[i] if S[i] > threshold else threshold
    J_safety = _jparse_svd_compose(U, S_safety, Vt)

    # Singular direction feedback term
    Phi_singular = np.zeros((m, m))

    if n_sing > 0:
        U_sing = np.empty((m, n_sing))
        phi_vals = np.empty(n_sing)
        idx = 0
        for i in range(k):
            if S[i] <= threshold:
                U_sing[:, idx] = U[:, i]
                phi_vals[idx] = (S[i] / sigma_max) / gamma
                idx += 1

        Phi_mat = np.diag(phi_vals)

        if pos_dims < 0 and ang_dims < 0:
            gains = np.full(m, sg_gain_pos)
        elif ang_dims < 0:
            gains = np.full(pos_dims, sg_gain_pos)
        elif pos_dims < 0:
            gains = np.full(ang_dims, sg_gain_ang)
        else:
            gains = np.empty(pos_dims + ang_dims)
            for i in range(pos_dims):
                gains[i] = sg_gain_pos
            for i in range(ang_dims):
                gains[pos_dims + i] = sg_gain_ang

        Kp = np.diag(gains)
        Phi_singular = U_sing @ Phi_mat @ U_sing.T @ Kp

    J_safety_pinv = np.linalg.pinv(J_safety)
    J_proj_pinv = np.linalg.pinv(J_proj)

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv
    if n_sing > 0:
        J_parse = J_parse + J_safety_pinv @ Phi_singular

    return J_parse, J_safety_pinv, J_safety


@jit(nopython=True)
def jparse_compute(
    jacobian,
    gamma=0.1,
    singular_direction_gain_position=1.0,
    singular_direction_gain_angular=1.0,
    position_dimensions=-1,
    angular_dimensions=-1,
):
    """
    Compute the J-PARSE pseudo-inverse of a Jacobian matrix.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix to invert.
    gamma : float, optional
        Singularity threshold (0 < gamma < 1). Default is 0.1.
    singular_direction_gain_position : float, optional
        Gain for position-related singular directions. Default is 1.0.
    singular_direction_gain_angular : float, optional
        Gain for orientation-related singular directions. Default is 1.0.
    position_dimensions : int, optional
        Number of rows corresponding to position (-1 = unspecified). Default is -1.
    angular_dimensions : int, optional
        Number of rows corresponding to orientation (-1 = unspecified). Default is -1.

    Returns
    -------
    ndarray
        The n x m J-PARSE pseudo-inverse matrix.

    Examples
    --------
    >>> J = np.array([[-0.707, -0.707], [0.707, 0.707]])
    >>> J_parse = jparse_compute(J, gamma=0.1)
    >>> desired_velocity = np.array([0.1, 0.1])
    >>> joint_velocities = J_parse @ desired_velocity
    """
    J_parse, _, _ = _jparse_compute_core(
        jacobian, gamma,
        singular_direction_gain_position, singular_direction_gain_angular,
        position_dimensions, angular_dimensions,
    )
    return J_parse


@jit(nopython=True)
def jparse_compute_with_nullspace(
    jacobian,
    gamma=0.1,
    singular_direction_gain_position=1.0,
    singular_direction_gain_angular=1.0,
    position_dimensions=-1,
    angular_dimensions=-1,
):
    """
    Compute the J-PARSE pseudo-inverse and nullspace projection matrix.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix to invert.
    gamma : float, optional
        Singularity threshold (0 < gamma < 1). Default is 0.1.
    singular_direction_gain_position : float, optional
        Gain for position-related singular directions. Default is 1.0.
    singular_direction_gain_angular : float, optional
        Gain for orientation-related singular directions. Default is 1.0.
    position_dimensions : int, optional
        Number of rows corresponding to position (-1 = unspecified). Default is -1.
    angular_dimensions : int, optional
        Number of rows corresponding to orientation (-1 = unspecified). Default is -1.

    Returns
    -------
    J_parse : ndarray
        The n x m J-PARSE pseudo-inverse matrix.
    nullspace : ndarray
        The n x n nullspace projection matrix for secondary objectives.
    """
    J_parse, J_safety_pinv, J_safety = _jparse_compute_core(
        jacobian, gamma,
        singular_direction_gain_position, singular_direction_gain_angular,
        position_dimensions, angular_dimensions,
    )
    nullspace = np.eye(J_safety.shape[1]) - J_safety_pinv @ J_safety
    return J_parse, nullspace


@jit(nopython=True)
def jparse_pinv(jacobian):
    """
    Compute the standard Moore-Penrose pseudo-inverse.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix.

    Returns
    -------
    ndarray
        The n x m pseudo-inverse matrix.
    """
    return np.linalg.pinv(jacobian.astype(np.float64))


@jit(nopython=True)
def jparse_damped_least_squares(jacobian, damping=0.01):
    """
    Compute the damped least squares (DLS) pseudo-inverse.

    Also known as Levenberg-Marquardt or SR-inverse.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix.
    damping : float, optional
        Damping factor (lambda). Default is 0.01.

    Returns
    -------
    ndarray
        The n x m damped least squares pseudo-inverse.
    """
    J = jacobian.astype(np.float64)
    return np.linalg.inv(J.T @ J + damping ** 2 * np.eye(J.shape[1])) @ J.T


@jit(nopython=True)
def jparse_damped_least_squares_with_nullspace(jacobian, damping=0.01):
    """
    Compute the DLS pseudo-inverse and nullspace projection matrix.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix.
    damping : float, optional
        Damping factor (lambda). Default is 0.01.

    Returns
    -------
    J_dls : ndarray
        The n x m damped least squares pseudo-inverse.
    nullspace : ndarray
        The n x n nullspace projection matrix.
    """
    J = jacobian.astype(np.float64)
    J_dls = np.linalg.inv(J.T @ J + damping ** 2 * np.eye(J.shape[1])) @ J.T
    nullspace = np.eye(J.shape[1]) - J_dls @ J
    return J_dls, nullspace


@jit(nopython=True)
def manipulability_measure(jacobian: NDArray[np.floating]) -> float:
    """
    Compute Yoshikawa's manipulability measure.

    The manipulability measure indicates how well-conditioned the Jacobian is
    at a given configuration. Higher values indicate better manipulability.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix.

    Returns
    -------
    float
        The manipulability measure: sqrt(det(J @ J.T))
    """
    J = np.asarray(jacobian)
    return float(np.sqrt(np.linalg.det(J @ J.T)))


@jit(nopython=True)
def inverse_condition_number(jacobian: NDArray[np.floating]) -> float:
    """
    Compute the inverse condition number of the Jacobian.

    The inverse condition number (sigma_min / sigma_max) indicates proximity
    to singularity. Values close to 0 indicate near-singular configurations,
    while values close to 1 indicate well-conditioned configurations.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix.

    Returns
    -------
    float
        The inverse condition number (sigma_min / sigma_max).
    """
    J = np.asarray(jacobian)
    _, S, _ = np.linalg.svd(J)
    return float(np.min(S) / np.max(S))

#### END JPARSE LIBRARY ####
