"""
Set of utilities for helping to execute robot control
"""

import torch as th

import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T


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
            joint positions (n-array): Joint positions in configuration space
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


@th.jit.script
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
    # Compute batch size
    batch_size = desired.numel() // 9  # Each 3x3 matrix has 9 elements

    desired_flat = desired.reshape(batch_size, 3, 3)
    current_flat = current.reshape(batch_size, 3, 3)

    rc1, rc2, rc3 = current_flat[:, :, 0], current_flat[:, :, 1], current_flat[:, :, 2]
    rd1, rd2, rd3 = desired_flat[:, :, 0], desired_flat[:, :, 1], desired_flat[:, :, 2]

    error = 0.5 * (th.linalg.cross(rc1, rd1) + th.linalg.cross(rc2, rd2) + th.linalg.cross(rc3, rd3))

    return error.reshape(desired.shape[:-2] + (3,))
