"""
Set of utilities for helping to execute robot control
"""
import lula
import numpy as np
from numba import jit
import omnigibson.utils.transform_utils as T
from omnigibson.utils.sim_utils import meets_minimum_isaac_version

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
        self.robot_description = lula.load_robot(robot_description_path, robot_urdf_path)
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
            link_position = pose3_lula.translation

            # get orientation
            rotation_lula = pose3_lula.rotation
            link_orientation = (
                rotation_lula.x(),
                rotation_lula.y(),
                rotation_lula.z(),
                rotation_lula.w(),
            )
            link_poses[link_name] =  (link_position, link_orientation)
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
        default_joint_pos,
    ):
        # Create robot description, kinematics, and config
        self.robot_description = lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.config = lula.CyclicCoordDescentIkConfig()
        self.eef_name = eef_name
        self.default_joint_pos = default_joint_pos

    def solve(
        self,
        target_pos,
        target_quat=None,
        tolerance_pos=0.002,
        tolerance_quat=0.01,
        weight_pos=1.0,
        weight_quat=0.05,
        max_iterations=150,
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
            max_iterations (int): Number of iterations used for each cyclic coordinate descent.
            initial_joint_pos (None or n-array): If specified, will set the initial cspace seed when solving for joint
                positions. Otherwise, will use self.default_joint_pos

        Returns:
            None or n-array: Joint positions for reaching desired target_pos and target_quat, otherwise None if no
                solution was found
        """
        pos = np.array(target_pos, dtype=np.float64).reshape(3, 1)
        rot = np.array(T.quat2mat(np.array([0, 0, 0, 1.0]) if target_quat is None else target_quat), dtype=np.float64)
        ik_target_pose = lula.Pose3(lula.Rotation3(rot), pos)

        # Set the cspace seed and tolerance
        initial_joint_pos = self.default_joint_pos if initial_joint_pos is None else np.array(initial_joint_pos)
        self.config.cspace_seeds = [initial_joint_pos]
        self.config.position_tolerance = tolerance_pos
        self.config.orientation_tolerance = 100.0 if target_quat is None else tolerance_quat

        if meets_minimum_isaac_version("2023.0.0"):
            self.config.ccd_position_weight = weight_pos
            self.config.ccd_orientation_weight = 0.0 if target_quat is None else weight_quat
            self.config.max_num_descents = max_iterations
        else:
            self.config.position_weight = weight_pos
            self.config.orientation_weight = 0.0 if target_quat is None else weight_quat
            self.config.max_iterations_per_descent = max_iterations

        # Compute target joint positions
        ik_results = lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        if ik_results.success:
            return np.array(ik_results.cspace_position)
        else:
            return None


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
