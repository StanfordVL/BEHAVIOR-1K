"""
Set of utilities for helping to execute robot control
"""
import lula
import numpy as np
import omnigibson.utils.transform_utils as T


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
        self.config.position_weight = weight_pos
        self.config.orientation_weight = 0.0 if target_quat is None else weight_quat
        self.config.max_iterations_per_descent = max_iterations

        # Compute target joint positions
        ik_results = lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        return np.array(ik_results.cspace_position)
