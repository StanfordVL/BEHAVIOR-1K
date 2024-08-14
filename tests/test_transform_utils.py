import math

import numpy as np
import pytest
import torch as th
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torch.testing import assert_close

from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.transform_utils import *

# Set the seed for PyTorch
th.manual_seed(0)


# Helper functions
def random_vector():
    return th.rand(3)


def random_matrix():
    return th.rand(3, 3)


def are_rotations_close(R1, R2, atol=1e-3):
    return (
        th.allclose(R1 @ R1.t(), th.eye(3), atol=atol)
        and th.allclose(R2 @ R2.t(), th.eye(3), atol=atol)
        and th.allclose(R1, R2, atol=atol)
    )


class TestQuaternionOperations:
    @pytest.mark.parametrize(
        "q",
        [
            th.tensor([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            th.tensor([1.0, 0.0, 0.0, 0.0]),  # 180 degree rotation around x
            th.tensor([0.0, 1.0, 0.0, 0.0]),  # 180 degree rotation around y
            th.tensor([0.0, 0.0, 1.0, 0.0]),  # 180 degree rotation around z
        ],
    )
    def test_quat2mat_special_cases(self, q):
        q_np = q.cpu().numpy()
        scipy_mat = R.from_quat(q_np).as_matrix()
        our_mat = quat2mat(q)
        assert_close(our_mat, th.from_numpy(scipy_mat.astype(NumpyTypes.FLOAT32)))

    def test_quat_mul(self):
        q1, q2 = random_quaternion().squeeze(), random_quaternion().squeeze()
        q1_scipy = q1.cpu().numpy()
        q2_scipy = q2.cpu().numpy()
        scipy_result = R.from_quat(q1_scipy) * R.from_quat(q2_scipy)
        scipy_quat = scipy_result.as_quat()
        our_quat = quat_mul(q1, q2)
        assert quaternions_close(our_quat, th.from_numpy(scipy_quat.astype(NumpyTypes.FLOAT32)))

    def test_quat_conjugate(self):
        q = random_quaternion().squeeze()
        q_scipy = q.cpu().numpy()
        scipy_conj = R.from_quat(q_scipy).inv().as_quat()
        our_conj = quat_conjugate(q)
        assert quaternions_close(our_conj, th.from_numpy(scipy_conj.astype(NumpyTypes.FLOAT32)))

    def test_quat_inverse(self):
        q = random_quaternion().squeeze()
        scipy_inv = R.from_quat(q.cpu().numpy()).inv().as_quat().astype(NumpyTypes.FLOAT32)
        our_inv = quat_inverse(q)
        assert quaternions_close(our_inv, th.from_numpy(scipy_inv))
        q_identity = quat_mul(q, our_inv)
        assert quaternions_close(q_identity, th.tensor([0.0, 0.0, 0.0, 1.0]))

    def test_quat_distance(self):
        q1, q2 = random_quaternion().squeeze(), random_quaternion().squeeze()
        dist = quat_distance(q1, q2)
        assert quaternions_close(quat_mul(dist, q2), q1)


class TestVectorOperations:
    def test_normalize(self):
        normalized = normalize(random_vector())
        assert_close(th.norm(normalized), th.tensor(1.0))

    @pytest.mark.parametrize("dim", [-1, 0])
    def test_dot_product(self, dim):
        v1, v2 = random_vector(), random_vector()
        assert_close(dot(v1, v2, dim=dim), th.dot(v1, v2))

    def test_l2_distance(self):
        v1, v2 = random_vector(), random_vector()
        dist = l2_distance(v1, v2)
        assert_close(dist, th.norm(v1 - v2))


class TestMatrixOperations:
    def test_rotation_matrix_properties(self):
        rand_quat = random_quaternion().squeeze()
        R_mat = quat2mat(rand_quat)
        scipy_R = R.from_quat(rand_quat.cpu().numpy()).as_matrix().astype(NumpyTypes.FLOAT32)
        assert_close(R_mat, th.from_numpy(scipy_R))
        assert_close(R_mat @ R_mat.t(), th.eye(3))
        assert_close(th.det(R_mat), th.tensor(1.0))

    @pytest.mark.parametrize("angle", [0, math.pi / 4, math.pi / 2, math.pi])
    def test_rotation_matrix(self, angle):
        direction = normalize(random_vector())
        R_mat = rotation_matrix(angle, direction)
        scipy_R = R.from_rotvec(angle * direction.cpu().numpy()).as_matrix().astype(NumpyTypes.FLOAT32)
        assert_close(R_mat, th.from_numpy(scipy_R))

        identity = th.eye(3, dtype=R_mat.dtype, device=R_mat.device)
        assert_close(R_mat @ R_mat.t(), identity)
        assert_close(R_mat.t() @ R_mat, identity)
        assert_close(th.det(R_mat), th.tensor(1.0, dtype=R_mat.dtype, device=R_mat.device))

        test_vector = th.randn(3, dtype=R_mat.dtype, device=R_mat.device)
        rotated_vector = R_mat @ test_vector
        assert_close(th.norm(rotated_vector), th.norm(test_vector))

        if angle != 0:
            axis = R_mat @ direction
            assert_close(axis, direction)
            perpendicular = th.cross(direction, th.randn(3, dtype=R_mat.dtype, device=R_mat.device))
            perpendicular = normalize(perpendicular)
            rotated_perpendicular = R_mat @ perpendicular
            cos_angle = th.dot(perpendicular, rotated_perpendicular)
            assert_close(cos_angle, th.cos(th.tensor(angle)))

    @pytest.mark.parametrize("angle", [0, math.pi / 4, math.pi / 2, math.pi])
    def test_transformation_matrix(self, angle):
        direction = normalize(random_vector())
        point = th.randn(3, dtype=th.float32)
        T = transformation_matrix(angle, direction, point)

        direction_np = direction.cpu().numpy()
        scipy_R = R.from_rotvec(angle * direction_np).as_matrix().astype(NumpyTypes.FLOAT32)
        scipy_T = np.eye(4, dtype=NumpyTypes.FLOAT32)
        scipy_T[:3, :3] = scipy_R
        scipy_T[:3, 3] = point.cpu().numpy() - np.dot(scipy_R, point.cpu().numpy())
        assert_close(T, th.from_numpy(scipy_T))

        rot = T[:3, :3]
        identity = th.eye(3, dtype=rot.dtype, device=rot.device)
        assert_close(rot @ rot.t(), identity)
        assert_close(rot.t() @ rot, identity)
        assert_close(th.det(rot), th.tensor(1.0, dtype=rot.dtype, device=rot.device))
        assert_close(T[3, :], th.tensor([0, 0, 0, 1], dtype=T.dtype, device=T.device))

    def test_transformation_matrix_no_point(self):
        direction = normalize(random_vector())
        angle = math.pi / 4
        T = transformation_matrix(angle, direction)

        scipy_R = R.from_rotvec(angle * direction.cpu().numpy()).as_matrix().astype(NumpyTypes.FLOAT32)
        scipy_T = np.eye(4, dtype=NumpyTypes.FLOAT32)
        scipy_T[:3, :3] = scipy_R
        assert_close(T, th.from_numpy(scipy_T))

        assert_close(T[:3, 3], th.zeros(3, dtype=T.dtype, device=T.device))
        rot = rotation_matrix(angle, direction)
        assert_close(T[:3, :3], rot)

    def test_matrix_inverse(self):
        M = random_matrix()
        M_inv = matrix_inverse(M)
        scipy_M_inv = np.linalg.inv(M.cpu().numpy()).astype(NumpyTypes.FLOAT32)
        assert_close(M_inv, th.from_numpy(scipy_M_inv), atol=1e-3, rtol=1e-3)
        assert_close(M @ M_inv, th.eye(3))


class TestCoordinateTransformations:
    def test_cartesian_to_polar(self):
        x, y = 3.0, 4.0
        rho, phi = cartesian_to_polar(th.tensor(x), th.tensor(y))
        np_rho, np_phi = np.hypot(x, y), np.arctan2(y, x)
        assert_close(rho, th.tensor(np_rho, dtype=th.float32))
        assert_close(phi, th.tensor(np_phi, dtype=th.float32))


class TestPoseTransformations:
    def test_pose2mat_and_mat2pose(self):
        pos, orn = random_vector(), random_quaternion().squeeze()
        T = pose2mat((pos, orn))

        scipy_R = R.from_quat(orn.cpu().numpy())
        scipy_T = np.eye(4, dtype=NumpyTypes.FLOAT32)
        scipy_T[:3, :3] = scipy_R.as_matrix()
        scipy_T[:3, 3] = pos.cpu().numpy()

        assert_close(T, th.from_numpy(scipy_T))

        recovered_pos, recovered_orn = mat2pose(T)
        assert_close(pos, recovered_pos)
        assert quaternions_close(orn, recovered_orn)

    def test_pose_inv(self):
        pos, orn = random_vector(), random_quaternion().squeeze()
        T = pose2mat((pos, orn))
        T_inv = pose_inv(T)

        scipy_R = R.from_quat(orn.cpu().numpy())
        scipy_T = np.eye(4, dtype=NumpyTypes.FLOAT32)
        scipy_T[:3, :3] = scipy_R.as_matrix()
        scipy_T[:3, 3] = pos.cpu().numpy()
        scipy_T_inv = np.linalg.inv(scipy_T)

        assert_close(T_inv, th.from_numpy(scipy_T_inv))
        assert_close(T @ T_inv, th.eye(4))

    def test_relative_pose_transform(self):
        pos0, orn0 = random_vector(), random_quaternion().squeeze()
        pos1, orn1 = random_vector(), random_quaternion().squeeze()
        rel_pos, rel_orn = relative_pose_transform(pos1, orn1, pos0, orn0)

        scipy_R0 = R.from_quat(orn0.cpu().numpy())
        scipy_R1 = R.from_quat(orn1.cpu().numpy())
        scipy_rel_R = scipy_R0.inv() * scipy_R1
        scipy_rel_pos = scipy_R0.inv().apply(pos1.cpu().numpy() - pos0.cpu().numpy())

        assert_close(rel_pos, th.from_numpy(scipy_rel_pos.astype(NumpyTypes.FLOAT32)))
        assert quaternions_close(rel_orn, th.from_numpy(scipy_rel_R.as_quat().astype(NumpyTypes.FLOAT32)))


class TestAxisAngleConversions:
    @pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, math.pi])
    def test_axisangle2quat_and_quat2axisangle(self, angle):
        axis = normalize(random_vector())
        axisangle = axis * angle
        quat = axisangle2quat(axisangle)

        scipy_R = R.from_rotvec(axisangle.cpu().numpy())
        scipy_quat = scipy_R.as_quat().astype(NumpyTypes.FLOAT32)

        assert quaternions_close(quat, th.from_numpy(scipy_quat))

        recovered_axisangle = quat2axisangle(quat)
        scipy_recovered_axisangle = scipy_R.as_rotvec().astype(NumpyTypes.FLOAT32)

        assert th.allclose(recovered_axisangle, th.from_numpy(scipy_recovered_axisangle)) or th.allclose(
            recovered_axisangle, -th.from_numpy(scipy_recovered_axisangle)
        ), f"Axis-angles not equivalent: {recovered_axisangle} vs {scipy_recovered_axisangle}"

    def test_vecs2axisangle(self):
        vec1 = th.tensor([1.0, 0.0, 0.0])
        vec2 = th.tensor([0.0, 1.0, 0.0])
        axisangle = vecs2axisangle(vec1, vec2)

        scipy_R = R.align_vectors(vec2.unsqueeze(0).cpu().numpy(), vec1.unsqueeze(0).cpu().numpy())[0]
        scipy_axisangle = scipy_R.as_rotvec().astype(NumpyTypes.FLOAT32)

        assert_close(axisangle, th.from_numpy(scipy_axisangle))

    def test_vecs2quat(self):
        vec1 = th.tensor([1.0, 0.0, 0.0])
        vec2 = th.tensor([0.0, 1.0, 0.0])
        quat = vecs2quat(vec1, vec2)

        scipy_R = R.align_vectors(vec2.unsqueeze(0).cpu().numpy(), vec1.unsqueeze(0).cpu().numpy())[0]
        scipy_quat = scipy_R.as_quat().astype(NumpyTypes.FLOAT32)

        assert quaternions_close(quat, th.from_numpy(scipy_quat))


class TestEulerAngleConversions:
    @pytest.mark.parametrize(
        "euler",
        [
            th.tensor([0.0, 0.0, 0.0]),
            th.tensor([math.pi / 4, math.pi / 3, math.pi / 2]),
        ],
    )
    def test_euler2quat_and_quat2euler(self, euler):
        quat = euler2quat(euler)
        scipy_R = R.from_euler("xyz", euler.cpu().numpy())
        scipy_quat = scipy_R.as_quat().astype(NumpyTypes.FLOAT32)
        assert quaternions_close(quat, th.from_numpy(scipy_quat))

        recovered_euler = quat2euler(quat)
        scipy_recovered_euler = scipy_R.as_euler("xyz").astype(NumpyTypes.FLOAT32)
        assert_close(recovered_euler, th.from_numpy(scipy_recovered_euler))

    @pytest.mark.parametrize(
        "euler",
        [
            th.tensor([0.0, 0.0, 0.0]),
            th.tensor([math.pi / 4, math.pi / 3, math.pi / 2]),
        ],
    )
    def test_euler2mat_and_mat2euler(self, euler):
        mat = euler2mat(euler)
        scipy_R = R.from_euler("xyz", euler.cpu().numpy())
        scipy_mat = scipy_R.as_matrix().astype(NumpyTypes.FLOAT32)
        assert_close(mat, th.from_numpy(scipy_mat))

        recovered_euler = mat2euler(mat)
        scipy_recovered_euler = scipy_R.as_euler("xyz").astype(NumpyTypes.FLOAT32)
        assert_close(recovered_euler, th.from_numpy(scipy_recovered_euler))


class TestQuaternionApplications:
    def test_quat_apply(self):
        quat = random_quaternion().squeeze()
        vec = random_vector()
        rotated_vec = quat_apply(quat, vec)

        scipy_R = R.from_quat(quat.cpu().numpy())
        scipy_rotated_vec = scipy_R.apply(vec.cpu().numpy()).astype(NumpyTypes.FLOAT32)

        assert rotated_vec.shape == (3,)
        assert_close(rotated_vec, th.from_numpy(scipy_rotated_vec))
        assert_close(th.norm(rotated_vec), th.norm(vec))

    def test_quat_slerp(self):
        q1, q2 = random_quaternion().squeeze(), random_quaternion().squeeze()
        t = th.rand(1)
        q_slerp = quat_slerp(q1, q2, t)

        key_rots = R.from_quat(np.stack([q1.cpu().numpy(), q2.cpu().numpy()]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        scipy_q_slerp = slerp([t]).as_quat()[0].astype(NumpyTypes.FLOAT32)

        assert quaternions_close(q_slerp, th.from_numpy(scipy_q_slerp))
        assert_close(th.norm(q_slerp), th.tensor(1.0))


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


class TestTransformPoints:
    def test_transform_points_2d(self):
        points = th.tensor([[1.0, 0.0], [0.0, 1.0]])
        matrix = th.tensor([[0.0, -1.0, 2.0], [1.0, 0.0, 3.0], [0.0, 0.0, 1.0]])
        transformed = transform_points(points, matrix)
        transformed_trimesh = th.tensor(trimesh.transform_points(points, matrix), dtype=th.float32)
        assert_close(transformed, transformed_trimesh)

    def test_transform_points_3d(self):
        points = th.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        matrix = th.eye(4)
        matrix[:3, 3] = th.tensor([1.0, 2.0, 3.0])
        transformed = transform_points(points, matrix)
        transformed_trimesh = th.tensor(trimesh.transform_points(points, matrix), dtype=th.float32)
        assert_close(transformed, transformed_trimesh)


class TestMiscellaneousFunctions:
    def test_convert_quat(self):
        quat_wxyz = th.tensor([1.0, 2.0, 3.0, 4.0])
        quat_xyzw = convert_quat(quat_wxyz, to="xyzw")
        assert_close(quat_xyzw, th.tensor([2.0, 3.0, 4.0, 1.0]))

    def test_random_quaternion(self):
        num_quats = 10
        quats = random_quaternion(num_quaternions=num_quats)
        assert quats.shape == (num_quats, 4)
        assert_close(th.norm(quats, dim=1), th.ones(num_quats))

    def test_random_axis_angle(self):
        axis, angle = random_axis_angle()
        assert_close(th.norm(axis), th.tensor(1.0))
        assert 0 <= angle <= 2 * math.pi

    def test_align_vector_sets(self):
        # Test case 1: 90-degree rotation
        vec_set1 = th.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=th.float32)
        vec_set2 = th.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=th.float32)
        quat = align_vector_sets(vec_set2, vec_set1)
        expected_quat = th.tensor([0.5, 0.5, 0.5, 0.5], dtype=th.float32)
        assert th.allclose(quat.abs(), expected_quat.abs(), atol=1e-6)

        # Test case 2: Identity rotation
        vec_set1 = th.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=th.float32)
        vec_set2 = vec_set1.clone()
        quat = align_vector_sets(vec_set2, vec_set1)
        expected_quat = th.tensor([0, 0, 0, 1], dtype=th.float32)
        assert th.allclose(quat.abs(), expected_quat.abs(), atol=1e-6)

        # Test case 3: 120-degree rotation around [1, 1, 1] axis
        vec_set1 = th.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=th.float32)
        vec_set2 = th.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=th.float32)
        quat = align_vector_sets(vec_set2, vec_set1)
        expected_quat = th.tensor([0.5, 0.5, 0.5, 0.5], dtype=th.float32)
        assert th.allclose(quat.abs(), expected_quat.abs(), atol=1e-6)

    def test_copysign(self):
        a = 2.0
        b = th.tensor([-1.0, 1.0])
        result = copysign(a, b)
        assert_close(result, th.tensor([-2.0, 2.0]))

    def test_anorm(self):
        x = th.tensor([[1.0, 2.0], [3.0, 4.0]])
        norm = anorm(x, dim=1)
        expected = th.tensor([math.sqrt(5), math.sqrt(25)])
        assert_close(norm, expected)

    def test_check_quat_right_angle(self):
        right_angle_quat = th.tensor([0.0, 0.707106781, 0.0, 0.707106781])
        assert check_quat_right_angle(right_angle_quat)

        non_right_angle_quat = th.tensor([0.1, 0.2, 0.3, 0.9])
        assert not check_quat_right_angle(non_right_angle_quat)

    def test_z_angle_from_quat(self):
        quat = euler2quat(th.tensor([0.0, 0.0, math.pi / 4]))
        angle = z_angle_from_quat(quat)
        assert_close(angle, th.tensor(math.pi / 4))

    def test_integer_spiral_coordinates(self):
        coords = [integer_spiral_coordinates(i) for i in range(5)]
        expected = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1)]
        assert coords == expected
