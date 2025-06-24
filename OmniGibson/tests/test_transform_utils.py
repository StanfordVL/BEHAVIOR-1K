import math

import numpy as np
import torch as th
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torch.testing import assert_close

from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.transform_utils import (
    align_vector_sets,
    anorm,
    axisangle2quat,
    cartesian_to_polar,
    check_quat_right_angle,
    convert_quat,
    copysign,
    dot,
    euler2mat,
    euler2quat,
    integer_spiral_coordinates,
    l2_distance,
    mat2euler,
    mat2pose,
    matrix_inverse,
    normalize,
    pose2mat,
    pose_inv,
    quat2axisangle,
    quat2euler,
    quat2mat,
    quat_apply,
    quat_conjugate,
    quat_distance,
    quat_inverse,
    quat_multiply,
    quat_slerp,
    quaternions_close,
    random_axis_angle,
    random_quaternion,
    rotation_matrix,
    transform_points,
    transformation_matrix,
    vecs2axisangle,
    vecs2quat,
    z_angle_from_quat,
)

# Create constants for vectors
RANDOM_VECTORS = [
    th.tensor([0.56853108, 0.53382016, 0.30716877], dtype=th.float32),
    th.tensor([0.52257347, 0.61831128, 0.83885363], dtype=th.float32),
    th.tensor([0.21115992, 0.21581846, 0.32354917], dtype=th.float32),
    th.tensor([0.29239464, 0.56651807, 0.18654108], dtype=th.float32),
]

# Create constants for matrices
RANDOM_MATRICES = [
    th.tensor(
        [
            [0.73807538, 0.34566713, 0.22840234],
            [0.6477331, 0.11909693, 0.54837387],
            [0.80250765, 0.98231487, 0.30666593],
        ],
        dtype=th.float32,
    ),
    th.tensor(
        [
            [0.49792992, 0.34195128, 0.97021054],
            [0.34943073, 0.94597711, 0.4247565],
            [0.33942933, 0.34367859, 0.12948883],
        ],
        dtype=th.float32,
    ),
    th.tensor(
        [
            [0.14253589, 0.0570198, 0.52688842],
            [0.13947784, 0.71386355, 0.25629677],
            [0.49064311, 0.72391959, 0.46148444],
        ],
        dtype=th.float32,
    ),
    th.tensor(
        [
            [0.37922823, 0.11913949, 0.97869396],
            [0.17461795, 0.55869352, 0.18168803],
            [0.88240868, 0.57003021, 0.09736692],
        ],
        dtype=th.float32,
    ),
]

# Create constants for quaternions
RANDOM_QUATERNIONS = [
    th.tensor([-0.67976515, 0.50242053, -0.18529368, -0.50115786], dtype=th.float32),
    th.tensor([0.7823932, -0.18596287, -0.43777126, 0.40203857], dtype=th.float32),
    th.tensor([-0.66576888, 0.56006078, 0.06682257, -0.4884859], dtype=th.float32),
    th.tensor([0.6827, 0.7298, 0.0191, 0.0301], dtype=th.float32),
]


def are_rotations_close(R1, R2, atol=1e-3):
    return (
        th.allclose(R1 @ R1.t(), th.eye(3), atol=atol)
        and th.allclose(R2 @ R2.t(), th.eye(3), atol=atol)
        and th.allclose(R1, R2, atol=atol)
    )


class TestQuaternionOperations:
    def test_quat2mat_special_cases(self):
        special_quats = [
            th.tensor([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            th.tensor([1.0, 0.0, 0.0, 0.0]),  # 180 degree rotation around x
            th.tensor([0.0, 1.0, 0.0, 0.0]),  # 180 degree rotation around y
            th.tensor([0.0, 0.0, 1.0, 0.0]),  # 180 degree rotation around z
        ]
        for q in special_quats:
            q_np = q.cpu().numpy()
            scipy_mat = R.from_quat(q_np).as_matrix()
            our_mat = quat2mat(q)
            assert_close(our_mat, th.from_numpy(scipy_mat.astype(NumpyTypes.FLOAT32)))

    def test_quat_multiply(self):
        for i in range(0, len(RANDOM_QUATERNIONS), 2):
            q1, q2 = RANDOM_QUATERNIONS[i], RANDOM_QUATERNIONS[i + 1]
            q1_scipy = q1.cpu().numpy()
            q2_scipy = q2.cpu().numpy()
            scipy_result = R.from_quat(q1_scipy) * R.from_quat(q2_scipy)
            scipy_quat = scipy_result.as_quat()
            our_quat = quat_multiply(q1, q2)
            assert quaternions_close(our_quat, th.from_numpy(scipy_quat.astype(NumpyTypes.FLOAT32)))

    def test_quat_conjugate(self):
        for q in RANDOM_QUATERNIONS:
            q_scipy = q.cpu().numpy()
            scipy_conj = R.from_quat(q_scipy).inv().as_quat()
            our_conj = quat_conjugate(q)
            assert quaternions_close(our_conj, th.from_numpy(scipy_conj.astype(NumpyTypes.FLOAT32)))

    def test_quat_inverse(self):
        for q in RANDOM_QUATERNIONS:
            scipy_inv = R.from_quat(q.cpu().numpy()).inv().as_quat().astype(NumpyTypes.FLOAT32)
            our_inv = quat_inverse(q)
            assert quaternions_close(our_inv, th.from_numpy(scipy_inv))
            q_identity = quat_multiply(q, our_inv)
            assert quaternions_close(q_identity, th.tensor([0.0, 0.0, 0.0, 1.0]))

    def test_quat_distance(self):
        for i in range(0, len(RANDOM_QUATERNIONS), 2):
            q1, q2 = RANDOM_QUATERNIONS[i], RANDOM_QUATERNIONS[i + 1]
            dist = quat_distance(q2, q1)
            assert quaternions_close(quat_multiply(dist, q1), q2)


class TestVectorOperations:
    def test_normalize(self):
        for v in RANDOM_VECTORS:
            normalized = normalize(v)
            assert_close(th.norm(normalized), th.tensor(1.0))

    def test_dot_product(self):
        for i in range(0, len(RANDOM_VECTORS), 2):
            v1, v2 = RANDOM_VECTORS[i], RANDOM_VECTORS[i + 1]
            for dim in [-1, 0]:
                assert_close(dot(v1, v2, dim=dim), th.dot(v1, v2))

    def test_l2_distance(self):
        for i in range(0, len(RANDOM_VECTORS), 2):
            v1, v2 = RANDOM_VECTORS[i], RANDOM_VECTORS[i + 1]
            dist = l2_distance(v1, v2)
            assert_close(dist, th.norm(v1 - v2))


class TestMatrixOperations:
    def test_rotation_matrix_properties(self):
        for rand_quat in RANDOM_QUATERNIONS:
            R_mat = quat2mat(rand_quat)
            scipy_R = R.from_quat(rand_quat.cpu().numpy()).as_matrix().astype(NumpyTypes.FLOAT32)
            assert_close(R_mat, th.from_numpy(scipy_R))
            assert_close(R_mat @ R_mat.t(), th.eye(3))
            assert_close(th.det(R_mat), th.tensor(1.0))

    def test_rotation_matrix(self):
        angles = [0, math.pi / 4, math.pi / 2, math.pi]
        for angle in angles:
            for direction in RANDOM_VECTORS:
                direction = normalize(direction)
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

    def test_transformation_matrix(self):
        angles = [0, math.pi / 4, math.pi / 2, math.pi]
        for angle in angles:
            for direction in RANDOM_VECTORS:
                for point in RANDOM_VECTORS:
                    direction = normalize(direction)
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
        angles = [0, math.pi / 4, math.pi / 2, math.pi]
        for angle in angles:
            for direction in RANDOM_VECTORS:
                direction = normalize(direction)
                T = transformation_matrix(angle, direction)

                scipy_R = R.from_rotvec(angle * direction.cpu().numpy()).as_matrix().astype(NumpyTypes.FLOAT32)
                scipy_T = np.eye(4, dtype=NumpyTypes.FLOAT32)
                scipy_T[:3, :3] = scipy_R
                assert_close(T, th.from_numpy(scipy_T))

                assert_close(T[:3, 3], th.zeros(3, dtype=T.dtype, device=T.device))
                rot = rotation_matrix(angle, direction)
                assert_close(T[:3, :3], rot)

    def test_matrix_inverse(self):
        for M in RANDOM_MATRICES:
            M_inv = matrix_inverse(M)
            scipy_M_inv = np.linalg.inv(M.cpu().numpy()).astype(NumpyTypes.FLOAT32)
            assert_close(M_inv, th.from_numpy(scipy_M_inv), atol=1e-3, rtol=1e-3)
            assert_close(M @ M_inv, th.eye(3))


class TestCoordinateTransformations:
    def test_cartesian_to_polar(self):
        for v in RANDOM_VECTORS:
            x, y = v[0], v[1]
            rho, phi = cartesian_to_polar(x, y)
            np_rho, np_phi = np.hypot(x, y), np.arctan2(y, x)
            assert_close(rho, th.tensor(np_rho, dtype=th.float32))
            assert_close(phi, th.tensor(np_phi, dtype=th.float32))


class TestPoseTransformations:
    def test_pose2mat_and_mat2pose(self):
        for pos in RANDOM_VECTORS:
            for orn in RANDOM_QUATERNIONS:
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
        for pos in RANDOM_VECTORS:
            for orn in RANDOM_QUATERNIONS:
                T = pose2mat((pos, orn))
                T_inv = pose_inv(T)

                scipy_R = R.from_quat(orn.cpu().numpy())
                scipy_T = np.eye(4, dtype=NumpyTypes.FLOAT32)
                scipy_T[:3, :3] = scipy_R.as_matrix()
                scipy_T[:3, 3] = pos.cpu().numpy()
                scipy_T_inv = np.linalg.inv(scipy_T)

                assert_close(T_inv, th.from_numpy(scipy_T_inv))
                assert_close(T @ T_inv, th.eye(4))


class TestAxisAngleConversions:
    def test_axisangle2quat_and_quat2axisangle(self):
        angles = [0.0, math.pi / 4, math.pi / 2, math.pi]
        for angle in angles:
            for axis in RANDOM_VECTORS:
                axis = normalize(axis)
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
    def test_euler2quat_and_quat2euler(self):
        euler_angles = [
            th.tensor([0.0, 0.0, 0.0]),
            th.tensor([math.pi / 4, math.pi / 3, math.pi / 2]),
        ]
        for euler in euler_angles:
            quat = euler2quat(euler)
            scipy_R = R.from_euler("xyz", euler.cpu().numpy())
            scipy_quat = scipy_R.as_quat().astype(NumpyTypes.FLOAT32)
            assert quaternions_close(quat, th.from_numpy(scipy_quat))

            recovered_euler = quat2euler(quat)
            scipy_recovered_euler = scipy_R.as_euler("xyz").astype(NumpyTypes.FLOAT32)
            assert_close(recovered_euler, th.from_numpy(scipy_recovered_euler))

    def test_euler2mat_and_mat2euler(self):
        euler_angles = [
            th.tensor([0.0, 0.0, 0.0]),
            th.tensor([math.pi / 4, math.pi / 3, math.pi / 2]),
        ]
        for euler in euler_angles:
            mat = euler2mat(euler)
            scipy_R = R.from_euler("xyz", euler.cpu().numpy())
            scipy_mat = scipy_R.as_matrix().astype(NumpyTypes.FLOAT32)
            assert_close(mat, th.from_numpy(scipy_mat))

            recovered_euler = mat2euler(mat)
            scipy_recovered_euler = scipy_R.as_euler("xyz").astype(NumpyTypes.FLOAT32)
            assert_close(recovered_euler, th.from_numpy(scipy_recovered_euler))


class TestQuaternionApplications:
    def test_quat_apply(self):
        for quat in RANDOM_QUATERNIONS:
            for vec in RANDOM_VECTORS:
                rotated_vec = quat_apply(quat, vec)

                scipy_R = R.from_quat(quat.cpu().numpy())
                scipy_rotated_vec = scipy_R.apply(vec.cpu().numpy()).astype(NumpyTypes.FLOAT32)

                assert rotated_vec.shape == (3,)
                assert_close(rotated_vec, th.from_numpy(scipy_rotated_vec), atol=1e-3, rtol=1e-3)
                assert_close(th.norm(rotated_vec), th.norm(vec), atol=1e-3, rtol=1e-3)

    def test_quat_slerp(self):
        for i in range(0, len(RANDOM_QUATERNIONS), 2):
            q1, q2 = RANDOM_QUATERNIONS[i], RANDOM_QUATERNIONS[i + 1]
            t = th.rand(1)
            q_slerp = quat_slerp(q1, q2, t)

            key_rots = R.from_quat(np.stack([q1.cpu().numpy(), q2.cpu().numpy()]))
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            scipy_q_slerp = slerp([t.item()]).as_quat()[0].astype(NumpyTypes.FLOAT32)

            assert quaternions_close(q_slerp, th.from_numpy(scipy_q_slerp))
            assert_close(th.norm(q_slerp), th.tensor(1.0))


class TestTransformPoints:
    def test_transform_points_2d(self):
        points = th.tensor([[1.0, 0.0], [0.0, 1.0]])
        for matrix in RANDOM_MATRICES:
            matrix_2d = th.eye(3)
            matrix_2d[:2, :2] = matrix[:2, :2]
            matrix_2d[:2, 2] = matrix[:2, 2]
            transformed = transform_points(points, matrix_2d)
            transformed_trimesh = th.tensor(trimesh.transform_points(points, matrix_2d), dtype=th.float32)
            assert_close(transformed, transformed_trimesh)

    def test_transform_points_3d(self):
        points = th.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for matrix in RANDOM_MATRICES:
            matrix_4d = th.eye(4)
            matrix_4d[:3, :3] = matrix
            matrix_4d[:3, 3] = th.tensor([1.0, 2.0, 3.0])
            transformed = transform_points(points, matrix_4d)
            transformed_trimesh = th.tensor(trimesh.transform_points(points, matrix_4d), dtype=th.float32)
            assert_close(transformed, transformed_trimesh)


class TestMiscellaneousFunctions:
    def test_convert_quat(self):
        for quat_wxyz in RANDOM_QUATERNIONS:
            quat_xyzw = convert_quat(quat_wxyz, to="xyzw")
            assert_close(quat_xyzw, th.tensor([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]))

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
