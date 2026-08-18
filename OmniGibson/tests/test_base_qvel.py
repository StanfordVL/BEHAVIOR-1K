from types import SimpleNamespace

import numpy as np
import torch as th

from omnigibson.eval.utils.eval_utils import holonomic_base_qvel_to_robot_frame
from omnigibson.robots.robot import Robot


def test_holonomic_base_qvel_to_robot_frame_supports_batched_3dof_and_6dof_inputs():
    yaws = np.array([0.0, np.pi / 2, -np.pi / 2, np.pi])
    world_qvel = np.array(
        [
            [1.0, 2.0, 0.1],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.3],
            [1.0, 2.0, 0.4],
        ]
    )
    expected = np.array(
        [
            [1.0, 2.0, 0.1],
            [0.0, -1.0, 0.2],
            [-1.0, 0.0, 0.3],
            [-1.0, -2.0, 0.4],
        ]
    )

    qpos_3dof = np.stack([np.zeros_like(yaws), np.zeros_like(yaws), yaws], axis=-1)
    np.testing.assert_allclose(holonomic_base_qvel_to_robot_frame(qpos_3dof, world_qvel), expected, atol=1e-7)

    qpos_6dof = np.zeros((len(yaws), 6))
    qpos_6dof[:, -1] = yaws
    qvel_6dof = np.zeros((len(yaws), 6))
    qvel_6dof[:, :2] = world_qvel[:, :2]
    qvel_6dof[:, -1] = world_qvel[:, -1]
    np.testing.assert_allclose(holonomic_base_qvel_to_robot_frame(qpos_6dof, qvel_6dof), expected, atol=1e-7)


def test_robot_holonomic_base_qvel_for_proprioception_uses_robot_frame():
    robot = SimpleNamespace(
        base_control_idx=th.tensor([0, 1, 5]),
        base_idx=th.arange(6),
        is_holonomic_base=True,
    )
    joint_positions = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    joint_velocities = th.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.25])

    actual = Robot._get_base_qvel_for_proprioception(robot, joint_positions, joint_velocities)

    th.testing.assert_close(actual, th.tensor([0.0, -1.0, 0.25]), atol=1e-6, rtol=0.0)


def test_robot_nonholonomic_base_qvel_for_proprioception_is_unchanged():
    robot = SimpleNamespace(
        base_control_idx=th.tensor([1, 3]),
        is_holonomic_base=False,
    )
    joint_positions = th.zeros(4)
    joint_velocities = th.tensor([10.0, 1.5, 20.0, -2.5])

    actual = Robot._get_base_qvel_for_proprioception(robot, joint_positions, joint_velocities)

    th.testing.assert_close(actual, th.tensor([1.5, -2.5]))
