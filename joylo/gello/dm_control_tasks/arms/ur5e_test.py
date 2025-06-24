"""Tests for ur5e.py."""

from absl.testing import absltest
from dm_control import mjcf

from gello.dm_control_tasks.arms import ur5e


class UR5eTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        robot = ur5e.UR5e()
        physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
        physics.step()

    def test_joints(self) -> None:
        robot = ur5e.UR5e()
        for joint in robot.joints:
            self.assertEqual(joint.tag, "joint")


if __name__ == "__main__":
    absltest.main()
