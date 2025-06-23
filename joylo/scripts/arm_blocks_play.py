from dataclasses import dataclass

import numpy as np
import tyro
from dm_control import composer, viewer

from gello.agents.gello_agent import DynamixelRobotConfig
from gello.dm_control_tasks.arms.ur5e import UR5e
from gello.dm_control_tasks.manipulation.arenas.floors import Floor
from gello.dm_control_tasks.manipulation.tasks.block_play import BlockPlay


@dataclass
class Args:
    use_gello: bool = False


config = DynamixelRobotConfig(
    joint_ids=(1, 2, 3, 4, 5, 6),
    joint_offsets=(
        -np.pi / 2,
        1 * np.pi / 2 + np.pi,
        np.pi / 2 + 0 * np.pi,
        0 * np.pi + np.pi / 2,
        np.pi - 2 * np.pi / 2,
        -1 * np.pi / 2 + 2 * np.pi,
    ),
    joint_signs=(1, 1, -1, 1, 1, 1),
    gripper_config=(7, 20, -22),
)


def main(args: Args) -> None:
    reset_joints_left = np.deg2rad([90, -90, -90, -90, 90, 0, 0])
    robot = UR5e()
    task = BlockPlay(robot, Floor(), reset_joints=reset_joints_left[:-1])
    # task = BlockPlay(robot, Floor())
    env = composer.Environment(task=task)

    action_space = env.action_spec()
    if args.use_gello:
        gello = config.make_robot(
            port="/dev/cu.usbserial-FT7WBEIA", start_joints=reset_joints_left
        )

    def policy(timestep) -> np.ndarray:
        if args.use_gello:
            joint_command = gello.get_joint_state()
            joint_command = np.array(joint_command).copy()

            joint_command[-1] = joint_command[-1] * 255
            return joint_command
        return np.random.uniform(action_space.minimum, action_space.maximum)

    viewer.launch(env, policy=policy)


if __name__ == "__main__":
    main(tyro.cli(Args))
