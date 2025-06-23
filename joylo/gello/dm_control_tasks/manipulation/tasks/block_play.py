"""A task where a walker must learn to stand."""

from typing import Optional

import numpy as np
from dm_control import mjcf
from dm_control.suite.utils.randomizers import random_limited_quaternion

from gello.dm_control_tasks.arms.manipulator import Manipulator
from gello.dm_control_tasks.manipulation.arenas.floors import FixedManipulationArena
from gello.dm_control_tasks.manipulation.tasks import base

_TARGET_COLOR = (0.8, 0.2, 0.2, 0.6)


class BlockPlay(base.ManipulationTask):
    """Task for a manipulator. Blocks are randomly placed in the scene."""

    def __init__(
        self,
        arm: Manipulator,
        arena: FixedManipulationArena,
        physics_timestep=base._PHYSICS_TIMESTEP,
        control_timestep=base._CONTROL_TIMESTEP,
        num_blocks: int = 10,
        size: float = 0.03,
        reset_joints: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(arm, arena, physics_timestep, control_timestep)

        # find key frame
        key_frames = self.root_entity.mjcf_model.find_all("key")
        if len(key_frames) == 0:
            key_frames = None
        else:
            key_frames = key_frames[0]

        # Create target.
        block_joints = []
        for i in range(num_blocks):
            # select random colors = np.random.uniform(0, 1, size=3)
            color = np.concatenate([np.random.uniform(0, 1, size=3), [1.0]])

            # attach a body for block i
            b = self.root_entity.mjcf_model.worldbody.add(
                "body", name=f"block_{i}", pos=(0, 0, 0)
            )

            # # add a freejoint to the block so it can be moved
            _joint = b.add("freejoint")
            block_joints.append(_joint)

            # add a geom to the block
            b.add(
                "geom",
                name=f"block_geom_{i}",
                type="box",
                size=(size, size, size),
                rgba=color,
                # contype=0,
                # conaffinity=0,
            )
            assert key_frames is not None
            key_frames.qpos = np.concatenate([key_frames.qpos, np.zeros(7)])

        # # save xml to file
        # _xml_string = self.root_entity.mjcf_model.to_xml_string()
        # with open("block_play.xml", "w") as f:
        #     f.write(_xml_string)

        self._block_joints = block_joints
        self._block_size = size
        self._reset_joints = reset_joints

    def initialize_episode(self, physics, random_state):
        # Randomly set feasible target position
        if self._reset_joints is not None:
            self._arm.set_joints(physics, self._reset_joints)
        else:
            self._arm.randomize_joints(physics, random_state)
        physics.forward()

        # check if arm is in collision with floor
        while self.in_collision(physics):
            self._arm.randomize_joints(physics, random_state)
            physics.forward()

        # Randomize block positions
        for block_j in self._block_joints:
            randomize_pose(
                block_j,
                physics,
                random_state=random_state,
                position_range=0.5,
                z_offset=self._block_size * 2,
            )

        physics.forward()

    def get_reward(self, physics):
        # flange position
        return 0


def randomize_pose(
    free_joint: mjcf.RootElement,
    physics: mjcf.Physics,
    random_state: np.random.RandomState,
    position_range: float = 0.5,
    z_offset: float = 0.0,
) -> None:
    """Randomize the pose of an entity."""
    entity_pos = random_state.uniform(-position_range, position_range, size=2)
    # make x, y farther than 0.1 from 0, 0
    while np.linalg.norm(entity_pos) < 0.3:
        entity_pos = random_state.uniform(-position_range, position_range, size=2)

    entity_pos = np.concatenate([entity_pos, [z_offset]])
    entity_quat = random_limited_quaternion(random_state, limit=np.pi)

    qpos = np.concatenate([entity_pos, entity_quat])

    physics.bind(free_joint).qpos = qpos
