"""A task where a walker must learn to stand."""

import numpy as np
from dm_control.suite.utils import randomizers
from dm_control.utils import rewards

from gello.dm_control_tasks.arms.manipulator import Manipulator
from gello.dm_control_tasks.manipulation.arenas.floors import FixedManipulationArena
from gello.dm_control_tasks.manipulation.tasks import base

_TARGET_COLOR = (0.8, 0.2, 0.2, 0.6)


class Reach(base.ManipulationTask):
    """Reach task for a manipulator."""

    def __init__(
        self,
        arm: Manipulator,
        arena: FixedManipulationArena,
        physics_timestep=base._PHYSICS_TIMESTEP,
        control_timestep=base._CONTROL_TIMESTEP,
        distance_tolerance: float = 0.5,
    ) -> None:
        super().__init__(arm, arena, physics_timestep, control_timestep)

        self._distance_tolerance = distance_tolerance

        # Create target.
        self._target = self.root_entity.mjcf_model.worldbody.add(
            "site",
            name="target",
            type="sphere",
            pos=(0, 0, 0),
            size=(0.1,),
            rgba=_TARGET_COLOR,
        )

    def initialize_episode(self, physics, random_state):
        # Randomly set feasible target position
        randomizers.randomize_limited_and_rotational_joints(physics, random_state)
        physics.forward()
        flange_position = physics.bind(self._arm.flange).xpos[:3]
        print(flange_position)

        # set target position to flange position
        physics.bind(self._target).pos = flange_position

        # Randomize initial position of the arm.
        randomizers.randomize_limited_and_rotational_joints(physics, random_state)
        physics.forward()

    def get_reward(self, physics):
        # flange position
        flange_pos = physics.bind(self._arm.flange).pos[:3]
        distance = np.linalg.norm(physics.bind(self._target).pos[:3] - flange_pos)
        return -rewards.tolerance(
            distance,
            bounds=(0, self._distance_tolerance),
            margin=self._distance_tolerance,
            value_at_margin=0,
            sigmoid="linear",
        )
