"""A abstract class for all walker tasks."""

from dm_control import composer, mjcf

from gello.dm_control_tasks.arms.manipulator import Manipulator
from gello.dm_control_tasks.manipulation.arenas.floors import FixedManipulationArena

# Timestep of the physics simulation.
_PHYSICS_TIMESTEP: float = 0.002

# Interval between agent actions, in seconds.
# We send a control signal every (_CONTROL_TIMESTEP / _PHYSICS_TIMESTEP) physics steps.
_CONTROL_TIMESTEP: float = 0.02  # 50 Hz.


class ManipulationTask(composer.Task):
    """Base composer task for walker robots."""

    def __init__(
        self,
        arm: Manipulator,
        arena: FixedManipulationArena,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
    ) -> None:
        self._arm = arm
        self._arena = arena

        self._arm_geoms = arm.root_body.find_all("geom")
        self._arena_geoms = arena.root_body.find_all("geom")
        arena.attach(arm, attach_site=arena.arm_attachment_site)

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

        # Enable all robot observables. Note: No additional entity observables should
        # be added in subclasses.
        for observable in self._arm.observables.proprioception:
            observable.enabled = True

    def in_collision(self, physics: mjcf.Physics) -> bool:
        """Checks if the arm is in collision with the floor (self._arena)."""
        arm_ids = [physics.bind(g).element_id for g in self._arm_geoms]
        arena_ids = [physics.bind(g).element_id for g in self._arena_geoms]

        return any(
            (con.geom1 in arm_ids and con.geom2 in arena_ids)  # arm and arena
            or (con.geom1 in arena_ids and con.geom2 in arm_ids)  # arena and arm
            or (con.geom1 in arm_ids and con.geom2 in arm_ids)  # self collision
            for con in physics.data.contact[: physics.data.ncon]
        )

    @property
    def root_entity(self):
        return self._arena
