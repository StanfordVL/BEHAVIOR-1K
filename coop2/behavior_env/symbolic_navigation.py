"""A cuRobo-free ``NAVIGATE_TO`` for the symbolic primitive set.

``SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO`` is in the enum and mapped to
a handler, but it cannot execute as shipped. Two independent defects on the
same path:

1. ``SymbolicSemanticActionPrimitives`` overrides only ``_navigate_to_pose``
   (making it a pure teleport). ``_navigate_to_obj`` and
   ``_sample_pose_near_object`` are inherited from
   ``StarterSemanticActionPrimitives``, and the sampler opens with
   ``self._motion_generator.update_obstacles()`` then validates candidates with
   cuRobo IK + collision checks. The symbolic constructor passes
   ``skip_curobo_initilization=True``, so ``self._motion_generator is None``
   and the call raises ``AttributeError: 'NoneType' object has no attribute
   'update_obstacles'``.
2. Even with a motion generator present, ``_navigate_to_obj`` calls
   ``self._navigate_to_pose(pose, skip_obstacle_update=...)`` while the
   symbolic override is declared ``_navigate_to_pose(self, pose_2d)`` -- no
   such keyword -- so it would raise ``TypeError``.

So NAVIGATE_TO is unusable in the symbolic set exactly the way OPEN / CLOSE /
TOGGLE_ON / TOGGLE_OFF are unusable in the physical set (those raise
``NotImplementedError``).

This module supplies the missing piece: sample a base pose around the target,
keep it in the same room, and teleport there. No cuRobo, no collision check,
no reachability check -- which is consistent with the rest of the symbolic set,
where ``_grasp`` teleports the object to the end-effector at any distance and
``_navigate_to_pose`` teleports the base with no collision checking either.

Why bother navigating at all when symbolic grasping ignores distance? Because
COOP2's **spatial** constraint is defined on where the agents are: "how many
agents are near the task". Without a working NAVIGATE_TO there is no way for a
symbolic-mode agent to satisfy it, and the spatial metric degenerates.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch as th

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
)

__all__ = ["NavigableSymbolicActionPrimitives"]


class NavigableSymbolicActionPrimitives(SymbolicSemanticActionPrimitives):
    """Symbolic primitives whose ``NAVIGATE_TO`` actually runs.

    Drop-in replacement for :class:`SymbolicSemanticActionPrimitives`; only the
    two broken navigation methods are overridden.

    Upstream samples the same 0-1.5 m annulus we do, but then runs every
    candidate through ``_validate_poses`` -- cuRobo IK plus a collision query --
    and returns only one that passes. That stage is what keeps the robot out of
    walls, not the distance range. Dropping cuRobo therefore means replacing it,
    and measurement says the replacement has to be *three* checks, because each
    covers a different obstacle class:

    * ``reach``/``clearance`` -- the target itself. cuRobo does not catch this:
      an apple is 5 cm on the floor and the base clears it, so ``check_collisions``
      called every pose from 0.05 m out collision-free (8/8). Symbolic grasp then
      teleports the object into a robot standing on top of it and the settle
      shakes it loose.
    * ``require_traversable`` -- static geometry. The scene's baked trav map,
      eroded by the robot's own radius, is what cuRobo's collision query buys us,
      for free and on the CPU.
    * ``robot_separation`` -- teammates. Static maps do not contain them. Without
      this, two robots teleport into the same spot, PhysX shoves them apart, and
      the physical assisted grasp latches onto the other *robot*.

    Args:
        env, robot: as the base class.
        reach: metres of usable annulus *beyond* the target's clearance radius.
            The sampling range is ``[clearance(obj), clearance(obj) + reach]``,
            so it widens automatically for a table and stays tight for an apple.
            A fixed range cannot do this: for a large object a 1.5 m upper bound
            is inside the object.
        clearance_margin: slack added to (target half-diagonal + robot radius).
        robot_radius: circumscribed radius. ``None`` derives it from
            ``reset_joint_pos_aabb_extent``, which is what the trav map's own
            erosion uses.
        robot_separation: minimum centre-to-centre distance to another robot.
            ``None`` means ``2 * robot_radius`` -- just-touching. Nothing more is
            needed: symbolic navigation teleports, so there is no path to keep
            clear, only bodies to keep from overlapping.
        require_traversable: reject poses where the eroded trav map says the
            robot does not fit. Silently inert on a scene with no trav map.
        sampling_attempts: how many candidates to try before giving up.
        require_same_room: reject candidates outside the target's room. Set
            False for scenes without a segmentation map (a plain ``Scene``
            rather than an ``InteractiveTraversableScene``).
        distance_range: absolute ``(lo, hi)`` override. When given it replaces
            the object-relative range entirely -- kept for the stubbed tests and
            for reproducing the old behaviour.
    """

    def __init__(
        self,
        env,
        robot,
        reach: float = 1.5,
        clearance_margin: float = 0.05,
        robot_radius: Optional[float] = None,
        robot_separation: Optional[float] = None,
        require_traversable: bool = True,
        sampling_attempts: int = 200,
        require_same_room: bool = True,
        distance_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(env, robot, **kwargs)
        self._nav_reach = float(reach)
        self._nav_clearance_margin = float(clearance_margin)
        self._nav_robot_radius = robot_radius
        self._nav_robot_separation = robot_separation
        self._nav_require_traversable = bool(require_traversable)
        self._nav_sampling_attempts = sampling_attempts
        self._nav_require_same_room = require_same_room
        self._nav_distance_range = distance_range
        self._nav_eroded_map = None

    # -- geometry ---------------------------------------------------------

    @property
    def robot_radius(self) -> float:
        """Circumscribed radius of the robot at its reset pose.

        ``reset_joint_pos_aabb_extent`` is the whole-robot AABB (arms included),
        so half its planar diagonal is the radius of a circle that contains the
        robot at *any* yaw. Conservative for a rectangular chassis; an oriented
        box test could pack tighter at the cost of real code.
        """
        if self._nav_robot_radius is None:
            try:
                extent = self.robot.reset_joint_pos_aabb_extent[:2]
                self._nav_robot_radius = float(th.norm(extent)) / 2.0
            except Exception:  # noqa: BLE001 - robots without the property
                self._nav_robot_radius = 0.6
        return self._nav_robot_radius

    @property
    def robot_separation(self) -> float:
        if self._nav_robot_separation is None:
            self._nav_robot_separation = 2.0 * self.robot_radius
        return self._nav_robot_separation

    def clearance_for(self, obj) -> float:
        """Smallest centre distance at which the robot does not overlap @obj."""
        try:
            extent = obj.aabb_extent[:2]
            half_diagonal = float(th.norm(extent)) / 2.0
        except Exception:  # noqa: BLE001 - objects without an aabb
            half_diagonal = 0.0
        return half_diagonal + self.robot_radius + self._nav_clearance_margin

    def sampling_range_for(self, obj) -> Tuple[float, float]:
        """``(lo, hi)`` metres to sample the base position in, for @obj."""
        if self._nav_distance_range is not None:
            return tuple(self._nav_distance_range)
        clearance = self.clearance_for(obj)
        return clearance, clearance + self._nav_reach

    # -- traversability ---------------------------------------------------

    def _trav_map(self):
        scene = self.robot.scene
        return getattr(scene, "trav_map", None)

    def _eroded_trav_map(self):
        """The floor map eroded by the robot radius, built once and cached."""
        if self._nav_eroded_map is None:
            trav_map = self._trav_map()
            floor_map = getattr(trav_map, "floor_map", None)
            if not floor_map:
                return None
            self._nav_eroded_map = trav_map._erode_trav_map(th.clone(floor_map[0]), robot=self.robot)
        return self._nav_eroded_map

    def _is_traversable(self, xy) -> bool:
        """Does the robot fit at @xy? True when the scene has no trav map."""
        if not self._nav_require_traversable:
            return True
        eroded = self._eroded_trav_map()
        if eroded is None:
            return True
        row, col = self._trav_map().world_to_map(th.tensor([float(xy[0]), float(xy[1])]))
        row, col = int(row), int(col)
        if not (0 <= row < eroded.shape[0] and 0 <= col < eroded.shape[1]):
            return False
        return bool(eroded[row][col] == 255)

    def _clear_of_other_robots(self, xy) -> bool:
        """Is @xy at least ``robot_separation`` from every other robot?"""
        separation = self.robot_separation
        scene = getattr(self.robot, "scene", None)
        robots = getattr(scene, "robots", None)
        if robots is None:
            robots = getattr(self.env, "robots", None) or []
        for other in robots:
            if other is self.robot:
                continue
            other_xy = other.get_position_orientation()[0][:2]
            if math.hypot(float(xy[0]) - float(other_xy[0]), float(xy[1]) - float(other_xy[1])) < separation:
                return False
        return True

    # -- helpers ----------------------------------------------------------

    def _seg_map(self):
        """The scene's segmentation map, or None on a non-traversable scene."""
        scene = self.robot.scene
        return getattr(scene, "seg_map", None) or getattr(scene, "_seg_map", None)

    def _room_of(self, xy) -> Optional[str]:
        seg_map = self._seg_map()
        if seg_map is None:
            return None
        # Note get_room_type_by_point is broken upstream (it indexes a dict
        # with a 0-dim tensor); the instance variant calls .item() and works.
        return seg_map.get_room_instance_by_point(xy)

    def _target_rooms(self, obj, target_xy) -> Sequence[Optional[str]]:
        """Rooms the target counts as being in.

        ``in_rooms`` is static scene metadata assigned at load time and is
        never updated when an object moves, and objects added through the env
        config have none at all -- so fall back to a live point query, exactly
        as the upstream sampler does.
        """
        in_rooms = getattr(obj, "in_rooms", None)
        if in_rooms:
            return list(in_rooms)
        return [self._room_of(target_xy)]

    def _facing_yaw_offset(self) -> float:
        """Yaw correction that leaves the target in front of the arm.

        Verbatim from the upstream sampler: ``yaw + pi - mean(workspace)``.
        """
        try:
            return math.pi - float(th.mean(self.robot.arm_workspace_range[self.arm]))
        except Exception:  # noqa: BLE001 - robots without a workspace range
            return math.pi

    # -- overrides --------------------------------------------------------

    def _sample_pose_near_object(
        self,
        obj,
        eef_pose=None,
        plan_with_open_gripper=False,
        sampling_attempts=None,
        skip_obstacle_update=False,
    ):
        """cuRobo-free replacement for the inherited sampler.

        Keeps the upstream geometry (polar sampling around the target, robot
        yawed to face it, same-room rejection) and drops the two cuRobo stages
        (``update_obstacles`` and ``_validate_poses``).

        Returns:
            th.Tensor or None: ``(x, y, yaw)``, or None if no candidate landed
            in the target's room.
        """
        if eef_pose is not None:
            target_position = eef_pose[0]
        else:
            target_position = obj.get_position_orientation()[0]
        target_xy = target_position[:2]

        target_rooms = self._target_rooms(obj, target_xy)
        yaw_offset = self._facing_yaw_offset()
        distance_lo, distance_hi = self.sampling_range_for(obj)
        attempts = self._nav_sampling_attempts if sampling_attempts is None else sampling_attempts

        for _ in range(attempts):
            distance = th.rand(1).item() * (distance_hi - distance_lo) + distance_lo
            yaw = th.rand(1).item() * 2.0 * math.pi - math.pi
            candidate = th.tensor(
                [
                    float(target_xy[0]) + distance * math.cos(yaw),
                    float(target_xy[1]) + distance * math.sin(yaw),
                    yaw + yaw_offset,
                ],
                dtype=th.float32,
            )
            # Room first: it is the cheapest of the three and, on a scene
            # where the target has no room, it is a no-op rather than a
            # rejection (target_rooms is then [None] and _room_of returns
            # None for anything off the map).
            if self._nav_require_same_room and self._seg_map() is not None:
                if self._room_of(candidate[:2]) not in target_rooms:
                    continue
            if not self._is_traversable(candidate[:2]):
                continue
            if not self._clear_of_other_robots(candidate[:2]):
                continue
            return candidate

        # No candidate satisfied all of the filters. Returning None makes
        # _navigate_to_obj raise PLANNING_ERROR -- the honest signal that this
        # target has no standable pose around it. Measured on Rs_int that is
        # 46% of targets, which is a property of the scene, not of this code:
        # see feasibility_verify/measure_teleport_risk.py. Previously the
        # sampler returned the first candidate regardless, so the robot
        # teleported into a wall and toppled instead of reporting failure.
        return None

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False):
        """Same as upstream, minus the keyword the symbolic override rejects.

        The inherited version forwards ``skip_obstacle_update`` into
        ``_navigate_to_pose``, but the symbolic override's signature is
        ``_navigate_to_pose(self, pose_2d)``, so that call is a ``TypeError``.
        """
        pose = self._sample_pose_near_object(obj, eef_pose=eef_pose)
        if pose is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find a valid base pose near the object in the same room",
                {"object": obj.name},
            )
        yield from self._navigate_to_pose(pose)
