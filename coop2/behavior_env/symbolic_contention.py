"""Resource contention for the symbolic primitive set.

The symbolic primitives are distance-blind and holder-blind. Every
``_navigate_if_needed`` call in
``SymbolicSemanticActionPrimitives`` is commented out, so:

* ``_grasp`` teleports the object to the end-effector **at any distance**, and
  its only precondition is ``self._get_obj_in_hand() is None`` -- which reads
  ``self.robot._ag_obj_in_hand[arm]``, i.e. *this* robot's own hand. There is
  no cross-robot check anywhere. ``_establish_grasp`` then creates its joint at
  ``{self.eef_links[arm].prim_path}/ag_constraint``, a path scoped to the
  *grasping* robot, so a second robot grasping the same object does not
  collide with the first one's joint: the object is yanked out of A's hand to
  B's end-effector, ends up carrying **two** FixedJoints, and both robots'
  ``_ag_obj_in_hand`` claim to hold it. Both post-condition checks pass. No
  exception, no log -- silent state corruption plus an over-constrained body.
* ``_place_with_predicate`` / ``_open_or_close`` / ``_toggle`` are the same
  story minus the joint: they set the state from across the room.

Consequence: symbolic mode has essentially no resource contention, which is
what COOP2's cooperation pressure is made of. This module puts it back with
three coupled rules:

1. **Interaction radius.** Acting on an object requires the robot base to be
   within ``interaction_radius`` metres of it, else ``TOO_FAR``.
2. **Travel time.** ``NAVIGATE_TO`` yields hold-position actions proportional
   to the distance actually travelled *before* teleporting, so being far away
   costs ticks.
3. **Claims.** Acting on an object another robot is holding fails with
   ``OBJECT_CLAIMED`` instead of corrupting the scene.

Together these make "who gets there first" depend on where the agents started,
which is exactly the allocation problem the centralized leader and the
broadcast chain are supposed to solve -- and going after a teammate's target is
a measurable net loss (travel ticks burned, then a failure).

Deliberately *not* a lock in the engine: the loser still burns a full navigate
and still issues its GRASP, so contention stays visible to the cognitive layer
as a wasted decision. A pre-assignment lock would hide the very signal the
topology layer is being measured on. See PORTING_PLAN.md 3.4 / 4.2 and the
"Deliberately not implemented" section of coop2/CLAUDE.md.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError

from coop2.behavior_env.symbolic_navigation import NavigableSymbolicActionPrimitives

__all__ = [
    "ContentiousSymbolicActionPrimitives",
    "DEFAULT_GATED_PRIMITIVES",
    "GATE_GRASP",
    "GATE_OPEN_CLOSE",
    "GATE_PLACE",
    "GATE_TOGGLE",
]

#: Gate labels, so callers can switch individual rules off without subclassing.
GATE_GRASP = "grasp"
GATE_PLACE = "place"
GATE_OPEN_CLOSE = "open_close"
GATE_TOGGLE = "toggle"

DEFAULT_GATED_PRIMITIVES = frozenset({GATE_GRASP, GATE_PLACE, GATE_OPEN_CLOSE, GATE_TOGGLE})

#: Headroom between the navigation sampler's upper bound and the interaction
#: radius. Without it a pose sampled exactly at the far edge of the annulus
#: would be borderline TOO_FAR, and the agent would loop
#: navigate -> TOO_FAR -> navigate on a float comparison.
DEFAULT_RADIUS_MARGIN = 0.35

#: Ticks of travel per metre. One env step is 1/``action_frequency`` seconds
#: (30 Hz by default), so 60 ticks/m is 2 s/m, i.e. a 0.5 m/s base -- roughly
#: what the physical primitives achieve. Tune it as an experiment variable:
#: this number sets how expensive distance is relative to a decision.
DEFAULT_TRAVEL_TICKS_PER_METER = 60.0

#: Ticks a ``wait`` holds for when the agent does not say. Long enough that a
#: teammate's NAVIGATE_TO (300-500 ticks here) makes real progress during it.
DEFAULT_WAIT_TICKS = 200

#: Cap on a single wait. An agent that yields the floor for the rest of the
#: episode cannot react to the thing it was waiting for.
MAX_WAIT_TICKS = 600


class ContentiousSymbolicActionPrimitives(NavigableSymbolicActionPrimitives):
    """Symbolic primitives with proximity, travel cost, and cross-robot claims.

    Drop-in replacement for :class:`NavigableSymbolicActionPrimitives`.

    Args:
        env, robot: as the base class.
        interaction_radius: fixed metres within which an object may be
            grasped / placed on / opened / toggled. ``None`` (the default and
            the sane choice) derives it **per object** from the navigation
            sampler's own range -- see :meth:`interaction_radius_for`.
        travel_ticks_per_meter: hold-position ticks emitted per metre of
            straight-line base travel before the teleport. 0 disables the
            travel cost.
        travel_ticks_range: ``(min, max)`` clamp on the emitted travel ticks.
            The max keeps a cross-scene navigate from eating the episode
            budget; the min gives every navigate a floor cost so that
            "navigate to where I already am" is not free.
        gated_primitives: which proximity gates to enforce. See
            :data:`DEFAULT_GATED_PRIMITIVES`.
        enforce_claims: reject actions on an object another robot holds.
        distance_range, sampling_attempts, require_same_room: passed to
            :class:`NavigableSymbolicActionPrimitives`.
    """

    def __init__(
        self,
        env,
        robot,
        interaction_radius: Optional[float] = None,
        travel_ticks_per_meter: float = DEFAULT_TRAVEL_TICKS_PER_METER,
        travel_ticks_range: Tuple[int, int] = (0, 3000),
        gated_primitives: Iterable[str] = DEFAULT_GATED_PRIMITIVES,
        enforce_claims: bool = True,
        **kwargs,
    ):
        super().__init__(env, robot, **kwargs)

        travel_min, travel_max = travel_ticks_range
        if travel_min < 0 or travel_max < travel_min:
            raise ValueError(f"travel_ticks_range must be a non-negative (min, max), got {travel_ticks_range!r}")

        self._interaction_radius = None if interaction_radius is None else float(interaction_radius)
        self.travel_ticks_per_meter = float(travel_ticks_per_meter)
        self.travel_ticks_range = (int(travel_min), int(travel_max))
        self.gated_primitives = frozenset(gated_primitives)
        self.enforce_claims = bool(enforce_claims)

    def interaction_radius_for(self, obj) -> float:
        """How close the base must be to act on @obj.

        Derived from the navigation sampler's range for this very object, so
        the two can never disagree: anything NAVIGATE_TO can produce is in
        range. A single global constant cannot hold for both an apple and a
        table -- the table's clearance alone exceeds what would be a sane
        radius for the apple, and an agent that navigated successfully would
        still be told TOO_FAR, forever.
        """
        if self._interaction_radius is not None:
            return self._interaction_radius
        return self.sampling_range_for(obj)[1] + DEFAULT_RADIUS_MARGIN

    # -- world queries ----------------------------------------------------

    def _peer_robots(self) -> Sequence[Any]:
        """Every robot in the scene, this one included.

        Prefers ``robot.scene.robots`` over ``env.robots`` so the class still
        works when driven without an Environment (tests, direct scene use).
        """
        scene = getattr(self.robot, "scene", None)
        robots = getattr(scene, "robots", None)
        if robots is None:
            robots = getattr(self.env, "robots", None)
        return list(robots or [])

    def holder_of(self, obj) -> Optional[Any]:
        """The robot currently holding ``obj``, or None.

        Asks every robot's public ``is_grasping(arm, candidate_obj)``. This is
        the cross-robot view ``_get_obj_in_hand`` does not give: that one is
        indexed by ``self.robot`` and ``self.arm`` only. Going through the
        public API also picks up the ``grasping_mode == "physical"`` branch that
        reading ``_ag_obj_in_hand`` directly would skip.
        """
        for robot in self._peer_robots():
            for arm in getattr(robot, "arm_names", []):
                try:
                    if robot.is_grasping(arm=arm, candidate_obj=obj):
                        return robot
                except Exception:  # noqa: BLE001 - non-manipulation robots
                    break
        return None

    def _base_xy(self) -> Tuple[float, float]:
        position = self.robot.get_position_orientation()[0]
        return float(position[0]), float(position[1])

    def _object_xy(self, obj) -> Tuple[float, float]:
        position = obj.get_position_orientation()[0]
        return float(position[0]), float(position[1])

    def distance_to(self, obj) -> float:
        """Planar base-to-object distance.

        Planar and base-relative on purpose: NAVIGATE_TO moves the *base* in
        the xy plane, so gating on anything else (end-effector, 3D distance)
        would make a successful navigate a non-guarantee of reachability.
        """
        base_x, base_y = self._base_xy()
        obj_x, obj_y = self._object_xy(obj)
        return math.hypot(obj_x - base_x, obj_y - base_y)

    # -- gates ------------------------------------------------------------

    def _error(self, reason_code: str, message: str, metadata: dict) -> ActionPrimitiveError:
        """A PRE_CONDITION error tagged with a COOP2 reason code.

        ``ReasonCode.from_primitive_error`` reads ``metadata['reason_code']``
        in preference to the ``ActionPrimitiveError.Reason`` enum, which only
        has five members and cannot express these two.
        """
        return ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            message,
            dict(metadata, reason_code=reason_code),
        )

    def _require_near(self, obj, verb: str, gate: str) -> None:
        if gate not in self.gated_primitives:
            return
        distance = self.distance_to(obj)
        radius = self.interaction_radius_for(obj)
        if distance <= radius:
            return
        raise self._error(
            "TOO_FAR",
            f"You are {distance:.2f} m from {obj.name}, too far to {verb} it "
            f"(you must be within {radius:.2f} m). Navigate to it first.",
            {"target object": obj.name, "distance": round(distance, 3), "radius": round(radius, 3)},
        )

    def _require_not_held_by_self(self, obj) -> None:
        """Refuse to grasp what is already in this robot's own gripper.

        Upstream re-grasps happily: the primitive settles for fifty ticks and
        reports success without changing anything. An agent that had achieved
        its goal therefore kept proposing ``holding(apple)`` and kept being
        told it had succeeded -- nineteen times in one episode -- because the
        environment never contradicted it. A no-op that scores as a success
        both wastes the agent's decisions and inflates Y_plan, so it fails, and
        the failure sends the agent back to reasoning where it can see that it
        is already holding the thing and pick something else to do.
        """
        if self.holder_of(obj) is self.robot:
            raise self._error(
                "ALREADY_HELD",
                f"You are already holding {obj.name}; grasping it again does nothing. "
                "If this was your goal, it is met -- choose a different task.",
                {"target object": obj.name},
            )

    def _require_unclaimed(self, obj, verb: str) -> None:
        if not self.enforce_claims:
            return
        holder = self.holder_of(obj)
        if holder is None or holder is self.robot:
            return
        # The holder's name is the agent id the cognitive layer knows it by,
        # and this prose goes straight into the prompt -- it is the main way a
        # decentralized agent finds out it needs to negotiate.
        raise self._error(
            "OBJECT_CLAIMED",
            f"{obj.name} is currently held by {holder.name}, so you cannot {verb} it. "
            "Pick a different target, or ask them to release it.",
            {"target object": obj.name, "held by": holder.name},
        )

    def _hold_action(self):
        """One "keep the current joint configuration" action.

        Identical to what ``_settle_robot`` yields, so the travel padding is
        indistinguishable from settling as far as the controller is concerned.
        """
        return self._postprocess_action(self.robot.q_to_action(self.robot.get_joint_positions()))

    def wait(self, ticks: Optional[int] = None):
        """Hold position for @ticks, then finish. A real primitive on purpose.

        ``wait`` used to be a "communication action": no primitive, completed
        the instant it was issued. That made it worse than useless. The plan
        loop freezes physics whenever any agent is not ready -- it returns
        without stepping the env -- so an instant wait put the agent straight
        back into reasoning, which stopped the world, which meant the teammate
        it was waiting for advanced by exactly nothing while the wait burned an
        LLM call. Yielding the floor is only meaningful if time passes, and time
        only passes while some agent holds an active primitive.

        Args:
            ticks: how long to hold. Defaults to DEFAULT_WAIT_TICKS, clamped to
                MAX_WAIT_TICKS.
        """
        held = DEFAULT_WAIT_TICKS if ticks is None else int(ticks)
        held = max(1, min(held, MAX_WAIT_TICKS))
        for _ in range(held):
            yield self._hold_action()

    def travel_ticks(self, distance: float) -> int:
        """Hold-position ticks charged for travelling ``distance`` metres."""
        ticks = int(round(max(0.0, distance) * self.travel_ticks_per_meter))
        low, high = self.travel_ticks_range
        return max(low, min(high, ticks))

    # -- overrides --------------------------------------------------------

    def _navigate_to_pose(self, pose_2d):
        """Charge travel time, then teleport.

        The padding must come **before** ``super()._navigate_to_pose``, which
        teleports on its first ``next()``: pad afterwards and the robot arrives
        instantly and then idles, so a teammate inspecting the world during
        those ticks sees it already at the destination and distance stops being
        a scarce resource. Padding first models "in transit".

        Measuring here rather than in the engine also means the distance is
        read before the teleport for free.
        """
        start_x, start_y = self._base_xy()
        distance = math.hypot(float(pose_2d[0]) - start_x, float(pose_2d[1]) - start_y)
        ticks = self.travel_ticks(distance)
        # One line per navigate -- there are tens per episode, not thousands.
        # Without it a 1435-tick NAVIGATE_TO to a target 3.7 m away is only
        # visible after the run, as a number with no way to attribute it
        # between travel padding and the settle loop.
        print(f"[nav] {getattr(self.robot, 'name', '?')} -> "
              f"({float(pose_2d[0]):.2f}, {float(pose_2d[1]):.2f}) "
              f"{distance:.1f} m, {ticks} travel ticks")
        for _ in range(ticks):
            yield self._hold_action()
        yield from super()._navigate_to_pose(pose_2d)

    # Each override below is a generator function, so the body -- and the gate
    # -- runs on the first next(), which is before the parent has touched the
    # scene. A rejected primitive therefore costs 0 ticks and moves nothing.

    def _grasp(self, obj):
        self._require_not_held_by_self(obj)
        self._require_unclaimed(obj, "grasp")
        self._require_near(obj, "grasp", GATE_GRASP)
        yield from super()._grasp(obj)

    def _place_with_predicate(self, obj, predicate, near_poses=None, near_poses_threshold=None):
        """Upstream's placement, with the object's velocity zeroed on arrival.

        obj is the *reference* here (the table), not the thing in hand.

        Reimplemented rather than delegated because of one line. Upstream does
        release -> set_position_orientation -> settle, and
        ``set_position_orientation`` moves a body without touching its
        velocity. The object has just spent the release's settle (50 + 200
        ticks) falling out of the gripper, so it arrives at the sampled pose
        carrying that fall, and the settle afterwards integrates it: one
        recorded episode had a task apple leave the house entirely, logged at
        12 m, then 27 m, then 36 m, then 38 m from the living room. Every
        subsequent navigate_to it then failed with NO_SPACE_AROUND_TARGET
        (200/200 candidates rejected by the same-room filter), which reads as a
        crowding problem and is really a missing object.

        ``keep_still()`` is upstream's own helper for exactly this.
        """
        self._require_unclaimed(obj, "place onto")
        self._require_near(obj, "place onto", GATE_PLACE)

        from omnigibson.action_primitives.action_primitive_set_base import (  # noqa: PLC0415
            ActionPrimitiveError as _Error,
        )

        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise _Error(
                _Error.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping an object first to place it somewhere.",
            )

        obj_pose = self._sample_pose_with_object_and_predicate(
            predicate, obj_in_hand, obj,
            near_poses=near_poses, near_poses_threshold=near_poses_threshold,
        )
        yield from self._release()

        obj_in_hand.set_position_orientation(*obj_pose)
        # The one line upstream is missing.
        obj_in_hand.keep_still()
        yield from self._settle_robot()

        if not obj_in_hand.states[predicate].get_value(obj):
            raise _Error(
                _Error.Reason.EXECUTION_ERROR,
                f"Failed to place {obj_in_hand.name} onto {obj.name}: it did not come to rest "
                "there. It has been released, so grasp it again before retrying.",
                {"dropped object": obj_in_hand.name, "target object": obj.name},
            )

    def _open_or_close(self, obj, should_open):
        verb = "open" if should_open else "close"
        self._require_unclaimed(obj, verb)
        self._require_near(obj, verb, GATE_OPEN_CLOSE)
        yield from super()._open_or_close(obj, should_open)

    def _toggle(self, obj, value):
        verb = "toggle on" if value else "toggle off"
        self._require_unclaimed(obj, verb)
        self._require_near(obj, verb, GATE_TOGGLE)
        yield from super()._toggle(obj, value)
