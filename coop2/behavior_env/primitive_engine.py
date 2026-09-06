"""L1c — Multi-agent primitive execution engine for BEHAVIOR-1K.

One :class:`StarterSemanticActionPrimitives` controller per robot. The engine
is a **stepper, not a driver**: the caller owns the main loop, exactly as
COOP2's ``PlanningEnvWrapper`` owns it in ma_crafter.

Isomorphism with COOP2
----------------------
COOP2's barrier lives at the **plan** boundary, not the action boundary
(``PlanningEnvWrapper.step``)::

    all_ready = all(self.agents[aid].ready for aid in self.agent_names)
    if managed_agents and not all_ready:
        return self._idle_step_return({"waiting_for_agents": waiting_for})
    for agent_id in self.agent_names:          # W -> X, shared timestamp
        ...
    for agent_id in self.agent_names:          # one action from each plan
        actions[agent_id] = self.plan_executors[agent_id].step(...)

An agent only becomes un-ready when its *plan* finishes or fails
(``set_unready('plan_terminated')``); while a plan is executing the agents are
not synchronised with each other at all. This engine reproduces that:

    while not done:
        if not all_ready:                          # plan-level barrier
            yield idle_step_return(); continue     # tick() NOT called: physics frozen
        for agent_id in executing:
            if not engine.has_active(agent_id):    # previous primitive just ended
                engine.assign(agent_id, *next_primitive_of(agent_id))
        outcomes = engine.tick()                   # exactly one env.step
        for agent_id, outcome in outcomes.items():
            ...advance / fail the plan, maybe set_unready

In ma_crafter one symbolic action is ~one ``env.step``. Here one primitive is
10^3-10^4 ``env.step`` calls, so the "one action per agent per step" rule
becomes "one low-level action per agent per tick, pulled from that agent's
active primitive generator". The layering is identical, only the timescale
differs. Python generators suspend and resume for free, so freezing physics
at the barrier while a primitive is mid-flight is safe.

Why this layer has to exist
---------------------------
1. ``apply_ref`` yields **one robot's** action tensor, while
   ``Environment._pre_step`` requires a key for **every** robot (a missing key
   is a ``KeyError``, not a no-op). Non-acting robots need an explicit idle
   action every tick.
2. cuRobo's collision world is a snapshot taken at plan time
   (``update_obstacles`` walks ``robot.scene.objects`` and skips only
   ``self.robot``), and ``_execute_motion_plan`` then runs for thousands of
   ticks without re-checking. With several robots moving at once every plan is
   stale from the first tick. See :class:`MotionMode`.

Not handled here, on purpose
----------------------------
**Target arbitration.** Two agents can be assigned the same object; nothing in
BEHAVIOR-1K prevents it and this engine does not either. The loser finds out
via a ``POST_CONDITION_ERROR`` ("An unexpected object was detected in hand")
after executing a full motion. That is deliberate: contention is a
*cooperation* problem, so it belongs to the topology layer (the leader's
allocation in centralized, the proposal chain in broadcast_chain), not to a
mechanical lock down here. The engine's job is to report the failure
faithfully.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    ActionPrimitiveErrorGroup,
)
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.robots import Robot

__all__ = [
    "MotionMode",
    "ReasonCode",
    "PrimitiveOutcome",
    "MultiAgentPrimitiveEngine",
]


# ---------------------------------------------------------------------------
# Configuration / result types
# ---------------------------------------------------------------------------


class MotionMode:
    """How much real concurrency the engine allows.

    ``CONCURRENT``
        Every active generator advances on every tick. Maximum overlap, and
        what the decentralized method needs -- but every cuRobo plan is
        computed against a stale snapshot of the teammates' poses. Pair with
        ``obstacle_refresh_every`` to at least refresh the world between the
        sub-motions of a single primitive.
    ``EXCLUSIVE``
        Only the earliest-assigned still-active agent advances; the others
        hold their joint configuration. Decisions stay concurrent (that
        happens in the cognitive layer), only *execution* is serialized, so
        the stale-obstacle problem disappears. This is the reliable control
        and a faithful match for the chain / centralized baselines.
    """

    CONCURRENT = "concurrent"
    EXCLUSIVE = "exclusive"

    ALL = (CONCURRENT, EXCLUSIVE)


class ReasonCode:
    """Structured failure codes handed to the cognitive layer.

    The first five mirror ``ActionPrimitiveError.Reason``; the rest are
    produced by this engine. COOP2's crafter implementation matched failure
    *strings* with ``startswith`` (``reason.startswith("requires ")``); use
    these codes instead and keep ``failure_reason`` as prose for the prompt.
    """

    OK = "OK"

    # Mapped 1:1 from ActionPrimitiveError.Reason
    PRE_CONDITION = "PRE_CONDITION"
    SAMPLING = "SAMPLING"
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    POST_CONDITION = "POST_CONDITION"

    # Produced by this engine
    INVALID_TARGET = "INVALID_TARGET"  # unknown name, or a Robot was passed
    TIMEOUT = "TIMEOUT"  # exceeded max_ticks_per_primitive
    ABORTED = "ABORTED"  # dropped by the caller (message interrupt, X -> I)
    CRASHED = "CRASHED"  # unexpected exception from the primitive

    # Raised by coop2.behavior_env.symbolic_contention, which tags them onto
    # metadata["reason_code"] because ActionPrimitiveError.Reason has only five
    # members and both of these would collapse into PRE_CONDITION. Both
    # terminate the plan and return the agent to reasoning -- see
    # TERMINATES_PLAN below.
    OBJECT_CLAIMED = "OBJECT_CLAIMED"  # target is held by another agent
    TOO_FAR = "TOO_FAR"  # outside the interaction radius; navigate first

    _FROM_PRIMITIVE_REASON = {
        "PRE_CONDITION_ERROR": PRE_CONDITION,
        "SAMPLING_ERROR": SAMPLING,
        "PLANNING_ERROR": PLANNING,
        "EXECUTION_ERROR": EXECUTION,
        "POST_CONDITION_ERROR": POST_CONDITION,
    }

    #: Codes that mean "this plan cannot proceed", not just "this action
    #: failed". Mirrors COOP2's ``terminate_plan`` flag, which ma_crafter sets
    #: for ``navigate_no_path`` and navigate timeouts. A target the planner
    #: cannot reach or sample a pose for will not become reachable by trying
    #: the next action of the same plan.
    #:
    #: OBJECT_CLAIMED and TOO_FAR are in here too, which is not obvious: both
    #: are individually recoverable (a teammate may release the object; walking
    #: closer fixes the distance). But the plan that produced them was written
    #: against a world that has since contradicted it, so continuing to its next
    #: action executes a stale intention. Sending the agent back to reasoning is
    #: also the *only* place cooperation can happen -- that is where it can
    #: negotiate for the contested object or choose a different target. Letting
    #: the plan grind on would turn contention into silent wasted motion instead
    #: of a decision the topology layer is measured on.
    TERMINATES_PLAN = frozenset(
        {PLANNING, SAMPLING, TIMEOUT, INVALID_TARGET, CRASHED, OBJECT_CLAIMED, TOO_FAR}
    )

    @classmethod
    def from_primitive_error(cls, error: BaseException) -> str:
        """Map an ActionPrimitiveError(Group) onto a reason code."""
        if isinstance(error, ActionPrimitiveErrorGroup):
            # apply_ref raises a group once every attempt has failed; the last
            # error is the most informative one.
            inner = list(getattr(error, "exceptions", ()) or ())
            if inner:
                return cls.from_primitive_error(inner[-1])
            return cls.EXECUTION
        # A coop2-raised precondition carries its own, finer code.
        override = (getattr(error, "metadata", None) or {}).get("reason_code")
        if isinstance(override, str):
            return override
        reason = getattr(error, "reason", None)
        return cls._FROM_PRIMITIVE_REASON.get(getattr(reason, "name", None), cls.EXECUTION)


@dataclass
class PrimitiveOutcome:
    """Result of one primitive execution.

    Field names follow COOP2's action-record vocabulary (``status``,
    ``failure_reason``, ``terminate_plan``, ``outcome``) so that
    ``PlanningEnvWrapper`` can consume :meth:`to_record` unmodified.
    ``reason_code`` is the one addition: a structured replacement for
    ma_crafter's failure-string matching.
    """

    agent_id: str
    primitive: str
    target: Optional[str]
    status: str  # "success" | "failed"
    reason_code: str = ReasonCode.OK
    failure_reason: str = ""
    terminate_plan: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    ticks: int = 0
    wall_seconds: float = 0.0
    started_env_step: int = 0
    ended_env_step: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "primitive": self.primitive,
            "target": self.target,
            "status": self.status,
            "reason_code": self.reason_code,
            "failure_reason": self.failure_reason,
            "terminate_plan": self.terminate_plan,
            "metadata": self.metadata,
            "ticks": self.ticks,
            "wall_seconds": round(self.wall_seconds, 3),
            "started_env_step": self.started_env_step,
            "ended_env_step": self.ended_env_step,
        }

    def to_record(self) -> Dict[str, Any]:
        """The shape ``PlanningEnvWrapper`` reads from ``get_action_records()[-1]``.

        ma_crafter reads ``status``, ``failure_reason``, ``primitive_action``,
        ``primitive_action_history`` and ``outcome`` off that record. Here the
        "primitive action" *is* the semantic primitive; the low-level action
        tensors are not worth logging, so ``primitive_action_history`` carries
        the single ``{env_step: primitive}`` entry that keeps the shape.
        """
        record = self.to_dict()
        record.update(
            {
                "primitive_action": self.primitive,
                "primitive_action_history": {self.started_env_step: self.primitive},
                "outcome": {
                    "status": self.status,
                    "reason_code": self.reason_code,
                    "reason": self.failure_reason,
                    **self.metadata,
                },
            }
        )
        return record

    def __str__(self) -> str:
        head = f"[{self.agent_id}] {self.primitive}({self.target}) -> {self.status}"
        timing = f"{self.ticks} ticks / {self.wall_seconds:.1f}s"
        if self.ok:
            return f"{head} in {timing}"
        tail = " [terminates plan]" if self.terminate_plan else ""
        return f"{head} ({self.reason_code}) in {timing}{tail} :: {self.failure_reason}"


@dataclass
class _ActiveRun:
    """One in-flight primitive generator."""

    agent_id: str
    primitive: str
    target: Optional[str]
    generator: Any
    started_env_step: int
    started_at: float
    ticks: int = 0
    is_cleanup: bool = False  # a retraction driven by abort(retract=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MultiAgentPrimitiveEngine:
    """Drives N per-robot primitive controllers over a single OmniGibson env.

    Construct this **after** the robots are at their intended reset pose:
    ``StarterSemanticActionPrimitives.__init__`` freezes ``_arm_targets`` and
    ``_reset_eef_pose`` from the joint state at construction time, and
    ``_reset_robot()`` will forever drive back to that pose.

    Args:
        env (og.Environment): the environment (already played).
        robots (list of Robot): defaults to ``env.robots``.
        agent_ids (list of str): COOP2-style ids, positionally zipped with
            ``robots``. Defaults to the robot names -- name them ``agent_0
            ... agent_{n-1}`` at config time and this is the identity map, so
            ``SymbolicEnvWrapper``'s name_map stays trivial.
        attempts (int): forwarded to ``apply_ref``. Keep at 1: the built-in
            5x retry is **not idempotent** (a failed place has already
            released the object, so attempt 2 fails its precondition) and it
            re-runs ``_reset_robot`` + ``_settle_robot`` each time, burning
            thousands of ticks. Failures should reach the LLM quickly.
        motion_mode (str): see :class:`MotionMode`.
        obstacle_refresh_every (int): if > 0, re-run ``update_obstacles()`` on
            the active controllers every N ticks. Best-effort only: it fixes
            the world used by the *next* plan inside a primitive, it cannot
            re-route a trajectory already in flight. 0 disables.
        max_ticks_per_primitive (int or None): hard safety cap per primitive;
            the run is abandoned with ``TIMEOUT``.
        enable_head_tracking (bool): MUST stay False for R1/R1Pro.
            ``_overwrite_head_action`` asserts ``robot.model == "tiago"`` and
            ``_grasp`` sets ``_tracking_object``, so True crashes on the first
            grasp.
        curobo_batch_size (int): CUDA graphs are captured at this size and it
            cannot be changed afterwards.
        progress_every (int): heartbeat every N ticks (0 = silent).
        verbose (bool): print each outcome as it lands.
        on_tick (callable): called with the new ``env_step`` after each tick.
    """

    def __init__(
        self,
        env,
        robots: Optional[Sequence[Robot]] = None,
        agent_ids: Optional[Sequence[str]] = None,
        attempts: int = 1,
        motion_mode: str = MotionMode.CONCURRENT,
        obstacle_refresh_every: int = 0,
        max_ticks_per_primitive: Optional[int] = 20000,
        enable_head_tracking: bool = False,
        curobo_batch_size: int = 1,
        progress_every: int = 500,
        verbose: bool = True,
        on_tick: Optional[Callable[[int], None]] = None,
        controllers: Optional[Dict[str, Any]] = None,
    ):
        if motion_mode not in MotionMode.ALL:
            raise ValueError(f"motion_mode must be one of {MotionMode.ALL}, got {motion_mode!r}")

        self.env = env
        self.robots: List[Robot] = list(robots if robots is not None else env.robots)
        if not self.robots:
            raise ValueError("No robots in the environment; check scene.include_robots and the robots config.")

        ids = list(agent_ids) if agent_ids is not None else [robot.name for robot in self.robots]
        if len(ids) != len(self.robots):
            raise ValueError(f"Got {len(ids)} agent ids for {len(self.robots)} robots.")
        self.agent_ids: List[str] = ids
        self.robots_by_id: Dict[str, Robot] = dict(zip(ids, self.robots))
        # Kept for callers that think in robot names (e.g. the demo script).
        self.robots_by_name: Dict[str, Robot] = {robot.name: robot for robot in self.robots}

        self.attempts = attempts
        self.motion_mode = motion_mode
        self.obstacle_refresh_every = obstacle_refresh_every
        self.max_ticks_per_primitive = max_ticks_per_primitive
        self.progress_every = progress_every
        self.verbose = verbose
        self.on_tick = on_tick

        if enable_head_tracking:
            models = {robot.model for robot in self.robots}
            if models - {"tiago"}:
                raise ValueError(
                    "enable_head_tracking=True is only valid for Tiago: _overwrite_head_action asserts "
                    "robot.model == 'tiago' and _grasp sets _tracking_object, so this crashes on the first "
                    f"grasp for {sorted(models)}."
                )

        # One controller per robot. The class holds no cross-instance state
        # (only self.env / self.robot), so this is safe -- but each one builds
        # its own CuRoboMotionGenerator (~3 MotionGen for R1, each warmed up
        # with CUDA-graph capture, plus a 2048-mesh collision cache). Startup
        # cost and VRAM scale linearly with N.
        # ``controllers`` lets a caller inject pre-built (or fake) controllers;
        # its only real use is the stubbed CPU test in feasibility_verify,
        # which exercises the assign/tick control flow with no Isaac at all.
        self.controllers: Dict[str, StarterSemanticActionPrimitives] = dict(controllers or {})
        for agent_id, robot in zip(self.agent_ids, self.robots):
            if agent_id in self.controllers:
                continue
            started = time.time()
            if self.verbose:
                print(f"[engine] building primitive controller for {agent_id} ({robot.name}, {robot.model}) ...")
            self.controllers[agent_id] = StarterSemanticActionPrimitives(
                env,
                robot,
                enable_head_tracking=enable_head_tracking,
                curobo_batch_size=curobo_batch_size,
            )
            if self.verbose:
                print(f"[engine] {agent_id} controller ready in {time.time() - started:.1f}s")

        self._active: Dict[str, _ActiveRun] = {}
        #: assignment order, so EXCLUSIVE mode is deterministic
        self._order: List[str] = []
        self._records: Dict[str, List[Dict[str, Any]]] = {agent_id: [] for agent_id in self.agent_ids}

        #: ticks, i.e. env.step calls. This is what Timeout(max_steps) counts.
        self.env_step: int = 0
        #: primitives issued. THIS is the denominator for COOP2's metrics --
        #: one primitive is ~10^3-10^4 ticks, so per-tick rates are meaningless.
        self.decision_count: int = 0
        #: the 5-tuple from the most recent env.step, for the L1 facade to
        #: build per-agent obs/info from.
        self.last_env_transition: Optional[Tuple[Any, Any, Any, Any, Any]] = None
        #: agent ids that contributed a real (non-idle) action to the most
        #: recent tick. ``len(last_advanced) >= 2`` is the definition of "these
        #: agents moved simultaneously", so this is what a concurrency
        #: measurement counts.
        self.last_advanced: List[str] = []

    # -- introspection -----------------------------------------------------

    @property
    def total_ticks(self) -> int:
        """Alias for :attr:`env_step`."""
        return self.env_step

    def has_active(self, agent_id: str) -> bool:
        """Whether this agent's previous primitive is still in flight."""
        return agent_id in self._active

    def active_agents(self) -> List[str]:
        """Agents with an in-flight primitive, in assignment order."""
        return [agent_id for agent_id in self._order if agent_id in self._active]

    def active_primitive(self, agent_id: str) -> Optional[str]:
        run = self._active.get(agent_id)
        return None if run is None else run.primitive

    def get_action_records(self, agent_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Per-agent primitive history, mirroring ``SymbolicEnvWrapper.get_action_records``."""
        if agent_id is not None:
            return {agent_id: list(self._records[agent_id])}
        return {aid: list(records) for aid, records in self._records.items()}

    def clear_action_history(self, agent_id: Optional[str] = None) -> None:
        for aid in [agent_id] if agent_id is not None else list(self._records):
            self._records[aid] = []

    def held_object(self, agent_id: str):
        """The object currently in that agent's hand, or None.

        Note this reads ``robot._ag_obj_in_hand[arm]``, so each controller only
        ever sees its **own** hand. There is no cross-agent "who holds what"
        view at this layer; build it from these values one layer up.
        """
        return self.controllers[agent_id]._get_obj_in_hand()

    def held_objects(self) -> Dict[str, Optional[str]]:
        """``{agent_id: held object name or None}`` across all agents."""
        result = {}
        for agent_id in self.agent_ids:
            held = self.held_object(agent_id)
            result[agent_id] = None if held is None else held.name
        return result

    # -- actions -----------------------------------------------------------

    def idle_action(self, robot: Robot) -> th.Tensor:
        """A "hold current joint configuration" action for one robot.

        This is what ``_settle_robot`` itself uses. Deliberately **not**
        ``controller._empty_action()``: that one actively servos the arm
        toward the ``_arm_targets`` frozen at controller construction, which
        would silently retract the arm of every waiting robot.
        """
        return robot.q_to_action(robot.get_joint_positions())

    def idle_action_dict(self) -> Dict[str, th.Tensor]:
        """A full action dict that keeps every robot where it is.

        Keyed by **robot name**, because that is what ``env.step`` wants.
        """
        return {robot.name: self.idle_action(robot) for robot in self.robots}

    def resolve_target(self, target: Any) -> Tuple[Optional[Any], Optional[str]]:
        """Resolve a target given as a name or an object handle.

        Returns:
            ``(object_or_None, error_message_or_None)``
        """
        if target is None:
            return None, None
        obj = target
        if isinstance(target, str):
            obj = self.env.scene.object_registry("name", target)
            if obj is None:
                return None, f"No object named {target!r} in the scene."
        # StarterSemanticActionPrimitives._grasp only checks
        # isinstance(obj, USDObject), and Robot satisfies that -- so guard
        # here. There is no in-sim handover primitive; agent-to-agent transfer
        # has to be modelled symbolically.
        if isinstance(obj, Robot):
            return None, f"{getattr(obj, 'name', obj)!r} is a robot; primitives cannot take a robot as a target."
        return obj, None

    def refresh_obstacles(self, agent_ids: Optional[Iterable[str]] = None) -> None:
        """Re-snapshot the cuRobo collision world for the given controllers.

        Only affects plans computed *after* the call; a trajectory already in
        flight is not re-routed.
        """
        for agent_id in agent_ids if agent_ids is not None else self.agent_ids:
            generator = getattr(self.controllers[agent_id], "_motion_generator", None)
            if generator is None:
                continue
            try:
                generator.update_obstacles()
            except Exception as error:  # noqa: BLE001 - best effort only
                if self.verbose:
                    print(f"[engine] update_obstacles failed for {agent_id}: {error}")

    # -- assign / tick / abort --------------------------------------------

    def assign(
        self,
        agent_id: str,
        primitive: StarterSemanticActionPrimitiveSet,
        target: Any = None,
    ) -> Optional[PrimitiveOutcome]:
        """Give one agent one primitive to start. Advances no simulation.

        Args:
            agent_id: must not already have an active run.
            primitive: e.g. ``StarterSemanticActionPrimitiveSet.GRASP``.
                Note ``OPEN``/``CLOSE``/``TOGGLE_ON``/``TOGGLE_OFF`` raise
                ``NotImplementedError`` in the physical primitive set; only
                GRASP / PLACE_ON_TOP / PLACE_INSIDE / NAVIGATE_TO / RELEASE
                are usable.
            target: object name, object handle, or None (``RELEASE``).

        Returns:
            ``None`` if the primitive was accepted and is now in flight, or a
            terminal :class:`PrimitiveOutcome` if it failed before anything
            moved (unknown target, robot passed as target).
        """
        if agent_id not in self.controllers:
            raise KeyError(f"Unknown agent {agent_id!r}; known: {sorted(self.controllers)}")
        if agent_id in self._active:
            raise RuntimeError(
                f"{agent_id} already has an active {self._active[agent_id].primitive}; "
                "call has_active() first, or abort() it."
            )

        obj, error = self.resolve_target(target)
        target_name = obj.name if obj is not None else (target if isinstance(target, str) else None)
        primitive_name = primitive.name

        if error is not None:
            return self._finish(
                agent_id,
                self._outcome(
                    agent_id,
                    primitive_name,
                    target_name,
                    status="failed",
                    reason_code=ReasonCode.INVALID_TARGET,
                    failure_reason=error,
                    ticks=0,
                    started_env_step=self.env_step,
                    started_at=time.time(),
                ),
            )

        generator = self.controllers[agent_id].apply_ref(
            primitive, *([] if obj is None else [obj]), attempts=self.attempts
        )
        self._active[agent_id] = _ActiveRun(
            agent_id=agent_id,
            primitive=primitive_name,
            target=target_name,
            generator=generator,
            started_env_step=self.env_step,
            started_at=time.time(),
        )
        if agent_id in self._order:
            self._order.remove(agent_id)
        self._order.append(agent_id)
        self.decision_count += 1
        if self.verbose:
            print(f"[engine] {agent_id} <- {primitive_name}({target_name})")
        return None

    def tick(self) -> Dict[str, PrimitiveOutcome]:
        """Advance exactly one ``env.step``.

        Every robot gets an action: active agents get the next value from
        their primitive generator, everyone else holds position.

        Returns:
            ``{agent_id: PrimitiveOutcome}`` for the primitives that
            **terminated on this tick** only. Runs still in flight do not
            appear, so the normal return value is ``{}``.
        """
        action = self.idle_action_dict()
        outcomes: Dict[str, PrimitiveOutcome] = {}
        advanced: List[str] = []

        for agent_id in self._advancing_agents():
            run = self._active[agent_id]
            if self.max_ticks_per_primitive is not None and run.ticks >= self.max_ticks_per_primitive:
                timed_out = self._outcome(
                    agent_id,
                    run.primitive,
                    run.target,
                    status="failed",
                    reason_code=ReasonCode.TIMEOUT,
                    failure_reason=f"Abandoned after {run.ticks} ticks (max_ticks_per_primitive).",
                    ticks=run.ticks,
                    started_env_step=run.started_env_step,
                    started_at=run.started_at,
                )
                self._finish(agent_id, timed_out, record=not run.is_cleanup)
                if not run.is_cleanup:
                    outcomes[agent_id] = timed_out
                continue
            try:
                # cuRobo planning happens INSIDE the generator body, so this
                # next() blocks for seconds at each sub-motion boundary. Only
                # wall-clock is serialized that way; the yielded actions still
                # land in the same env.step, so the robots move simultaneously
                # in simulation time.
                action[self.robots_by_id[agent_id].name] = next(run.generator)
                run.ticks += 1
                advanced.append(agent_id)
                continue
            except StopIteration:
                outcome = self._success(run)
            except (ActionPrimitiveError, ActionPrimitiveErrorGroup) as error:
                # apply_ref raises from INSIDE the generator, so this surfaces
                # here at next(), not at the assign() call.
                outcome = self._failure(run, error)
            except Exception as error:  # noqa: BLE001 - never let one agent kill the episode
                outcome = self._crash(run, error)
            # A cleanup run (abort(retract=True)) is bookkeeping, not a
            # decision: it is neither reported nor recorded.
            self._finish(agent_id, outcome, record=not run.is_cleanup)
            if not run.is_cleanup:
                outcomes[agent_id] = outcome

        obs, rewards, terminated, truncated, info = self.env.step(action)
        self.last_env_transition = (obs, rewards, terminated, truncated, info)
        self.last_advanced = advanced
        self.env_step += 1

        if self.on_tick is not None:
            self.on_tick(self.env_step)
        if self.obstacle_refresh_every and self.env_step % self.obstacle_refresh_every == 0:
            self.refresh_obstacles(self.active_agents())
        if self.progress_every and self.env_step % self.progress_every == 0 and self._active:
            running = ", ".join(f"{aid}:{run.primitive}@{run.ticks}" for aid, run in self._active.items())
            print(f"[engine] env_step {self.env_step} | running: {running}")
        if self.verbose:
            for outcome in outcomes.values():
                print(f"[engine] {outcome}")
        return outcomes

    def abort(self, agent_id: str, retract: bool = False) -> Optional[PrimitiveOutcome]:
        """Drop an in-flight primitive (COOP2's X -> I message interrupt).

        Discarding the generator is enough to stop issuing actions, but the
        robot stays wherever it was -- arm extended, possibly holding
        something. Set ``retract=True`` to queue the controller's own
        ``_reset_robot()`` as a cleanup run, which is driven by subsequent
        :meth:`tick` calls like any other primitive and costs real ticks; its
        termination is not reported as an outcome.

        Returns:
            The terminal outcome for the aborted primitive, or ``None`` if the
            agent had nothing running.
        """
        run = self._active.get(agent_id)
        if run is None:
            return None
        outcome = self._outcome(
            agent_id,
            run.primitive,
            run.target,
            status="failed",
            reason_code=ReasonCode.ABORTED,
            failure_reason="Primitive aborted by the cognitive layer (interrupted).",
            ticks=run.ticks,
            started_env_step=run.started_env_step,
            started_at=run.started_at,
        )
        outcome = self._finish(agent_id, outcome, record=not run.is_cleanup)
        if retract:
            self._active[agent_id] = _ActiveRun(
                agent_id=agent_id,
                primitive="RETRACT",
                target=None,
                generator=self.controllers[agent_id]._reset_robot(),
                started_env_step=self.env_step,
                started_at=time.time(),
                is_cleanup=True,
            )
            if agent_id in self._order:
                self._order.remove(agent_id)
            self._order.append(agent_id)
        return outcome

    # -- convenience (demos / unit tests) ---------------------------------

    def macro_step(
        self,
        assignments: Dict[str, Tuple[StarterSemanticActionPrimitiveSet, Any]],
    ) -> Dict[str, PrimitiveOutcome]:
        """Assign one primitive per agent and tick until all of them finish.

        A thin wrapper over :meth:`assign` / :meth:`tick` for scripted demos
        and tests. **Do not build the COOP2 runner on this**: it aligns the
        agents at every primitive boundary, whereas COOP2's barrier is at the
        *plan* boundary only (see the module docstring).
        """
        outcomes: Dict[str, PrimitiveOutcome] = {}
        pending = set()
        for agent_id, (primitive, target) in assignments.items():
            immediate = self.assign(agent_id, primitive, target)
            if immediate is None:
                pending.add(agent_id)
            else:
                outcomes[agent_id] = immediate
        while pending:
            for agent_id, outcome in self.tick().items():
                if agent_id in pending:
                    pending.discard(agent_id)
                    outcomes[agent_id] = outcome
        return outcomes

    def run_until_idle(self, max_ticks: Optional[int] = None) -> Dict[str, List[PrimitiveOutcome]]:
        """Tick until no primitive is in flight. Useful after :meth:`abort`."""
        collected: Dict[str, List[PrimitiveOutcome]] = {}
        ticks = 0
        while self._active:
            if max_ticks is not None and ticks >= max_ticks:
                break
            for agent_id, outcome in self.tick().items():
                collected.setdefault(agent_id, []).append(outcome)
            ticks += 1
        return collected

    # -- internals ---------------------------------------------------------

    def _advancing_agents(self) -> List[str]:
        """Which active agents get to advance on this tick."""
        active = self.active_agents()
        if self.motion_mode == MotionMode.EXCLUSIVE and active:
            return active[:1]
        return active

    def _finish(self, agent_id: str, outcome: PrimitiveOutcome, record: bool = True) -> PrimitiveOutcome:
        """Retire an active run and append it to the agent's history."""
        self._active.pop(agent_id, None)
        if agent_id in self._order:
            self._order.remove(agent_id)
        if record:
            self._records[agent_id].append(outcome.to_record())
        return outcome

    def _outcome(
        self,
        agent_id: str,
        primitive: str,
        target: Optional[str],
        status: str,
        reason_code: str,
        failure_reason: str,
        ticks: int,
        started_env_step: int,
        started_at: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PrimitiveOutcome:
        held = self.held_object(agent_id)
        payload = dict(metadata or {})
        payload.setdefault("held_object", None if held is None else held.name)
        return PrimitiveOutcome(
            agent_id=agent_id,
            primitive=primitive,
            target=target,
            status=status,
            reason_code=reason_code,
            failure_reason=failure_reason,
            terminate_plan=reason_code in ReasonCode.TERMINATES_PLAN,
            metadata=payload,
            ticks=ticks,
            wall_seconds=time.time() - started_at,
            started_env_step=started_env_step,
            ended_env_step=self.env_step,
        )

    def _success(self, run: _ActiveRun) -> PrimitiveOutcome:
        return self._outcome(
            run.agent_id,
            run.primitive,
            run.target,
            status="success",
            reason_code=ReasonCode.OK,
            failure_reason="",
            ticks=run.ticks,
            started_env_step=run.started_env_step,
            started_at=run.started_at,
        )

    def _failure(self, run: _ActiveRun, error: BaseException) -> PrimitiveOutcome:
        return self._outcome(
            run.agent_id,
            run.primitive,
            run.target,
            status="failed",
            reason_code=ReasonCode.from_primitive_error(error),
            # ActionPrimitiveError messages are already written as LLM-facing
            # natural language and several suggest a recovery; pass verbatim.
            failure_reason=str(error),
            ticks=run.ticks,
            started_env_step=run.started_env_step,
            started_at=run.started_at,
            metadata=dict(getattr(error, "metadata", {}) or {}),
        )

    def _crash(self, run: _ActiveRun, error: BaseException) -> PrimitiveOutcome:
        traceback.print_exc()
        return self._outcome(
            run.agent_id,
            run.primitive,
            run.target,
            status="failed",
            reason_code=ReasonCode.CRASHED,
            failure_reason=f"{type(error).__name__}: {error}",
            ticks=run.ticks,
            started_env_step=run.started_env_step,
            started_at=run.started_at,
        )
