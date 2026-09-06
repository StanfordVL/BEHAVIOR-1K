"""L2 for BEHAVIOR-1K: symbolic action -> engine primitive.

Replaces ma_crafter's crafter controllers. The architecture is kept --
``ActionRecord``, ``SymbolicActionStatus`` and the
pending/success/failed/terminate_plan contract of
``check_termination_condition`` are imported from :mod:`.action` unchanged,
because L3 is built on them and the whole point of the port is to leave the
upper layers alone.

What changes is the shape of a "primitive". Crafter's executor returns one
action **string per environment step** (``"move_left"``, ``"do"``): symbolic
actions are one or a few ticks. Here a primitive is a generator that runs for
10^2-10^3 ticks inside :class:`MultiAgentPrimitiveEngine`, and the caller owns
the loop. So ``execute()`` does not return a per-tick action -- it *assigns*
the primitive and returns its name, and ``check_termination_condition``
consumes the outcome the engine reports when the primitive finally terminates.

The other job of this layer is **grounding**: the LLM speaks in the type-local
ids L1b hands it (``apple#1``), while the engine addresses objects by their
scene name (``apple_agveuv_0``). Translating here, rather than letting either
side see the other's names, is what keeps prompts readable and the engine
unaware that an LLM exists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from coop2.behavior_env.primitive_engine import ReasonCode
from coop2.cognitive.action.action import ActionRecord, SymbolicActionStatus

__all__ = [
    "BEHAVIOR_ACTION_SCHEMA",
    "BEHAVIOR_ACTION_TO_PRIMITIVE",
    "BehaviorActionExecutor",
    "COMMUNICATION_ACTIONS",
]

#: symbolic action -> ``SymbolicSemanticActionPrimitiveSet`` member name.
#: All nine exist in the symbolic set. The physical set implements only the
#: first five and raises NotImplementedError for OPEN/CLOSE/TOGGLE_*, which is
#: why dependency constraints ("open the fridge first") are expressible at all
#: in symbolic mode and were not in physical mode.
BEHAVIOR_ACTION_TO_PRIMITIVE = {
    "navigate_to": "NAVIGATE_TO",
    "grasp": "GRASP",
    "place_on_top": "PLACE_ON_TOP",
    "place_inside": "PLACE_INSIDE",
    "release": "RELEASE",
    "open": "OPEN",
    "close": "CLOSE",
    "toggle_on": "TOGGLE_ON",
    "toggle_off": "TOGGLE_OFF",
}

#: Actions with no physical effect. They must still be first-class: COOP2's
#: plans coordinate through them, and an agent with nothing useful to do has to
#: be able to say so rather than being forced into a pointless primitive.
COMMUNICATION_ACTIONS = ("wait", "share")

#: The LLM-facing vocabulary. Mirrors ``cognitive/constants.py:ACTION_SCHEMA``
#: in shape so the prompt builder needs no special-casing, but every target is a
#: single ``entity_id`` -- crafter's (object_type, object_id) pair does not apply
#: when every object already has a unique type-local id from L1b.
BEHAVIOR_ACTION_SCHEMA = {
    "navigate_to": [{"type": "entity_id", "field": "target"}],
    "grasp": [{"type": "entity_id", "field": "target"}],
    "place_on_top": [{"type": "entity_id", "field": "target"}],
    "place_inside": [{"type": "entity_id", "field": "target"}],
    "release": [],
    "open": [{"type": "entity_id", "field": "target"}],
    "close": [{"type": "entity_id", "field": "target"}],
    "toggle_on": [{"type": "entity_id", "field": "target"}],
    "toggle_off": [{"type": "entity_id", "field": "target"}],
    "wait": [],
    "share": [
        {"type": "agent", "field": "target_agent", "allow_self": False},
        {"type": "text", "field": "message"},
    ],
}


class BehaviorActionExecutor:
    """Grounds one agent's symbolic actions into engine primitives.

    Args:
        agent_id: the robot name, which is also the engine's key for it.
        engine: :class:`MultiAgentPrimitiveEngine`. The executor only ever
            *assigns*; it never ticks. Ticking is the plan loop's job (L3), so
            that the barrier can sit at the plan boundary rather than at every
            primitive -- see coop2/CLAUDE.md.
        world_state: :class:`BehaviorWorldState`, used to translate entity ids.
    """

    def __init__(self, agent_id: str, engine=None, world_state=None):
        self.agent_id = agent_id
        self.engine = engine
        self.world_state = world_state

        self.current_symbolic_action: Optional[ActionRecord] = None
        self.current_env_step: int = 0
        self.action_history: List[ActionRecord] = []
        self._last_outcome: Optional[Dict[str, Any]] = None

    # -- grounding ---------------------------------------------------------

    def resolve_target(self, target: Optional[str]) -> Optional[str]:
        """``'apple#1'`` -> ``'apple_agveuv_0'``.

        Passes through anything already a scene name, so a caller that knows
        the real name (tests, scripted demos) does not have to invent an id.
        """
        if target is None or self.world_state is None:
            return target
        for name, entity_id in self.world_state._ids.items():
            if entity_id == target:
                return name
        return target

    def _primitive_enum(self, primitive_name: str):
        from omnigibson.action_primitives.symbolic_semantic_action_primitives import (  # noqa: PLC0415
            SymbolicSemanticActionPrimitiveSet,
        )

        return getattr(SymbolicSemanticActionPrimitiveSet, primitive_name)

    # -- execution ---------------------------------------------------------

    def execute(self, action_type: str, current_step: Optional[int] = None, **kwargs) -> str:
        """Start one symbolic action. Returns the primitive name issued.

        Unlike crafter's executor this advances **nothing**: the primitive is
        handed to the engine and the plan loop ticks it. Returning early here is
        what lets several agents have primitives in flight simultaneously.
        """
        if current_step is not None:
            self.current_env_step = current_step

        record = ActionRecord(action_type, dict(kwargs), self.current_env_step)
        self.current_symbolic_action = record
        self.action_history.append(record)
        self._last_outcome = None

        if action_type in COMMUNICATION_ACTIONS:
            # No primitive, no ticks. Completed immediately so the plan advances
            # on the next check rather than stalling on an agent that is talking.
            record.primitive_action = action_type
            self._last_outcome = {
                "status": "success",
                "reason_code": ReasonCode.OK,
                "reason": "",
                "metadata": dict(kwargs),
                "effects": {},
            }
            return action_type

        primitive_name = BEHAVIOR_ACTION_TO_PRIMITIVE.get(action_type)
        if primitive_name is None:
            record.primitive_action = "invalid"
            self._last_outcome = self._failure(
                ReasonCode.INVALID_TARGET,
                f"Unknown action {action_type!r}. Valid actions: "
                f"{sorted(list(BEHAVIOR_ACTION_TO_PRIMITIVE) + list(COMMUNICATION_ACTIONS))}.",
            )
            return "invalid"

        record.primitive_action = primitive_name
        if self.engine is None:
            return primitive_name

        target = self.resolve_target(kwargs.get("target"))
        immediate = self.engine.assign(self.agent_id, self._primitive_enum(primitive_name), target)
        if immediate is not None:
            # assign() rejects unknown targets before anything moves, and
            # reports them as a terminal outcome rather than raising.
            self._last_outcome = immediate.to_dict()
        return primitive_name

    def _failure(self, reason_code: str, reason: str) -> Dict[str, Any]:
        return {
            "status": "failed",
            "reason_code": reason_code,
            "reason": reason,
            "metadata": {},
            "effects": {},
        }

    # -- termination -------------------------------------------------------

    def submit_outcome(self, outcome: Dict[str, Any]) -> None:
        """Hand the engine's terminal outcome for this agent to the executor."""
        self._last_outcome = outcome

    def check_termination_condition(
        self,
        world_state: Optional[Dict] = None,
        action_outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """pending / success / failed for the action in flight.

        Same contract as the crafter executor, so L3 needs no changes. The
        source of truth is different: there, termination was inferred from the
        world; here the engine reports it, because only the engine knows when a
        generator raised or ran out.
        """
        record = self.current_symbolic_action
        if record is None or record.status != SymbolicActionStatus.PENDING:
            return {"status": "pending"}

        outcome = action_outcome if action_outcome is not None else self._last_outcome
        if outcome is None:
            # Still in flight. The engine returns outcomes only on the tick a
            # primitive terminates, so "no outcome" is the normal case.
            if self.engine is not None and self.engine.has_active(self.agent_id):
                return {"status": "pending"}
            if record.action_type in COMMUNICATION_ACTIONS:
                return self._complete(SymbolicActionStatus.SUCCESS, None, None)
            return {"status": "pending"}

        if outcome.get("status") == "failed":
            reason = outcome.get("reason") or outcome.get("failure_reason") or "Action failed"
            result = self._complete(SymbolicActionStatus.FAILED, reason, outcome)
            # terminate_plan mirrors COOP2's flag. Contention codes terminate
            # too: the plan was written against a world that has since
            # contradicted it, and reasoning is the only stage where the agent
            # can negotiate for the contested object or retarget.
            result["terminate_plan"] = bool(
                outcome.get("terminate_plan")
                or outcome.get("reason_code") in ReasonCode.TERMINATES_PLAN
            )
            return result
        return self._complete(SymbolicActionStatus.SUCCESS, None, outcome)

    def _complete(
        self,
        status: SymbolicActionStatus,
        reason: Optional[str],
        outcome: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        record = self.current_symbolic_action
        record.complete(self.current_env_step, status, reason, outcome)
        self._last_outcome = None
        return {
            "status": status.value,
            "failure_reason": reason,
            "outcome": outcome,
            "terminate_plan": False,
        }

    # -- surface the upstream wrappers call -------------------------------
    # SymbolicEnvWrapper and PlanningEnvWrapper reach into the executor for
    # these. They are part of the contract just as much as execute() is, and
    # missing one shows up as an AttributeError deep inside a reset.

    def update_env_step(self, env_step: int) -> None:
        self.current_env_step = int(env_step)

    def reset_current_action(self) -> None:
        self.current_symbolic_action = None
        self._last_outcome = None

    def clear_history(self) -> None:
        self.action_history.clear()

    def get_action_records(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.action_history]

    def get_action_value(self, primitive_action: str) -> str:
        """Crafter mapped primitive names to integer env actions; here the
        primitive name *is* the value, because the facade takes symbolic
        actions rather than an action index."""
        return primitive_action

    #: Crafter side-channels for share/place/collect requests. The wrapper
    #: polls these every step; ours never populates them because the facade
    #: takes the symbolic action itself.
    pending_share = None
    pending_place = None
    pending_collect = None

    def complete_current_action(
        self,
        status: SymbolicActionStatus,
        failure_reason: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Force-complete, e.g. when the plan is abandoned from outside."""
        if self.current_symbolic_action is not None:
            self.current_symbolic_action.complete(self.current_env_step, status, failure_reason, outcome)
            self._last_outcome = None
