"""No-op stand-in for ma_crafter's ``coop2_repair`` package.

Repair is explicitly **not** ported (see coop2/CLAUDE.md). But
``cognitive/plan/plan_env_wrapper.py`` imports and calls the repair gate
unconditionally, and the point of the port is to keep COOP2's upper layers
byte-similar to the original so fixes can flow both ways. So instead of editing
the wrapper, this package satisfies its imports with a gate that is always
disabled.

The contract that matters is one line in ``plan_env_wrapper``::

    evaluation = self.coop2_repair_controller.before_execution(...)
    if evaluation is not None and evaluation.should_repair:

Upstream's own controller already returns ``None`` from ``before_execution``
when ``self.enabled`` is false, so "always disabled" is not a behavioural fork
-- it is exactly what upstream does with repair switched off.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from coop2._repair_shim.message_protocol import (
    COOP2_REPAIR_CONTENT_TYPE,
    COOP2_REPAIR_MESSAGE_TYPE,
    COOP2_REPAIR_SENDER_ID,
    MESSAGE_TYPE_METADATA_KEY,
    coop2_repair_metadata,
)

__all__ = [
    "COOP2_REPAIR_CONTENT_TYPE",
    "COOP2_REPAIR_MESSAGE_TYPE",
    "COOP2_REPAIR_SENDER_ID",
    "MESSAGE_TYPE_METADATA_KEY",
    "coop2_repair_metadata",
    "AgentPlanView",
    "Coop2RepairController",
    "Coop2TraceLogger",
    "MacrafterCoopAdapter",
    "ParallelEnvAdapter",
    "PettingZooParallelAdapter",
    "PreExecutionConstraintEvaluator",
    "PreExecutionEvaluation",
]


class AgentPlanView:
    """Placeholder for the per-agent plan snapshot the evaluator consumed."""

    def __init__(self, agent_id: str = "", **kwargs: Any):
        self.agent_id = agent_id
        self.__dict__.update(kwargs)


class PreExecutionEvaluation:
    """Always reports "nothing to repair"."""

    def __init__(self, env_step: int = 0, **kwargs: Any):
        self.env_step = env_step
        self.plan_views: List[AgentPlanView] = []
        self.results: List[Any] = []
        self.failures: List[Any] = []
        self.affected_agents: List[str] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def should_repair(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {"env_step": self.env_step, "should_repair": False, "shim": True}


class PreExecutionConstraintEvaluator:
    def __init__(self, *args: Any, **kwargs: Any):
        pass


class ParallelEnvAdapter:
    def __init__(self, *args: Any, **kwargs: Any):
        pass


class PettingZooParallelAdapter(ParallelEnvAdapter):
    pass


class MacrafterCoopAdapter(ParallelEnvAdapter):
    pass


class Coop2TraceLogger:
    """Swallows trace events; keeps them in memory so callers can still read."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.events: List[Any] = []

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_event(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset(self) -> None:
        self.events.clear()

    def to_list(self) -> List[Any]:
        return list(self.events)


class Coop2RepairController:
    """A repair gate that is permanently off.

    ``set_enabled(True)`` is accepted but does nothing: turning it on would
    promise predictive repair that this port does not implement, and silently
    doing nothing under a True flag is less confusing than raising from deep
    inside the plan loop. The flag is readable so a caller can notice.
    """

    def __init__(self, adapter: Any = None, trace_logger: Any = None, enabled: bool = False, **kwargs: Any):
        self.adapter = adapter
        self.trace_logger = trace_logger
        self.enabled = False
        self.requested_enabled = bool(enabled)
        self.evaluator = None
        self.task_cooldown_steps = kwargs.get("task_cooldown_steps", 0)

    def set_enabled(self, enabled: bool) -> None:
        self.requested_enabled = bool(enabled)
        self.enabled = False

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        self.set_enabled(False)

    def set_adapter(self, adapter: Any) -> None:
        self.adapter = adapter

    def reset(self) -> None:
        return None

    def build_plan_views(self, agents: Optional[Dict[str, Any]] = None) -> List[AgentPlanView]:
        return []

    def before_execution(
        self,
        env_step: int,
        agents: Optional[Dict[str, Any]] = None,
        current_info: Optional[Dict[str, Any]] = None,
        message_broker: Optional[Any] = None,
        plan_views: Optional[Sequence[AgentPlanView]] = None,
        repair_dispatcher: Optional[Any] = None,
    ) -> Optional[PreExecutionEvaluation]:
        return None
