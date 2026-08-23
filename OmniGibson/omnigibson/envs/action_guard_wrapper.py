"""Fail-closed action guard for OmniGibson environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from omnigibson.envs.env_wrapper import EnvironmentWrapper


@dataclass(frozen=True)
class ActionGuardDecision:
    """Result returned by a pre-action guard."""

    allowed: bool
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ActionRejectedError(PermissionError):
    """Raised before simulation when an action guard denies an action."""

    def __init__(self, reason: str, metadata: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.metadata = dict(metadata or {})


class ActionGuardWrapper(EnvironmentWrapper):
    """Evaluate a caller-supplied guard immediately before ``env.step``.

    The guard receives ``(env, action)`` and may return a boolean or any object
    with ``allowed``, optional ``reason``, and optional ``metadata`` attributes.
    A denial, malformed result, or guard exception prevents the simulator from
    stepping and raises :class:`ActionRejectedError`.
    """

    def __init__(self, env, guard: Callable[[Any, Any], Any]):
        if not callable(guard):
            raise TypeError("guard must be callable")
        self._guard = guard
        self.last_decision = None
        super().__init__(env=env)

    @staticmethod
    def _normalize_decision(value: Any) -> ActionGuardDecision:
        if isinstance(value, bool):
            return ActionGuardDecision(value, "allowed" if value else "rejected")
        allowed = getattr(value, "allowed", None)
        if not isinstance(allowed, bool):
            raise TypeError("guard result must be a bool or expose a boolean 'allowed' attribute")
        reason = getattr(value, "reason", "")
        metadata = getattr(value, "metadata", {})
        if not isinstance(reason, str) or not isinstance(metadata, Mapping):
            raise TypeError("guard result reason and metadata must be a string and mapping")
        return ActionGuardDecision(allowed, reason, dict(metadata))

    def step(self, action, n_render_iterations=1):
        try:
            decision = self._normalize_decision(self._guard(self.env, action))
        except Exception as exc:
            self.last_decision = ActionGuardDecision(
                False,
                "action_guard_error",
                {"error_type": type(exc).__name__},
            )
            raise ActionRejectedError(self.last_decision.reason, self.last_decision.metadata) from exc

        self.last_decision = decision
        if not decision.allowed:
            raise ActionRejectedError(decision.reason or "rejected", decision.metadata)
        return self.env.step(action, n_render_iterations=n_render_iterations)
