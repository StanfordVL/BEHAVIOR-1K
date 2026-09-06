"""Visualization stubs.

ma_crafter's viz renders the crafter grid world; none of it transfers. The
runners import these names unconditionally, so this module keeps the surface
and drops the drawing. ``RealtimeVisualizationWrapper`` in particular must stay
a working *pass-through*: the runner wraps the plan env in it and then drives
the wrapper, so a stub that dropped calls would silently disable the episode.

The real thing here will be OmniGibson video capture --
:mod:`coop2.behavior_env.recording` already writes third-person video off the
engine's per-tick hook.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RealtimeAgentVisualizer",
    "RealtimeVisualizationWrapper",
    "build_repair_intervention_report",
    "print_plan_summary",
    "visualize_all_agents_progress",
    "visualize_comprehensive_timeline",
    "visualize_plan_timeline",
    "visualize_repair_interventions",
]


class RealtimeVisualizationWrapper:
    """Transparent pass-through around the plan env.

    Every attribute that is not defined here is forwarded, so the runner's
    ``env.step`` / ``env.agents`` / ``env.message_broker`` all reach the wrapped
    environment untouched.
    """

    def __init__(self, plan_env: Any, record_video: bool = False, show: bool = True):
        self._env = plan_env
        self.record_video = record_video
        self.show = show

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found normally, so _env itself is safe.
        return getattr(self._env, name)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()


class RealtimeAgentVisualizer:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def visualize_plan_timeline(*args: Any, **kwargs: Any) -> None:
    return None


def visualize_all_agents_progress(*args: Any, **kwargs: Any) -> None:
    return None


def print_plan_summary(*args: Any, **kwargs: Any) -> None:
    return None


def visualize_comprehensive_timeline(*args: Any, **kwargs: Any) -> None:
    return None


def build_repair_intervention_report(*args: Any, **kwargs: Any) -> dict:
    return {}


def visualize_repair_interventions(*args: Any, **kwargs: Any) -> None:
    return None
