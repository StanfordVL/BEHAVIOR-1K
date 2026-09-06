"""L1d task tracking. **Not implemented yet (M6).**

PORTING_PLAN 4 maps COOP2's ``TaskState`` four-tuple (spatial / temporal /
dependency / participation) onto BEHAVIOR predicates, and says to carry
``StepMetrics``, ``CapabilityChange``, ``convert_to_serializable`` and
``plot_metrics_timeline`` over unchanged so ``compute_constraint_metrics`` and
``build_results_table`` need no edits at all.

Only ``plot_metrics_timeline`` is imported at module scope by the runners, so
only it needs to exist right now. It is a no-op rather than a raise: it is
called at the *end* of an episode purely to draw a figure, and killing a
finished episode over a missing plot would be the wrong trade.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["plot_metrics_timeline"]


def plot_metrics_timeline(metrics_history: Any, output_path: Optional[str] = None, show: bool = False) -> None:
    """No-op stand-in for the metrics figure (M6)."""
    print(f"[coop2] plot_metrics_timeline is a stub (M6); would have written {output_path!r}")
    return None
