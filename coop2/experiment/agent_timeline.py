"""Draw each agent's FSM state over wall-clock time, one lane per agent.

Replaces metrics_timeline.png, which plotted constraint counters this project
does not use.

The x axis is wall clock, not env_step, and that is the point of the figure.
The plan loop does not step the environment while any agent is not ready, so an
agent in R or I freezes the whole world: those spans occupy real seconds while
env_step does not move at all. Plotted against env_step they would collapse to
zero width, hiding the one cost the figure exists to show.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["plot_agent_state_timeline"]

#: FSM state -> colour. Deliberately loud for R and I: those are the spans that
#: stop every other agent.
STATE_COLOURS = {
    "reasoning": "#d1495b",    # R -- LLM call, world frozen
    "interrupted": "#edae49",  # I -- message arrived, world frozen
    "waiting": "#8d99ae",      # W -- ready, waiting for the others
    "executing": "#2a9d8f",    # X -- primitive advancing
}


def _spans(transitions: List[Any], end_time: float) -> List[Tuple[float, float, str, int]]:
    """``[(start, end, state, env_step), ...]`` from a list of transitions.

    agent_states.json records the moment a state was *entered*, so a span runs
    to the next entry, and the last one to the end of the episode.
    """
    spans = []
    for index, entry in enumerate(transitions):
        timestamp, env_step, state = float(entry[0]), int(entry[1]), str(entry[2])
        stop = float(transitions[index + 1][0]) if index + 1 < len(transitions) else end_time
        if stop > timestamp:
            spans.append((timestamp, stop, state, env_step))
    return spans


def plot_agent_state_timeline(
    agent_states: Dict[str, List[Any]],
    output_path: str,
    title: Optional[str] = None,
) -> Optional[str]:
    """Write a Gantt-style figure of agent states. Returns the path, or None.

    Args:
        agent_states: ``{agent_id: [[wall_clock, env_step, state], ...]}``, the
            contents of agent_states.json.
        output_path: where to write the PNG.
        title: figure title; defaults to the run directory's name.
    """
    if not agent_states:
        return None

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    agents = sorted(agent_states)
    end_time = max(
        (float(entry[0]) for entries in agent_states.values() for entry in entries),
        default=0.0,
    )
    if end_time <= 0:
        return None
    # The last state of every agent runs to the end of the run, and the run is
    # at least as long as the last transition anyone made.
    end_time *= 1.02

    figure, axes = plt.subplots(figsize=(14, 1.4 + 0.9 * len(agents)))
    seen_states = []
    for lane, agent_id in enumerate(agents):
        for start, stop, state, env_step in _spans(agent_states[agent_id], end_time):
            axes.barh(
                lane, stop - start, left=start, height=0.55,
                color=STATE_COLOURS.get(state, "#cccccc"),
                edgecolor="white", linewidth=0.5,
            )
            if state not in seen_states:
                seen_states.append(state)
            # env_step inside the span, where it fits: it is how a reader ties
            # this figure back to plan_logs.json.
            if stop - start > end_time * 0.04:
                axes.text(
                    (start + stop) / 2, lane, f"{env_step}",
                    ha="center", va="center", fontsize=7, color="white",
                )

    axes.set_yticks(range(len(agents)))
    axes.set_yticklabels(agents)
    axes.set_ylim(-0.6, len(agents) - 0.4)
    axes.invert_yaxis()
    axes.set_xlim(0, end_time)
    axes.set_xlabel("wall clock (s) -- labels inside the bars are env_step")
    axes.set_title(title or os.path.basename(os.path.dirname(os.path.abspath(output_path))))
    axes.grid(axis="x", alpha=0.3, linestyle=":")

    order = [s for s in ("reasoning", "interrupted", "waiting", "executing") if s in seen_states]
    axes.legend(
        handles=[mpatches.Patch(color=STATE_COLOURS[s], label=s) for s in order],
        loc="upper center", bbox_to_anchor=(0.5, -0.28),
        ncol=len(order) or 1, frameon=False,
    )

    figure.tight_layout()
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_from_run_dir(run_dir: str, filename: str = "agent_timeline.png") -> Optional[str]:
    """Draw the timeline for an existing run directory."""
    states_path = os.path.join(run_dir, "agent_states.json")
    if not os.path.exists(states_path):
        return None
    with open(states_path) as handle:
        agent_states = json.load(handle)
    return plot_agent_state_timeline(
        agent_states, os.path.join(run_dir, filename), title=os.path.basename(run_dir)
    )


if __name__ == "__main__":
    import sys

    for directory in sys.argv[1:] or ["."]:
        written = plot_from_run_dir(directory)
        print(f"{directory}: {written or 'no agent_states.json'}")
