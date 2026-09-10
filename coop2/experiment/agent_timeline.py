"""Draw each agent's FSM state over wall-clock time, one lane per agent.

Replaces metrics_timeline.png, which plotted constraint counters this project
does not use.

Inter-agent messages are overlaid as arrows from the sender's lane to each
recipient's, at the moment they were sent. Content is deliberately not drawn --
the question this figure answers is *when* an agent talked and *to whom*, which
is what ties a frozen world (the I and R spans) to the thing that froze it. The
message clock and the state clock are both seconds since the run started, so
they share the x axis directly.

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

#: Messages are drawn in one colour on purpose: the arrow already carries the
#: direction, and colouring by metadata["type"] would compete with the state
#: colours for the reader's attention.
MESSAGE_COLOUR = "#22223b"


def _coalesce(spans, min_width: float):
    """Absorb sub-pixel spans into their neighbour, then merge like with like.

    A chain wave wakes agent_i once per upstream relay, and with the turn gate
    those wake-ups return immediately: measured at 34 ms of `interrupted`
    across 42 of them in an 877 s episode. Drawn as bars they are 1e-5 of the
    axis, but each still paints its white 0.5 pt edge, so agent_5's lane came
    out shredded by events that together cost a thirtieth of a second. Widening
    them to a visible pixel would overstate them by five orders of magnitude,
    so they are folded into the span around them instead -- the messages are
    still on the figure as arrows, which is where a wave is legible anyway.

    Returns ``(spans, absorbed_count)``.
    """
    kept, absorbed = [], 0
    for start, stop, state, first_step, last_step in spans:
        if stop - start < min_width and kept:
            # Hand the time back to the span it interrupted.
            prev = kept[-1]
            kept[-1] = (prev[0], stop, prev[2], prev[3], last_step)
            absorbed += 1
            continue
        kept.append((start, stop, state, first_step, last_step))

    merged = []
    for span in kept:
        if merged and merged[-1][2] == span[2] and abs(merged[-1][1] - span[0]) < 1e-9:
            prev = merged[-1]
            merged[-1] = (prev[0], span[1], prev[2], prev[3], span[4])
            continue
        merged.append(span)
    return merged, absorbed


def _spans(
    transitions: List[Any], end_time: float, end_step: Optional[int] = None,
) -> List[Tuple[float, float, str, int, int]]:
    """``[(start, end, state, env_step_in, env_step_out), ...]``.

    agent_states.json records the moment a state was *entered*, so a span runs
    to the next entry, and the last one to the end of the episode. Both ends of
    the env_step range are carried because only the range is informative: an
    entry step alone reads as "this span cost 0 steps" on the first span of the
    run, which starts at env_step 0 and can run for thousands of ticks.
    """
    spans = []
    for index, entry in enumerate(transitions):
        timestamp, env_step, state = float(entry[0]), int(entry[1]), str(entry[2])
        following = transitions[index + 1] if index + 1 < len(transitions) else None
        stop = float(following[0]) if following is not None else end_time
        stop_step = int(following[1]) if following is not None else (
            end_step if end_step is not None else env_step
        )
        if stop > timestamp:
            spans.append((timestamp, stop, state, env_step, stop_step))
    return spans


def _step_label(first: int, last: int) -> str:
    """``"280"`` when the world did not move, ``"0-280"`` when it did.

    A degenerate range is the point, not a defect: R, W and I all freeze the
    env, so a single number *is* the reading, and seeing the same number on
    three consecutive bars is how the barrier shows up in the figure.
    """
    return str(first) if last <= first else f"{first}-{last}"



def _draw_messages(axes, messages, lane_of, end_time) -> int:
    """Overlay sender -> recipient arrows. Returns how many were drawn.

    One arrow per (message, recipient), so a leader's broadcast to eight
    followers is eight arrowheads leaving one point -- which is the shape of the
    cost, and reads correctly at a glance. Slightly curved so that several
    messages exchanged at almost the same instant do not collapse into a single
    vertical stroke.
    """
    drawn = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        try:
            when = float(message.get("timestamp"))
        except (TypeError, ValueError):
            continue
        sender = message.get("sender")
        if sender not in lane_of or not 0.0 <= when <= end_time:
            continue
        recipients = [
            r for r in (message.get("recipients") or [])
            if r in lane_of and r != sender
        ]
        if not recipients:
            continue
        # A message that stops a teammate mid-primitive and one that waits in
        # their buffer cost completely different amounts -- the first is an LLM
        # round trip, the second is free -- so they must not be the same stroke.
        # `interrupts_execution` is the field the broker itself reads.
        interrupting = bool((message.get("metadata") or {}).get("interrupts_execution", True))
        style = "-" if interrupting else (0, (2, 2))
        axes.plot(
            [when], [lane_of[sender]], marker="o", markersize=3.5,
            color=MESSAGE_COLOUR, zorder=5,
        )
        for recipient in recipients:
            axes.annotate(
                "",
                xy=(when, lane_of[recipient]), xytext=(when, lane_of[sender]),
                arrowprops={
                    "arrowstyle": "-|>", "color": MESSAGE_COLOUR,
                    "linewidth": 1.0, "shrinkA": 1.5, "shrinkB": 1.5,
                    "connectionstyle": "arc3,rad=0.12",
                    "linestyle": style, "alpha": 1.0 if interrupting else 0.55,
                },
                zorder=5, annotation_clip=False,
            )
            drawn[interrupting] = drawn.get(interrupting, 0) + 1
    return drawn


def plot_agent_state_timeline(
    agent_states: Dict[str, List[Any]],
    output_path: str,
    title: Optional[str] = None,
    end_step: Optional[int] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Write a Gantt-style figure of agent states. Returns the path, or None.

    Args:
        agent_states: ``{agent_id: [[wall_clock, env_step, state], ...]}``, the
            contents of agent_states.json.
        output_path: where to write the PNG.
        title: figure title; defaults to the run directory's name.
        end_step: env_step the episode ended on, for the last span of each
            agent. Without it that span's range is left degenerate rather than
            guessed at.
        messages: message_log.json's contents, or None. Only ``timestamp``,
            ``sender`` and ``recipients`` are read; the content is not drawn.
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
    # Half a pixel at the figure's own width: below this a bar is nothing but
    # its own edge stroke.
    min_width = end_time / (14 * 140 * 2)
    absorbed_total = 0
    for lane, agent_id in enumerate(agents):
        spans, absorbed = _coalesce(
            _spans(agent_states[agent_id], end_time, end_step), min_width
        )
        absorbed_total += absorbed
        for start, stop, state, first_step, last_step in spans:
            axes.barh(
                lane, stop - start, left=start, height=0.55,
                color=STATE_COLOURS.get(state, "#cccccc"),
                edgecolor="white", linewidth=0.5,
            )
            if state not in seen_states:
                seen_states.append(state)
            # The env_step range inside the span, where it fits: it is how a
            # reader ties this figure back to plan_logs.json, and the width of
            # the range is how much simulation the span actually bought.
            if stop - start > end_time * 0.05:
                axes.text(
                    (start + stop) / 2, lane, _step_label(first_step, last_step),
                    ha="center", va="center", fontsize=7, color="white",
                )

    lane_of = {agent_id: lane for lane, agent_id in enumerate(agents)}
    drawn_messages = _draw_messages(axes, messages or [], lane_of, end_time)
    message_count = sum(drawn_messages.values())

    axes.set_yticks(range(len(agents)))
    axes.set_yticklabels(agents)
    axes.set_ylim(-0.6, len(agents) - 0.4)
    axes.invert_yaxis()
    # A sliver of left margin: a leader's opening broadcast is sent at t~0, and
    # against xlim=(0, ...) its marker and arrowhead sit on the spine.
    axes.set_xlim(-end_time * 0.012, end_time)
    xlabel = "wall clock (s) -- labels inside the bars are the env_step range"
    if absorbed_total:
        xlabel += f"  |  {absorbed_total} sub-pixel spans folded into their neighbour"
    axes.set_xlabel(xlabel)
    axes.set_title(title or os.path.basename(os.path.dirname(os.path.abspath(output_path))))
    axes.grid(axis="x", alpha=0.3, linestyle=":")

    order = [s for s in ("reasoning", "interrupted", "waiting", "executing") if s in seen_states]
    handles = [mpatches.Patch(color=STATE_COLOURS[s], label=s) for s in order]
    if message_count:
        # Proxy artists, because an annotate() arrow is not a legend handle.
        from matplotlib.lines import Line2D  # noqa: PLC0415

        for interrupting, label in ((True, "message, interrupts"), (False, "message, buffered")):
            count = drawn_messages.get(interrupting, 0)
            if not count:
                continue
            handles.append(Line2D(
                [0], [0], color=MESSAGE_COLOUR, marker="o", markersize=4, linewidth=1.0,
                linestyle="-" if interrupting else (0, (2, 2)),
                alpha=1.0 if interrupting else 0.55,
                label=f"{label} (n={count})",
            ))
    axes.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.28),
        ncol=len(handles) or 1, frameon=False,
    )

    figure.tight_layout()
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _episode_end_step(run_dir: str) -> Optional[int]:
    """Last env_step of the episode, from plan_logs.json.

    agent_states.json only holds transitions, so the final span of every agent
    has no recorded end. Plans are closed out with the episode (they are marked
    interrupted at the step it ended on), so the largest ``end_step`` there is
    that step.
    """
    path = os.path.join(run_dir, "plan_logs.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if isinstance(payload, list):
        plans = payload
    else:
        # The saver writes {"plan_history": [...], "current_plans": {...}}.
        plans = payload.get("plan_history") or payload.get("plans") or []
    steps = [p.get("end_step") for p in plans if isinstance(p, dict)]
    steps = [int(s) for s in steps if isinstance(s, (int, float))]
    return max(steps) if steps else None


def _messages(run_dir: str) -> List[Dict[str, Any]]:
    """message_log.json, or an empty list.

    Absent or empty for the `individual` topology, which is not a failure: that
    topology has no communication channel at all, so the figure simply carries
    no arrows.
    """
    path = os.path.join(run_dir, "message_log.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    return payload if isinstance(payload, list) else []


def plot_from_run_dir(run_dir: str, filename: str = "agent_timeline.png") -> Optional[str]:
    """Draw the timeline for an existing run directory."""
    states_path = os.path.join(run_dir, "agent_states.json")
    if not os.path.exists(states_path):
        return None
    with open(states_path) as handle:
        agent_states = json.load(handle)
    return plot_agent_state_timeline(
        agent_states,
        os.path.join(run_dir, filename),
        title=os.path.basename(run_dir),
        end_step=_episode_end_step(run_dir),
        messages=_messages(run_dir),
    )


if __name__ == "__main__":
    import sys

    for directory in sys.argv[1:] or ["."]:
        written = plot_from_run_dir(directory)
        print(f"{directory}: {written or 'no agent_states.json'}")
