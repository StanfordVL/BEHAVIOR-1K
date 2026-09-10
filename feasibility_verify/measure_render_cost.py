"""How much of an episode's wall clock is the episode recording?

`recording.py` claims rendering is the whole cost of a symbolic run, because the
primitives are teleports plus a settle and are nearly free. That is a plausible
claim and worth a number rather than a belief, because the dial it implies --
`--no-video` -- is the difference between a sweep that takes an afternoon and one
that takes an hour.

The recorder captures every `every` ticks, and each capture renders once per
agent **plus one throwaway** (the first view of a pass comes back stale, see
MultiViewRecorder.capture). So the cost should scale with the agent count, which
is why this measures at whatever `--agents` you pass rather than quoting one
figure.

No LLM and no plans here on purpose: every agent is given one long `wait`, so the
engine ticks exactly like a real episode while nothing varies between the two
conditions except whether the recorder is attached. API latency and the model's
own choices would otherwise swamp the thing being measured.

Run both halves (video_path is a constructor argument, and og.sim is a process
singleton, so they cannot share a process):

    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_render_cost.py --video
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_render_cost.py --no-video
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--steps", type=int, default=600, help="ticks to time")
    parser.add_argument("--scene", default="hall_glass_ceiling")
    parser.add_argument("--room", default="empty_room_0")
    parser.add_argument("--bddl-activity", default="coop_nine_apples_hall")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", dest="video", action="store_true")
    group.add_argument("--no-video", dest="video", action="store_false")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv
    from coop2.behavior_env.primitive_engine import WAIT

    video_path = None
    if args.video:
        scratch = os.environ.get("COOP2_RENDER_SCRATCH", "/tmp")
        video_path = os.path.join(scratch, "render_cost_probe.mp4")

    build_started = time.monotonic()
    env = CooperativeBehaviorEnv(
        n_agents=args.agents, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=args.steps * 4,
        video_path=video_path,
    )
    env.reset()
    build_seconds = time.monotonic() - build_started

    engine = env.engine
    # One long wait each: the engine then ticks exactly as it would mid-episode,
    # and the recorder (if attached) fires on the same schedule.
    for agent_id in env.agent_names:
        engine.assign(agent_id, WAIT, None, primitive_kwargs={"ticks": args.steps * 2})

    started = time.monotonic()
    for _ in range(args.steps):
        engine.tick()
    elapsed = time.monotonic() - started

    label = "WITH video" if args.video else "NO video"
    per_step_ms = 1000.0 * elapsed / args.steps
    print(f"\n===== {label}: {args.agents} agents, {args.steps} ticks")
    print(f"  scene build + reset : {build_seconds:6.1f} s")
    print(f"  {args.steps} ticks          : {elapsed:6.1f} s")
    print(f"  per tick            : {per_step_ms:6.2f} ms   ({args.steps / elapsed:.1f} ticks/s)")
    if args.video:
        recorder = env.recorder
        frames = getattr(recorder, "frames", None)
        every = getattr(recorder, "every", None)
        views = len(getattr(recorder, "views", {}) or {})
        print(f"  recorder            : every={every}, {views} views, {frames} frames per view")
        if every:
            renders = (args.steps // every) * (views + 1)
            print(f"  renders in the window: ~{renders} "
                  f"({args.steps}/{every} passes x ({views} views + 1 throwaway))")
    print(f"\nRESULT {label}: {per_step_ms:.2f} ms/tick")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
