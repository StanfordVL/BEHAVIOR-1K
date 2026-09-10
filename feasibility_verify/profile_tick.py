"""Where does one engine tick go?

An episode's wall clock is (scene load) + (LLM latency, world frozen) + (ticking).
This prices the last one, because the other two are answered elsewhere and the
question "is there anything left to cut" needs to know whether ticking is
physics-bound or paying for something coop2 does per tick.

The workload is a long ``wait`` on every agent -- it occupies the engine exactly
like any other primitive but does no motion planning, so what remains is the
per-tick floor: ``env.step``, the engine's own bookkeeping, and whatever
callbacks are chained onto ``on_tick``.

Reads as: cumulative time by function, with the OmniGibson/coop2 split called
out. A past regression in this project (two all-pairs scans per macro-step, 75x
slower env_step) was invisible until someone profiled instead of guessing, which
is the reason this exists as a script rather than a one-off.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/profile_tick.py \
        --agents 6 --scene hall_glass_ceiling --room empty_room_0 \
        --bddl-activity coop_nine_apples_hall
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--scene", default="hall_glass_ceiling")
    parser.add_argument("--room", default="empty_room_0")
    parser.add_argument("--bddl-activity", default="coop_nine_apples_hall")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--rows", type=int, default=22)
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import omnigibson as og

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv
    from coop2.behavior_env.primitive_engine import WAIT

    env = CooperativeBehaviorEnv(
        n_agents=args.agents, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=10_000_000,
        video_path=None,
    )
    env.reset()
    engine = env.engine

    def keep_busy(count):
        for agent_id in env.agent_names:
            if not engine.has_active(agent_id):
                engine.assign(agent_id, WAIT, None, primitive_kwargs={"ticks": count + 10})

    keep_busy(args.ticks)
    for _ in range(50):  # warm-up, so shader/pipeline setup is not in the profile
        engine.tick()

    keep_busy(args.ticks)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    for _ in range(args.ticks):
        engine.tick()
    profiler.disable()
    elapsed = time.perf_counter() - start

    print(f"\n{args.ticks} ticks in {elapsed:.2f} s -> {1000 * elapsed / args.ticks:.1f} ms/tick "
          f"({args.agents} agents, {args.scene})\n")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(args.rows)

    # The split that decides whether there is anything to cut on our side.
    ours = sum(
        entry[3] for key, entry in stats.stats.items()
        if "/coop2/" in key[0]
    )
    print(f"\ncumulative time inside coop2/ frames: {ours:.2f} s of {elapsed:.2f} s "
          f"({100 * ours / elapsed:.0f}%)")
    print("(cumulative, so it includes the env.step called from inside them -- "
          "read the table above for where it actually goes)")
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
