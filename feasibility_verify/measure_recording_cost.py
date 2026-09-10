"""How much wall clock does episode recording actually cost?

``MultiViewRecorder`` captures one view per robot every ``every`` ticks, plus one
throwaway render per pass (the first view of a pass comes back showing the scene
from before the camera moved -- see recording.py). So a 6-robot run renders
``(steps / 4) * 7`` frames, and ``recording.py`` claims rendering is the whole
cost of a symbolic run because the primitives are teleports and settles.

That claim is worth a number rather than a belief, and the number decides whether
``--no-video`` is worth passing.

Method: build the env **once** with recording enabled -- the viewer camera has to
exist before the Environment is built -- then time the same workload twice, with
the recorder detached from ``engine.on_tick`` and attached. Same scene, same
robots, same primitives, one process: the difference is the recorder and nothing
else. Every agent is given a long ``wait`` so both phases tick an identical
workload; ``wait`` occupies the engine exactly like any other primitive.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_recording_cost.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="hall_glass_ceiling")
    parser.add_argument("--room", default="empty_room_0")
    parser.add_argument("--bddl-activity", default="coop_nine_apples_hall")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--ticks", type=int, default=400, help="ticks timed per phase")
    parser.add_argument("--episode-steps", type=int, default=9242,
                        help="a real episode length, to project the total onto")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv
    from coop2.behavior_env.primitive_engine import WAIT
    from coop2.behavior_env.recording import chain

    scratch = os.environ.get("COOP2_MEASURE_VIDEO", "/tmp/coop2_recording_cost/episode.mp4")
    os.makedirs(os.path.dirname(scratch), exist_ok=True)

    env = CooperativeBehaviorEnv(
        n_agents=args.agents, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=200000,
        video_path=scratch,
    )
    env.reset()
    engine = env.engine
    recorder = env.recorder
    if recorder is None:
        print("FAIL: no recorder was built, so there is nothing to compare against")
        return 1
    print(f"\n{args.agents} robots -> {len(recorder.views)} views, capture every "
          f"{recorder.every} ticks, {len(recorder.views) + 1} renders per pass "
          "(the extra one is the throwaway slot)")

    def timed_phase(label, on_tick):
        """Tick @args.ticks with an identical workload and report seconds/tick."""
        engine.on_tick = on_tick
        # Re-arm every agent so both phases run the same primitive.
        for agent_id in env.agent_names:
            if not engine.has_active(agent_id):
                engine.assign(agent_id, WAIT, None,
                              primitive_kwargs={"ticks": args.ticks + 50})
        start = time.perf_counter()
        for _ in range(args.ticks):
            engine.tick()
        elapsed = time.perf_counter() - start
        per_tick = elapsed / args.ticks
        print(f"  {label:<24} {elapsed:7.2f} s for {args.ticks} ticks   "
              f"{per_tick * 1000:7.2f} ms/tick   {1.0 / per_tick:6.1f} ticks/s")
        return per_tick

    print("\ntiming the same workload twice, one process, one scene:")
    # Recording OFF first: the recorder is what we are adding, so measuring the
    # cheaper configuration first cannot be flattered by warm caches.
    off = timed_phase("recording OFF", None)
    on = timed_phase("recording ON", chain(None, recorder))
    # And again, alternating, because a one-shot pair cannot tell a real
    # difference from drift in whatever else the machine is doing.
    off2 = timed_phase("recording OFF (repeat)", None)
    on2 = timed_phase("recording ON (repeat)", chain(None, recorder))

    off_mean = (off + off2) / 2
    on_mean = (on + on2) / 2
    overhead = on_mean - off_mean
    print(f"\nmean off {off_mean * 1000:.2f} ms/tick, mean on {on_mean * 1000:.2f} ms/tick")
    print(f"recording costs {overhead * 1000:.2f} ms/tick "
          f"-> {on_mean / off_mean:.2f}x slower overall")

    projected_off = off_mean * args.episode_steps
    projected_on = on_mean * args.episode_steps
    print(f"\nprojected onto a {args.episode_steps}-step episode:")
    print(f"  with recording   {projected_on / 60:6.1f} min")
    print(f"  without          {projected_off / 60:6.1f} min")
    print(f"  saved by --no-video {(projected_on - projected_off) / 60:6.1f} min "
          f"({100 * (projected_on - projected_off) / projected_on:.0f} % of the episode)")
    print("\nNote: this is episode wall clock only. Scene load (~1-2 min) and the "
          "LLM's own latency are unchanged by --no-video.")

    import omnigibson as og
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
