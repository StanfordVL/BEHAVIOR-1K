"""Does ``--no-video`` actually stop rendering?

Short answer, from the code: no, only partly. ``--no-video`` leaves
``video_path`` unset, which (a) skips ``enable_viewer_rendering()``, (b) leaves
``gm.RENDER_VIEWER_CAMERA`` False, and (c) never attaches ``MultiViewRecorder``
to ``engine.on_tick``, so the recorder's explicit ``og.sim.render()`` calls --
one per robot plus a throwaway, every ``every`` ticks -- are gone.

What it does **not** touch is the render inside every simulator step.
``Simulator.__init__`` sets ``self._render_on_step = True`` unconditionally, and
``Simulator.step`` reads it:

    render = self._render_on_step
    ...
    if render: self._sim_context.step(render=True)
    else:      for _ in range(n_physics_timesteps_per_render): step(render=False)

The only thing that flips it is the ``og.sim.render_on_step(False)`` context
manager, and coop2 never uses it. ``gm.RENDER_VIEWER_CAMERA=False`` only hides
the Viewport *window* (simulator.py's ``hide_window_names``); it does not stop
the renderer.

So this measures the part ``--no-video`` cannot remove: the same tick workload
with the per-step render left on (what a ``--no-video`` run does today) against
it switched off. Both conditions run in one process with ``video_path=None``, so
the recorder is absent from both and the only difference is that switch.

The workload is a long ``wait`` on every agent: it occupies the engine exactly
like any other primitive, ticks identically in both conditions, and involves no
LLM, whose latency would swamp what is being measured.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_no_video_render.py
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_no_video_render.py \
        --agents 9 --scene hall_glass_ceiling --room empty_room_0 \
        --bddl-activity coop_nine_apples_hall
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--scene", default="Pomaria_1_int")
    parser.add_argument("--room", default="living_room_0")
    parser.add_argument("--bddl-activity", default="coop_two_apples_pomaria")
    parser.add_argument("--ticks", type=int, default=400, help="ticks per timed phase")
    parser.add_argument("--rounds", type=int, default=3, help="A/B pairs, to average out drift")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import omnigibson as og
    from omnigibson.macros import gm

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv
    from coop2.behavior_env.primitive_engine import WAIT

    env = CooperativeBehaviorEnv(
        n_agents=args.agents, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=10_000_000,
        video_path=None,  # exactly what --no-video produces
    )
    env.reset()
    engine = env.engine

    print(f"\nscene={args.scene}  agents={args.agents}")
    print(f"gm.RENDER_VIEWER_CAMERA = {gm.RENDER_VIEWER_CAMERA}   (--no-video leaves this False)")
    print(f"gm.HEADLESS             = {gm.HEADLESS}")
    print(f"og.sim._render_on_step  = {og.sim._render_on_step}   "
          f"<-- this is what decides whether every step renders")
    print(f"recorder attached       = {env.recorder is not None}")
    print(f"viewer camera exists    = {og.sim.viewer_camera is not None}\n")

    def tick_phase(count):
        """Keep every agent busy with a wait, then tick `count` times."""
        for agent_id in env.agent_names:
            if not engine.has_active(agent_id):
                engine.assign(agent_id, WAIT, None, primitive_kwargs={"ticks": count + 10})
        start = time.perf_counter()
        for _ in range(count):
            engine.tick()
        return time.perf_counter() - start

    # A warm-up phase, so the first timed phase does not pay for shader/pipeline
    # setup and hand the win to whichever condition happens to run second.
    tick_phase(args.ticks)

    rendering, physics_only = [], []
    for round_index in range(args.rounds):
        rendering.append(tick_phase(args.ticks))
        with og.sim.render_on_step(False):
            physics_only.append(tick_phase(args.ticks))
        print(f"  round {round_index + 1}: render-on-step {rendering[-1]:6.2f} s   "
              f"physics-only {physics_only[-1]:6.2f} s")

    render_total = sum(rendering)
    physics_total = sum(physics_only)
    ticks = args.ticks * args.rounds
    print(f"\n{ticks} ticks per condition, {args.rounds} rounds:")
    print(f"  as --no-video runs today (render_on_step=True): "
          f"{render_total:6.2f} s  ->  {ticks / render_total:6.1f} ticks/s")
    print(f"  with the per-step render off:                   "
          f"{physics_total:6.2f} s  ->  {ticks / physics_total:6.1f} ticks/s")
    if physics_total > 0:
        speedup = render_total / physics_total
        share = 1.0 - physics_total / render_total
        print(f"\n  => {speedup:.2f}x faster, i.e. {100 * share:.0f}% of a --no-video "
              f"run's tick time is rendering nobody looks at")
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
