"""How many ticks does each primitive cost?

"Latency" is not one number per verb. Three different things set it:

* ``navigate_to`` is dominated by the **travel charge** -- hold-position ticks
  emitted in proportion to the straight-line distance before the teleport
  (``DEFAULT_TRAVEL_TICKS_PER_METER``, 60 ticks/m). It is a function of distance,
  so a single median is meaningless without the distances it came from.
* the manipulation verbs are dominated by ``_settle_robot``, i.e. physics coming
  to rest, capped by ``MAX_STEPS_FOR_SETTLING``.
* a **rejected precondition is not free**: ``apply_ref`` runs its settle block
  after catching the error, so TOO_FAR and friends still cost ticks.

One tick is one ``env.step``, i.e. 1/``action_frequency`` s (30 Hz by default),
so seconds = ticks / 30.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_primitive_latency.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Pomaria_1_int")
    parser.add_argument("--room", default="living_room_0")
    parser.add_argument("--bddl-activity", default="coop_two_apples_pomaria")
    parser.add_argument("--repeats", type=int, default=4, help="grasp/place cycles to time")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import torch as th

    from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
        SymbolicSemanticActionPrimitiveSet as P,
    )

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv
    from coop2.behavior_env.primitive_engine import WAIT

    env = CooperativeBehaviorEnv(
        n_agents=2, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=200000,
    )
    env.reset()
    engine = env.engine
    controller = env.controllers["agent_0"]
    scope = env.env.task.object_scope

    def entity(key):
        value = scope[key]
        return getattr(value, "wrapped_obj", value)

    apple = entity("apple.n.01_1")
    table = entity("coffee_table.n.01_1")
    robot = env.env.robots[0]

    def run_one(primitive, target=None, **kwargs):
        """Issue one primitive, tick to completion, return (ticks, reason)."""
        before = engine.env_step
        engine.assign("agent_0", primitive, target, primitive_kwargs=kwargs or None)
        outcome = None
        for _ in range(20000):
            if not engine.has_active("agent_0"):
                break
            results = engine.tick()
            if "agent_0" in results:
                outcome = results["agent_0"]
                break
        ticks = engine.env_step - before
        reason = getattr(outcome, "reason_code", None) if outcome else None
        return ticks, reason

    print(f"\ntravel charge: {controller.travel_ticks_per_meter:.0f} ticks/m; "
          f"one tick = 1/30 s\n")

    # -- navigate_to, as a function of distance ----------------------------
    print("navigate_to (the cost is the travel charge, so distance is the variable)")
    print(f"  {'target':<22}{'distance':>10}{'ticks':>8}{'sec':>7}{'ticks/m':>9}")
    rows = []
    for key, target in (("apple.n.01_1", apple), ("coffee_table.n.01_1", table),
                        ("apple.n.01_2", entity("apple.n.01_2")), ("coffee_table.n.01_1", table)):
        start = robot.get_position_orientation()[0][:2]
        goal = target.get_position_orientation()[0][:2]
        distance = float(th.norm(goal - start))
        ticks, reason = run_one(P.NAVIGATE_TO, target)
        per_m = ticks / distance if distance > 0.01 else float("nan")
        flag = "" if reason in (None, "OK") else f"  ({reason})"
        print(f"  {key:<22}{distance:>10.2f}{ticks:>8}{ticks / 30.0:>7.1f}{per_m:>9.0f}{flag}")
        rows.append((distance, ticks))
    # Deliberately no linear fit against these distances. The travel charge is
    # levied on the distance to the *sampled standing pose*, not to the object's
    # centre, and the pose is anywhere in the annulus [clearance, clearance +
    # reach] -- so fitting ticks against centre distance invents a slope. The
    # `[nav]` line printed by the controller reports the distance actually
    # charged and the ticks for it; read those. Measured there, it is exactly
    # `travel_ticks_per_meter` (3.8 m -> 227, 2.1 m -> 129, 0.9 m -> 53).
    print(f"  (the [nav] lines above show the charged distance: exactly "
          f"{controller.travel_ticks_per_meter:.0f} ticks/m of it, plus settle)")

    # -- grasp / place_on_top ----------------------------------------------
    print("\ngrasp and place_on_top (cost is settle; distance does not enter)")
    timings = {"grasp": [], "place_on_top": []}
    for _ in range(args.repeats):
        run_one(P.NAVIGATE_TO, apple)
        ticks, reason = run_one(P.GRASP, apple)
        if reason in (None, "OK"):
            timings["grasp"].append(ticks)
        run_one(P.NAVIGATE_TO, table)
        ticks, reason = run_one(P.PLACE_ON_TOP, table)
        if reason in (None, "OK"):
            timings["place_on_top"].append(ticks)
    for name, values in timings.items():
        if values:
            print(f"  {name:<16}n={len(values)}  min={min(values)}  "
                  f"median={int(statistics.median(values))}  max={max(values)}  "
                  f"({statistics.median(values) / 30.0:.1f} s)")
        else:
            print(f"  {name:<16}no successful sample")

    # -- wait ---------------------------------------------------------------
    print("\nwait (exactly what it is asked for -- it must occupy the engine, or "
          "the world stops while an agent yields)")
    for requested in (10, 100):
        ticks, _ = run_one(WAIT, None, ticks=requested)
        print(f"  wait(ticks={requested:<4}) -> {ticks} ticks ({ticks / 30.0:.1f} s)")

    # -- a rejected precondition -------------------------------------------
    print("\nrejected preconditions are NOT free (apply_ref settles after catching)")
    far = th.tensor([float(v) for v in table.get_position_orientation()[0][:2]]) + th.tensor([6.0, 0.0])
    robot.set_position_orientation(
        position=th.tensor([float(far[0]), float(far[1]), 0.05]),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    robot.keep_still()
    ticks, reason = run_one(P.GRASP, apple)
    print(f"  grasp from 6 m away -> {reason} after {ticks} ticks")

    print("\nNote: an episode's --steps budget is spent in these units, so the "
          "cost of one apple is navigate + grasp + navigate + place.")
    import omnigibson as og
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
