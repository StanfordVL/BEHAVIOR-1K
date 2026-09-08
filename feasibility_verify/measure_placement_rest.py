"""Does a pose the OnTop sampler returns actually leave the object OnTop?

`place_on_top` failed with EXECUTION_ERROR "it did not come to rest there" on the
*second* apple of `coop_two_apples_pomaria`, after the first had already been
delivered to the same coffee table. Three mechanisms could do that and they need
different fixes:

  * the sampled pose overlaps the apple already resting there, so the newcomer is
    pushed off during the settle;
  * the sampler lifts the object PREDICATE_SAMPLING_Z_OFFSET (0.02 m) above the
    surface, and an apple is a sphere -- it lands, rolls, and leaves the top;
  * the sampler returns poses that were never on the table to begin with.

So measure the pipeline in isolation: sample a pose the same way the primitive
does, teleport, keep_still, settle, then ask OnTop. Report the displacement
between the sampled pose and where it ended, and the distance to the other
apple, in both conditions. No robot and no LLM are involved -- if placement is
unreliable here it is unreliable for every topology.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/measure_placement_rest.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _unwrap(entity):
    """`object_scope` holds BDDLEntity wrappers; the sampler needs the object."""
    return getattr(entity, "wrapped_obj", entity)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Pomaria_1_int")
    parser.add_argument("--room", default="living_room_0")
    parser.add_argument("--bddl-activity", default="coop_two_apples_pomaria")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--probe-steps", type=int, default=60,
                        help="consecutive single-step OnTop reads of a motionless apple")
    parser.add_argument("--probe-only", action="store_true",
                        help="run only the flicker probe, skipping the placement surveys")
    parser.add_argument("--settle-steps", type=int, default=200,
                        help="matches tune_primitive_macros' MAX_STEPS_FOR_SETTLING")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import torch as th

    import omnigibson as og
    from omnigibson import object_states

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv

    env = CooperativeBehaviorEnv(
        n_agents=2, seed=0, scene_model=args.scene, room=args.room,
        bddl_activity=args.bddl_activity or None, length=40000,
    )
    env.reset()

    scope = getattr(env.env.task, "object_scope", None) or {}
    missing = [k for k in ("apple.n.01_1", "apple.n.01_2", "coffee_table.n.01_1") if k not in scope]
    if missing:
        print(f"FAIL: object_scope is missing {missing}; it has {sorted(scope)}")
        return 1

    apple_1 = _unwrap(scope["apple.n.01_1"])
    apple_2 = _unwrap(scope["apple.n.01_2"])
    table = _unwrap(scope["coffee_table.n.01_1"])
    controller = env.controllers["agent_0"]

    print(f"table  {table.name} at {[round(float(v), 3) for v in table.get_position_orientation()[0]]}")
    print(f"apple1 {apple_1.name}")
    print(f"apple2 {apple_2.name}")
    print(f"settle = {args.settle_steps} steps, PREDICATE_SAMPLING_Z_OFFSET applies to every sample\n")

    def settle():
        for _ in range(args.settle_steps):
            og.sim.step()

    def park(obj, pose):
        obj.set_position_orientation(*pose)
        obj.keep_still()

    def trial(moving, other, label):
        """One sample -> teleport -> settle -> OnTop, with attribution."""
        try:
            pose = controller._sample_pose_with_object_and_predicate(
                object_states.OnTop, moving, table,
            )
        except Exception as exc:  # SAMPLING_ERROR is itself an answer
            return {"outcome": "sample_failed", "detail": type(exc).__name__}

        sampled_xyz = [float(v) for v in pose[0]]
        park(moving, pose)
        # The neighbour must be held in place too, or its own drift is scored
        # against this trial.
        if other is not None:
            other.keep_still()
        settle()

        final_xyz = [float(v) for v in moving.get_position_orientation()[0]]
        drift = float(th.norm(th.tensor(final_xyz) - th.tensor(sampled_xyz)))
        on_top = bool(moving.states[object_states.OnTop].get_value(table))
        to_other = None
        if other is not None:
            other_xyz = [float(v) for v in other.get_position_orientation()[0]]
            to_other = float(th.norm(th.tensor(sampled_xyz) - th.tensor(other_xyz)))

        # OnTop = Touching(table) AND table below AND table not above. Report the
        # conjuncts rather than the verdict: "it did not come to rest" reads as a
        # physics failure and the apple is demonstrably resting on the surface.
        touching = bool(moving.states[object_states.Touching].get_value(table))
        adjacency = moving.states[object_states.VerticalAdjacency].get_value()
        below = [o.name for o in adjacency.negative_neighbors]
        above = [o.name for o in adjacency.positive_neighbors]
        table_xy = th.tensor([float(v) for v in table.get_position_orientation()[0][:2]])
        return {
            "outcome": "on_top" if on_top else "fell",
            "sampled": sampled_xyz,
            "final": final_xyz,
            "drift": drift,
            "dz": final_xyz[2] - sampled_xyz[2],
            "to_other": to_other,
            "touching": touching,
            "table_below": table.name in below,
            "table_above": table.name in above,
            "below": below,
            "above": above,
            "r_from_table": float(th.norm(th.tensor(final_xyz[:2]) - table_xy)),
        }

    def survey(label, moving, other, other_pose=None):
        print(f"--- {label}")
        if other_pose is not None:
            park(other, other_pose)
            settle()
            resting = bool(other.states[object_states.OnTop].get_value(table))
            print(f"    neighbour {other.name} OnTop(table) = {resting} before the survey starts")
            if not resting:
                print("    NOTE: the neighbour itself would not stay on the table")

        rows = [trial(moving, other, label) for _ in range(args.trials)]
        counts = {}
        for row in rows:
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
        n = len(rows)
        on_top = counts.get("on_top", 0)
        print(f"    OnTop after settle: {on_top}/{n} ({100.0 * on_top / n:.0f}%)   {counts}")

        landed = [r for r in rows if "drift" in r]
        if landed:
            drifts = [r["drift"] for r in landed]
            print(f"    drift sampled->final: median {statistics.median(drifts):.3f} m, "
                  f"max {max(drifts):.3f} m")
        fell = [r for r in landed if r["outcome"] == "fell"]
        held = [r for r in landed if r["outcome"] == "on_top"]

        # Which conjunct of OnTop is the one that fails?
        def conjunct_tally(rows):
            return {
                "touching": sum(1 for r in rows if r["touching"]),
                "table_below": sum(1 for r in rows if r["table_below"]),
                "table_above": sum(1 for r in rows if r["table_above"]),
                "n": len(rows),
            }

        if fell:
            print(f"    failed  conjuncts: {conjunct_tally(fell)}")
        if held:
            print(f"    ok      conjuncts: {conjunct_tally(held)}")
        if fell and held:
            print(f"    radius from table centre: failed median "
                  f"{statistics.median([r['r_from_table'] for r in fell]):.3f} m vs succeeded "
                  f"{statistics.median([r['r_from_table'] for r in held]):.3f} m")
        for r in fell[:6]:
            near = f", {r['to_other']:.3f} m from the neighbour" if r["to_other"] is not None else ""
            print(f"      fell: z {r['sampled'][2]:.3f}->{r['final'][2]:.3f}, r={r['r_from_table']:.3f} m, "
                  f"touching={r['touching']} below={r['below']} above={r['above']}{near}")
        for r in held[:3]:
            print(f"      ok:   z {r['sampled'][2]:.3f}->{r['final'][2]:.3f}, r={r['r_from_table']:.3f} m, "
                  f"below={r['below']} above={r['above']}")
        if fell and any(r["to_other"] is not None for r in fell):
            near_dists = [r["to_other"] for r in fell if r["to_other"] is not None]
            ok_dists = [r["to_other"] for r in landed
                        if r["outcome"] == "on_top" and r["to_other"] is not None]
            if near_dists and ok_dists:
                print(f"    distance to neighbour: failed median {statistics.median(near_dists):.3f} m "
                      f"vs succeeded median {statistics.median(ok_dists):.3f} m")
        print()
        return on_top, n

    def probe(moving, steps):
        """A settled apple, queried once per step: how often does OnTop say yes?

        `Touching` reads RigidContactAPI's *current step* contact matrix, so a
        body at rest is at the mercy of whether PhysX emitted a contact for that
        particular step. If this fraction is not 1.0 the predicate is unreliable
        for everyone who reads it -- our placement check AND BDDL's check_goal,
        which decides whether the activity is solved.
        """
        print(f"--- C: probe a settled apple for {steps} consecutive steps")
        try:
            pose = controller._sample_pose_with_object_and_predicate(
                object_states.OnTop, moving, table,
            )
        except Exception as exc:
            print(f"    could not seat the apple: {exc}\n")
            return
        park(moving, pose)
        settle()

        readings = []
        for _ in range(steps):
            og.sim.step()
            readings.append((
                bool(moving.states[object_states.Touching].get_value(table)),
                bool(moving.states[object_states.OnTop].get_value(table)),
            ))
        touch_true = sum(1 for t, _ in readings if t)
        top_true = sum(1 for _, o in readings if o)
        final_z = float(moving.get_position_orientation()[0][2])
        print(f"    the apple never moves (final z={final_z:.3f}), yet over {steps} steps:")
        print(f"      Touching True: {touch_true}/{steps} ({100.0 * touch_true / steps:.0f}%)")
        print(f"      OnTop    True: {top_true}/{steps} ({100.0 * top_true / steps:.0f}%)")
        pattern = "".join("T" if o else "." for _, o in readings)
        print(f"      per-step OnTop: {pattern}")
        if top_true not in (0, steps):
            print("    => the predicate flickers on a motionless object: any single-step read "
                  "of OnTop is a coin flip, including check_goal's.")

        # Touching reads the *current step* matrix. If the accumulated one
        # disagrees, the contact exists and only the current-step read misses it.
        from omnigibson.utils.usd_utils import RigidContactAPI

        cur = RigidContactAPI.is_in_contact(
            scene_idx=table.scene.idx, query_set=[moving], with_set=[table],
            ignore_set=None, current_only=True,
        )
        acc = RigidContactAPI.is_in_contact(
            scene_idx=table.scene.idx, query_set=[moving], with_set=[table],
            ignore_set=None, current_only=False,
        )
        print(f"      contact with the table: current_only=True -> {cur}, False -> {acc}")
        any_cur = RigidContactAPI.is_in_contact(
            scene_idx=table.scene.idx, query_set=[moving], with_set=None,
            ignore_set=None, current_only=True,
        )
        vel = [round(float(v), 5) for v in moving.get_linear_velocity()]
        print(f"      apple in contact with ANYTHING this step: {any_cur}")
        print(f"      apple linear velocity: {vel}")
        print(f"      apple kinematic_only={moving.kinematic_only}, "
              f"table kinematic_only={table.kinematic_only}")
        print(f"      apple is_asleep = {moving.is_asleep}, sleep_threshold = {moving.sleep_threshold}")

        # A sleeping actor emits no contact reports, so wake it and step once:
        # if the predicate flips with the object not having moved, sleep is the
        # entire cause and nothing about the placement geometry is wrong.
        moving.wake()
        og.sim.step()
        woke_z = float(moving.get_position_orientation()[0][2])
        print(f"      after wake() + 1 step: is_asleep={moving.is_asleep}, z={woke_z:.3f}, "
              f"Touching={bool(moving.states[object_states.Touching].get_value(table))}, "
              f"OnTop={bool(moving.states[object_states.OnTop].get_value(table))}")

        # And the durable version: an object that is never allowed to sleep.
        moving.sleep_threshold = 0.0
        moving.wake()
        for _ in range(30):
            og.sim.step()
        print(f"      with sleep_threshold=0 after 30 steps: is_asleep={moving.is_asleep}, "
              f"z={float(moving.get_position_orientation()[0][2]):.3f}, "
              f"Touching={bool(moving.states[object_states.Touching].get_value(table))}, "
              f"OnTop={bool(moving.states[object_states.OnTop].get_value(table))}")

        if not cur and acc:
            print("    => the contact is real; only the current-step read misses it. Touching's "
                  "current_only=True is the whole failure.")
        elif not cur and not acc and not any_cur:
            print("    => PhysX reports no contact at all for this apple: it is resting on "
                  "nothing the contact API sees, so no re-read will fix it.")
        print()

    # Park apple_1 far from the table so condition A is a genuinely empty top.
    away = ([float(v) for v in table.get_position_orientation()[0]], None)
    away_xyz = th.tensor(away[0]) + th.tensor([3.0, 3.0, 0.0])
    park(apple_1, (away_xyz, th.tensor([0.0, 0.0, 0.0, 1.0])))
    settle()

    probe(apple_2, args.probe_steps)
    if args.probe_only:
        og.shutdown()
        return 0

    a_ok, a_n = survey("A: table top empty, placing apple_2", apple_2, None)

    # Condition B: apple_1 resting on the table, exactly the state agent_1 met.
    try:
        pose_1 = controller._sample_pose_with_object_and_predicate(
            object_states.OnTop, apple_1, table,
        )
    except Exception as exc:
        print(f"FAIL: could not seat the first apple to build condition B: {exc}")
        return 1
    b_ok, b_n = survey("B: apple_1 already resting on the table, placing apple_2",
                       apple_2, apple_1, other_pose=pose_1)

    print("=" * 70)
    print(f"empty top:          {a_ok}/{a_n} placements come to rest")
    print(f"one apple already:  {b_ok}/{b_n} placements come to rest")
    if a_n and b_n:
        if a_ok == a_n and b_ok < b_n:
            print("=> the neighbour is the cause: placement is reliable only on an empty top.")
        elif a_ok < a_n and b_ok < b_n:
            print("=> not about the neighbour: the sampler returns poses that do not hold "
                  "even on an empty table.")
        elif a_ok == a_n and b_ok == b_n:
            print("=> placement is reliable in both states; the run failure is not reproduced "
                  "here, so look at what the robot adds (release timing, arm contact).")
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
