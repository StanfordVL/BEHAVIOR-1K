"""Sample and cache the ``coop_nine_apples_hall`` instance, the native way.

Nine straight chairs and one ``coffee_table-cjjayg`` in ``hall_glass_ceiling``'s
``empty_room_0``, an apple on each chair, goal = all nine apples on the table.
Intended for nine R1s (``--agents 9``).

Follows the documented BEHAVIOR procedure
(behavior.stanford.edu/behavior_components/behavior_tasks.html): build a config
with ``online_object_sampling: True``, construct the ``Environment`` -- which is
where BDDLSampler places every object named in the ``:init`` conditions -- and
call ``env.task.save_task()``. Nothing here arranges the scene by hand; the
layout is whatever the sampler produced.

Two things are *checked* rather than fixed, because caching a layout that
violates its own init conditions is worse than caching nothing:

* every apple must still be on its own chair after a settle, and
* the goal must not already hold at t=0.

The task objects are woken before those checks. ``OnTop`` is ``Touching``,
``Touching`` is a contact-report query, and a sleeping PhysX actor reports no
contacts -- so a correctly seated apple that has gone to sleep reads as *not* on
its chair, and the check would throw away a perfectly good layout.

One post-step is coop2-specific and not part of sampling: the robot entries are
stripped from the saved json. ``save_task`` dumps whatever was loaded and the
template *is* the scene file every later run loads, so leaving them in would make
every episode run the sampler's robots and would pin the robot count, breaking
``--agents``.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/sample_nine_apples_hall.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm, macros

from feasibility_verify.sample_coop_task_instance import strip_robots_from_template

ACTIVITY = "coop_nine_apples_hall"
SCENE_MODEL = "hall_glass_ceiling"
N_CHAIRS = 9
#: Robots in the *sampling* env. Only ``agent.n.01_1`` is declared in the BDDL
#: and only it is ever bound, so one is enough; the episode's robot count comes
#: from ``--agents`` at run time, which is why the robots are stripped below.
N_ROBOTS = 1

#: ``{synset: {category: {model: None-or-bbox}}}`` -- a dict, despite the
#: BehaviorTask docstring describing a list of valid models (bddl_utils calls
#: ``.keys()`` on it). The coffee table is pinned because the task names that
#: exact model; the apple and chair are pinned so a re-sample differs only in
#: layout rather than in which objects exist.
SAMPLING_WHITELIST = {
    "coffee_table.n.01": {"coffee_table": {"cjjayg": None}},
    "apple.n.01": {"apple": {"omzprq": None}},
    "straight_chair.n.01": {"straight_chair": {"amgwaw": None}},
}

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 5
macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 5


def build_config(instance_id):
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": None},
        "scene": {
            "type": "InteractiveTraversableScene",
            # No load_room_types filter. It bakes into the template, which *is*
            # the scene file every later run loads, and floors are not exempt
            # from the room filter (STRUCTURE_CATEGORIES - GROUND_CATEGORIES),
            # so filtering here leaves the rest of the building with no ground.
            "scene_model": SCENE_MODEL,
            "seg_map_resolution": 0.1,
        },
        "robots": [
            {
                "model": "r1",
                "name": f"agent_{i}",
                "obs_modalities": [],
                "default_reset_mode": "tuck",
                # Parked outside the floor plan so it is never in the sampler's
                # way; stripped from the template afterwards regardless.
                "position": [-80.0 - 2.0 * i, -80.0, 0.05],
            }
            for i in range(N_ROBOTS)
        ],
        "task": {
            "type": "BehaviorTask",
            "activity_name": ACTIVITY,
            "activity_definition_id": 0,
            "activity_instance_id": instance_id,
            "online_object_sampling": True,
            "sampling_whitelist": SAMPLING_WHITELIST,
            "use_presampled_robot_pose": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--settle", type=int, default=300)
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    import torch as th

    env = og.Environment(configs=build_config(args.instance_id))
    assert env.task.feedback is None, f"Sampling failed: {env.task.feedback}"
    print(f"\nSampling succeeded for {ACTIVITY} on {SCENE_MODEL}")

    scope = env.task.object_scope
    table = scope["coffee_table.n.01_1"]
    assert table.model == "cjjayg", f"expected coffee_table-cjjayg, got {table.category}-{table.model}"
    assert scope["agent.n.01_1"] is env.robots[0], "agent.n.01_1 is not env.robots[0]"
    chairs = [scope[f"straight_chair.n.01_{i}"] for i in range(1, N_CHAIRS + 1)]
    apples = [scope[f"apple.n.01_{i}"] for i in range(1, N_CHAIRS + 1)]
    print(f"table  {table.name} ({table.category}-{table.model})")
    print(f"chairs {[c.name for c in chairs]}")

    og.sim.play()
    env.task.reset(env)

    # Wake the task objects, or the checks below read a lie.
    robot_names = {r.name for r in env.robots}
    for entity in scope.values():
        obj = getattr(entity, "wrapped_obj", entity)
        if obj is None or getattr(obj, "name", None) in robot_names:
            continue
        if getattr(obj, "kinematic_only", True):
            continue
        try:
            obj.sleep_threshold = 0.0
            obj.wake()
        except Exception:  # noqa: BLE001 - never block sampling on this
            pass

    for _ in range(args.settle):
        og.sim.step()

    print(f"\nlayout after a {args.settle}-step settle:")
    table_xy = table.get_position_orientation()[0][:2]
    print(f"  coffee_table.n.01_1 at ({float(table_xy[0]):+.2f}, {float(table_xy[1]):+.2f})")
    stable = True
    distances = []
    for index, (apple, chair) in enumerate(zip(apples, chairs), start=1):
        on_chair = bool(apple.states[object_states.OnTop].get_value(chair))
        chair_xy = chair.get_position_orientation()[0][:2]
        distance = float(th.norm(chair_xy - table_xy))
        distances.append(distance)
        print(f"  apple.n.01_{index} ontop straight_chair.n.01_{index}: {on_chair!s:<5} "
              f"chair at ({float(chair_xy[0]):+7.2f}, {float(chair_xy[1]):+7.2f}), "
              f"{distance:6.2f} m from the table")
        stable = stable and on_chair
    if distances:
        # 60 travel ticks per metre, each way, so this is the number that decides
        # whether the activity is finishable inside a step budget at all.
        total = 2 * sum(distances)
        print(f"\n  chair-to-table: min {min(distances):.2f} m, max {max(distances):.2f} m; "
              f"all nine round trips = {total:.0f} m ~ {60 * total:.0f} travel ticks "
              f"of work to divide between the agents")

    if not stable:
        print("\nUNSTABLE LAYOUT -- not saving; re-run to draw another one")
        og.shutdown()
        sys.exit(1)

    goal_met, breakdown = env.task.compiled_task.check_goal(env.task._evaluate_predicate)
    print(f"\ngoal already met: {goal_met}  {breakdown}   (must be False)")
    assert not goal_met, "the goal holds at t=0 -- the task would be trivial"

    if args.no_save:
        print("\n--no_save given; nothing written")
    else:
        env.scene.update_initial_file()
        env.task.save_task(env=env, save_dir=args.save_dir, override=True, task_relevant_only=False)
        fname = env.task.get_cached_activity_scene_filename(
            scene_model=SCENE_MODEL,
            activity_name=ACTIVITY,
            activity_definition_id=0,
            activity_instance_id=args.instance_id,
        )
        save_dir = args.save_dir or os.path.join(
            gm.DATA_PATH, "2026-challenge-task-instances", "scenes", SCENE_MODEL, "json"
        )
        written = os.path.join(save_dir, fname + ".json")
        removed = strip_robots_from_template(written)
        print(f"\nwrote {written}")
        print(f"stripped {len(removed)} robot entries (coop2 brings its own): {removed}")

    og.shutdown()


if __name__ == "__main__":
    main()
