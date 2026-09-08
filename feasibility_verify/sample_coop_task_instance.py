"""Sample and cache the ``coop_two_apples_pomaria`` BDDL task instance on Pomaria_1_int.

Two R1s stand in ``living_room_0``; one apple sits on each of the two armchairs and the
goal is to get both onto ``coffee_table-gpkbiw`` (the whitelist pins that exact model --
the room holds two coffee tables).

The BDDL declares a **single** agent on purpose, even though the env runs two robots:
the goal never references an agent, upstream's BDDLSampler only ever binds
``agent.n.01_1`` (to ``env.robots[0]``), and a second declared agent would need a patch
to ``bddl_utils.py``. Robot poses come from ``coop2.behavior_env.placement`` after
sampling, so the BDDL's ``(ontop agent.n.01_1 floor.n.01_1)`` only has to give the
sampler something satisfiable -- it does not constrain where the robots end up.

Sampling is unseeded, so every run lays the apples out differently; the point of this
script is to freeze **one** layout into a template json that every later experiment loads,
so cross-topology comparisons are not polluted by sampling noise.

Run:
    OMNIGIBSON_HEADLESS=1 python feasibility_verify/sample_coop_task_instance.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm, macros

ACTIVITY = "coop_two_apples_pomaria"
SCENE_MODEL = "Pomaria_1_int"
# living_room_0 holds coffee_table-gcollb and coffee_table-gpkbiw; BDDL only knows the
# synset, so the whitelist is the only way to pin the one we want.
# Format is {synset: {category: {model: None-or-bbox}}}. The BehaviorTask docstring says
# "list of valid models", but the code calls .keys() on it -- it must be a dict.
SAMPLING_WHITELIST = {
    "coffee_table.n.01": {"coffee_table": {"gpkbiw": None}},
    # Apple models are picked per instance, and some of them roll off the armchair
    # cushion during settling. Pin one that stays put so the layout is reproducible.
    "apple.n.01": {"apple": {"omzprq": None}},
}
N_ROBOTS = 2  # robots in the env; only agent.n.01_1 is declared in the BDDL
ROOM_INSTANCE = "living_room_0"

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
macros.systems.micro_particle_system.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = 0.5
macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 5
macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 5


def build_config(instance_id):
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": None},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": SCENE_MODEL,
            "load_room_types": ["living_room"],
            "seg_map_resolution": 0.1,
        },
        "robots": [
            {
                "model": "r1",
                "name": f"agent_{i}",
                "obs_modalities": [],
                "default_reset_mode": "tuck",
                # Parked off-scene; the sampler moves every robot away anyway, and we
                # place them properly once sampling has finished.
                "position": [-50.0 - 2.0 * i, -50.0, 0.0],
            }
            for i in range(N_ROBOTS)
        ],
        "objects": [],
        "task": {
            "type": "BehaviorTask",
            "activity_name": ACTIVITY,
            "activity_definition_id": 0,
            "activity_instance_id": instance_id,
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
            "sampling_whitelist": SAMPLING_WHITELIST,
        },
    }


def report_scope(task):
    print("\n--- object scope ---")
    for inst, entity in task.object_scope.items():
        if entity is None:
            print(f"  {inst:24s} -> None")
            continue
        model = getattr(entity, "model", None)
        pos = entity.get_position_orientation()[0]
        print(f"  {inst:24s} -> {entity.name:28s} model={model}  pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_id", type=int, default=0, help="activity_instance_id to write")
    parser.add_argument("--save_dir", type=str, default=None, help="defaults to the 2026 task-instance dataset")
    parser.add_argument("--no_save", action="store_true", help="sample and report, but do not write the json")
    args = parser.parse_args()

    env = og.Environment(configs=build_config(args.instance_id))

    assert env.task.feedback is None, f"Sampling failed: {env.task.feedback}"
    print(f"\nSampling succeeded for {ACTIVITY} on {SCENE_MODEL}")

    # Verify the whitelist actually pinned the intended coffee table.
    table = env.task.object_scope["coffee_table.n.01_1"]
    assert table.model == "gpkbiw", f"Expected coffee_table-gpkbiw, got {table.category}-{table.model}"

    # The one declared agent must be bound; the other robots are deliberately invisible to BDDL.
    assert env.task.object_scope["agent.n.01_1"] is env.robots[0], "agent.n.01_1 is not env.robots[0]"
    assert "agent.n.01_2" not in env.task.object_scope, "BDDL should declare exactly one agent"
    assert len(env.robots) == N_ROBOTS, f"expected {N_ROBOTS} robots, got {len(env.robots)}"

    og.sim.play()
    env.task.reset(env)
    for _ in range(300):
        og.sim.step()

    # scene.get_random_point() samples the whole floor, which put the robots 25 m away in
    # another room; use the room-scoped placement coop2 already has.
    from coop2.behavior_env.placement import place_robots

    poses, room = place_robots(env, room=ROOM_INSTANCE, seed=args.instance_id)
    print(f"placed {len(poses)} robots in {room}: {[(round(x, 2), round(y, 2)) for x, y in poses]}")
    for _ in range(30):
        og.sim.step()

    report_scope(env.task)

    # NB: do not call check_initial_conditions() here -- the init block contains `inroom`,
    # which has no entry in PREDICATE_TO_STATE and would raise KeyError. Check the two
    # kinematic facts we actually care about directly instead.
    stable = True
    for i in (1, 2):
        apple, chair = env.task.object_scope[f"apple.n.01_{i}"], env.task.object_scope[f"armchair.n.01_{i}"]
        on_chair = apple.states[object_states.OnTop].get_value(chair)
        print(f"apple.n.01_{i} ontop armchair.n.01_{i}: {on_chair}")
        stable = stable and on_chair
    if not stable:
        # Sampling is unseeded; an apple that rolled off the cushion just means this draw
        # is unusable. Exit non-zero and let the caller re-run rather than saving a layout
        # whose own initial conditions are already violated.
        print("\nUNSTABLE LAYOUT -- not saving; re-run to draw another one")
        og.shutdown()
        sys.exit(1)

    goal_met, breakdown = env.task.compiled_task.check_goal(env.task._evaluate_predicate)
    print(f"goal already met       : {goal_met}  {breakdown}   (must be False)")
    assert not goal_met, "Goal is already satisfied at t=0 -- the task would be trivial"

    if args.no_save:
        print("\n--no_save given; nothing written")
    else:
        final_state = env.scene.dump_state()
        env.scene.load_state(final_state)
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
        print(f"\nwrote {os.path.join(save_dir, fname + '.json')}")

    og.shutdown()


if __name__ == "__main__":
    main()
