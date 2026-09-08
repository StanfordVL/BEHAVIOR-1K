"""Load the cached ``coop_two_apples_pomaria`` template and prove the task is usable.

This is the acceptance test for the sampled instance: it exercises the *cached* path
(``online_object_sampling=False``), which is the one every experiment will use.

The BDDL declares one agent while the env runs two robots -- see the sampling script for
why. Both robots still load (from the template json) and both are drivable; BDDL simply
does not track the second one, which is fine because the goal never mentions an agent.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/verify_coop_task_instance.py
"""

import os
import sys

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

ACTIVITY = "coop_two_apples_pomaria"
SCENE_MODEL = "Pomaria_1_int"
N_ROBOTS = 2  # only agent.n.01_1 is declared in the BDDL

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

CONFIG = {
    "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": None},
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": SCENE_MODEL,
        "seg_map_resolution": 0.1,
        # scene_instance is filled in by BehaviorTask.verify_scene_and_task_config
    },
    # Left empty on purpose: the template json carries agent_0 / agent_1, and _load_robots
    # only falls back to this list when the scene brought no robots of its own.
    "robots": [],
    "objects": [],
    "task": {
        "type": "BehaviorTask",
        "activity_name": ACTIVITY,
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "online_object_sampling": False,
        "use_presampled_robot_pose": False,
    },
}


def main():
    env = og.Environment(configs=CONFIG)
    task = env.task
    failures = []

    print(f"\nloaded scene_instance = {env.scene.scene_file}")
    print(f"robots: {[r.name for r in env.robots]}")

    print("\n--- object scope ---")
    for inst, entity in task.object_scope.items():
        if entity is None:
            print(f"  {inst:24s} -> None")
            failures.append(f"{inst} is unbound")
            continue
        pos = entity.get_position_orientation()[0]
        print(
            f"  {inst:24s} -> {entity.name:26s} model={getattr(entity, 'model', '-'):8s} "
            f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
        )

    if len(env.robots) != N_ROBOTS:
        failures.append(f"expected {N_ROBOTS} robots in the template, got {len(env.robots)}")
    if task.object_scope["agent.n.01_1"] is not env.robots[0]:
        failures.append("agent.n.01_1 is not bound to env.robots[0]")
    if "agent.n.01_2" in task.object_scope:
        failures.append("BDDL declares more than one agent")

    table = task.object_scope["coffee_table.n.01_1"]
    if table.model != "gpkbiw":
        failures.append(f"coffee_table.n.01_1 is {table.model}, expected gpkbiw")
    if task.object_scope["armchair.n.01_1"] is task.object_scope["armchair.n.01_2"]:
        failures.append("both armchair instances resolved to the same object")

    print("\n--- initial kinematics ---")
    for i in (1, 2):
        apple = task.object_scope[f"apple.n.01_{i}"]
        chair = task.object_scope[f"armchair.n.01_{i}"]
        on_chair = apple.states[object_states.OnTop].get_value(chair)
        print(f"  apple.n.01_{i} ontop armchair.n.01_{i}: {on_chair}")
        if not on_chair:
            failures.append(f"apple.n.01_{i} is not on armchair.n.01_{i}")

    goal_met, breakdown = task.compiled_task.check_goal(task._evaluate_predicate)
    print(f"\ngoal at t=0: {goal_met}  {breakdown}")
    if goal_met:
        failures.append("goal is already satisfied at t=0")

    # Now force the goal and re-check: proves the goal is reachable and that check_goal
    # actually tracks it, which is what L1d will hang its score on.
    print("\n--- forcing the goal ---")
    for i in (1, 2):
        apple = task.object_scope[f"apple.n.01_{i}"]
        ok = apple.states[object_states.OnTop].set_value(table, True)
        print(f"  put apple.n.01_{i} on coffee_table.n.01_1: {ok}")
        if not ok:
            failures.append(f"could not place apple.n.01_{i} on the coffee table")
    for _ in range(30):
        og.sim.step()

    goal_met, breakdown = task.compiled_task.check_goal(task._evaluate_predicate)
    print(f"goal after forcing: {goal_met}  {breakdown}")
    if not goal_met:
        failures.append("goal did not register as satisfied after placing both apples")

    print("\n" + ("FAILED:\n  " + "\n  ".join(failures) if failures else "ALL CHECKS PASSED"))
    og.shutdown()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
