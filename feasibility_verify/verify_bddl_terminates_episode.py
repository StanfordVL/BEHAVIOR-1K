"""M9 acceptance: BDDL's own goal expression is what ends a COOP2 episode.

Until now ``terminated`` was hardcoded False for every agent and the task was
``DummyTask``, which evaluates nothing: an episode could only ever end by
running out of steps, and a robot could hold the goal object with nothing
noticing. ``--succeed-when-all-hold`` was a stand-in for that; this replaces it
with the activity's real goal.

What this proves, in order:

1. The facade loads the cached ``coop_two_apples_pomaria`` instance, so the
   BehaviorTask's object_scope is bound and both robots are drivable.
2. ``terminated`` is False while the goal is unmet -- otherwise a run would
   stop at step 0 and every metric would be empty.
3. Forcing the goal (both apples onto the coffee table) flips
   ``compiled_task.check_goal``, and the facade reports ``terminated`` True on
   the next tick. This is the half that cannot be inferred: if check_goal never
   flipped, a real run's terminated=False would be indistinguishable from
   "the agents did not manage it".
4. The step at which it happened is recorded, and stays recorded.

Run:
    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/verify_bddl_terminates_episode.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTIVITY = "coop_two_apples_pomaria"
SCENE_MODEL = "Pomaria_1_int"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    from omnigibson import object_states

    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    env = CooperativeBehaviorEnv(
        n_agents=2,
        seed=0,
        scene_model=SCENE_MODEL,
        bddl_activity=ACTIVITY,
        bddl_instance_id=0,
        room="living_room_0",
        length=40000,
    )
    try:
        env.reset()
        task = env.env.task
        print(f"\ntask: {type(task).__name__}  activity={getattr(task, 'activity_name', '?')}")

        print("\n1. the cached instance is loaded and bound")
        check(type(task).__name__ == "BehaviorTask", "a BehaviorTask is loaded, not DummyTask")
        scope = getattr(task, "object_scope", {}) or {}
        unbound = sorted(name for name, entity in scope.items() if entity is None)
        check(not unbound, f"every object_scope entry is bound (unbound: {unbound})")
        check(len(env.env.robots) == 2, f"both robots loaded ({len(env.env.robots)})")

        print("\n2. an unmet goal does not terminate the episode")
        _, _, terminated, truncated, _ = env.step(None)
        check(not any(terminated.values()), f"terminated is False at the start {terminated}")
        check(not any(truncated.values()), "truncated is False at the start")
        check(env.goal_reached_at is None, "no goal step recorded yet")

        print("\n3. forcing the goal flips check_goal and terminates the episode")
        table = scope["coffee_table.n.01_1"]
        for i in (1, 2):
            apple = scope[f"apple.n.01_{i}"]
            placed = apple.states[object_states.OnTop].set_value(table, True)
            check(bool(placed), f"apple.n.01_{i} placed onto the coffee table")

        # set_value writes the pose; the kinematic states it feeds read from the
        # physics view, so the value only becomes visible after a step.
        met = False
        for _ in range(20):
            _, _, terminated, _, _ = env.step(None)
            met = any(terminated.values())
            if met:
                break
        check(met, "terminated becomes True once both apples are on the table")
        check(env.goal_reached_at is not None, f"the goal step is recorded ({env.goal_reached_at})")

        print("\n4. termination is sticky")
        recorded = env.goal_reached_at
        _, _, terminated, _, _ = env.step(None)
        check(any(terminated.values()), "terminated stays True on the next tick")
        check(env.goal_reached_at == recorded, "the recorded step does not move")

        print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S):"))
        for failure in failures:
            print(f"  - {failure}")
        return 1 if failures else 0
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
