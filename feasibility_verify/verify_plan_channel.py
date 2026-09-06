"""M5 acceptance: one agent completes "put the apple on the table".

Drives :class:`CooperativeBehaviorEnv` the way COOP2's ``PlanningEnvWrapper``
does -- issue a symbolic action, tick until the engine reports an outcome, read
``info[agent_id]["action_outcome"]``, advance the plan -- without any LLM. If
this works, the only thing between here and a real episode is the cognitive
layer choosing the actions instead of a hard-coded list.

Run:
    OMNIGIBSON_HEADLESS=1 python feasibility_verify/verify_plan_channel.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.coop_env import CooperativeBehaviorEnv

APPLE = {"type": "DatasetObject", "name": "apple_0", "category": "apple", "model": "agveuv",
         "position": [0.0, 0.0, 0.05], "orientation": [0.0, 0.0, 0.0, 1.0]}


def find_receptacle(info, agent_id):
    """A placeable surface from the agent's own target hints."""
    from coop2.behavior_env.symbolic_view import is_receptacle  # noqa: PLC0415

    observation = info[agent_id]["symbolic_world_state"]
    for entity in sorted(observation.entities.values(), key=lambda e: e.entity_id):
        if not entity.is_robot and is_receptacle(entity):
            return entity.entity_id
    return None


def run_action(env, agent_id, action, max_ticks=8000):
    """Issue one symbolic action and tick until it terminates."""
    started = time.time()
    info = None
    before = env.decision_count
    for tick in range(max_ticks):
        _, _, _, _, info = env.step({agent_id: action} if tick == 0 else {})
        if tick == 0 and env.decision_count == before and action["action_type"] not in ("wait", "share"):
            raise RuntimeError(
                f"{action['action_type']} was not issued -- the agent still had a primitive in flight."
            )
        outcome = info[agent_id].get("action_outcome")
        if outcome is not None:
            status = outcome.get("status")
            reason = outcome.get("reason") or outcome.get("failure_reason") or ""
            print(
                f"  {action['action_type']:<14}{status:<9}{outcome.get('ticks', tick):>6} ticks "
                f"{time.time() - started:>6.1f}s  {outcome.get('reason_code', '')}"
            )
            if reason:
                print(f"      {reason.splitlines()[0][:110]}")
            return status == "success", info
    print(f"  {action['action_type']:<14}TIMED OUT after {max_ticks} ticks")
    return False, info


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    env = CooperativeBehaviorEnv(n_agents=1, seed=0, objects=[APPLE], length=40000)
    agent_id = "agent_0"
    try:
        started = time.time()
        _, info = env.reset()
        print(f"[timing] env ready in {time.time() - started:.1f}s, room={env.placement_room}")

        print("\n--- what the agent is told (truncated) ---")
        print("\n".join(info[agent_id]["symbolic_view"].splitlines()[:8]))

        target = find_receptacle(info, agent_id)
        print(f"\n--- plan: navigate_to(apple#1) -> grasp(apple#1) -> navigate_to({target}) -> place_on_top({target})")
        if target is None:
            raise RuntimeError("No receptacle in this room; nothing to place onto.")

        plan = [
            {"action_type": "navigate_to", "target": "apple#1"},
            {"action_type": "grasp", "target": "apple#1"},
            {"action_type": "navigate_to", "target": target},
            {"action_type": "place_on_top", "target": target},
        ]
        print(f"\n{'action':<16}{'status':<9}{'ticks':>6}{'wall':>8}  reason_code")
        print("-" * 72)
        for action in plan:
            success, info = run_action(env, agent_id, action)
            if not success:
                print("\nPLAN FAILED -- stopping here, as PlanningEnvWrapper would.")
                break
        else:
            print("\nPLAN COMPLETE")

        held = env.world.held_objects()
        print(f"\nfinal held_objects: {held}")
        print(f"decision_count={env.decision_count}  env_step={env.current_step}")
        facts = [str(f) for f in info[agent_id]["symbolic_world_state"].facts if "apple" in str(f).lower()]
        print("apple relations: " + (", ".join(facts[:5]) if facts else "(none)"))
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
