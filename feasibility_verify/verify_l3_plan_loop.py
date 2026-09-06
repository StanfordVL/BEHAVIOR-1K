"""Drive COOP2's real plan layer (L3) against CooperativeBehaviorEnv.

Everything up to now has bypassed L3: ``verify_plan_channel.py`` hand-advances a
hard-coded list. This runs the actual ``PlanningEnvWrapper`` -- its ready
barrier, its plan lifecycle, its logging -- with a scripted agent standing in
for the LLM. If this works, the only remaining piece is an agent that *chooses*
the actions instead of reciting them.

What it checks, which the hand-rolled loop could not:

* a plan of arbitrary length runs to completion without the driver advancing it
* the barrier really freezes physics while any agent is un-ready (COOP2 puts it
  at the plan boundary, not the primitive boundary)
* a failure returns the agent to reasoning and a *new* plan is generated
* ``needs_new_plan`` / ``set_unready('plan_terminated')`` fire as L3 expects

Run:
    OMNIGIBSON_HEADLESS=1 python feasibility_verify/verify_l3_plan_loop.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.action.behavior_env_wrapper import BehaviorSymbolicEnvWrapper
from coop2.cognitive.plan import plan_env_wrapper as _plan_env_wrapper

# PlanningEnvWrapper constructs SymbolicEnvWrapper by name inside __init__.
# Swapping the name is a two-line shim; subclassing would mean duplicating the
# whole constructor, and the upstream file is kept byte-identical on purpose.
_plan_env_wrapper.SymbolicEnvWrapper = BehaviorSymbolicEnvWrapper

from coop2.behavior_env.coop_env import CooperativeBehaviorEnv  # noqa: E402
from coop2.cognitive.agent.agent import Agent  # noqa: E402
from coop2.cognitive.plan.plan import SymbolicAction, SymbolicPlan  # noqa: E402
from coop2.cognitive.plan.plan_env_wrapper import PlanningEnvWrapper  # noqa: E402

APPLE = {"type": "DatasetObject", "name": "apple_0", "category": "apple", "model": "agveuv",
         "position": [0.0, 0.0, 0.05], "orientation": [0.0, 0.0, 0.0, 1.0]}


class ScriptedAgent(Agent):
    """Stands in for the LLM: emits a fixed multi-step plan, re-plans on failure.

    The point is not the plan's content but that L3 drives it: the agent hands
    over a list and never touches it again until L3 says it needs a new one.
    """

    def __init__(self, agent_id: str, plans):
        super().__init__(agent_id=agent_id)
        self._scripts = list(plans)
        self._issued = 0
        self.plans_generated = []

    def observe(self, observation, env_step: int):
        self.observation = observation
        self.env_step = env_step

    def generate_plan(self) -> SymbolicPlan:
        script = self._scripts[min(self._issued, len(self._scripts) - 1)]
        self._issued += 1
        actions = [SymbolicAction(action_type=a["action_type"], args={k: v for k, v in a.items() if k != "action_type"})
                   for a in script]
        self.plans_generated.append([a["action_type"] for a in script])
        print(f"  [{self.agent_id}] generated plan #{self._issued}: "
              + " -> ".join(f"{a['action_type']}({a.get('target', '')})" for a in script))
        return SymbolicPlan(
            specification=f"scripted plan {self._issued}",
            actions=actions,
            plan_id=self._issued,
            agent_id=self.agent_id,
            created_at_step=self.env_step,
        )

    def handle_reasoning(self):
        self.plan = self.generate_plan()
        self.set_ready(env_step=self.env_step)

    def handle_interrupt(self):
        self.handle_reasoning()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    base = CooperativeBehaviorEnv(n_agents=1, seed=0, objects=[APPLE], length=40000)
    agent_id = "agent_0"
    try:
        started = time.time()
        base.reset()
        print(f"[timing] env ready in {time.time() - started:.1f}s, room={base.placement_room}")

        # A 3-action plan, then a deliberately impossible one to force a
        # failure -> reasoning -> replan cycle, then a recovery plan.
        scripts = [
            [
                {"action_type": "navigate_to", "target": "apple#1"},
                {"action_type": "grasp", "target": "apple#1"},
                {"action_type": "release"},
            ],
            [{"action_type": "grasp", "target": "ghost#99"}],
            [{"action_type": "navigate_to", "target": "apple#1"}],
        ]
        agent = ScriptedAgent(agent_id, scripts)

        wrapper = PlanningEnvWrapper(base, agent_names=[agent_id], agents={agent_id: agent})
        wrapper.reset()
        agent.handle_reasoning()  # first plan, so the barrier opens

        print(f"\n{'step':>7}  event")
        print("-" * 68)
        frozen_ticks = 0
        last_decisions = base.decision_count
        deadline = time.time() + 600  # a plan that cannot finish in 10 min is stuck
        stall_since = base.decision_count
        stall_ticks = 0
        for step in range(60000):
            if time.time() > deadline:
                print(f"\nWALL-CLOCK LIMIT at env_step {base.current_step}")
                break
            # A plan that keeps issuing primitives but never advances looks
            # identical to a slow one from the outside; the first version of
            # this script ran 30 minutes before a timeout killed it. Catch the
            # loop explicitly instead.
            if base.decision_count > stall_since + 12:
                print(f"\nSTALLED: {base.decision_count - stall_since} primitives issued without the "
                      f"plan advancing past action {agent.plan.current_action_index if agent.plan else '?'}")
                break
            wrapper.step()
            if base.decision_count == last_decisions and not agent.ready:
                frozen_ticks += 1
            last_decisions = base.decision_count

            if agent.plan is not None and agent.plan.current_action_index != getattr(main, "_seen_index", -1):
                main._seen_index = agent.plan.current_action_index
                stall_since = base.decision_count

            if not agent.ready:
                # L3 sent it back to reasoning; the LLM would think here.
                before = base.current_step
                agent.handle_reasoning()
                print(f"{before:>7}  replanned (plan #{agent._issued}), env_step frozen at {before}")
                if agent._issued > len(scripts):
                    break
            if agent._issued >= len(scripts) and agent.plan and agent.plan.is_complete():
                break

        print("\n--- result ---")
        print(f"plans generated : {agent.plans_generated}")
        print(f"decision_count  : {base.decision_count}   env_step: {base.current_step}")
        print(f"held_objects    : {base.world.held_objects()}")
        print(f"ticks with the barrier closed (physics frozen): {frozen_ticks}")
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        base.close()


if __name__ == "__main__":
    main()
