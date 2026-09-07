"""Drive L6's runner against this environment with no API key.

The runner, L5's topology and L4's agent FSM have never been executed here --
only L3 has. This substitutes StubLLMClient for LLMClient.from_env and runs
run_individual_experiment end to end, so a later failure with real credentials
is unambiguously the model rather than the plumbing.

Run:
    OMNIGIBSON_HEADLESS=1 python feasibility_verify/verify_runner_offline.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feasibility_verify.stub_llm import StubLLMClient  # noqa: E402

# TWO substitutions are needed, and forgetting either one fails silently.
#
# PlanningEnvWrapper constructs SymbolicEnvWrapper by name; the crafter one
# converts each symbolic action into a crafter *integer* action id, which this
# facade cannot execute. Patch the name in the consumer's namespace -- the
# module did `from ..action.action_env_wrapper import SymbolicEnvWrapper`, so
# rebinding the source module has no effect.
import coop2.cognitive.action.behavior_env_wrapper as _l2w  # noqa: E402
import coop2.cognitive.agent.llm_client as _llm  # noqa: E402
import coop2.cognitive.plan.plan_env_wrapper as _plan_env_wrapper  # noqa: E402

_plan_env_wrapper.SymbolicEnvWrapper = _l2w.BehaviorSymbolicEnvWrapper
_llm.LLMClient = StubLLMClient
import coop2.experiment.run_individual as run_individual  # noqa: E402

run_individual.LLMClient = StubLLMClient

# The runner inherited crafter's constructor call, which places no objects at
# all. The goal is "pick up an apple", so without these the scene has no apple:
# every plan is rejected by L2 grounding with INVALID_TARGET before a primitive
# is ever assigned, the agent replans every step, and `constraints` stays empty
# because decision_count never leaves zero.
APPLES = [
    {"type": "DatasetObject", "name": f"apple_{i}", "category": "apple",
     "model": "agveuv", "position": [0.4 * i, 0.0, 0.05],
     "orientation": [0.0, 0.0, 0.0, 1.0]}
    for i in range(2)
]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    out = os.path.join(
        os.environ.get("COOP2_OUT", "/tmp/coop2_offline_run"), "individual_stub"
    )
    try:
        run_individual.run_individual_experiment(
            n_agents=2,
            max_steps=1500,
            verbose=True,
            show_viz=False,
            llm_verbose=False,
            goal_instruction="Pick up an apple.",
            time_limit_seconds=600,
            seed=0,
            record_video=False,
            output_root=out,
            objects=APPLES,
        )
        print("\nRUNNER COMPLETED")
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        env = getattr(run_individual, "_last_base_env", None)
        if env is not None:
            print(f"  env_step={env.current_step} decisions={env.decision_count} "
                  f"outcomes_seen={env.outcomes_seen} "
                  f"task_history={len(env.task_tracker.get_history()) if env.task_tracker else 'n/a'}")
        for name in ("coop2_metrics.json", "task_states.json", "plan_log.json"):
            for root, _, files in os.walk(out):
                if name in files:
                    print(f"  wrote {os.path.join(root, name)}")
                    break


if __name__ == "__main__":
    main()
