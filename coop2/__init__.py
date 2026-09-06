"""COOP2 multi-agent cooperation architecture, ported to BEHAVIOR-1K.

See ``coop2/PORTING_PLAN.md`` for the seven-layer design. Package layout
mirrors ``coop2-llm-mas/ma_crafter`` so the two trees stay diffable:

    coop2/behavior_env/   L0-L1   env, world model, text observation,
                                  primitive execution, task tracking
    coop2/cognitive/      L2-L4   action grounding, plan lifecycle, agents
    coop2/comm_topology/  L5      individual / chain / centralized / decentralized
    coop2/experiment/     L6      runners, grid, metrics
"""

__all__ = ["behavior_env"]
