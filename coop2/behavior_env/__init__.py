"""L1 — the BEHAVIOR-1K side of the COOP2 port.

Replaces ``ma_crafter/macrafter``. Sub-layers, per PORTING_PLAN.md:

==========================  ====================================================
module                      layer
==========================  ====================================================
``env_setup``               L0  N-robot config + startup ritual + sanity checks
``primitive_engine``        L1c concurrent primitive execution, idle padding
``symbolic_navigation``     L1  cuRobo-free NAVIGATE_TO for the symbolic set
``symbolic_contention``     L1  proximity gate, travel cost, cross-robot claims
``recording``               --  third-person video off the engine's on_tick hook
``world_state``             L1a entity registry, type-local IDs, room membership
``symbolic_view``           L1b scene -> text, target_hints
``placement``               --  scene-derived robot/object placement
``cooperative_tasks``(todo) L1d TaskState spatial/temporal/dependency/participation
``coop_env`` (todo)         L1  CooperativeBehaviorEnv, the PettingZoo-shaped facade
==========================  ====================================================
"""

from coop2.behavior_env.env_setup import (
    assert_multi_robot_sanity,
    build_multi_robot_config,
    prepare_robots,
    tune_primitive_macros,
)
from coop2.behavior_env.primitive_engine import (
    MotionMode,
    MultiAgentPrimitiveEngine,
    PrimitiveOutcome,
    ReasonCode,
)
from coop2.behavior_env.placement import TraversabilityIndex, place_objects, place_robots
from coop2.behavior_env.recording import ViewerRecorder, chain, enable_viewer_rendering
from coop2.behavior_env.symbolic_contention import (
    DEFAULT_GATED_PRIMITIVES,
    ContentiousSymbolicActionPrimitives,
)
from coop2.behavior_env.symbolic_navigation import NavigableSymbolicActionPrimitives
from coop2.behavior_env.symbolic_view import render_symbolic_view, target_hints
from coop2.behavior_env.world_state import (
    BehaviorWorldState,
    EntityObservation,
    PredicateFact,
    SymbolicObservation,
)

__all__ = [
    "BehaviorWorldState",
    "EntityObservation",
    "PredicateFact",
    "SymbolicObservation",
    "TraversabilityIndex",
    "place_objects",
    "place_robots",
    "render_symbolic_view",
    "target_hints",
    "ViewerRecorder",
    "chain",
    "enable_viewer_rendering",
    "ContentiousSymbolicActionPrimitives",
    "DEFAULT_GATED_PRIMITIVES",
    "NavigableSymbolicActionPrimitives",
    "assert_multi_robot_sanity",
    "build_multi_robot_config",
    "prepare_robots",
    "tune_primitive_macros",
    "MotionMode",
    "MultiAgentPrimitiveEngine",
    "PrimitiveOutcome",
    "ReasonCode",
]
