"""L1d: cooperative task tracking and the COOP2 constraint metrics.

The metric containers are carried over from ma_crafter/macrafter verbatim
-- TaskStatus, CapabilityChange, StepMetrics, StepTaskSummary,
plot_metrics_timeline, convert_to_serializable and the two log savers.
PORTING_PLAN.md 4 calls for exactly that: keep them unchanged and
compute_constraint_metrics / build_results_table need no edits at all.
Their field names are the contract -- compute_metrics.py reads
metrics.constraint_changes.spatial.improved_tasks by path.

What is rewritten is the tracker and the task itself. A crafter task is a
collectable resource instance at a grid cell; here it is a
(target_object, goal_predicate) pair evaluated against the simulator, and
the four COOP2 constraints map across as:

===============  =========================================  ==========================================
constraint       crafter                                    here
===============  =========================================  ==========================================
spatial          agents within 2 cells >= required_agents   agents within distance_threshold m
temporal         agents issuing collect on one step     agents acting on the task in one macro-step
dependency       agents holding required_tool           the task's precondition predicate holds
participation    participants / required_agents             unchanged
===============  =========================================  ==========================================

Goal predicates are evaluated through OmniGibson's object states, addressed by
BDDL token, so a task goal is written the way an activity definition writes it:
("ontop", "apple.n.01_1", "breakfast_table.n.01_1"). When M9 swaps this
evaluation for compiled_task.check_goal, nothing above L1d changes -- which
is the main reason for skipping BDDL first.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np



class TaskStatus(Enum):
    """Status of a collection task."""
    PENDING = "pending"  # Resource exists, not being collected
    IN_PROGRESS = "in_progress"  # Some agents nearby/attempting
    READY = "ready"  # All requirements met, can collect
    COMPLETED = "completed"  # Resource collected
    FAILED = "failed"  # Collection attempt failed (requirements not met)

@dataclass
class CapabilityChange:
    """Record of a capability change for an agent."""
    agent_id: str
    item_name: str
    old_count: int
    new_count: int
    change_type: str  # "gain" or "loss"
    env_step: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "item_name": self.item_name,
            "old_count": self.old_count,
            "new_count": self.new_count,
            "change_type": self.change_type,
            "env_step": self.env_step
        }

@dataclass
class StepMetrics:
    """Aggregated metrics comparing current step to previous step."""
    env_step: int
    
    # Task set changes
    task_set_size: int = 0  # Current number of tasks
    tasks_added: int = 0  # New tasks this step
    tasks_removed: int = 0  # Tasks removed (collected)
    task_set_delta: int = 0  # Net change in task set size
    
    # Current constraint satisfaction counts (tasks with constraint met)
    spatial_satisfied: int = 0  # Tasks where spatial_count >= required_agents
    temporal_satisfied: int = 0  # Tasks where temporal_count >= required_agents
    dependency_satisfied: int = 0  # Tasks where dependency_ratio >= 1.0
    participation_satisfied: int = 0  # Tasks where participation_ratio >= 1.0
    total_satisfied: int = 0  # Tasks where ALL constraints are met
    
    # Average participation ratio across all tasks
    participation_ratio_avg: float = 0.0
    
    # Constraint changes - count of unique tasks with each constraint type change
    spatial_improved: int = 0  # Tasks where spatial_count increased
    spatial_worsened: int = 0  # Tasks where spatial_count decreased
    temporal_improved: int = 0  # Tasks where temporal_count increased
    temporal_worsened: int = 0  # Tasks where temporal_count decreased
    dependency_improved: int = 0  # Tasks where dependency changed False->True
    dependency_worsened: int = 0  # Tasks where dependency changed True->False
    participation_improved: int = 0  # Tasks where participation_ratio increased
    participation_worsened: int = 0  # Tasks where participation_ratio decreased
    
    # Task IDs for each constraint type (for tracking unique tasks across episode)
    spatial_improved_tasks: List[str] = field(default_factory=list)
    spatial_worsened_tasks: List[str] = field(default_factory=list)
    temporal_improved_tasks: List[str] = field(default_factory=list)
    temporal_worsened_tasks: List[str] = field(default_factory=list)
    dependency_improved_tasks: List[str] = field(default_factory=list)
    dependency_worsened_tasks: List[str] = field(default_factory=list)
    participation_improved_tasks: List[str] = field(default_factory=list)
    participation_worsened_tasks: List[str] = field(default_factory=list)
    
    # Aggregate constraint changes
    total_improved: int = 0  # Total tasks with any positive constraint change
    total_worsened: int = 0  # Total tasks with any negative constraint change
    
    # Capability changes (resources/tools, not health/food/drink/energy)
    capability_increased: int = 0  # Number of capability gains
    capability_decreased: int = 0  # Number of capability losses
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_step": self.env_step,
            "task_set": {
                "size": self.task_set_size,
                "added": self.tasks_added,
                "removed": self.tasks_removed,
                "delta": self.task_set_delta
            },
            "constraint_satisfaction": {
                "spatial": self.spatial_satisfied,
                "temporal": self.temporal_satisfied,
                "dependency": self.dependency_satisfied,
                "participation": self.participation_satisfied,
                "participation_ratio_avg": self.participation_ratio_avg,
                "total": self.total_satisfied
            },
            "constraint_changes": {
                "spatial": {"improved": self.spatial_improved, "worsened": self.spatial_worsened,
                           "improved_tasks": self.spatial_improved_tasks, "worsened_tasks": self.spatial_worsened_tasks},
                "temporal": {"improved": self.temporal_improved, "worsened": self.temporal_worsened,
                            "improved_tasks": self.temporal_improved_tasks, "worsened_tasks": self.temporal_worsened_tasks},
                "dependency": {"improved": self.dependency_improved, "worsened": self.dependency_worsened,
                              "improved_tasks": self.dependency_improved_tasks, "worsened_tasks": self.dependency_worsened_tasks},
                "participation": {"improved": self.participation_improved, "worsened": self.participation_worsened,
                                 "improved_tasks": self.participation_improved_tasks, "worsened_tasks": self.participation_worsened_tasks},
                "total_improved": self.total_improved,
                "total_worsened": self.total_worsened
            },
            "capability_changes": {
                "increased": self.capability_increased,
                "decreased": self.capability_decreased
            }
        }

@dataclass
class StepTaskSummary:
    """Summary of all task states for a single env step."""
    env_step: int
    # Copied verbatim; the annotation named crafter's TaskState, which this
    # module does not define. Widened rather than renamed: to_dict() and the
    # runner read the field by name.
    tasks: Dict[Any, Any] = field(default_factory=dict)
    successful_collections: List[int] = field(default_factory=list)  # resource_ids collected
    failed_attempts: List[int] = field(default_factory=list)  # resource_ids with failed attempts
    metrics: Optional[StepMetrics] = None  # Aggregated metrics for this step
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_step": self.env_step,
            "tasks": {rid: t.to_dict() for rid, t in self.tasks.items()},
            "successful_collections": self.successful_collections,
            "failed_attempts": self.failed_attempts,
            "metrics": self.metrics.to_dict() if self.metrics else None
        }

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj

def save_task_states_log(task_states_history: List[Dict[str, Any]], output_path: str):
    """
    Save task states history to JSON file.
    
    Snapshots come from ``CoopTaskTracker.get_task_states_history()``, whose
    keys are 'env_step', 'tasks' (already-serialised task dicts) and 'metrics'.
    The crafter original read 'step'/'task_states' and dropped 'metrics'
    altogether -- compute_metrics reads constraint changes out of 'metrics', so
    that version wrote a file scoring zero constraints for every run.

    Args:
        task_states_history: List of dicts with 'env_step', 'tasks', 'metrics'
        output_path: Path to save JSON file
    """
    task_states_serializable = []
    for snapshot in task_states_history:
        tasks = snapshot.get('tasks', [])
        if isinstance(tasks, dict):
            tasks = list(tasks.values())
        task_states_serializable.append({
            'step': int(snapshot.get('env_step', snapshot.get('step', 0))),
            'task_states': {
                str(task['task_id']): convert_to_serializable(task) for task in tasks
            },
            'metrics': convert_to_serializable(snapshot.get('metrics')),
        })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(task_states_serializable, f, indent=2)
    print(f"Saved {len(task_states_history)} task state snapshots to {output_path}")

def save_capability_log(capability_history: List['CapabilityChange'], output_path: str):
    """
    Save capability change history to JSON file.
    
    Args:
        capability_history: List of CapabilityChange objects
        output_path: Path to save JSON file
    """
    capability_data = [convert_to_serializable(change.to_dict()) for change in capability_history]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(capability_data, f, indent=2)
    print(f"Saved {len(capability_history)} capability changes to {output_path}")

def plot_metrics_timeline(metrics_list: List['StepMetrics'], output_path: Optional[str] = None, show: bool = True):
    """
    Plot cooperative task metrics timeline in row-based format.
    
    Args:
        metrics_list: List of StepMetrics from task tracker history
        output_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - only this function plots
    import matplotlib.patches as mpatches  # noqa: PLC0415 - legend handles below
    if not metrics_list:
        print("No metrics to plot")
        return
    
    # Use all metrics (including step 1 with initial task set)
    metrics_to_plot = metrics_list
    
    if not metrics_to_plot:
        print("Not enough metrics to plot")
        return
    
    # Extract data
    steps = [m.env_step for m in metrics_to_plot]
    task_set_delta = [m.task_set_delta for m in metrics_to_plot]
    total_improved = [m.total_improved for m in metrics_to_plot]
    total_worsened = [m.total_worsened for m in metrics_to_plot]
    capability_increased = [m.capability_increased for m in metrics_to_plot]
    capability_decreased = [m.capability_decreased for m in metrics_to_plot]
    
    # Individual constraint types
    spatial_improved = [m.spatial_improved for m in metrics_to_plot]
    spatial_worsened = [m.spatial_worsened for m in metrics_to_plot]
    temporal_improved = [m.temporal_improved for m in metrics_to_plot]
    temporal_worsened = [m.temporal_worsened for m in metrics_to_plot]
    dependency_improved = [m.dependency_improved for m in metrics_to_plot]
    dependency_worsened = [m.dependency_worsened for m in metrics_to_plot]
    participation_improved = [m.participation_improved for m in metrics_to_plot]
    participation_worsened = [m.participation_worsened for m in metrics_to_plot]
    
    # Satisfaction counts
    total_satisfied = [m.total_satisfied for m in metrics_to_plot]
    spatial_satisfied = [m.spatial_satisfied for m in metrics_to_plot]
    temporal_satisfied = [m.temporal_satisfied for m in metrics_to_plot]
    dependency_satisfied = [m.dependency_satisfied for m in metrics_to_plot]
    participation_satisfied = [m.participation_satisfied for m in metrics_to_plot]
    participation_ratio_avg = [m.participation_ratio_avg for m in metrics_to_plot]
    
    # Create visualization - 8 rows
    fig, axes = plt.subplots(8, 1, figsize=(16, 12), sharex=True)
    
    # Row 1: Env Step
    for step in steps:
        axes[0].barh(0, 1, left=step, height=0.6, color='lightsteelblue', edgecolor='white', linewidth=0.5)
        axes[0].text(step + 0.5, 0, str(step), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Row 2: Task Set Change (delta)
    for i, step in enumerate(steps):
        delta = task_set_delta[i]
        color = 'green' if delta > 0 else 'red' if delta < 0 else 'gray'
        axes[1].barh(0, 1, left=step, height=0.6, color=color, edgecolor='white', linewidth=0.5, alpha=0.7)
        if delta != 0:
            axes[1].text(step + 0.5, 0, str(delta), ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Helper function for combined improved/worsened rows (2 levels)
    def plot_combined_row(ax, improved_data, worsened_data):
        for i, step in enumerate(steps):
            imp = improved_data[i]
            wor = worsened_data[i]
            # Upper half: improved (green)
            color_imp = 'green' if imp > 0 else 'lightgray'
            alpha_imp = 0.3 + (0.7 * min(imp / 10, 1.0)) if imp > 0 else 0.3
            ax.barh(0.25, 1, left=step, height=0.4, color=color_imp, edgecolor='white', linewidth=0.5, alpha=alpha_imp)
            if imp > 0:
                ax.text(step + 0.5, 0.25, str(imp), ha='center', va='center', fontsize=7, fontweight='bold', color='white')
            # Lower half: worsened (red)
            color_wor = 'red' if wor > 0 else 'lightgray'
            alpha_wor = 0.3 + (0.7 * min(wor / 10, 1.0)) if wor > 0 else 0.3
            ax.barh(-0.25, 1, left=step, height=0.4, color=color_wor, edgecolor='white', linewidth=0.5, alpha=alpha_wor)
            if wor > 0:
                ax.text(step + 0.5, -0.25, str(wor), ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Helper function for 3-level rows: satisfied (top), improved (middle), worsened (bottom)
    def plot_triple_row(ax, sat_data, improved_data, worsened_data):
        for i, step in enumerate(steps):
            sat = sat_data[i]
            imp = improved_data[i]
            wor = worsened_data[i]
            # Top third: satisfied count (blue)
            color_sat = 'steelblue' if sat > 0 else 'lightgray'
            alpha_sat = 0.3 + (0.7 * min(sat / 100, 1.0)) if sat > 0 else 0.3
            ax.barh(0.33, 1, left=step, height=0.28, color=color_sat, edgecolor='white', linewidth=0.5, alpha=alpha_sat)
            if sat > 0:
                ax.text(step + 0.5, 0.33, str(sat), ha='center', va='center', fontsize=6, fontweight='bold', color='white')
            # Middle third: improved (green)
            color_imp = 'green' if imp > 0 else 'lightgray'
            alpha_imp = 0.3 + (0.7 * min(imp / 10, 1.0)) if imp > 0 else 0.3
            ax.barh(0, 1, left=step, height=0.28, color=color_imp, edgecolor='white', linewidth=0.5, alpha=alpha_imp)
            if imp > 0:
                ax.text(step + 0.5, 0, str(imp), ha='center', va='center', fontsize=6, fontweight='bold', color='white')
            # Bottom third: worsened (red)
            color_wor = 'red' if wor > 0 else 'lightgray'
            alpha_wor = 0.3 + (0.7 * min(wor / 10, 1.0)) if wor > 0 else 0.3
            ax.barh(-0.33, 1, left=step, height=0.28, color=color_wor, edgecolor='white', linewidth=0.5, alpha=alpha_wor)
            if wor > 0:
                ax.text(step + 0.5, -0.33, str(wor), ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    
    # Row 3: Capability Increased/Decreased (combined, 2 levels)
    plot_combined_row(axes[2], capability_increased, capability_decreased)
    
    # Row 4: Tasks (improved/worsened only, 2 levels)
    plot_combined_row(axes[3], total_improved, total_worsened)
    
    # Row 5: Spatial (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[4], spatial_satisfied, spatial_improved, spatial_worsened)
    
    # Row 6: Temporal (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[5], temporal_satisfied, temporal_improved, temporal_worsened)
    
    # Row 7: Dependency (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[6], dependency_satisfied, dependency_improved, dependency_worsened)
    
    # Row 8: Participation (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[7], participation_satisfied, participation_improved, participation_worsened)
    
    # Configure axes
    axes[0].set_ylabel('Env Step', fontsize=9, fontweight='bold')
    axes[1].set_ylabel('Task Set\n(+added/-removed)', fontsize=9, fontweight='bold')
    axes[2].set_ylabel('Capability\n(+gained/-lost)', fontsize=9, fontweight='bold')
    axes[3].set_ylabel('# Tasks\n(any +/-)', fontsize=9, fontweight='bold')
    axes[4].set_ylabel('# Tasks\n(spatial sat/+/-)', fontsize=9, fontweight='bold')
    axes[5].set_ylabel('# Tasks\n(temporal sat/+/-)', fontsize=9, fontweight='bold')
    axes[6].set_ylabel('# Tasks\n(depend. sat/+/-)', fontsize=9, fontweight='bold')
    axes[7].set_ylabel('# Tasks\n(particip. sat/+/-)', fontsize=9, fontweight='bold')
    
    for ax in axes:
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)
        ax.set_ylim(-0.5, 0.5)
    
    axes[-1].set_xlabel('Environment Step', fontsize=12)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='steelblue', alpha=0.7, label='Satisfied (top)'),
        mpatches.Patch(color='green', alpha=0.7, label='Improved (middle)'),
        mpatches.Patch(color='red', alpha=0.7, label='Worsened (bottom)'),
        mpatches.Patch(color='lightgray', alpha=0.5, label='Zero'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle('Cooperative Task Metrics Timeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


@dataclass
class BehaviorTaskState:
    """One cooperative task: get @goal true for @target.

    Replaces crafter's resource-instance task. ``goal`` is a BDDL-token
    predicate over entity ids, e.g. ``("ontop", "apple.n.01_1",
    "breakfast_table.n.01_1")``, so a task reads the way an activity definition
    writes it and M9 can hand the same tuple to ``check_goal``.
    """

    task_id: str
    target_id: str
    goal: Tuple[str, ...]
    env_step: int = 0
    required_agents: int = 1
    #: Predicate that must hold before the goal is achievable at all -- the
    #: dependency constraint. crafter expressed this as "hold this tool"; here
    #: it is a predicate, which is what lets "open the fridge first" be stated.
    precondition: Optional[Tuple[str, ...]] = None
    distance_threshold: float = 2.0

    agents_nearby: List[str] = field(default_factory=list)
    spatial_count: int = 0
    acting_agents: List[str] = field(default_factory=list)
    temporal_count: int = 0
    dependency_met: bool = True
    participating_agents: List[str] = field(default_factory=list)
    participation_count: int = 0
    participation_ratio: float = 0.0
    satisfied: bool = False
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_id": self.target_id,
            "goal": list(self.goal),
            "env_step": self.env_step,
            "required_agents": self.required_agents,
            "precondition": list(self.precondition) if self.precondition else None,
            "agents_nearby": list(self.agents_nearby),
            "spatial_count": self.spatial_count,
            "acting_agents": list(self.acting_agents),
            "temporal_count": self.temporal_count,
            "dependency_met": self.dependency_met,
            "participating_agents": list(self.participating_agents),
            "participation_count": self.participation_count,
            "participation_ratio": self.participation_ratio,
            "satisfied": self.satisfied,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
        }


#: Tokens meaning "some agent has this in a gripper". BDDL has no unary
#: `holding` predicate -- it writes inhandofrobot(obj, agent) -- but the
#: cooperative tasks ported from crafter are stated over the object alone.
_HELD_TOKENS = frozenset({"holding", "held", "inhandofrobot"})


class CoopTaskTracker:
    """Evaluates the tasks each macro-step and produces :class:`StepMetrics`.

    Args:
        world: :class:`~coop2.behavior_env.world_state.BehaviorWorldState`,
            which supplies entity ids, positions and the fact list.
        tasks: the task set. Ids in ``goal`` / ``target_id`` are BDDL entity ids.

    ``step()`` is called once per macro-step, not per tick: the constraints are
    defined over decisions, and ``decision_count`` -- not ticks -- is COOP2's
    metric denominator.
    """

    def __init__(self, world, tasks: Sequence[BehaviorTaskState] = ()):
        self.world = world
        self.tasks: Dict[str, BehaviorTaskState] = {task.task_id: task for task in tasks}
        self.metrics_history: List[StepMetrics] = []
        self.task_history: List[StepTaskSummary] = []
        self.task_states_history: List[Dict[str, Any]] = []
        self.capability_history: List[CapabilityChange] = []
        self._previous: Dict[str, BehaviorTaskState] = {}
        self._previous_capability: Dict[str, int] = {}

    # -- goal evaluation ---------------------------------------------------

    def _holds(self, predicate: Optional[Tuple[str, ...]]) -> bool:
        """Is @predicate true right now?

        Binary predicates are evaluated one at a time through the world model.
        This used to build the set of *every* true relation in the scene and
        test membership, which is quadratic in the entity count: 9.4 s per
        macro-step on house_single_floor, to answer at most a few questions.
        """
        if predicate is None:
            return True
        if len(predicate) == 2:  # unary: (token, entity)
            token, entity = predicate
            observation_entities = self.world.entities()
            entity_obs = observation_entities.get(entity)
            if entity_obs is None:
                return False
            # "holding" is not an object state -- it is the grasp relation,
            # carried on the observation as held_by. Without this the default
            # ("holding", "apple.n.01_1") task could never be satisfied: an
            # agent held the apple for a thousand steps while every snapshot
            # recorded satisfied=False, so success was invisible to the metrics
            # and the episode had no reason to stop.
            if token in _HELD_TOKENS:
                return getattr(entity_obs, "held_by", None) is not None
            # Other unary predicates live on the entity's own state dict, under
            # the OmniGibson state class name.
            for name, value in entity_obs.states.items():
                if self.world.predicate_token(name) == token:
                    return bool(value)
            return False
        if len(predicate) != 3:
            return False
        return self.world.relation_holds(*predicate)

    # -- stepping ----------------------------------------------------------

    def step(self, env_step: int, acting: Optional[Dict[str, str]] = None) -> StepMetrics:
        """Refresh every task and diff against the previous macro-step.

        Args:
            env_step: tick index, for the record.
            acting: ``{agent_id: task_id}`` -- who issued a primitive against
                which task this macro-step. This is the temporal constraint's
                only possible source: "acting simultaneously" is not visible in
                world state, only in what was decided.
        """
        acting = acting or {}
        entities = self.world.entities()
        robots = {name: entities[eid] for eid, name in ((e.entity_id, e.name) for e in entities.values())
                  if entities[eid].is_robot}

        for task in self.tasks.values():
            target = entities.get(task.target_id)
            nearby: List[str] = []
            if target is not None:
                for entity in entities.values():
                    if not entity.is_robot:
                        continue
                    distance = math.hypot(
                        entity.position[0] - target.position[0], entity.position[1] - target.position[1]
                    )
                    if distance <= task.distance_threshold:
                        nearby.append(entity.name)

            task.agents_nearby = sorted(nearby)
            task.spatial_count = len(nearby)
            task.acting_agents = sorted(a for a, t in acting.items() if t == task.task_id)
            task.temporal_count = len(task.acting_agents)
            task.dependency_met = self._holds(task.precondition)
            task.participating_agents = sorted(set(task.agents_nearby) | set(task.acting_agents))
            task.participation_count = len(task.participating_agents)
            task.participation_ratio = (
                task.participation_count / task.required_agents if task.required_agents else 0.0
            )
            task.satisfied = self._holds(task.goal)
            task.status = TaskStatus.COMPLETED if task.satisfied else TaskStatus.PENDING

        metrics = self._diff(env_step)
        self.metrics_history.append(metrics)
        self.task_history.append(
            StepTaskSummary(
                env_step=env_step,
                # Snapshot as *copies*, not dicts and not references. The
                # saver both diffs these by attribute (curr.spatial_count) and
                # calls .to_dict() on them, so dicts break it; and the live
                # task objects are rewritten every step, so references would
                # make every historical snapshot show the latest values.
                tasks={
                    task_id: BehaviorTaskState(**{**task.__dict__})
                    for task_id, task in self.tasks.items()
                },
                successful_collections=[t.task_id for t in self.tasks.values() if t.satisfied],
                failed_attempts=[],
                metrics=metrics,
            )
        )
        self.task_states_history.append(
            {
                "env_step": env_step,
                "tasks": [task.to_dict() for task in self.tasks.values()],
                "metrics": metrics.to_dict(),
            }
        )
        self._previous = {
            task_id: BehaviorTaskState(**{**task.__dict__}) for task_id, task in self.tasks.items()
        }
        return metrics

    def _diff(self, env_step: int) -> StepMetrics:
        """Improved/worsened per constraint, against the previous macro-step."""
        metrics = StepMetrics(env_step=env_step)
        metrics.task_set_size = len(self.tasks)

        for task_id, task in self.tasks.items():
            before = self._previous.get(task_id)
            if task.spatial_count >= task.required_agents:
                metrics.spatial_satisfied += 1
            if task.temporal_count >= task.required_agents:
                metrics.temporal_satisfied += 1
            if task.dependency_met:
                metrics.dependency_satisfied += 1
            if task.participation_ratio >= 1.0:
                metrics.participation_satisfied += 1
            if task.satisfied:
                metrics.total_satisfied += 1

            if before is None:
                continue
            if task.spatial_count > before.spatial_count:
                metrics.spatial_improved += 1
                metrics.spatial_improved_tasks.append(task_id)
            elif task.spatial_count < before.spatial_count:
                metrics.spatial_worsened += 1
                metrics.spatial_worsened_tasks.append(task_id)
            if task.temporal_count > before.temporal_count:
                metrics.temporal_improved += 1
                metrics.temporal_improved_tasks.append(task_id)
            elif task.temporal_count < before.temporal_count:
                metrics.temporal_worsened += 1
                metrics.temporal_worsened_tasks.append(task_id)
            if task.dependency_met and not before.dependency_met:
                metrics.dependency_improved += 1
                metrics.dependency_improved_tasks.append(task_id)
            elif before.dependency_met and not task.dependency_met:
                metrics.dependency_worsened += 1
                metrics.dependency_worsened_tasks.append(task_id)
            if task.participation_ratio > before.participation_ratio:
                metrics.participation_improved += 1
                metrics.participation_improved_tasks.append(task_id)
            elif task.participation_ratio < before.participation_ratio:
                metrics.participation_worsened += 1
                metrics.participation_worsened_tasks.append(task_id)

        if self.tasks:
            metrics.participation_ratio_avg = sum(
                t.participation_ratio for t in self.tasks.values()
            ) / len(self.tasks)
        improved = (
            set(metrics.spatial_improved_tasks)
            | set(metrics.temporal_improved_tasks)
            | set(metrics.dependency_improved_tasks)
            | set(metrics.participation_improved_tasks)
        )
        worsened = (
            set(metrics.spatial_worsened_tasks)
            | set(metrics.temporal_worsened_tasks)
            | set(metrics.dependency_worsened_tasks)
            | set(metrics.participation_worsened_tasks)
        )
        metrics.total_improved = len(improved)
        metrics.total_worsened = len(worsened)
        return metrics

    # -- what plan_log_saver reads ----------------------------------------

    def get_history(self) -> List[StepTaskSummary]:
        """Per-step summaries, as L6's runner expects.

        ``run_individual`` does
        ``[s.metrics for s in task_tracker.get_history() if s.metrics]``, so the
        summaries must carry a populated ``metrics`` -- the name is the contract.
        """
        return list(self.task_history)

    def get_task_states_history(self) -> List[Dict[str, Any]]:
        return list(self.task_states_history)

    def get_capability_history(self) -> List[CapabilityChange]:
        return list(self.capability_history)

    def summary(self) -> Dict[str, Any]:
        satisfied = sum(1 for task in self.tasks.values() if task.satisfied)
        return {
            "total": len(self.tasks),
            "satisfied": satisfied,
            "score": satisfied / len(self.tasks) if self.tasks else 0.0,
        }
