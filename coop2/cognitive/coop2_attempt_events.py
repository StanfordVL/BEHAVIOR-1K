"""Shared COOP2 attempt-event extraction for result tables and process plots."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .resource_utils import SCORED_RESOURCES, normalize_process_resource


ATTEMPT_CONSTRAINTS = ("spatial", "temporal", "dependency")


def extract_attempt_constraint_events(process_log: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return constraint scores only for concrete task attempts.

    The process log stores per-step state. This function converts it into
    task-attempt rows so downstream summaries do not average over navigation or
    idle planning time. At present, MA-Crafter's observed COOP2 constraints are
    resource-collection constraints, so a task attempt is a collect action that
    can be tied to a known resource task.
    """
    events: List[Dict[str, Any]] = []
    for entry in process_log or []:
        if not isinstance(entry, dict):
            continue
        # Prefer recomputing from the full per-agent process row when possible.
        # Older traces may contain stored attempt rows created before target
        # grounding was strict enough; recomputing keeps plots/tables consistent.
        if entry.get("agents") and entry.get("active_tasks"):
            events.extend(extract_attempt_constraint_events_from_entry(entry))
        else:
            stored = entry.get("constraint_attempts")
            if isinstance(stored, list):
                events.extend(dict(event) for event in stored if isinstance(event, dict))
    return events


def extract_attempt_constraint_events_from_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return grouped attempt events for one process-log step."""
    step = _safe_int(entry.get("env_step"))
    if step is None:
        return []

    tasks = _active_tasks_by_id(entry)
    if not tasks:
        return []

    constraints = _constraints_by_task(entry)
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for agent_view in entry.get("agents") or []:
        if not isinstance(agent_view, dict) or _action_type(agent_view) != "collect":
            continue

        agent_id = str(agent_view.get("agent_id") or "unknown")
        target_ids = _collect_target_ids(agent_view, tasks)
        if not target_ids:
            target_ids = _target_ids_from_collection_outcome(agent_view, tasks)
        if not target_ids:
            continue

        reason = _collect_reason(agent_view)
        for task_id in target_ids:
            task = tasks.get(str(task_id))
            if task is None:
                continue
            success = _collect_succeeded_for_task(agent_view, task)
            key = (str(task_id), "collect")
            event = grouped.setdefault(
                key,
                {
                    "env_step": step,
                    "action_type": "collect",
                    "task_id": str(task_id),
                    "attempt_label": _attempt_label(task),
                    "resource_type": task.get("resource_type"),
                    "target_resource_type": task.get("resource_type"),
                    "target_position": task.get("position"),
                    "required_agents": task.get("required_agents", 1),
                    "required_tool": task.get("required_tool"),
                    "attempting_agents": set(),
                    "successful_agents": set(),
                    "blocked_agents": set(),
                    "block_reasons": set(),
                    "actual_collections": [],
                    "success": False,
                    "constraint_scores": {},
                    "constraint_violations": {},
                },
            )
            event["attempting_agents"].add(agent_id)
            event["actual_collections"].extend(_actual_collection_summaries(agent_view))
            if success:
                event["success"] = True
                event["successful_agents"].add(agent_id)
            else:
                event["blocked_agents"].add(agent_id)
                event["block_reasons"].add(str(reason or _target_miss_reason(agent_view, task)))

    events = []
    for (task_id, _group_action_type), event in sorted(grouped.items(), key=lambda item: item[0]):
        task = tasks[task_id]
        required_agents = max(_safe_int(task.get("required_agents"), 1) or 1, 1)
        event["constraint_scores"] = _attempt_constraint_scores(
            event=event,
            task=task,
            task_constraints=constraints.get(task_id, {}),
            required_agents=required_agents,
        )
        event["constraint_violations"] = {
            name: max(0.0, 1.0 - min(score, 1.0))
            for name, score in event["constraint_scores"].items()
        }
        for key in ("attempting_agents", "successful_agents", "blocked_agents", "block_reasons"):
            event[key] = sorted(event[key])
        event["actual_collections"] = _dedupe_actual_collections(event.get("actual_collections") or [])
        events.append(event)
    return events


def aggregate_attempt_violation_rates(events: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate attempt-event scores into violation-rate columns."""
    event_list = list(events or [])
    totals = {constraint: 0 for constraint in ATTEMPT_CONSTRAINTS}
    deficits = {constraint: 0.0 for constraint in ATTEMPT_CONSTRAINTS}

    for event in event_list:
        scores = event.get("constraint_scores") or {}
        for constraint in ATTEMPT_CONSTRAINTS:
            if constraint not in scores:
                continue
            totals[constraint] += 1
            deficits[constraint] += max(0.0, 1.0 - min(_safe_float(scores[constraint]), 1.0))

    rates: Dict[str, float] = {"attempt_constraint_events": float(len(event_list))}
    for constraint in ATTEMPT_CONSTRAINTS:
        total = totals[constraint]
        violation_rate = deficits[constraint] / total if total else 0.0
        rates[f"{constraint}_violation_rate"] = violation_rate
        rates[f"{constraint}_score"] = 1.0 - violation_rate if total else 0.0
        rates[f"{constraint}_checks"] = total
    return rates


def _attempt_constraint_scores(
    event: Dict[str, Any],
    task: Dict[str, Any],
    task_constraints: Dict[str, Dict[str, Any]],
    required_agents: int,
) -> Dict[str, float]:
    if event.get("success"):
        return {constraint: 1.0 for constraint in ATTEMPT_CONSTRAINTS}

    scores = {}
    for constraint in ATTEMPT_CONSTRAINTS:
        result = task_constraints.get(constraint)
        if result is not None:
            scores[constraint] = _constraint_score(result, required_agents)

    # Temporal is defined at the attempt event: agents issuing collect for this
    # task divided by required participation.
    scores["temporal"] = len(event.get("attempting_agents") or []) / required_agents

    # No-tool tasks have dependency satisfied by definition. This keeps simple
    # wood attempts from looking like dependency failures when no agent is near.
    if not _has_dependency_requirement(task.get("required_tool")):
        scores["dependency"] = 1.0

    for constraint in ATTEMPT_CONSTRAINTS:
        scores.setdefault(constraint, 0.0)
        scores[constraint] = max(0.0, float(scores[constraint]))
    return scores


def _active_tasks_by_id(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    for task in entry.get("active_tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        resource = normalize_process_resource(task.get("resource_type"))
        if task_id is None or resource not in SCORED_RESOURCES:
            continue
        task_copy = dict(task)
        task_copy["task_id"] = str(task_id)
        task_copy["resource_type"] = resource
        task_copy["required_agents"] = max(_safe_int(task.get("required_agents"), 1) or 1, 1)
        tasks[str(task_id)] = task_copy
    return tasks


def _constraints_by_task(entry: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for result in entry.get("observed_constraints") or []:
        if not isinstance(result, dict):
            continue
        task_id = result.get("task_id")
        constraint = result.get("constraint_type")
        if task_id is None or constraint not in ATTEMPT_CONSTRAINTS:
            continue
        grouped[str(task_id)][str(constraint)] = result
    for result in entry.get("grounded_action_constraints") or []:
        if not isinstance(result, dict):
            continue
        task_id = result.get("task_id")
        constraint = result.get("constraint_type")
        if task_id is None or constraint not in ATTEMPT_CONSTRAINTS:
            continue
        current = grouped[str(task_id)].get(str(constraint))
        if current is None or _constraint_score(result, 1) > _constraint_score(current, 1):
            grouped[str(task_id)][str(constraint)] = result
    return grouped


def _collect_target_ids(agent_view: Dict[str, Any], tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    args = _action_args(agent_view)
    target_ids = []
    for key in ("task_id", "resource_id", "target_id", "item_id"):
        value = args.get(key)
        if value is not None and str(value) in tasks:
            target_ids.append(str(value))
    if target_ids:
        return sorted(set(target_ids))

    active_ids = [str(task_id) for task_id in agent_view.get("active_target_ids") or [] if str(task_id) in tasks]
    if len(active_ids) == 1:
        return active_ids
    return []


def _target_ids_from_collection_outcome(
    agent_view: Dict[str, Any],
    tasks: Dict[str, Dict[str, Any]],
) -> List[str]:
    outcome = agent_view.get("action_outcome") or {}
    effects = outcome.get("effects") or {}
    matched_ids: List[str] = []
    for collection in effects.get("collections") or []:
        if not _agent_participated(agent_view.get("agent_id"), collection.get("participating_agents")):
            continue
        collection_pos = _position_tuple(collection.get("position"))
        collection_resource = normalize_process_resource(collection.get("resource_type"))
        for task_id, task in tasks.items():
            if collection_resource and collection_resource != task.get("resource_type"):
                continue
            if collection_pos is not None and collection_pos == _position_tuple(task.get("position")):
                matched_ids.append(task_id)
    return sorted(set(matched_ids))


def _action_type(agent_view: Dict[str, Any]) -> str:
    action = agent_view.get("action") or {}
    symbolic = agent_view.get("symbolic_action") or {}
    return str(action.get("action_type") or symbolic.get("action_type") or "").lower()


def _action_args(agent_view: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for source in (agent_view.get("symbolic_action") or {}, agent_view.get("action") or {}):
        args = source.get("args") if isinstance(source, dict) else None
        if isinstance(args, dict):
            merged.update(args)
        if isinstance(source, dict):
            merged.update({key: value for key, value in source.items() if key not in {"action_type", "args"}})
    return merged


def _collect_succeeded_for_task(agent_view: Dict[str, Any], task: Dict[str, Any]) -> bool:
    outcome = agent_view.get("action_outcome") or {}
    effects = outcome.get("effects") or {}
    if outcome.get("status") != "success":
        return False
    task_pos = _position_tuple(task.get("position"))
    task_resource = normalize_process_resource(task.get("resource_type"))
    for collection in effects.get("collections") or []:
        if not _agent_participated(agent_view.get("agent_id"), collection.get("participating_agents")):
            continue
        collection_resource = normalize_process_resource(collection.get("resource_type"))
        collection_pos = _position_tuple(collection.get("position"))
        if task_resource and collection_resource != task_resource:
            continue
        if task_pos is not None and collection_pos != task_pos:
            continue
        return True
    return False


def _attempt_label(task: Dict[str, Any]) -> str:
    resource = normalize_process_resource(task.get("resource_type")) or "resource"
    task_id = task.get("task_id")
    if task_id is None:
        return f"collect({resource})"
    return f"collect({resource}#{task_id})"


def _actual_collection_summaries(agent_view: Dict[str, Any]) -> List[Dict[str, Any]]:
    outcome = agent_view.get("action_outcome") or {}
    effects = outcome.get("effects") or {}
    summaries = []
    for collection in effects.get("collections") or []:
        if not _agent_participated(agent_view.get("agent_id"), collection.get("participating_agents")):
            continue
        raw_resource = collection.get("resource_type")
        summaries.append(
            {
                "agent_id": str(agent_view.get("agent_id") or "unknown"),
                "resource_type": normalize_process_resource(raw_resource),
                "raw_resource_type": raw_resource,
                "position": collection.get("position"),
            }
        )
    return summaries


def _dedupe_actual_collections(collections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for collection in collections:
        key = (
            collection.get("agent_id"),
            collection.get("resource_type"),
            collection.get("raw_resource_type"),
            tuple(collection.get("position") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(collection)
    return sorted(
        deduped,
        key=lambda item: (
            str(item.get("agent_id") or ""),
            str(item.get("resource_type") or ""),
            str(item.get("raw_resource_type") or ""),
            str(item.get("position") or ""),
        ),
    )


def _collect_reason(agent_view: Dict[str, Any]) -> Optional[str]:
    outcome = agent_view.get("action_outcome") or {}
    return outcome.get("reason") if isinstance(outcome, dict) else None


def _target_miss_reason(agent_view: Dict[str, Any], task: Dict[str, Any]) -> str:
    outcome = agent_view.get("action_outcome") or {}
    effects = outcome.get("effects") or {}
    collections = effects.get("collections") or []
    if not collections:
        return "Target resource was not collected"
    task_resource = normalize_process_resource(task.get("resource_type")) or "target"
    collected = sorted(
        {
            str(collection.get("resource_type") or normalize_process_resource(collection.get("resource_type")) or "unknown")
            for collection in collections
            if _agent_participated(agent_view.get("agent_id"), collection.get("participating_agents"))
        }
    )
    if collected:
        return f"Collected {', '.join(collected)} instead of target {task_resource}"
    return "Target resource was not collected by this agent"


def _constraint_score(result: Dict[str, Any], required_agents: int) -> float:
    score = result.get("score")
    if score is not None:
        return _safe_float(score)
    agents = result.get("agents") or []
    if isinstance(agents, list):
        return len(agents) / max(required_agents, 1)
    return 1.0 if result.get("satisfied") else 0.0


def _has_dependency_requirement(required_tool: Any) -> bool:
    if required_tool is None:
        return False
    if isinstance(required_tool, str):
        return required_tool.strip().lower() not in {"", "none", "null", "[]"}
    if isinstance(required_tool, (list, tuple, set)):
        return any(_has_dependency_requirement(item) for item in required_tool)
    return bool(required_tool)


def _agent_participated(agent_id: Any, participants: Any) -> bool:
    normalized = _normalize_agent_id(agent_id)
    return any(_normalize_agent_id(participant) == normalized for participant in participants or [])


def _normalize_agent_id(agent_id: Any) -> str:
    text = str(agent_id or "")
    if text.startswith("agent_"):
        return text.split("_", 1)[1]
    return text


def _position_tuple(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
