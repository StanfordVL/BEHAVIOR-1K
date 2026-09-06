"""COOP2 process trace builder for plan-execution case studies."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..resource_utils import (
    RESOURCE_TARGET_PATTERN,
    SCORED_RESOURCES,
    normalize_process_resource,
    process_resource_from_text,
)
from ..coop2_attempt_events import extract_attempt_constraint_events_from_entry


class Coop2ProcessLogger:
    """Build compact per-step COOP2 traces from wrapper/env state."""

    def __init__(
        self,
        symbolic_env: Any,
        agent_names: List[str],
        agents: Dict[str, Any],
        coop2_adapter: Any,
    ):
        self.symbolic_env = symbolic_env
        self.agent_names = agent_names
        self.agents = agents
        self.coop2_adapter = coop2_adapter
        self.records: List[Dict[str, Any]] = []
        self._task_cache: Dict[str, Dict[str, Any]] = {}
        self._last_score_event_count = 0

    def reset(self, info: Optional[Dict[str, Any]] = None) -> None:
        """Clear previous episode records and seed the task cache from reset info."""
        self.records = []
        self._task_cache = {}
        self._last_score_event_count = 0
        if info is not None:
            self._update_task_cache(info)

    def build_agent_views(self, actions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Snapshot the active plan/action selected for this primitive step."""
        views = []
        for agent_id in self.agent_names:
            agent = self.agents.get(agent_id)
            plan = getattr(agent, "plan", None) if agent is not None else None
            action = actions.get(agent_id) or {"action_type": "noop"}
            current_action = plan.get_current_action() if plan is not None else None
            plan_status = self._enum_value(getattr(plan, "status", None)) if plan is not None else None
            specification = getattr(plan, "specification", "") if plan is not None else ""
            active_target_ids = self._extract_target_ids(plan)
            views.append(
                {
                    "agent_id": agent_id,
                    "plan_id": getattr(plan, "plan_id", None) if plan is not None else None,
                    "specification": specification,
                    "plan_status": plan_status,
                    "current_action_index": getattr(plan, "current_action_index", None) if plan is not None else None,
                    "action": dict(action),
                    "symbolic_action": current_action.to_dict() if current_action is not None else None,
                    "resource_focus": self._action_focus(action, specification),
                    "active_target_ids": sorted(active_target_ids),
                }
            )
        return views

    def record_step(
        self,
        env_step: int,
        info: Dict[str, Any],
        agent_views: List[Dict[str, Any]],
    ) -> None:
        """Record one process row after a real environment step completes."""
        self._update_task_cache(info)
        action_outcomes = self._action_outcomes(info)
        observed_constraints = self._observed_constraints(info)
        score_summary, score_events = self._score_events()

        for view in agent_views:
            agent_id = view["agent_id"]
            view["action_outcome"] = action_outcomes.get(agent_id)
            agent = self.agents.get(agent_id)
            plan = getattr(agent, "plan", None) if agent is not None else None
            if plan is not None and getattr(plan, "plan_id", None) == view.get("plan_id"):
                view["plan_status_after_step"] = self._enum_value(getattr(plan, "status", None))
                view["current_action_index_after_step"] = getattr(plan, "current_action_index", None)
            else:
                view["plan_status_after_step"] = view.get("plan_status")
                view["current_action_index_after_step"] = view.get("current_action_index")

        record = {
            "env_step": env_step,
            "agents": agent_views,
            "active_tasks": self._active_tasks(agent_views),
            "observed_constraints": observed_constraints,
            "score": score_summary,
            "score_events": score_events,
        }
        record["constraint_attempts"] = extract_attempt_constraint_events_from_entry(record)
        self.records.append(record)

    def save(self, output_path: str) -> None:
        """Save the compact per-step COOP2 process trace."""
        if not self.records:
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._json_safe(self.records), f, indent=2)
        print(f"Saved {len(self.records)} COOP2 process records to {output_path}")

    def _update_task_cache(self, info: Dict[str, Any]) -> None:
        task_states = self._first_info_value(info, "task_states") or {}
        if not isinstance(task_states, dict):
            return
        for task_id, task in task_states.items():
            task_dict = task.to_dict() if hasattr(task, "to_dict") else task
            if isinstance(task_dict, dict):
                self._task_cache[str(task_id)] = task_dict

    def _action_outcomes(self, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        outcomes = {}
        get_action_outcomes = getattr(self.coop2_adapter, "get_action_outcomes", None)
        if get_action_outcomes is not None:
            for agent_id, outcome in (get_action_outcomes(info) or {}).items():
                outcomes[str(agent_id)] = outcome.to_dict() if hasattr(outcome, "to_dict") else dict(outcome)
        for agent_id, agent_info in (info or {}).items():
            raw = agent_info.get("action_outcome") if isinstance(agent_info, dict) else None
            if isinstance(raw, dict):
                outcomes[str(agent_id)] = dict(raw)
        return outcomes

    def _observed_constraints(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        get_constraint_results = getattr(self.coop2_adapter, "get_constraint_results", None)
        if get_constraint_results is None:
            return []
        results = []
        for result in get_constraint_results(info) or []:
            if hasattr(result, "to_dict"):
                results.append(result.to_dict())
            elif isinstance(result, dict):
                results.append(dict(result))
        return results

    def _active_tasks(self, agent_views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        agents_by_task: Dict[str, List[str]] = {}
        plans_by_task: Dict[str, List[Any]] = {}
        for view in agent_views:
            for task_id in view.get("active_target_ids") or []:
                task_id = str(task_id)
                agents_by_task.setdefault(task_id, []).append(str(view["agent_id"]))
                plan_id = view.get("plan_id")
                if plan_id is not None:
                    plans_by_task.setdefault(task_id, []).append(plan_id)

        active_tasks = []
        for task_id in sorted(agents_by_task):
            task = self._task_cache.get(task_id) or {}
            resource = normalize_process_resource(task.get("resource_type"))
            active_tasks.append(
                {
                    "task_id": task_id,
                    "resource_type": resource or task.get("resource_type"),
                    "required_agents": self._safe_positive_int(task.get("required_agents"), default=1),
                    "required_tool": task.get("required_tool"),
                    "required_tool_mode": task.get("required_tool_mode"),
                    "distance_threshold": task.get("distance_threshold"),
                    "position": task.get("position"),
                    "active_agents": sorted(set(agents_by_task[task_id])),
                    "plan_ids": sorted(set(plans_by_task.get(task_id, []))),
                }
            )
        return active_tasks

    def _score_events(self):
        base_env = self.symbolic_env.env
        score_summary = {}
        events: List[Dict[str, Any]] = []
        if hasattr(base_env, "get_team_score_breakdown"):
            score_summary = base_env.get_team_score_breakdown(include_events=True) or {}
            all_events = score_summary.get("events") or []
            events = [
                dict(event)
                for event in all_events[self._last_score_event_count:]
                if isinstance(event, dict)
            ]
            self._last_score_event_count = len(all_events)
            score_summary = dict(score_summary)
            score_summary.pop("events", None)
        return score_summary, events

    @staticmethod
    def _first_info_value(info: Dict[str, Any], key: str) -> Any:
        if not isinstance(info, dict):
            return None
        for agent_info in info.values():
            if isinstance(agent_info, dict) and key in agent_info:
                return agent_info.get(key)
        return None

    def _extract_target_ids(self, plan: Any) -> set[str]:
        if plan is None:
            return set()
        specification = str(getattr(plan, "specification", "") or "")
        for target_type, target_id in RESOURCE_TARGET_PATTERN.findall(specification):
            if normalize_process_resource(target_type) in SCORED_RESOURCES:
                return {str(target_id)}
        metadata = getattr(plan, "metadata", {}) or {}
        task_id = getattr(plan, "task_id", None) or metadata.get("task_id")
        return {str(task_id)} if task_id is not None else set()

    def _action_focus(self, action: Dict[str, Any], specification: str) -> str:
        args = self._action_args(action)
        for key in ("target", "resource_type", "object_type", "item", "item_type"):
            resource = normalize_process_resource(args.get(key))
            if resource:
                return resource
        action_type = str(action.get("action_type") or "")
        if action_type in {"craft", "place"}:
            return "tools"
        return process_resource_from_text(specification) or "other"

    @staticmethod
    def _action_args(action: Dict[str, Any]) -> Dict[str, Any]:
        args = action.get("args")
        if isinstance(args, dict):
            return args
        return {
            key: value
            for key, value in action.items()
            if key not in {"action_type", "status", "start_step", "end_step"}
        }

    @staticmethod
    def _safe_positive_int(value: Any, default: int = 1) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    def _json_safe(self, value: Any) -> Any:
        """Convert runtime values in process records into plain JSON values."""
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, set):
            return sorted(self._json_safe(item) for item in value)
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return self._json_safe(value.to_dict())
        return value
