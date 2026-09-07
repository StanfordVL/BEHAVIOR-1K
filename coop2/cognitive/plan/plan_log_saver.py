"""Log-saving helpers for PlanningEnvWrapper."""

from __future__ import annotations

import json
import os
from typing import Any


class PlanningLogSaver:
    """Persist plan-wrapper logs without bloating the execution wrapper."""

    def __init__(self, wrapper: Any):
        self.wrapper = wrapper

    def save_agent_log(self, output_path: str) -> None:
        """Save agent state timelines to JSON."""
        agent_states = {}
        for agent_id, agent in self.wrapper.agents.items():
            if agent is not None:
                timeline = agent.get_state_timeline()
                agent_states[agent_id] = [
                    (timestamp, step, state.value) for timestamp, step, state in timeline
                ]

        with open(output_path, "w") as f:
            json.dump(agent_states, f, indent=2)
        print(f"Saved agent state timelines to {output_path}")

    def save_message_log(self, output_path: str) -> None:
        """Save inter-agent messages to JSON."""
        message_broker = self.wrapper.message_broker
        if message_broker is not None:
            message_log = message_broker.get_message_log()
            with open(output_path, "w") as f:
                json.dump(message_log, f, indent=2)
            print(f"Saved {len(message_log)} messages to {output_path}")

    def save_task_log(self, output_path: str) -> None:
        """Save task states history to JSON (only changed tasks to reduce file size)."""
        from coop2.behavior_env.cooperative_tasks import convert_to_serializable

        base_env = self.wrapper.symbolic_env.env
        if not hasattr(base_env, "task_tracker"):
            return

        task_history = base_env.task_tracker.get_history()
        if not task_history:
            return

        lightweight_data = []
        previous_tasks = {}

        for summary in task_history:
            step_data = {
                "step": summary.env_step,
                "metrics": convert_to_serializable(summary.metrics.to_dict()) if summary.metrics else None,
                "changed_tasks": {},
                "collected": summary.successful_collections,
                "failed_attempts": summary.failed_attempts,
            }

            current_task_ids = set(summary.tasks.keys())
            previous_task_ids = set(previous_tasks.keys())
            changed_task_ids = set(current_task_ids - previous_task_ids)
            changed_task_ids.update(previous_task_ids - current_task_ids)

            for task_id in current_task_ids & previous_task_ids:
                curr = summary.tasks[task_id]
                prev = previous_tasks[task_id]
                if (
                    curr.spatial_count != prev.spatial_count
                    or curr.temporal_count != prev.temporal_count
                    or curr.dependency_met != prev.dependency_met
                    or curr.status != prev.status
                ):
                    changed_task_ids.add(task_id)

            for task_id in changed_task_ids:
                if task_id in summary.tasks:
                    step_data["changed_tasks"][str(task_id)] = convert_to_serializable(
                        summary.tasks[task_id].to_dict()
                    )

            lightweight_data.append(step_data)
            previous_tasks = summary.tasks.copy()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(lightweight_data, f, indent=2)
        print(f"Saved {len(lightweight_data)} task state snapshots (changed tasks only) to {output_path}")

    def save_capability_log(self, output_path: str) -> None:
        """Save capability change history to JSON (if CooperativeEnv is used)."""
        from coop2.behavior_env.cooperative_tasks import save_capability_log

        base_env = self.wrapper.symbolic_env.env
        if hasattr(base_env, "task_tracker"):
            capability_history = base_env.task_tracker.capability_history
            if capability_history:
                save_capability_log(capability_history, output_path)

    def save_team_score_log(self, output_path: str) -> None:
        """Save team resource score summary if the base environment provides one."""
        base_env = self.wrapper.symbolic_env.env
        if hasattr(base_env, "save_team_score_log"):
            base_env.save_team_score_log(output_path)

    def save_all(self, output_dir: str) -> None:
        """Save all experiment logs and data."""
        os.makedirs(output_dir, exist_ok=True)

        self.wrapper.save_plan_logs(os.path.join(output_dir, "plan_logs.json"))
        self.save_agent_log(os.path.join(output_dir, "agent_states.json"))
        self.save_message_log(os.path.join(output_dir, "message_log.json"))
        self.save_task_log(os.path.join(output_dir, "task_states.json"))
        self.save_capability_log(os.path.join(output_dir, "capability_changes.json"))
        self.save_team_score_log(os.path.join(output_dir, "team_score.json"))
        self.wrapper.coop2_trace.save(os.path.join(output_dir, "coop2_trace.json"))
        self.wrapper.save_coop2_process_log(os.path.join(output_dir, "coop2_process_log.json"))
