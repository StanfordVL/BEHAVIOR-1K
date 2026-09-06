"""
Compute COOP² metrics from experiment logs:
- D: Decision Overhead Metrics
- L: Communication Load Metrics
- P: Planning Dynamics Metrics
- ΔZ: Embodied Effects Metrics
- C: Cooperative Constraint Metrics
- Y: Task Success Metric
"""

import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional


def _step_value(value: Any, default: int = 0) -> int:
    """Return a numeric step for logs that may contain None for pending plans."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_logs(results_dir: str) -> Dict[str, Any]:
    """Load all log files from a results directory."""
    logs = {}

    # Load agent states
    agent_states_path = os.path.join(results_dir, "agent_states.json")
    if os.path.exists(agent_states_path):
        with open(agent_states_path) as f:
            logs['agent_states'] = json.load(f)

    # Load message log
    message_log_path = os.path.join(results_dir, "message_log.json")
    if os.path.exists(message_log_path):
        with open(message_log_path) as f:
            logs['messages'] = json.load(f)

    # Load plan logs
    plan_logs_path = os.path.join(results_dir, "plan_logs.json")
    if os.path.exists(plan_logs_path):
        with open(plan_logs_path) as f:
            logs['plans'] = json.load(f)

    # Load task states
    task_states_path = os.path.join(results_dir, "task_states.json")
    if os.path.exists(task_states_path):
        with open(task_states_path) as f:
            logs['task_states'] = json.load(f)

    # Load capability changes
    capability_path = os.path.join(results_dir, "capability_changes.json")
    if os.path.exists(capability_path):
        with open(capability_path) as f:
            logs['capability_changes'] = json.load(f)

    # Load COOP2 trace
    coop2_trace_path = os.path.join(results_dir, "coop2_trace.json")
    if os.path.exists(coop2_trace_path):
        with open(coop2_trace_path) as f:
            logs['coop2_trace'] = json.load(f)

    # Load team resource score
    team_score_path = os.path.join(results_dir, "team_score.json")
    if os.path.exists(team_score_path):
        with open(team_score_path) as f:
            logs['team_score'] = json.load(f)

    # Load LLM usage statistics
    llm_usage_path = os.path.join(results_dir, "llm_usage.json")
    if os.path.exists(llm_usage_path):
        with open(llm_usage_path) as f:
            logs['llm_usage'] = json.load(f)

    return logs


def compute_decision_overhead_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Decision Overhead Metrics (D).

    Uses agent_states log which contains [timestamp, env_step, state] entries.
    Decision time is time spent in R (reasoning) or I (interrupted)
    states. Wait time is tracked separately from W (waiting). X
    (executing) is not counted as either decision or wait time.
    """
    agent_states = logs.get('agent_states', {})

    if not agent_states:
        return {}

    DECISION_STATES = {'reasoning', 'interrupted'}
    WAIT_STATES = {'waiting'}

    # Get all agents
    agents = list(agent_states.keys())
    n_agents = len(agents)

    # Compute per-agent decision overhead D_i (count of R and I entries only)
    D_i = {}
    for agent_id, states in agent_states.items():
        # Count only reasoning (R) and interrupted (I) state entries
        D_i[agent_id] = sum(1 for entry in states if entry[2] in DECISION_STATES)

    # Compute per-agent and per-step time metrics
    wait_time_per_agent = defaultdict(float)
    decision_time_per_agent = defaultdict(float)
    wait_time_per_step = defaultdict(float)
    decision_time_per_step = defaultdict(float)

    for agent_id, states in agent_states.items():
        for i, entry in enumerate(states):
            timestamp, env_step, state = entry
            # Compute duration until next state transition
            if i + 1 < len(states):
                next_timestamp = states[i + 1][0]
                duration = next_timestamp - timestamp

                if state in WAIT_STATES:
                    wait_time_per_agent[agent_id] += duration
                    wait_time_per_step[env_step] += duration
                elif state in DECISION_STATES:
                    decision_time_per_agent[agent_id] += duration
                    decision_time_per_step[env_step] += duration

    # Ensure all agents have entries (even if 0)
    for agent_id in agents:
        if agent_id not in wait_time_per_agent:
            wait_time_per_agent[agent_id] = 0.0
        if agent_id not in decision_time_per_agent:
            decision_time_per_agent[agent_id] = 0.0

    # Compute aggregates
    wait_times = list(wait_time_per_agent.values())
    wait_time_std = np.std(wait_times) if wait_times else 0
    wait_time_avg = np.mean(wait_times) if wait_times else 0

    decision_times = list(decision_time_per_agent.values())
    decision_time_std = np.std(decision_times) if decision_times else 0
    decision_time_avg = np.mean(decision_times) if decision_times else 0

    # Get max step from states
    max_step = 0
    for agent_id, states in agent_states.items():
        for entry in states:
            max_step = max(max_step, entry[1])
    T = max_step + 1 if max_step > 0 else 1

    # Convert per-step dicts to lists
    wait_time_per_step_list = [wait_time_per_step.get(t, 0.0) for t in range(T)]
    decision_time_per_step_list = [decision_time_per_step.get(t, 0.0) for t in range(T)]

    # Compute per-step decision overhead count d_t (only R and I states)
    d_t = defaultdict(int)
    for agent_id, states in agent_states.items():
        for entry in states:
            if entry[2] in DECISION_STATES:
                step = entry[1]
                d_t[step] += 1

    # Convert to list
    d_t_list = [d_t.get(t, 0) for t in range(T)]

    # Average decision overhead D_δ
    D_delta = sum(d_t_list) / T if T > 0 else 0

    # Decision overhead std dev D_std
    D_i_values = list(D_i.values())
    D_std = np.std(D_i_values) if D_i_values else 0

    return {
        'D_i': D_i,  # Per-agent decision overhead count
        'd_t': d_t_list,  # Per-step decision overhead count
        'D_delta': D_delta,  # Average decision overhead
        'D_std': float(D_std),  # Decision overhead std dev
        'T': T,  # Total steps
        # Per-agent time metrics
        'wait_time_per_agent': dict(wait_time_per_agent),
        'decision_time_per_agent': dict(decision_time_per_agent),
        'wait_time_avg': float(wait_time_avg),
        'wait_time_std': float(wait_time_std),
        'decision_time_avg': float(decision_time_avg),
        'decision_time_std': float(decision_time_std),
        # Per-step time metrics
        'wait_time_per_step': wait_time_per_step_list,
        'decision_time_per_step': decision_time_per_step_list
    }


def compute_communication_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Communication Load Metrics (L).

    Uses message_log which contains sender, recipients, content, etc.
    """
    messages = logs.get('messages', [])
    agent_states = logs.get('agent_states', {})

    # Initialize all agents with 0 messages
    messages_per_agent = {agent_id: 0 for agent_id in agent_states.keys()}

    if not messages:
        return {'L_tot': 0, 'L_var': 0, 'messages_per_agent': messages_per_agent}

    # Total communication load L_tot
    L_tot = len(messages)

    # Messages sent per agent
    for msg in messages:
        sender = msg.get('sender', 'unknown')
        if sender in messages_per_agent:
            messages_per_agent[sender] += 1
        else:
            messages_per_agent[sender] = 1

    # Communication load std dev L_std
    sent_counts = list(messages_per_agent.values())
    L_std = np.std(sent_counts) if sent_counts else 0

    return {
        'L_tot': L_tot,
        'L_std': float(L_std),
        'messages_per_agent': dict(messages_per_agent)
    }


def compute_planning_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Planning Dynamics Metrics (P).

    Uses plan_logs which contains plan history with status, actions, etc.
    """
    plans_data = logs.get('plans', {})
    plan_history = plans_data.get('plan_history', [])

    if not plan_history:
        return {}

    # Count plans per agent
    plans_per_agent = defaultdict(list)
    for plan in plan_history:
        agent_id = plan.get('agent_id', 'unknown')
        plans_per_agent[agent_id].append(plan)

    # Plan outcome counts. Keep interruption separate from ordinary execution
    # failure so individual agents do not look message-interrupted.
    P_int = 0
    P_failed = 0
    P_abandoned = 0
    P_episode_ended = 0
    for plan in plan_history:
        status = plan.get('status', '')
        if status == 'interrupted':
            if plan.get('failure_reason') == 'Episode ended':
                P_episode_ended += 1
            else:
                P_int += 1
        elif status == 'failed':
            P_failed += 1
        elif status == 'abandoned':
            P_abandoned += 1
    P_non_success = P_int + P_failed + P_abandoned

    # Track what happens after each interruption:
    # - Resume: next plan has same spec as interrupted plan
    # - Replan: next plan has different spec
    P_res = 0  # Resumption count
    P_re_total = 0  # Replan count

    for agent_id, plans in plans_per_agent.items():
        # Sort plans by start_step to ensure chronological order
        sorted_plans = sorted(
            plans,
            key=lambda p: (
                _step_value(p.get('start_step')),
                _step_value(p.get('created_step')),
                p.get('plan_id') or 0,
            ),
        )

        for i, plan in enumerate(sorted_plans[:-1]):  # All but last plan
            status = plan.get('status', '')
            if status == 'interrupted' and plan.get('failure_reason') != 'Episode ended':
                # This plan was interrupted - check what the next plan did
                current_spec = plan.get('specification', '')
                next_plan = sorted_plans[i + 1]
                next_spec = next_plan.get('specification', '')

                if current_spec == next_spec:
                    P_res += 1  # Resumed same spec
                else:
                    P_re_total += 1  # Replanned with different spec

    P_re_avg = P_re_total / len(plans_per_agent) if plans_per_agent else 0

    # Plan success rate
    successful = sum(1 for p in plan_history if p.get('status') == 'success')
    total = len(plan_history)
    success_rate = successful / total if total > 0 else 0

    # Plan Coherence: at each step, what fraction of agents share the same specification?
    # Find max step across all plans
    max_step = 0
    for plan in plan_history:
        max_step = max(max_step, _step_value(plan.get('end_step')))

    coherence_per_step = []
    for step in range(max_step + 1):
        # Find active plan specification for each agent at this step
        active_specs = {}
        for plan in plan_history:
            agent_id = plan.get('agent_id')
            start = _step_value(plan.get('start_step'))
            end = _step_value(plan.get('end_step'), default=start)
            if start <= step <= end:
                active_specs[agent_id] = plan.get('specification', '')

        if len(active_specs) >= 2:
            # Count how many agents share the most common specification
            spec_counts = defaultdict(int)
            for spec in active_specs.values():
                spec_counts[spec] += 1
            max_count = max(spec_counts.values())
            # Coherence: 0 if no agents share spec, else fraction with most common
            if max_count == 1:
                coherence = 0  # No agents share the same spec
            else:
                coherence = max_count / len(active_specs)
            coherence_per_step.append(coherence)
        elif len(active_specs) == 1:
            coherence_per_step.append(1.0)  # Single agent is trivially coherent

    avg_coherence = np.mean(coherence_per_step) if coherence_per_step else 0

    return {
        'P_int': P_int,  # Plan interruption count
        'P_failed': P_failed,  # Ordinary failed plan count
        'P_abandoned': P_abandoned,  # Abandoned plan count
        'P_episode_ended': P_episode_ended,  # Plans truncated by episode end
        'P_non_success': P_non_success,  # Failed + interrupted + abandoned
        'P_re_total': P_re_total,  # Total replanning events
        'P_re_avg': P_re_avg,  # Average replanning per agent
        'P_res': P_res,  # Plan resumption count
        'success_rate': success_rate,
        'plans_per_agent': {k: len(v) for k, v in plans_per_agent.items()},
        'total_plans': len(plan_history),
        'coherence': avg_coherence  # Average plan coherence
    }


def compute_capability_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Embodied Effects Metrics (ΔZ).

    Uses capability_changes log which tracks resource/tool gains and losses.
    """
    capability_changes = logs.get('capability_changes', [])

    if not capability_changes:
        return {'total_gains': 0, 'total_losses': 0, 'delta_Z': 0}

    # Count gains and losses
    total_gains = sum(1 for c in capability_changes if c.get('change_type') == 'gain')
    total_losses = sum(1 for c in capability_changes if c.get('change_type') == 'loss')

    # Per-agent capability changes
    gains_per_agent = defaultdict(int)
    losses_per_agent = defaultdict(int)

    for change in capability_changes:
        agent_id = change.get('agent_id', 'unknown')
        if change.get('change_type') == 'gain':
            gains_per_agent[agent_id] += 1
        else:
            losses_per_agent[agent_id] += 1

    # Per-step capability changes
    gains_per_step = defaultdict(int)
    losses_per_step = defaultdict(int)

    for change in capability_changes:
        step = change.get('env_step', 0)
        if change.get('change_type') == 'gain':
            gains_per_step[step] += 1
        else:
            losses_per_step[step] += 1

    # Net capability change ΔZ
    delta_Z = total_gains - total_losses

    return {
        'total_gains': total_gains,
        'total_losses': total_losses,
        'delta_Z': delta_Z,  # Net capability change
        'gains_per_agent': dict(gains_per_agent),
        'losses_per_agent': dict(losses_per_agent),
        'gains_per_step': dict(gains_per_step),
        'losses_per_step': dict(losses_per_step),
        'capability_changes': capability_changes  # Raw data
    }


def compute_constraint_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Cooperative Constraint Metrics (C⁺, C⁻).

    Uses task_states log for the current COOP2 constraint definitions:
    spatial, temporal, and dependency.
    Counts unique tasks that had each constraint type improve/worsen across the episode.
    """
    task_states = logs.get('task_states', [])

    if not task_states:
        return {}

    # Track unique task IDs across all steps for each constraint type
    spatial_improved_tasks = set()
    spatial_worsened_tasks = set()
    temporal_improved_tasks = set()
    temporal_worsened_tasks = set()
    dependency_improved_tasks = set()
    dependency_worsened_tasks = set()

    total_capability_increased = 0
    total_capability_decreased = 0

    tasks_with_any_improvement = set()
    tasks_with_any_worsening = set()

    for step_data in task_states:
        metrics = step_data.get('metrics', {})
        constraint_changes = metrics.get('constraint_changes', {})
        capability_changes = metrics.get('capability_changes', {})

        # Collect unique task IDs for each constraint type
        spatial_improved_tasks.update(constraint_changes.get('spatial', {}).get('improved_tasks', []))
        spatial_worsened_tasks.update(constraint_changes.get('spatial', {}).get('worsened_tasks', []))
        temporal_improved_tasks.update(constraint_changes.get('temporal', {}).get('improved_tasks', []))
        temporal_worsened_tasks.update(constraint_changes.get('temporal', {}).get('worsened_tasks', []))
        dependency_improved_tasks.update(constraint_changes.get('dependency', {}).get('improved_tasks', []))
        dependency_worsened_tasks.update(constraint_changes.get('dependency', {}).get('worsened_tasks', []))
        # Capability
        total_capability_increased += capability_changes.get('increased', 0)
        total_capability_decreased += capability_changes.get('decreased', 0)

    # Count unique tasks for each constraint type
    total_spatial_improved = len(spatial_improved_tasks)
    total_spatial_worsened = len(spatial_worsened_tasks)
    total_temporal_improved = len(temporal_improved_tasks)
    total_temporal_worsened = len(temporal_worsened_tasks)
    total_dependency_improved = len(dependency_improved_tasks)
    total_dependency_worsened = len(dependency_worsened_tasks)
    # Aggregate: unique tasks with any improvement/worsening
    tasks_with_any_improvement = spatial_improved_tasks | temporal_improved_tasks | dependency_improved_tasks
    tasks_with_any_worsening = spatial_worsened_tasks | temporal_worsened_tasks | dependency_worsened_tasks

    # C⁺: Total unique tasks with any constraint satisfaction
    C_plus = len(tasks_with_any_improvement)

    # C⁻: Total unique tasks with any constraint violation
    C_minus = len(tasks_with_any_worsening)

    return {
        'C_plus': C_plus,  # Total unique tasks with any constraint satisfaction
        'C_minus': C_minus,  # Total unique tasks with any constraint violation
        'spatial': {'improved': total_spatial_improved, 'worsened': total_spatial_worsened},
        'temporal': {'improved': total_temporal_improved, 'worsened': total_temporal_worsened},
        'dependency': {'improved': total_dependency_improved, 'worsened': total_dependency_worsened},
        'capability_increased': total_capability_increased,
        'capability_decreased': total_capability_decreased
    }


def compute_task_success_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Task Success Metric (Y_g).

    Uses plan_logs to determine plan completion success.
    """
    plans_data = logs.get('plans', {})
    plan_history = plans_data.get('plan_history', [])
    capability_changes = logs.get('capability_changes', [])

    if not plan_history:
        return {}

    # Plan-based success: count successful plans
    successful_plans = sum(1 for p in plan_history if p.get('status') == 'success')
    total_plans = len(plan_history)
    plan_success_rate = successful_plans / total_plans if total_plans > 0 else 0

    # Resource-based success: count resources collected
    resources_collected = defaultdict(int)
    for change in capability_changes:
        if change.get('change_type') == 'gain':
            item = change.get('item_name', 'unknown')
            amount = int(change.get('new_count', 0)) - int(change.get('old_count', 0))
            resources_collected[item] += max(1, amount)

    return {
        'Y_plan': plan_success_rate,  # Plan success rate
        'successful_plans': successful_plans,
        'total_plans': total_plans,
        'resources_collected': dict(resources_collected)
    }


def compute_team_score_metrics(logs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the fixed-horizon team resource score."""
    team_score = logs.get('team_score', {})
    if not team_score:
        return {
            'enabled': False,
            'total_score': 0.0,
            'resource_counts': {},
            'resource_scores': {},
            'events': [],
        }

    return {
        'enabled': bool(team_score.get('enabled', True)),
        'total_score': float(team_score.get('total_score', 0.0)),
        'resource_counts': team_score.get('resource_counts', {}),
        'resource_scores': team_score.get('resource_scores', {}),
        'events': team_score.get('events', []),
        'terminate_on_diamond': team_score.get('terminate_on_diamond', False),
        'score_shared_resources_per_agent': team_score.get('score_shared_resources_per_agent', True),
        'score_lifetime_collections': team_score.get('score_lifetime_collections', True),
        'max_episode_steps': team_score.get('max_episode_steps'),
        'current_step': team_score.get('current_step'),
        'wall_clock_limit_seconds': team_score.get('wall_clock_limit_seconds'),
        'elapsed_wall_clock_seconds': team_score.get('elapsed_wall_clock_seconds'),
        'remaining_wall_clock_seconds': team_score.get('remaining_wall_clock_seconds'),
    }


def compute_failure_attribution(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Failure Attribution Metrics (ρ).

    Analyzes failed plans to attribute failures to constraint types.
    """
    plans_data = logs.get('plans', {})
    plan_history = plans_data.get('plan_history', [])

    failed_plans = [p for p in plan_history if p.get('status') in ['failed', 'interrupted', 'abandoned']]

    if not failed_plans:
        return {'total_failures': 0}

    # Categorize failures by reason
    failure_reasons = defaultdict(int)
    for plan in failed_plans:
        reason = plan.get('failure_reason', 'unknown')
        if reason:
            failure_reasons[reason] += 1
        else:
            failure_reasons['unknown'] += 1

    total_failures = len(failed_plans)

    # Compute proportions ρ
    rho = {reason: count / total_failures for reason, count in failure_reasons.items()}

    return {
        'total_failures': total_failures,
        'failure_reasons': dict(failure_reasons),
        'rho': rho  # Failure proportions
    }


def compute_scaling_efficiency_metrics(logs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived metrics for agent-count scaling comparisons."""
    agent_states = logs.get('agent_states', {})
    llm_usage = logs.get('llm_usage', {})
    agent_count = len(agent_states)
    if agent_count <= 0:
        agent_count = len(llm_usage.get('agents', {}))

    total_score = float(metrics.get('team_score', {}).get('total_score', 0.0))
    decision_times = metrics.get('decision_overhead', {}).get('decision_time_per_agent', {})
    total_decision_seconds = float(sum(decision_times.values())) if decision_times else 0.0
    total_messages = int(metrics.get('communication', {}).get('L_tot', 0))
    total_tokens = int(llm_usage.get('total_tokens', 0) or 0)
    total_api_calls = int(llm_usage.get('total_api_calls', 0) or 0)

    def per_score(value: float) -> Optional[float]:
        return float(value) / total_score if total_score > 0 else None

    return {
        'agent_count': agent_count,
        'score_per_agent': total_score / agent_count if agent_count > 0 else None,
        'total_decision_seconds': total_decision_seconds,
        'decision_seconds_per_score': per_score(total_decision_seconds),
        'messages_per_score': per_score(total_messages),
        'tokens_per_score': per_score(total_tokens),
        'api_calls_per_score': per_score(total_api_calls),
        'total_tokens': total_tokens,
        'total_api_calls': total_api_calls,
    }


def compute_all_metrics(results_dir: str) -> Dict[str, Any]:
    """Compute all COOP² metrics from a results directory."""
    logs = load_logs(results_dir)

    metrics = {
        'decision_overhead': compute_decision_overhead_metrics(logs),
        'communication': compute_communication_metrics(logs),
        'planning': compute_planning_metrics(logs),
        'capability': compute_capability_metrics(logs),
        'constraints': compute_constraint_metrics(logs),
        'task_success': compute_task_success_metrics(logs),
        'team_score': compute_team_score_metrics(logs),
        'failure_attribution': compute_failure_attribution(logs)
    }
    metrics['scaling_efficiency'] = compute_scaling_efficiency_metrics(logs, metrics)

    return metrics


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print a formatted summary of computed metrics."""
    print("\n" + "="*70)
    print("COOP² METRICS SUMMARY")
    print("="*70)

    # Decision Overhead (D)
    print("\n--- Decision Overhead Metrics (D) ---")
    d = metrics.get('decision_overhead', {})
    print(f"  D_δ (avg per step): {d.get('D_delta', 0):.2f}")
    print(f"  Per-agent D_i: {d.get('D_i', {})}")
    print(f"  Decision time avg: {d.get('decision_time_avg', 0):.2f}s")
    print(f"  Decision time std: {d.get('decision_time_std', 0):.2f}s")
    print(f"  Decision time per agent: {{{', '.join(f'{k}: {v:.2f}s' for k, v in d.get('decision_time_per_agent', {}).items())}}}")
    print(f"  Wait time avg: {d.get('wait_time_avg', 0):.2f}s")
    print(f"  Wait time std: {d.get('wait_time_std', 0):.2f}s")
    print(f"  Wait time per agent: {{{', '.join(f'{k}: {v:.2f}s' for k, v in d.get('wait_time_per_agent', {}).items())}}}")

    # Communication (L)
    print("\n--- Communication Load Metrics (L) ---")
    l = metrics.get('communication', {})
    print(f"  L_tot (total messages): {l.get('L_tot', 0)}")
    print(f"  L_std (std dev): {l.get('L_std', 0):.2f}")
    print(f"  Messages per agent: {l.get('messages_per_agent', {})}")

    # Planning (P)
    print("\n--- Planning Dynamics Metrics (P) ---")
    p = metrics.get('planning', {})
    print(f"  P_int (interruptions): {p.get('P_int', 0)}")
    print(f"  P_failed (plan failures): {p.get('P_failed', 0)}")
    print(f"  P_episode_ended: {p.get('P_episode_ended', 0)}")
    print(f"  P_res (resumptions): {p.get('P_res', 0)}")
    print(f"  Plan success rate: {p.get('success_rate', 0):.2%}")
    print(f"  Plans per agent: {p.get('plans_per_agent', {})}")

    # Capability (ΔZ)
    print("\n--- Embodied Effects Metrics (ΔZ) ---")
    z = metrics.get('capability', {})
    print(f"  Total gains: {z.get('total_gains', 0)}")
    print(f"  Total losses: {z.get('total_losses', 0)}")

    # Team Score (S)
    print("\n--- Team Resource Score (S) ---")
    s = metrics.get('team_score', {})
    e = metrics.get('scaling_efficiency', {})
    print(f"  S_team: {s.get('total_score', 0):.2f}")
    score_per_agent = e.get('score_per_agent')
    if score_per_agent is not None:
        print(f"  S_team per agent: {score_per_agent:.2f}")
    print(f"  Scored resources: {s.get('resource_counts', {})}")
    if s.get('max_episode_steps') is not None:
        print(f"  Budget: step {s.get('current_step')}/{s.get('max_episode_steps')}, "
              f"{s.get('elapsed_wall_clock_seconds', 0):.1f}s elapsed")
    if e:
        print(f"  Agents: {e.get('agent_count', 0)}")
        print(f"  Messages per score: {e.get('messages_per_score')}")
        print(f"  Decision seconds per score: {e.get('decision_seconds_per_score')}")
        print(f"  Tokens per score: {e.get('tokens_per_score')}")

    # Constraints (C)
    print("\n--- Cooperative Constraint Metrics (C⁺, C⁻) ---")
    c = metrics.get('constraints', {})
    print(f"  C⁺ (satisfactions): {c.get('C_plus', 0)}")
    print(f"  C⁻ (violations): {c.get('C_minus', 0)}")
    print(f"  Spatial: +{c.get('spatial', {}).get('improved', 0)} / -{c.get('spatial', {}).get('worsened', 0)}")
    print(f"  Temporal: +{c.get('temporal', {}).get('improved', 0)} / -{c.get('temporal', {}).get('worsened', 0)}")
    print(f"  Dependency: +{c.get('dependency', {}).get('improved', 0)} / -{c.get('dependency', {}).get('worsened', 0)}")

    # Task Success (Y)
    print("\n--- Task Success Metrics (Y) ---")
    y = metrics.get('task_success', {})
    print(f"  Y_plan (success rate): {y.get('Y_plan', 0):.2%}")
    print(f"  Successful/Total plans: {y.get('successful_plans', 0)}/{y.get('total_plans', 0)}")
    print(f"  Resources collected: {y.get('resources_collected', {})}")

    # Failure Attribution (ρ)
    print("\n--- Failure Attribution Metrics (ρ) ---")
    f = metrics.get('failure_attribution', {})
    print(f"  Total failures: {f.get('total_failures', 0)}")
    print(f"  Failure proportions: {f.get('rho', {})}")

    print("\n" + "="*70)


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """Save computed metrics to JSON file."""
    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    metrics_clean = convert(metrics)

    with open(output_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"Saved metrics to: {output_path}")


def save_metrics_csv(metrics: Dict[str, Any], output_path: str):
    """Save computed metrics to CSV file (flat format for easy analysis)."""
    import csv

    # Flatten metrics into rows
    rows = []

    # Decision Overhead (D)
    d = metrics.get('decision_overhead', {})
    rows.append(['D_delta', 'Average decision overhead per step', d.get('D_delta', 0)])
    rows.append(['decision_time_avg', 'Average decision time across agents (seconds)', d.get('decision_time_avg', 0)])
    rows.append(['decision_time_std', 'Decision time std dev across agents (seconds)', d.get('decision_time_std', 0)])
    rows.append(['wait_time_avg', 'Average wait time across agents (seconds)', d.get('wait_time_avg', 0)])
    rows.append(['wait_time_std', 'Wait time std dev across agents (seconds)', d.get('wait_time_std', 0)])

    # Communication (L)
    l = metrics.get('communication', {})
    rows.append(['L_tot', 'Total messages exchanged', l.get('L_tot', 0)])
    rows.append(['L_std', 'Communication load std dev', l.get('L_std', 0)])
    # Average messages per agent
    msg_counts = list(l.get('messages_per_agent', {}).values())
    rows.append(['L_avg', 'Average messages per agent', np.mean(msg_counts) if msg_counts else 0])

    # Planning (P)
    p = metrics.get('planning', {})
    total_plans = p.get('total_plans', 0)
    P_int = p.get('P_int', 0)
    P_res = p.get('P_res', 0)
    P_re_total = p.get('P_re_total', 0)
    rows.append(['P_total', 'Total plans', total_plans])
    rows.append(['P_int', 'Plan interruption count', P_int])
    rows.append(['P_failed', 'Failed plan count', p.get('P_failed', 0)])
    rows.append(['P_episode_ended', 'Plans truncated by episode end', p.get('P_episode_ended', 0)])
    rows.append(['P_non_success', 'Failed + interrupted + abandoned plans', p.get('P_non_success', P_int)])
    # Resume rate = resumes / interruptions (after interrupt, how often did agent resume same spec?)
    P_res_rate = P_res / P_int if P_int > 0 else 0
    rows.append(['P_res_rate', 'Plan resumption rate', P_res_rate])
    # Replan rate = complement of resume rate (they sum to 1)
    rows.append(['P_re_rate', 'Replanning rate', 1 - P_res_rate if P_int > 0 else 0])
    rows.append(['P_coherence', 'Average plan coherence per step', p.get('coherence', 0)])

    # Capability (ΔZ)
    z = metrics.get('capability', {})
    T = d.get('T', 1)
    rows.append(['Z_gain_rate', 'Capability gains per step', z.get('total_gains', 0) / T if T > 0 else 0])

    # Team Resource Score (S)
    s = metrics.get('team_score', {})
    e = metrics.get('scaling_efficiency', {})
    rows.append(['S_team', 'Total team resource score', s.get('total_score', 0)])
    rows.append(['S_agent_count', 'Number of agents in the episode', e.get('agent_count', 0)])
    rows.append(['S_team_per_agent', 'Team resource score per agent', e.get('score_per_agent')])
    rows.append(['S_decision_seconds_per_score', 'Decision seconds per team score point', e.get('decision_seconds_per_score')])
    rows.append(['S_messages_per_score', 'Messages per team score point', e.get('messages_per_score')])
    rows.append(['S_tokens_per_score', 'LLM tokens per team score point', e.get('tokens_per_score')])
    for item_name, count in sorted(s.get('resource_counts', {}).items()):
        rows.append([f'S_count_{item_name}', f'Scored team resource count: {item_name}', count])

    # Constraints (C) - improvement ratio = improved / (improved + worsened)
    c = metrics.get('constraints', {})

    def improvement_ratio(improved, worsened):
        total = improved + worsened
        return improved / total if total > 0 else 0.5

    spatial_imp = c.get('spatial', {}).get('improved', 0)
    spatial_wor = c.get('spatial', {}).get('worsened', 0)
    temporal_imp = c.get('temporal', {}).get('improved', 0)
    temporal_wor = c.get('temporal', {}).get('worsened', 0)
    dependency_imp = c.get('dependency', {}).get('improved', 0)
    dependency_wor = c.get('dependency', {}).get('worsened', 0)
    rows.append(['C_spatial_ratio', 'Spatial constraint improvement ratio', improvement_ratio(spatial_imp, spatial_wor)])
    rows.append(['C_temporal_ratio', 'Temporal constraint improvement ratio', improvement_ratio(temporal_imp, temporal_wor)])
    rows.append(['C_dependency_ratio', 'Dependency constraint improvement ratio', improvement_ratio(dependency_imp, dependency_wor)])

    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'description', 'value'])
        writer.writerows(rows)

    print(f"Saved metrics CSV to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute COOP² metrics from experiment logs")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--save", "-s", help="Save metrics to JSON file (also saves CSV with same name)")

    args = parser.parse_args()

    # Convert to absolute path
    results_dir = os.path.abspath(args.results_dir)

    # Compute metrics
    metrics = compute_all_metrics(results_dir)

    # Print summary
    print_metrics_summary(metrics)

    # Save if requested
    if args.save:
        json_path = os.path.abspath(args.save)
        csv_path = json_path.replace('.json', '.csv') if json_path.endswith('.json') else json_path + '.csv'
    else:
        # Default: save to results directory
        json_path = os.path.join(results_dir, "coop2_metrics.json")
        csv_path = os.path.join(results_dir, "coop2_metrics.csv")

    save_metrics(metrics, json_path)
    save_metrics_csv(metrics, csv_path)
