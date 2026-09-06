"""Shared LLM usage aggregation for experiment scripts."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional


RoleGetter = Callable[[str, Any], str]
ExtraGetter = Callable[[str, Any], Mapping[str, Any]]


def _agent_usage(agent: Any) -> Dict[str, Any]:
    if hasattr(agent, "get_usage_stats"):
        return dict(agent.get_usage_stats())
    api_calls = int(getattr(agent, "api_calls", 0) or 0)
    latency = float(getattr(agent, "api_latency_seconds", 0.0) or 0.0)
    return {
        "api_calls": api_calls,
        "total_tokens": int(getattr(agent, "total_tokens_used", 0) or 0),
        "prompt_tokens": int(getattr(agent, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(agent, "completion_tokens", 0) or 0),
        "api_latency_seconds": latency,
        "avg_api_latency_seconds": latency / api_calls if api_calls else 0.0,
        "max_api_latency_seconds": float(getattr(agent, "max_api_latency_seconds", 0.0) or 0.0),
        "api_retries": int(getattr(agent, "api_retries", 0) or 0),
        "api_rate_limit_retries": int(getattr(agent, "api_rate_limit_retries", 0) or 0),
        "api_error_retries": int(getattr(agent, "api_error_retries", 0) or 0),
        "llm_errors": int(getattr(agent, "llm_errors", 0) or 0),
        "guard_filter_events": int(getattr(agent, "guard_filter_events", 0) or 0),
    }


def collect_llm_usage(
    agents: Mapping[str, Any],
    role_getter: Optional[RoleGetter] = None,
    extra_getter: Optional[ExtraGetter] = None,
) -> Dict[str, Any]:
    """Return consistent total and per-agent LLM usage telemetry."""
    per_agent: Dict[str, Dict[str, Any]] = {}
    totals = {
        "total_api_calls": 0,
        "total_tokens": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_api_latency_seconds": 0.0,
        "max_api_latency_seconds": 0.0,
        "total_api_retries": 0,
        "total_api_rate_limit_retries": 0,
        "total_api_error_retries": 0,
        "total_llm_errors": 0,
        "total_guard_filter_events": 0,
    }

    for agent_id, agent in agents.items():
        usage = _agent_usage(agent)
        payload = {
            "role": role_getter(agent_id, agent) if role_getter else "agent",
            "plan_count": int(getattr(agent, "plan_count", 0) or 0),
            **usage,
        }
        if extra_getter:
            payload.update(dict(extra_getter(agent_id, agent)))
        per_agent[agent_id] = payload

        totals["total_api_calls"] += usage["api_calls"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["total_prompt_tokens"] += usage["prompt_tokens"]
        totals["total_completion_tokens"] += usage["completion_tokens"]
        totals["total_api_latency_seconds"] += usage["api_latency_seconds"]
        totals["max_api_latency_seconds"] = max(
            totals["max_api_latency_seconds"],
            usage["max_api_latency_seconds"],
        )
        totals["total_api_retries"] += usage["api_retries"]
        totals["total_api_rate_limit_retries"] += usage["api_rate_limit_retries"]
        totals["total_api_error_retries"] += usage["api_error_retries"]
        totals["total_llm_errors"] += usage["llm_errors"]
        totals["total_guard_filter_events"] += usage.get("guard_filter_events", 0)

    totals["avg_api_latency_seconds"] = (
        totals["total_api_latency_seconds"] / totals["total_api_calls"]
        if totals["total_api_calls"]
        else 0.0
    )
    totals["per_agent"] = per_agent
    return totals


def print_llm_usage_summary(
    agents: Mapping[str, Any],
    role_getter: Optional[RoleGetter] = None,
    extra_getter: Optional[ExtraGetter] = None,
) -> Dict[str, Any]:
    """Print and return a consistent LLM usage summary."""
    summary = collect_llm_usage(agents, role_getter=role_getter, extra_getter=extra_getter)
    print("\nLLM Usage Statistics:")
    for agent_id, stats in summary["per_agent"].items():
        print(f"  {agent_id} ({stats['role']}):")
        print(f"    Plans generated: {stats['plan_count']}")
        print(f"    API calls: {stats['api_calls']}")
        print(f"    Total tokens: {stats['total_tokens']}")
        print(f"    Prompt tokens: {stats['prompt_tokens']}")
        print(f"    Completion tokens: {stats['completion_tokens']}")
        print(f"    Avg API latency: {stats['avg_api_latency_seconds']:.2f}s")
        print(f"    Retries: {stats['api_retries']} (rate-limit: {stats['api_rate_limit_retries']})")
        print(f"    LLM errors: {stats['llm_errors']}")
        print(f"    Guard/filter events: {stats.get('guard_filter_events', 0)}")

    print(f"\n  TOTAL API calls: {summary['total_api_calls']}")
    print(f"  TOTAL tokens: {summary['total_tokens']}")
    print(f"  AVG API latency: {summary['avg_api_latency_seconds']:.2f}s")
    print(f"  MAX API latency: {summary['max_api_latency_seconds']:.2f}s")
    print(
        f"  TOTAL retries: {summary['total_api_retries']} "
        f"(rate-limit: {summary['total_api_rate_limit_retries']})"
    )
    print(f"  TOTAL LLM errors: {summary['total_llm_errors']}")
    print(f"  TOTAL guard/filter events: {summary['total_guard_filter_events']}")
    return summary
