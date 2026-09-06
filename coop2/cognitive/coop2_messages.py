"""Shared helpers for recognizing and summarizing COOP2 repair messages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from coop2._repair_shim.message_protocol import (
    COOP2_REPAIR_CONTENT_TYPE,
    COOP2_REPAIR_MESSAGE_TYPE,
    MESSAGE_TYPE_METADATA_KEY,
)


def is_coop2_repair_content(content: Any) -> bool:
    """Return True when a message payload is a structured COOP2 repair request."""
    return isinstance(content, dict) and content.get("type") == COOP2_REPAIR_CONTENT_TYPE


def is_coop2_repair_message(message: Any) -> bool:
    """Return True when a message record carries a COOP2 repair request."""
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata", {}) if isinstance(message.get("metadata"), dict) else {}
    if metadata.get(MESSAGE_TYPE_METADATA_KEY) == COOP2_REPAIR_MESSAGE_TYPE:
        return True
    return is_coop2_repair_content(message.get("content"))


def has_coop2_repair_message(messages: Optional[Iterable[Dict[str, Any]]]) -> bool:
    """Return True when any message in an iterable is a COOP2 repair request."""
    return any(is_coop2_repair_message(message) for message in messages or [])


def get_coop2_repair_context(
    messages: Optional[Iterable[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Return the latest structured COOP2 repair payload from message records."""
    context = None
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if is_coop2_repair_content(content):
            context = content
    return context


def summarize_coop2_repair_content(content: Dict[str, Any]) -> str:
    """Return a compact human-readable summary of a COOP2 repair payload."""
    failures = content.get("failures") or []
    channel = content.get("repair_channel") or {}
    statements = channel.get("statements") or []
    return (
        f"COOP2 repair request at step {content.get('env_step', '?')} "
        f"({len(failures)} predicted failure(s), {len(statements)} repair statement(s))."
    )


def summarize_coop2_repair_for_panel(content: Dict[str, Any]) -> str:
    """Return compact text suitable for realtime visualization panels."""
    failures = content.get("failures") or []
    counts: Dict[str, int] = {}
    for failure in failures:
        key = str(failure.get("constraint_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    count_text = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
    channel = content.get("repair_channel") or {}
    order = " -> ".join(str(agent_id) for agent_id in channel.get("order") or [])
    parts = [f"COOP2 repair: {len(failures)} predicted failures"]
    if count_text:
        parts.append(f"constraints {count_text}")
    if order:
        parts.append(f"order {order}")
    parts.append("revise before env step")
    return "; ".join(parts)
