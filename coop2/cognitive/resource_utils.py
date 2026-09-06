"""Shared resource naming helpers for cognitive logs, actions, and plots."""

from __future__ import annotations

import re
from typing import Any, Optional


SCORED_RESOURCES = ("wood", "stone", "coal", "iron", "diamond")
PROCESS_RESOURCE_ORDER = (*SCORED_RESOURCES, "tools")
RESOURCE_TARGET_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)#([A-Za-z0-9_-]+)")

RESOURCE_ALIASES = {
    "tree": "wood",
    "wood": "wood",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
    "table": "tools",
    "crafting_table": "tools",
    "wood_pickaxe": "tools",
    "stone_pickaxe": "tools",
    "iron_pickaxe": "tools",
    "furnace": "tools",
    "water": "survival",
    "drink": "survival",
    "cow": "survival",
    "food": "survival",
}


def normalize_resource_name(value: Any) -> Optional[str]:
    """Normalize resource-like values from strings, enums, and pydantic fields."""
    if value is None:
        return None
    raw = getattr(value, "value", value)
    text = str(raw).strip().lower().replace(" ", "_")
    if not text:
        return None
    return text


def normalize_process_resource(value: Any) -> Optional[str]:
    """Normalize values into process/plot resource categories."""
    text = normalize_resource_name(value)
    return RESOURCE_ALIASES.get(text, text) if text else None


def process_resource_from_text(text: Any) -> Optional[str]:
    """Infer a plotting/process resource category from task specification text."""
    lower = str(text or "").lower()
    for resource in SCORED_RESOURCES:
        if re.search(rf"\b{re.escape(resource)}\b", lower) or f"collect_{resource}" in lower:
            return resource
    if any(token in lower for token in ["table", "pickaxe", "furnace", "craft", "place_"]):
        return "tools"
    if any(token in lower for token in ["cow", "food", "water", "drink"]):
        return "survival"
    return None
