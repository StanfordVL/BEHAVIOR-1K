"""Placeholder for ma_crafter's grid A* navigation, which is not ported.

PORTING_PLAN drops this module entirely: crafter navigates a discrete grid,
while here NAVIGATE_TO is a primitive -- a cuRobo motion plan in the physical
set, a filtered teleport in the symbolic one
(:mod:`coop2.behavior_env.symbolic_navigation`). ``action.py`` still imports
these three names at module scope, so they exist and fail loudly if called.

They should disappear when L2 is rewritten in M5, along with the callers.
"""

from __future__ import annotations

from typing import Any

__all__ = ["find_object_position", "get_move_towards_target", "is_near_target"]

_MESSAGE = (
    "{name}() is ma_crafter's grid A*, which is not ported: navigation here is the "
    "NAVIGATE_TO primitive. If you reached this, an L2 controller still routes through "
    "grid pathfinding and needs rewriting (PORTING_PLAN M5)."
)


def find_object_position(*args: Any, **kwargs: Any):
    raise NotImplementedError(_MESSAGE.format(name="find_object_position"))


def is_near_target(*args: Any, **kwargs: Any):
    raise NotImplementedError(_MESSAGE.format(name="is_near_target"))


def get_move_towards_target(*args: Any, **kwargs: Any):
    raise NotImplementedError(_MESSAGE.format(name="get_move_towards_target"))
