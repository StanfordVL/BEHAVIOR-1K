"""
Symbolic action interface for MA-Crafter.

This module provides symbolic actions (data structures) and their executor
that translates symbolic actions to primitive actions.
"""

from typing import List, Optional, Dict, Tuple, Literal, Any
from dataclasses import dataclass, field
from enum import Enum
from ..constants import ACTION_NAME_TO_VALUE
from ..resource_utils import normalize_resource_name
from .navigation import find_object_position, is_near_target, get_move_towards_target
import numpy as np


@dataclass
class SymbolicAction:
    """A single symbolic action within a plan."""
    action_type: str
    args: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    status: Optional[str] = None  # "success", "failed", "executing"
    failure_reason: Optional[str] = None
    primitive_action: Optional[str] = None  # Step-level action name (e.g., "move_left", "do")
    primitive_action_history: Dict[int, str] = field(default_factory=dict)  # Per-step primitive actions
    outcome: Optional[Dict[str, Any]] = None  # Environment-backed outcome/effects

    def __post_init__(self):
        """Set step-level action name based on action type and args."""
        if self.primitive_action is None:
            self.primitive_action = self._map_to_step_action()

    def _map_to_step_action(self) -> str:
        """Map symbolic action to step-level action name.
        Only remaps actions where the names differ."""
        if self.action_type == 'collect':
            return 'do'
        elif self.action_type == 'craft' and self.args.get('object_type'):
            return f"make_{self.args['object_type']}"
        elif self.action_type == 'place' and self.args.get('object_type'):
            return f"place_{self.args['object_type']}"
        else:
            # For actions where symbolic name == step-level name (sleep, noop)
            # and multi-step actions (move, navigate), return action type
            # Multi-step actions will have this updated during execution
            return self.action_type

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "args": self.args,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "duration": (
                self.end_step - self.start_step
                if self.end_step is not None and self.start_step is not None
                else None
            ),
            "primitive_action": self.primitive_action,
            "primitive_action_history": self.primitive_action_history,
            "outcome": self.outcome,
        }


class SymbolicActionStatus(Enum):
    """Status of symbolic action execution."""
    PENDING = "pending"  # Action started but not complete
    SUCCESS = "success"  # Action completed successfully
    FAILED = "failed"    # Action failed


class ActionRecord:
    """Record of a symbolic action execution."""
    def __init__(self, action_type: str, args: Dict, start_step: int):
        self.action_type = action_type
        self.args = args
        self.start_step = start_step
        self.end_step: Optional[int] = None
        self.status: SymbolicActionStatus = SymbolicActionStatus.PENDING
        self.failure_reason: Optional[str] = None
        self.primitive_action: Optional[str] = None
        self.primitive_action_history: Dict[int, str] = {}  # Per-step primitive actions
        self.outcome: Optional[Dict[str, Any]] = None
        self.attempt_count: int = 0

    def complete(
        self,
        end_step: int,
        status: SymbolicActionStatus,
        failure_reason: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ):
        """Mark action as complete."""
        self.end_step = end_step
        self.status = status
        self.failure_reason = failure_reason
        self.outcome = outcome

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "action_type": self.action_type,
            "args": self.args,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "status": self.status.value,
            "failure_reason": self.failure_reason,
            "duration": (
                self.end_step - self.start_step
                if self.end_step is not None and self.start_step is not None
                else None
            ),
            "primitive_action": self.primitive_action,
            "primitive_action_history": self.primitive_action_history,
            "outcome": self.outcome,
            "attempt_count": self.attempt_count,
        }


class SymbolicActionExecutor:
    """
    Action executor that translates symbolic actions to primitive actions.

    Key differences from old PrimitiveActions:
    - No leader agent - all participants benefit equally
    - No pre/execution/post checks
    - Clean symbolic action -> primitive action mapping
    - Tracks multi-step action execution and status
    """

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id
        self.actions = {
            "noop": self.noop,
            "move": self.move,
            "collect": self.collect,
            "craft": self.craft,
            "sleep": self.sleep,
            "place": self.place,
            "share": self.share,
            "navigate": self.navigate,
        }

        # State for multi-step actions
        self.current_symbolic_action: Optional[ActionRecord] = None
        self.current_env_step: int = 0
        self.action_history: List[ActionRecord] = []

        # Navigate state
        self.navigate_target_pos: Optional[Tuple[int, int]] = None
        self.navigate_world_state: Optional[Dict] = None
        self.navigate_radius: int = 1

        # Move state
        self.move_steps_remaining: int = 0
        self.move_direction: Optional[str] = None

        # Share state (pending share to be executed by wrapper)
        self.pending_share: Optional[Dict[str, Any]] = None
        self.last_share_result: Optional[Dict[str, Any]] = None

        # Place state (pending symbolic place to be executed by wrapper/env)
        self.pending_place: Optional[Dict[str, Any]] = None

        # Collect state (pending symbolic collect metadata for task tracking)
        self.pending_collect: Optional[Dict[str, Any]] = None

    def _process_object_type(self, object_type: str) -> str:
        """Normalize object type string."""
        return object_type.lower().replace(" ", "_")

    def start_symbolic_action(self, action_type: str, args: Dict):
        """Start tracking a new symbolic action."""
        # Move completed action to history if any
        if self.current_symbolic_action and self.current_symbolic_action.status != SymbolicActionStatus.PENDING:
            self.action_history.append(self.current_symbolic_action)

        # Only start a new action if there's no current pending action
        if self.current_symbolic_action and self.current_symbolic_action.status == SymbolicActionStatus.PENDING:
            # Don't interrupt a pending action - this should not happen in normal flow
            return

        # Start new action at current step
        self.current_symbolic_action = ActionRecord(action_type, args, self.current_env_step)

    def complete_current_action(
        self,
        status: SymbolicActionStatus,
        failure_reason: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ):
        """Complete the current symbolic action."""
        if self.current_symbolic_action:
            self.current_symbolic_action.complete(self.current_env_step, status, failure_reason, outcome)
            self.action_history.append(self.current_symbolic_action)
            self.current_symbolic_action = None

    def update_env_step(self, step: int):
        """Update the current environment step counter."""
        self.current_env_step = step

    def reset_current_action(self):
        """Drop pending symbolic-action state after an interrupted/replaced plan."""
        self.current_symbolic_action = None
        self.navigate_target_pos = None
        self.navigate_world_state = None
        self.navigate_no_path = False
        self.navigate_radius = 1
        self.move_steps_remaining = 0
        self.move_direction = None
        self.pending_share = None
        self.pending_place = None
        self.pending_collect = None

    def check_termination_condition(
        self,
        world_state: Optional[Dict] = None,
        action_outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check if current symbolic action's termination condition is met.
        Called AFTER the environment step executes.

        Args:
            world_state: Current world state (needed for navigate)

        Returns:
            Dict with 'status' ('pending', 'success', 'failed') and optional 'failure_reason'
        """
        if not self.current_symbolic_action or self.current_symbolic_action.status != SymbolicActionStatus.PENDING:
            return {"status": "pending"}

        action_type = self.current_symbolic_action.action_type
        args = self.current_symbolic_action.args

        # Most symbolic actions are single-step. Collect is a bounded action
        # window and can keep issuing primitive do for a few steps.
        if action_type in ['noop', 'collect', 'craft', 'place', 'sleep', 'share']:
            if action_type == "collect":
                self.current_symbolic_action.attempt_count = max(
                    self.current_symbolic_action.attempt_count,
                    self._collect_elapsed_steps(),
                )
            if action_outcome and action_outcome.get("status") == "failed":
                reason = action_outcome.get("reason", "Action failed")
                if action_type == "collect" and self._should_continue_collect_after_failure(args, reason):
                    return {
                        "status": "pending",
                        "failure_reason": reason,
                        "outcome": action_outcome,
                    }
                self.complete_current_action(SymbolicActionStatus.FAILED, reason, action_outcome)
                return {"status": "failed", "failure_reason": reason, "outcome": action_outcome}
            if action_type == "collect" and action_outcome is not None:
                mismatch_reason = self._collect_target_mismatch_reason(args, action_outcome)
                if mismatch_reason:
                    grounded_outcome = dict(action_outcome)
                    grounded_outcome["environment_status"] = action_outcome.get("status", "success")
                    grounded_outcome["status"] = "failed"
                    grounded_outcome["reason"] = mismatch_reason
                    self.complete_current_action(
                        SymbolicActionStatus.FAILED,
                        mismatch_reason,
                        grounded_outcome,
                    )
                    return {
                        "status": "failed",
                        "failure_reason": mismatch_reason,
                        "outcome": grounded_outcome,
                    }
            self.complete_current_action(SymbolicActionStatus.SUCCESS, outcome=action_outcome)
            return {"status": "success", "outcome": action_outcome}

        # Move action - check if all steps completed
        elif action_type == 'move':
            if self.move_steps_remaining <= 0:
                self.complete_current_action(SymbolicActionStatus.SUCCESS)
                return {"status": "success"}
            return {"status": "pending"}

        # Navigate action - check if reached target, no path, or timeout
        elif action_type == 'navigate':
            elapsed = self.current_env_step - self.current_symbolic_action.start_step
            timeout = args.get('timeout', 64)

            # Check if no path exists (set during navigate() call)
            if hasattr(self, 'navigate_no_path') and self.navigate_no_path:
                failure_reason = f"No path to target (unreachable)"
                self.complete_current_action(SymbolicActionStatus.FAILED, failure_reason)
                return {"status": "failed", "failure_reason": failure_reason, "terminate_plan": True}

            # Check timeout
            if elapsed >= timeout:
                failure_reason = f"Navigate timeout after {elapsed} steps"
                self.complete_current_action(SymbolicActionStatus.FAILED, failure_reason)
                return {"status": "failed", "failure_reason": failure_reason, "terminate_plan": True}

            # Check if reached the interaction zone around the target. MA-Crafter
            # cooperative collection does not require facing the resource.
            if world_state and self.navigate_target_pos:
                if is_near_target(
                    world_state,
                    self.agent_id,
                    self.navigate_target_pos,
                    radius=getattr(self, "navigate_radius", 1),
                ):
                    self.complete_current_action(SymbolicActionStatus.SUCCESS)
                    return {"status": "success"}

            return {"status": "pending"}

        # Unknown action type - complete immediately
        return {"status": "pending"}

    def _collect_target_mismatch_reason(
        self,
        args: Dict[str, Any],
        action_outcome: Dict[str, Any],
    ) -> Optional[str]:
        expected = normalize_resource_name(
            args.get("target")
            or args.get("resource_type")
            or args.get("object_type")
        )
        if not expected:
            return None

        actual = self._collected_resource_names(action_outcome)
        if expected in actual:
            return None

        actual_text = ", ".join(sorted(actual)) if actual else "nothing"
        return f"Expected to collect {expected}, but collected {actual_text}"

    def _collected_resource_names(self, action_outcome: Dict[str, Any]) -> set[str]:
        effects = action_outcome.get("effects") or {}
        actual: set[str] = set()

        for item_name, delta in (effects.get("inventory_delta") or {}).items():
            try:
                changed = float(delta) != 0
            except (TypeError, ValueError):
                changed = bool(delta)
            if changed:
                normalized = normalize_resource_name(item_name)
                if normalized:
                    actual.add(normalized)

        for achievement, delta in (effects.get("achievement_delta") or {}).items():
            try:
                changed = float(delta) != 0
            except (TypeError, ValueError):
                changed = bool(delta)
            if not changed:
                continue
            achievement_name = normalize_resource_name(achievement)
            if achievement_name and achievement_name.startswith("collect_"):
                actual.add(achievement_name.removeprefix("collect_"))
            elif achievement_name == "eat_cow":
                actual.add("food")

        for collection in effects.get("collections") or []:
            resource_type = normalize_resource_name(collection.get("resource_type"))
            if resource_type:
                actual.add(resource_type)
                if resource_type == "tree":
                    actual.add("wood")
                elif resource_type == "cow":
                    actual.add("food")
            for item_name, amount in (collection.get("received") or {}).items():
                try:
                    received = float(amount) != 0
                except (TypeError, ValueError):
                    received = bool(amount)
                if received:
                    normalized = normalize_resource_name(item_name)
                    if normalized:
                        actual.add(normalized)

        return actual

    def _collect_elapsed_steps(self) -> int:
        if self.current_symbolic_action is None:
            return 0
        return max(1, self.current_env_step - self.current_symbolic_action.start_step)

    def _collect_step_limit(self, args: Dict[str, Any]) -> int:
        try:
            steps = int(args.get("steps", 3) or 3)
        except (TypeError, ValueError):
            steps = 3
        return max(1, steps)

    def _should_continue_collect_after_failure(self, args: Dict[str, Any], reason: str) -> bool:
        """Keep a bounded collect action open for near-miss synchronization."""
        if self.current_symbolic_action is None:
            return False
        if self._collect_elapsed_steps() >= self._collect_step_limit(args):
            return False

        reason_text = str(reason or "").lower()
        return reason_text.startswith("requires ") and "agent" in reason_text

    def process_step_completions(self):
        """
        Process action completions after step counter is updated.
        This is now just a wrapper that calls check_termination_condition.
        """
        if self.current_symbolic_action and self.current_symbolic_action.status == SymbolicActionStatus.PENDING:
            # Store world state if available for termination checking
            result = self.check_termination_condition(self.navigate_world_state)
            # Clear world state after use
            self.navigate_world_state = None
            return result
        return {"status": "pending"}

    def noop(self) -> str:
        """Do nothing."""
        self.start_symbolic_action("noop", {})
        return "noop"

    def move(self, direction: str, num_steps: int = 1) -> str:
        """
        Move in a direction for a specified number of steps.

        Args:
            direction: Direction to move (left, right, up, down)
            num_steps: Number of steps to move (default: 1)

        Returns:
            Primitive action string
        """
        direction = direction.lower()

        # Check if we need to start a new move action
        need_new_action = (
            # No current action
            not self.current_symbolic_action or
            # Current action is not a move
            self.current_symbolic_action.action_type != "move" or
            # Direction has changed
            self.move_direction != direction or
            # Previous move action has completed
            self.current_symbolic_action.status != SymbolicActionStatus.PENDING or
            # Move parameters have changed
            (self.current_symbolic_action.args and
             self.current_symbolic_action.args.get("num_steps") != num_steps) or
            # Steps remaining is already 0 or negative (corrupted state)
            self.move_steps_remaining <= 0
        )

        if need_new_action:
            # Start completely fresh move action
            self.start_symbolic_action("move", {"direction": direction, "num_steps": num_steps})
            self.move_steps_remaining = num_steps
            self.move_direction = direction

        # Execute one step of movement - decrement remaining steps
        self.move_steps_remaining -= 1

        # Return primitive action (termination will be checked after step)
        return f"move_{direction}"

    def collect(
        self,
        target: Optional[str] = None,
        resource_type: Optional[str] = None,
        task_id: Optional[Any] = None,
        resource_id: Optional[Any] = None,
        target_id: Optional[Any] = None,
        item_id: Optional[Any] = None,
        steps: int = 3,
    ) -> str:
        """
        Attempt to collect a resource for a bounded number of primitive steps.
        All participating agents receive the resource if participation requirements are met.

        Returns:
            Primitive action string
        """
        expected = normalize_resource_name(target or resource_type)
        args = {"target": expected} if expected else {}
        for key, value in {
            "task_id": task_id,
            "resource_id": resource_id,
            "target_id": target_id,
            "item_id": item_id,
        }.items():
            if value is not None:
                args[key] = value
        try:
            step_limit = int(steps or 3)
        except (TypeError, ValueError):
            step_limit = 3
        args["steps"] = max(1, step_limit)
        self.start_symbolic_action("collect", args)
        self.pending_collect = dict(args)
        return "do"

    def craft(self, object_type: str, participating_agents: Optional[List[int]] = None) -> str:
        """
        Craft an item. All participating agents receive the crafted item
        if participation requirements are met.

        Args:
            object_type: Type of object to craft (wood_pickaxe, stone_sword, etc.)
            participating_agents: List of agent IDs participating (optional)

        Returns:
            Primitive action string
        """
        object_type = self._process_object_type(object_type)
        self.start_symbolic_action("craft", {"object_type": object_type})
        return f"make_{object_type}"

    def place(self, object_type: str) -> str:
        """
        Place an object in the environment.

        Args:
            object_type: Type of object to place (stone, table, furnace, plant)

        Returns:
            Primitive action string
        """
        object_type = self._process_object_type(object_type)
        self.pending_place = {"object_type": object_type}
        self.start_symbolic_action("place", {"object_type": object_type})
        return f"place_{object_type}"

    def sleep(self) -> str:
        """Sleep to recover energy."""
        self.start_symbolic_action("sleep", {})
        return "sleep"

    def share(self, recipient_agent_id: str, resource_type: str, quantity: int = 1) -> str:
        """
        Share resources/tools with another agent.

        Args:
            recipient_agent_id: ID of the agent to share with (e.g., 'agent_1' or '1')
            resource_type: Type of resource/tool to share (anything except 'health')
            quantity: Amount to share (default 1)

        Returns:
            Primitive action string 'share' - actual transfer handled by wrapper
        """
        # Normalize resource type
        resource_type = resource_type.lower().replace(" ", "_")

        # Store share parameters for the wrapper to execute
        self.pending_share = {
            "recipient_agent_id": recipient_agent_id,
            "resource_type": resource_type,
            "quantity": quantity
        }

        self.start_symbolic_action("share", {
            "recipient_agent_id": recipient_agent_id,
            "resource_type": resource_type,
            "quantity": quantity
        })

        return "share"

    def navigate(
        self,
        object_type: str,
        item_id: int,
        timeout: int = 64,
        world_state: Optional[Any] = None,
        radius: int = 1,
        distance_threshold: Optional[int] = None,
        interaction_radius: Optional[int] = None,
    ) -> str:
        """
        Navigate to a specific object. This is a multi-step action.

        Args:
            object_type: Type of entity ('material', 'object', 'agent')
            item_id: Stable ID of the item to navigate to
            timeout: Maximum number of steps before failing (default 64)
            world_state: SymbolicWorldState instance with object positions
            radius: Distance from target at which navigation is complete

        Returns:
            Primitive action string for current step, or "no_path" if navigation is impossible
        """
        nav_radius = max(
            0,
            int(
                interaction_radius
                if interaction_radius is not None
                else distance_threshold
                if distance_threshold is not None
                else radius
            ),
        )

        # Start new navigate action if not already navigating
        if not self.current_symbolic_action or self.current_symbolic_action.action_type != "navigate":
            self.start_symbolic_action("navigate", {
                "object_type": object_type,
                "item_id": item_id,
                "timeout": timeout,
                "radius": nav_radius,
            })
            self.navigate_target_pos = None
            self.navigate_no_path = False
            self.navigate_radius = nav_radius

            # Try to find target position from world state
            if world_state:
                target_pos = find_object_position(world_state, item_id, object_type=object_type)
                if target_pos is None:
                    # Object not found - mark as no path
                    self.navigate_target_pos = None
                    self.navigate_world_state = world_state
                    self.navigate_no_path = True
                    return "no_path"
                self.navigate_target_pos = target_pos

        # Store world state for termination checking after step
        self.navigate_world_state = world_state

        # If no target position, can't navigate
        if self.navigate_target_pos is None:
            return "no_path"

        if world_state and is_near_target(
            world_state,
            self.agent_id,
            self.navigate_target_pos,
            radius=getattr(self, "navigate_radius", 1),
        ):
            return "noop"

        # Calculate next move toward the nearest reachable interaction cell.
        if world_state:
            next_action = get_move_towards_target(
                world_state,
                self.agent_id,
                self.navigate_target_pos,
                interaction_radius=getattr(self, "navigate_radius", 1),
            )
            if next_action is None:
                # No path exists - mark for failure
                self.navigate_no_path = True
                return "no_path"
            return next_action

        # No world state available
        return "no_path"

    def get_action_value(self, action_name: str) -> int:
        """
        Convert step-level action name to primitive action value for the environment.

        Args:
            action_name: Step-level action name (e.g., "move_left", "do")

        Returns:
            Integer primitive action value recognized by environment
        """
        return ACTION_NAME_TO_VALUE.get(action_name, ACTION_NAME_TO_VALUE["noop"])

    def execute(self, action_type: str, world_state: Optional[Dict] = None,
                current_step: Optional[int] = None, **kwargs) -> str:
        """
        Execute a symbolic action and return the step-level action string.

        Args:
            action_type: Type of symbolic action
            world_state: Current world state (needed for navigate)
            current_step: Current environment step (for tracking primitive_action_history)
            **kwargs: Action-specific arguments

        Returns:
            Step-level action string (e.g., "move_left", "do")
        """
        action_func = self.actions.get(action_type)
        if action_func is None:
            return "noop"

        # Call the action function with provided arguments
        import inspect
        sig = inspect.signature(action_func)

        # Filter kwargs to only include parameters that the function accepts
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Add world_state if function accepts it
        if 'world_state' in sig.parameters and world_state is not None:
            valid_kwargs['world_state'] = world_state

        step_action = action_func(**valid_kwargs)

        # Store the step-level action in current record
        if self.current_symbolic_action:
            # For multi-step actions like navigate, update primitive_action each step
            # For single-step actions, only set once
            if self.current_symbolic_action.action_type in ['navigate', 'move']:
                # Always update for multi-step actions to show current step action
                self.current_symbolic_action.primitive_action = step_action
                # Record per-step history for visualization
                if current_step is not None:
                    self.current_symbolic_action.primitive_action_history[current_step] = step_action
            elif self.current_symbolic_action.primitive_action is None:
                # For single-step actions, set once
                self.current_symbolic_action.primitive_action = step_action
                # Also record in history for consistency
                if current_step is not None:
                    self.current_symbolic_action.primitive_action_history[current_step] = step_action

        return step_action

    def get_action_records(self) -> List[Dict]:
        """Get all completed action records."""
        records = [record.to_dict() for record in self.action_history]
        if self.current_symbolic_action:
            records.append(self.current_symbolic_action.to_dict())
        return records

    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()
        self.reset_current_action()
