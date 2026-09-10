"""
Prompt construction utilities for LLM agents.

This module provides clean, modular functions for building prompts
that describe the environment, agent states, and observations.
"""

import json
from typing import Dict, List, Any, Optional
from enum import Enum

from ..coop2_messages import (
    get_coop2_repair_context,
    has_coop2_repair_message,
    is_coop2_repair_content,
    summarize_coop2_repair_content,
)


# ============================================================================
# Environment Description
# ============================================================================

ENV_DESCRIPTION = """You are one of several robots working together in a house.

You act through high-level primitives, not joint commands. Each one takes many
simulation steps: driving across a room costs roughly 60 ticks per metre, so
distance is the main cost you control. You cannot see -- you are given a
symbolic description of the room you are standing in, and only that room.

Rules that decide whether an action succeeds:
- To grasp, place, open or toggle an object you must be closer to it than that
  object's own distance threshold; beyond it the action fails with TOO_FAR.
  Under "You can do:" a distance in metres is shown ONLY for objects that are
  currently out of range -- no distance means you are already close enough to
  act on it now.
- An out-of-range object is marked "unreachable" and offers navigate_to and
  NOTHING else. You cannot grasp, place, open or toggle an unreachable object,
  however close it looks in the room listing: you must navigate_to it first.
- If there is nothing useful to do right now -- a teammate is already handling
  the only target you could work on, or you are waiting on something to
  finish -- issue wait rather than acting anyway. wait holds your position for
  the number of ticks you give it, during which the others make progress.
- One object at a time: grasp needs an empty gripper, place needs a full one.
- Objects are exclusive. If a teammate is holding something, your grasp fails
  with "held by <agent>". Going after a target a teammate already has costs you
  the whole trip for nothing.
- Refer to objects only by the ids listed under "You can do:", which are the
  objects in the room you are standing in. They look like apple.n.01_1. Never
  invent or guess one. An entry marked "blocked" is in that room but currently
  unavailable -- the bracketed note says why.

When an action fails, your plan is abandoned and you are asked to think again.
Read the failure reason before replanning -- it tells you whether to wait,
approach, or pick a different target."""


def get_env_description() -> str:
    """Get the environment description."""
    return ENV_DESCRIPTION


# ============================================================================
# Agent State Formatting
# ============================================================================

class AgentStateCode(str, Enum):
    """Single-letter codes for agent states."""
    REASONING = "R"      # Agent is thinking/planning
    INTERRUPTED = "I"    # Agent was interrupted by message
    EXECUTING = "X"      # Agent is executing a plan
    WAITING = "W"        # Agent is waiting (ready for next step)


def get_state_code(state_value: str) -> str:
    """Convert agent state to single-letter code."""
    state_map = {
        "reasoning": "R",
        "interrupted": "I",
        "executing": "X",
        "waiting": "W",
        "idle": "W",
    }
    return state_map.get(state_value.lower(), "?")


def format_agent_states(
    current_agent_id: str,
    all_agents: Dict[str, Any],
) -> str:
    """
    Format information about all agents and their states.
    
    Args:
        current_agent_id: ID of the agent receiving this prompt
        all_agents: Dict mapping agent_id -> agent object (with .state attribute)
    
    Returns:
        Formatted string describing agent states
    """
    lines = []
    lines.append(f"TEAM ({len(all_agents)} agents):")
    lines.append(f"You are: {current_agent_id}")
    
    # Explicitly list collaborators
    collaborators = [a for a in all_agents.keys() if a != current_agent_id]
    if collaborators:
        lines.append(f"Your collaborators: {collaborators}")
    
    lines.append("")
    lines.append("Agent States (R=Reasoning, I=Interrupted, X=Executing, W=Waiting):")
    
    for agent_id, agent in all_agents.items():
        # Get state code
        if hasattr(agent, 'state'):
            state_value = agent.state.value if hasattr(agent.state, 'value') else str(agent.state)
            state_code = get_state_code(state_value)
        else:
            state_code = "?"
        
        # Mark current agent
        marker = " <- YOU" if agent_id == current_agent_id else ""
        lines.append(f"  {agent_id}: [{state_code}]{marker}")
    
    return "\n".join(lines)


def format_agent_states_simple(
    current_agent_id: str,
    agent_names: List[str],
    agent_states: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format agent states from simple lists/dicts.
    
    Args:
        current_agent_id: ID of the agent receiving this prompt
        agent_names: List of all agent IDs
        agent_states: Optional dict mapping agent_id -> state string
    
    Returns:
        Formatted string describing agent states
    """
    lines = []
    lines.append(f"TEAM ({len(agent_names)} agents):")
    lines.append(f"You are: {current_agent_id}")
    
    # Explicitly list collaborators
    collaborators = [a for a in agent_names if a != current_agent_id]
    if collaborators:
        lines.append(f"Your collaborators: {collaborators}")
    
    if agent_states:
        lines.append("")
        lines.append("Agent States (R=Reasoning, I=Interrupted, X=Executing, W=Waiting):")
        for agent_id in agent_names:
            state = agent_states.get(agent_id, "unknown")
            state_code = get_state_code(state)
            marker = " <- YOU" if agent_id == current_agent_id else ""
            lines.append(f"  {agent_id}: [{state_code}]{marker}")
    
    return "\n".join(lines)


# ============================================================================
# Agent Status Formatting (Health, Tools, Resources)
# ============================================================================

def format_agent_status(
    holding: Optional[str] = None,
    room: Optional[str] = None,
    last_error: Optional[str] = None,
    teammates: Optional[Dict[str, Any]] = None,
    **legacy: Any,
) -> str:
    """Format the agent's current status.

    crafter reported health/food/drink/energy and an inventory. None of that
    exists here: a robot has no vitals, and it carries at most one object. What
    replaces it is what actually gates the next action -- what is in the
    gripper, which room the agent is in, and why the last action failed.

    ``**legacy`` swallows the old keyword names so a caller that still passes
    ``health=`` gets a status block rather than a TypeError.

    Args:
        holding: entity id in the gripper, or None.
        room: room instance the agent is standing in.
        last_error: failure reason from the previous action.
        teammates: ``{agent_id: what they hold}`` -- the cross-agent view that
            makes contention visible before it costs a trip.
    """
    lines = []
    lines.append("YOUR STATUS:")
    lines.append(f"  Holding: {holding}" if holding else "  Holding: nothing")
    lines.append(f"  Room: {room}" if room else "  Room: unknown")
    if teammates:
        for agent_id, held in sorted(teammates.items()):
            lines.append(f"  {agent_id} is holding: {held or 'nothing'}")
    if last_error:
        lines.append(f"  Last action failed: {last_error}")
    inventory = legacy.get("inventory")
    
    # Inventory/Resources
    if inventory:
        resources = [f"{item}={count}" for item, count in inventory.items() if count > 0]
        if resources:
            lines.append(f"  Resources: {', '.join(resources)}")
        else:
            lines.append("  Resources: empty")
    
    return "\n".join(lines)


def format_status_from_dict(status_dict: Dict[str, Any]) -> str:
    """
    Format agent status from a dictionary (e.g., from observation).
    
    Expected keys: health, food, drink, energy, inventory, tools/equipment
    """
    return format_agent_status(
        health=status_dict.get('health'),
        food=status_dict.get('food'),
        drink=status_dict.get('drink'),
        energy=status_dict.get('energy'),
        tools=status_dict.get('tools') or status_dict.get('equipment'),
        inventory=status_dict.get('inventory'),
    )


# ============================================================================
# Observation Formatting
# ============================================================================

def format_visible_area(semantic_grid: Any) -> str:
    """
    Format the visible area from a semantic grid.
    
    Args:
        semantic_grid: Grid representation of what the agent sees
    
    Returns:
        Formatted string describing visible objects
    """
    if semantic_grid is None:
        return "Visible area: unknown"
    
    # If it's a simple string or already formatted
    if isinstance(semantic_grid, str):
        return f"Visible area:\n{semantic_grid}"
    
    # If it's a dict of object counts
    if isinstance(semantic_grid, dict):
        items = [f"{k}: {v}" for k, v in semantic_grid.items() if v > 0]
        if items:
            return f"Nearby objects: {', '.join(items)}"
        return "Nearby objects: none visible"
    
    # Fallback
    return f"Visible area: {semantic_grid}"


def format_position(position: Any, facing: Optional[str] = None) -> str:
    """Format agent position and facing direction."""
    lines = []
    if position is not None:
        lines.append(f"Position: {position}")
    if facing is not None:
        lines.append(f"Facing: {facing}")
    return "\n".join(lines) if lines else ""


def format_memory(memory_events: List[Dict], max_events: int = 5) -> str:
    """
    Format memory events for inclusion in prompts.
    
    Args:
        memory_events: List of event dicts from AgentMemory.get_events()
        max_events: Maximum number of recent events to include
        
    Returns:
        Formatted string describing recent memory
    """
    if not memory_events:
        return ""
    
    # Take most recent events
    recent = memory_events[-max_events:] if len(memory_events) > max_events else memory_events
    
    lines = ["RECENT MEMORY:"]
    for event in recent:
        step = event.get('env_step', '?')
        event_type = event.get('type', '')
        
        if event_type == 'message_in':
            sender = event.get('sender', '?')
            content = format_message_content_for_prompt(event.get('content', ''))
            lines.append(f"  [Step {step}] Received from {sender}: {content}")
        elif event_type == 'message_out':
            recipients = event.get('recipients', [])
            content = event.get('content', '')
            lines.append(f"  [Step {step}] Sent to {recipients}: {content}")
        elif event_type == 'plan':
            spec = event.get('specification', '?')
            plan_id = event.get('plan_id', '?')
            actions = event.get('actions', [])
            lines.append(f"  [Step {step}] Plan #{plan_id}: {spec}")
            if actions:
                action_str = ", ".join(str(a) for a in actions[:3])
                if len(actions) > 3:
                    action_str += f"... ({len(actions)} actions)"
                lines.append(f"    Actions: {action_str}")
        elif event_type == 'plan_failure':
            plan_id = event.get('plan_id', '?')
            spec = event.get('specification', '?')
            failed_action = event.get('failed_action', '?')
            reason = event.get('failure_reason', '?')
            lines.append(
                f"  [Step {step}] Plan #{plan_id} failed: {spec}; "
                f"failed_action={failed_action}; reason={reason}"
            )
            if "No path" in str(reason) or "unreachable" in str(reason).lower():
                lines.append("    Avoid retrying that same item_id unless a new observation shows a better path.")
    
    return "\n".join(lines)


def format_message_content_for_prompt(content: Any) -> str:
    """Keep structured repair messages readable instead of dumping a huge dict inline."""
    if is_coop2_repair_content(content):
        return summarize_coop2_repair_content(content)
    if isinstance(content, (dict, list)):
        return json.dumps(content, default=str)
    return str(content)


def _format_compact_mapping(value: Any, max_chars: int = 220) -> str:
    text = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def format_coop2_repair_context(messages: Optional[List[Dict]]) -> str:
    """Format structured repair evidence for replanning."""
    context = get_coop2_repair_context(messages)
    if not context:
        return ""

    lines = ["COOP2 REPAIR EVIDENCE:"]
    lines.append(f"- Repair step: {context.get('env_step', '?')}")

    guidance = context.get("repair_guidance") or {}
    if guidance:
        lines.append("- Soft repair guidance:")
        strategy = guidance.get("strategy")
        if strategy:
            lines.append(f"  strategy: {strategy}")
        for instruction in guidance.get("instructions") or []:
            lines.append(f"  - {instruction}")
        recommended = guidance.get("recommended_target")
        if recommended:
            lines.append(
                "  recommended shared target: "
                f"{_format_compact_mapping(recommended, 420)}"
            )
            participants = recommended.get("recommended_participants") or []
            timeout = recommended.get("recommended_timeout")
            if participants:
                lines.append(f"  recommended participants: {_format_compact_mapping(participants, 180)}")
            if timeout:
                lines.append(f"  use navigate timeout at least {timeout} for this repair target")
        recommended_plans = guidance.get("recommended_plans") or {}
        if recommended_plans:
            lines.append("  recommended per-agent plan skeletons:")
            for agent_id, plan in list(recommended_plans.items())[:6]:
                lines.append(f"    - {agent_id}: {_format_compact_mapping(plan, 700)}")
        policy = guidance.get("recommendation_policy")
        if policy:
            lines.append(f"  recommendation policy: {policy}")
        candidates = guidance.get("reachable_shared_targets") or []
        if candidates:
            lines.append("  reachable shared target candidates:")
            for candidate in candidates[:5]:
                lines.append(f"    - {_format_compact_mapping(candidate, 420)}")
        dependency_hints = guidance.get("dependency_prerequisites") or []
        if dependency_hints:
            lines.append("  dependency prerequisite hints:")
            for hint in dependency_hints[:5]:
                lines.append(f"    - {_format_compact_mapping(hint, 520)}")
        blocked_targets = guidance.get("blocked_targets_after_prerequisites") or []
        if blocked_targets:
            lines.append("  blocked targets to retry only after prerequisites:")
            for candidate in blocked_targets[:3]:
                lines.append(f"    - {_format_compact_mapping(candidate, 420)}")

    failures = context.get("failures") or []
    lines.append(f"- Predicted failures: {len(failures)}")
    for index, failure in enumerate(failures[:8], start=1):
        task_id = failure.get("task_id") or failure.get("task") or "unknown_task"
        constraint = failure.get("constraint_type") or "unknown_constraint"
        agents = failure.get("agents") or failure.get("affected_agents") or []
        details = failure.get("details") or failure.get("metadata") or {}
        lines.append(
            f"  {index}. {task_id}: {constraint}; agents={_format_compact_mapping(agents, 120)}; "
            f"details={_format_compact_mapping(details, 260)}"
        )
    if len(failures) > 8:
        lines.append(f"  ... {len(failures) - 8} additional failure(s) omitted")

    plan_views = context.get("plan_views") or []
    if plan_views:
        lines.append("- Committed plan views before repair:")
        for view in plan_views[:12]:
            agent_id = view.get("agent_id", "?")
            task = view.get("task") or view.get("specification") or view.get("goal") or "unknown"
            status = (view.get("metadata") or {}).get("plan_status", view.get("status", "?"))
            actions = view.get("remaining_actions") or view.get("actions") or []
            compact_actions = [_format_compact_mapping(action, 90) for action in actions[:4]]
            if len(actions) > 4:
                compact_actions.append(f"... +{len(actions) - 4} more")
            lines.append(f"  - {agent_id}: status={status}, task={task}, actions={compact_actions}")

    channel = context.get("repair_channel") or {}
    statements = channel.get("statements") or []
    if statements:
        lines.append("- Ordered repair-channel statements:")
        for statement in statements:
            agent_id = statement.get("agent_id", "?")
            text = str(statement.get("statement", "")).strip()
            if len(text) > 280:
                text = text[:277] + "..."
            lines.append(f"  - {agent_id}: {text}")

    revision_instruction = channel.get("revision_instruction")
    if revision_instruction:
        lines.append(f"- Channel revision instruction: {revision_instruction}")

    return "\n".join(lines)


def format_coop2_repair_instruction(messages: Optional[List[Dict]]) -> str:
    """Add direct repair-planning guidance when a COOP2 precheck interrupted the agent."""
    if not has_coop2_repair_message(messages):
        return ""
    return "\n".join([
        "COOP2 REPAIR OBJECTIVE:",
        "- Repair the predicted failure while keeping your system/topology role.",
        "- Use the evidence and repair guidance as recommendations; preserve useful progress from previous plans.",
        "- If guidance names a shared target, participants, or timeout, prefer those values unless current observation makes them infeasible.",
        "- For spatial/temporal failures, converge on one object_id and synchronize collect.",
        "- For dependency failures, get the required tools/resources first; ready agents may move to the target and wait.",
        "- Avoid failed, collected, or unreachable item_id targets unless they are reachable now.",
    ])


# ============================================================================
# Complete Prompt Building
# ============================================================================

def build_system_prompt(
    agent_id: str,
    max_actions: int = 6,
    include_env_description: bool = True,
) -> str:
    """
    Build the system prompt for an LLM agent.
    
    Args:
        agent_id: The agent's identifier
        max_actions: Maximum actions per plan
        include_env_description: Whether to include full env description
    
    Returns:
        Complete system prompt
    """
    parts = []
    
    # Identity
    # "crafting game" was crafter's wording and survived the port. It contradicted
    # the very next line of ENV_DESCRIPTION -- "one of several robots working
    # together in a house" -- so the prompt opened by telling the model two
    # different things about what it was.
    parts.append(f"You are agent '{agent_id}' in a multi-agent cooperative household task.")
    parts.append("")
    
    # Environment description
    if include_env_description:
        parts.append(ENV_DESCRIPTION)
        parts.append("")
    
    # Wrapped to the same ~78 columns as the rules above, with two-space
    # continuations. These four bullets were the only unwrapped lines in the
    # whole system prompt (89, 87 and 95 characters against a hard wrap
    # everywhere else), which is what made the line breaks look arbitrary.
    parts.append(f"""PLAN RESPONSE:
- Return a structured plan that matches the response schema.
- Choose one task and at most {max_actions} actions; 3-6 short executable
  actions are usually enough.
- Use exact item_id values from the current prompt for navigation and
  resource targets.
- Coordinate when the cooperative config, messages, or repair evidence
  require multiple agents.
- Briefly explain why this plan is the next useful step.""")
    
    return "\n".join(parts)


def build_observation_prompt(
    env_step: int,
    agent_id: str,
    all_agents: Optional[Dict[str, Any]] = None,
    agent_names: Optional[List[str]] = None,
    agent_states: Optional[Dict[str, str]] = None,
    status: Optional[Dict[str, Any]] = None,
    position: Any = None,
    facing: Optional[str] = None,
    visible_area: Any = None,
    messages: Optional[List[Dict]] = None,
    memory: Optional[List[Dict]] = None,
    coop_config: Optional[str] = None,
    symbolic_view: Optional[str] = None,
    target_hints: Optional[str] = None,
) -> str:
    """
    Build the observation/user prompt for plan generation.
    
    Args:
        env_step: Current environment step
        agent_id: The agent's identifier
        all_agents: Dict of agent_id -> agent object (alternative to agent_names)
        agent_names: List of agent IDs (alternative to all_agents)
        agent_states: Dict of agent_id -> state string
        status: Agent status dict (health, inventory, etc.)
        position: Agent position
        facing: Agent facing direction
        visible_area: Semantic grid or visible objects
        messages: List of received messages
        memory: List of memory events from AgentMemory.get_events()
        coop_config: Formatted cooperative configuration string from env.get_config_observation()
        symbolic_view: Symbolic view text from coop_env info['symbolic_view']
        target_hints: Compact current target list from env info['target_hints']
    
    Returns:
        Complete observation prompt
    """
    parts = []
    parts.append(f"=== STEP {env_step} ===")
    parts.append("")
    
    # Cooperative configuration (important for understanding requirements)
    if coop_config:
        parts.append(coop_config)
        parts.append("")
    
    # Agent states
    if all_agents:
        parts.append(format_agent_states(agent_id, all_agents))
    elif agent_names:
        parts.append(format_agent_states_simple(agent_id, agent_names, agent_states))
    parts.append("")
    
    # Agent status
    if status:
        parts.append(format_status_from_dict(status))
        parts.append("")
    
    # Position
    pos_str = format_position(position, facing)
    if pos_str:
        parts.append(pos_str)
        parts.append("")
    
    # Visible area
    if visible_area is not None:
        parts.append(format_visible_area(visible_area))
        parts.append("")
    
    # Symbolic view (detailed view with entity IDs). Its own tail is the action
    # catalogue under "You can do:", built from the same target_hints() call the
    # env renders into info["target_hints"] -- so appending both put every id in
    # the room twice, and each out-of-range one four times (a navigate_to line
    # and an unreachable line, both repeating the distance). The grouped form is
    # the one ENV_DESCRIPTION tells the agent to read ids from, so it wins; the
    # flat form stays as the fallback for a caller that has hints but no view.
    if symbolic_view:
        parts.append("SYMBOLIC VIEW:")
        parts.append(symbolic_view)
        parts.append("")
    elif target_hints:
        parts.append("YOU CAN DO:")
        parts.append(target_hints)
        parts.append("")

    # Memory (recent events)
    if memory:
        memory_str = format_memory(memory)
        if memory_str:
            parts.append(memory_str)
            parts.append("")
    
    # Messages from other agents (current step - may overlap with memory)
    if messages:
        parts.append("MESSAGES RECEIVED:")
        for msg in messages:
            sender = msg.get('sender', 'unknown')
            content = format_message_content_for_prompt(msg.get('content', ''))
            parts.append(f"  From {sender}: {content}")
        parts.append("")

    repair_context = format_coop2_repair_context(messages)
    if repair_context:
        parts.append(repair_context)
        parts.append("")

    repair_instruction = format_coop2_repair_instruction(messages)
    if repair_instruction:
        parts.append(repair_instruction)
        parts.append("")

    if coop_config and "TEAM SCORE OBJECTIVE" in coop_config:
        parts.append("Generate the next short executable plan using the current context.")
    else:
        parts.append("Generate the next short executable plan.")
    
    return "\n".join(parts)


def build_message_prompt(
    env_step: int,
    agent_id: str,
    other_agents: List[str],
    current_task: Optional[str] = None,
    status: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> str:
    """
    Build a prompt for message generation.
    
    Args:
        env_step: Current environment step
        agent_id: The agent's identifier
        other_agents: List of other agent IDs that can receive messages
        current_task: Current task being worked on
        status: Agent status dict
        context: Additional context for message generation
    
    Returns:
        Prompt for message generation
    """
    parts = []
    parts.append(f"=== STEP {env_step} ===")
    parts.append(f"You are: {agent_id}")
    parts.append(f"Other agents you can message: {other_agents}")
    parts.append("")
    
    if current_task:
        parts.append(f"Your current task: {current_task}")
    
    if status:
        parts.append(format_status_from_dict(status))
    
    if context:
        parts.append("")
        parts.append(f"Context: {context}")
    
    parts.append("")
    parts.append("Generate a brief coordination message only if it helps the current task.")
    
    return "\n".join(parts)
