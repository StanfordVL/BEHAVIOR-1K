"""
Symbolic wrapper for ma-crafter environment.

This package provides a high-level interface for interacting with the ma-crafter environment
using symbolic actions instead of low-level action IDs.
"""

from .action import SymbolicEnvWrapper, SymbolicActionExecutor, SymbolicAction
from .plan import PlanningEnvWrapper, SymbolicPlan, SymbolicPlanStatus, SymbolicPlanLogger, SymbolicPlanExecutor
from .constants import ACTION_NAME_TO_VALUE, ACTION_SCHEMA
from .agent import (
    Agent,
    SimpleAgent,
    AgentState,
    AgentMemory,
    BaseLLMAgent,
    LLMClient,
    LLMPlanResponse,
    LLMMessageResponse,
    LLMInterruptResponse,
    InterruptDecision,
    load_env_file,
    build_system_prompt,
    build_observation_prompt,
    build_message_prompt,
    get_env_description,
    format_agent_states,
    format_agent_status,
    build_plan_prompt,
    build_interrupt_prompt,
    parse_plan_response,
    parse_interrupt_response,
)
from .messages import MessageBroker
from .viz import (
    visualize_plan_timeline,
    visualize_all_agents_progress,
    print_plan_summary,
    visualize_comprehensive_timeline,
    build_repair_intervention_report,
    visualize_repair_interventions,
)
from .compute_metrics import (
    compute_all_metrics,
    compute_decision_overhead_metrics,
    compute_communication_metrics,
    compute_planning_metrics,
    compute_capability_metrics,
    compute_constraint_metrics,
    compute_task_success_metrics,
    compute_failure_attribution,
    print_metrics_summary,
    save_metrics,
    save_metrics_csv,
    load_logs,
)

# Simple wrapper functions for convenient action creation
def move(direction):
    """Create a move action. Usage: move('left')"""
    return {'action_type': 'move', 'args': {'direction': direction}}

def collect(leader_agent, object_type, object_id, collaborating_agents=None):
    """Create a collect action."""
    if collaborating_agents is None:
        collaborating_agents = []
    return {
        'action_type': 'collect',
        'args': {
            'leader_agent': leader_agent,
            'object_type': object_type,
            'object_id': object_id,
            'collaborating_agents': collaborating_agents
        }
    }

def craft(leader_agent, object_type, collaborating_agents=None):
    """Create a craft action."""
    if collaborating_agents is None:
        collaborating_agents = []
    return {
        'action_type': 'craft',
        'args': {
            'leader_agent': leader_agent,
            'object_type': object_type,
            'collaborating_agents': collaborating_agents
        }
    }

def place(object_type):
    """Create a place action."""
    return {'action_type': 'place', 'args': {'object_type': object_type}}

def sleep():
    """Create a sleep action."""
    return {'action_type': 'sleep', 'args': {}}

def share(recipient_agent_id, resource_type, quantity=1):
    """
    Create a share action to transfer resources/tools to another agent.
    
    Args:
        recipient_agent_id: ID of the agent to share with (e.g., 'agent_1' or '1')
        resource_type: Type of resource/tool to share (anything except 'health')
        quantity: Amount to share (default 1)
    
    Returns:
        Share action dict
    """
    return {
        'action_type': 'share',
        'args': {
            'recipient_agent_id': recipient_agent_id,
            'resource_type': resource_type,
            'quantity': quantity
        }
    }

def noop():
    """Create a no-operation action."""
    return {'action_type': 'noop', 'args': {}}

__all__ = [
    'SymbolicEnvWrapper',
    'SymbolicActionExecutor',
    'ACTION_NAME_TO_VALUE',
    'ACTION_SCHEMA',
    'SymbolicPlan',
    'SymbolicAction',
    'SymbolicPlanStatus',
    'SymbolicPlanLogger',
    'SymbolicPlanExecutor',
    'Agent',
    'SimpleAgent',
    'MessageBroker',
    'visualize_plan_timeline',
    'visualize_all_agents_progress',
    'print_plan_summary',
    'visualize_comprehensive_timeline',
    'build_repair_intervention_report',
    'visualize_repair_interventions',
    # Metrics
    'compute_all_metrics',
    'compute_decision_overhead_metrics',
    'compute_communication_metrics',
    'compute_planning_metrics',
    'compute_capability_metrics',
    'compute_constraint_metrics',
    'compute_task_success_metrics',
    'compute_failure_attribution',
    'print_metrics_summary',
    'save_metrics',
    'save_metrics_csv',
    'load_logs',
    # Helper functions
    'move',
    'collect', 
    'craft',
    'place',
    'sleep',
    'share',
    'noop'
]
