"""
Agent module for MA-Crafter.

This module contains:
- Base Agent class and SimpleAgent implementation
- LLM client and response models
- Cognitive utilities for prompt building and response parsing
- Agent memory system
- Prompt construction utilities
"""

from .agent import Agent, SimpleAgent, AgentState
from .memory import AgentMemory
from .base_llm_agent import BaseLLMAgent
from .llm_client import (
    LLMClient,
    LLMPlanResponse,
    LLMMessageResponse,
    LLMInterruptResponse,
    InterruptDecision,
    Direction,
    Resource,
    PlaceableItem,
    CraftableItem,
    Task,
    TaskSpecification,
    MoveAction,
    CollectAction,
    PlaceAction,
    CraftAction,
    SleepAction,
    NoopAction,
    NavigateAction,
    ShareAction,
    LLMAction,
    load_env_file,
)
from .prompts import (
    build_system_prompt,
    build_observation_prompt,
    build_message_prompt,
    get_env_description,
    format_agent_states,
    format_agent_status,
    format_memory,
)
from .cognitive_agent import (
    build_plan_prompt,
    build_interrupt_prompt,
    parse_plan_response,
    parse_interrupt_response,
    extract_status,
    extract_position,
    extract_facing,
    extract_visible_area,
    extract_action_parameters,
)

__all__ = [
    # Base agent
    'Agent',
    'SimpleAgent',
    'AgentState',
    'AgentMemory',
    'BaseLLMAgent',
    # LLM client and models
    'LLMClient',
    'LLMPlanResponse',
    'LLMMessageResponse',
    'LLMInterruptResponse',
    'InterruptDecision',
    'Direction',
    'Resource',
    'PlaceableItem',
    'CraftableItem',
    'Task',
    'TaskSpecification',
    'MoveAction',
    'CollectAction',
    'PlaceAction',
    'CraftAction',
    'SleepAction',
    'NoopAction',
    'NavigateAction',
    'ShareAction',
    'LLMAction',
    'load_env_file',
    # Prompts
    'build_system_prompt',
    'build_observation_prompt',
    'build_message_prompt',
    'get_env_description',
    'format_agent_states',
    'format_agent_status',
    'format_memory',
    # Cognitive utilities
    'build_plan_prompt',
    'build_interrupt_prompt',
    'parse_plan_response',
    'parse_interrupt_response',
    'extract_status',
    'extract_position',
    'extract_facing',
    'extract_visible_area',
    'extract_action_parameters',
]
