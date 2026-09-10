"""
LLM Client and Response Models for MA-Crafter.

This module provides:
- Pydantic models for structured LLM responses (plans, messages, interrupts)
- LLMClient wrapper for Azure OpenAI and OpenAI-compatible API calls
"""

import os
import time
from pathlib import Path
from typing import Any, Optional, List, Dict, Union, Literal
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI, RateLimitError, APIError
from pydantic import BaseModel, Field


# Load local credentials without overriding variables exported by the caller.
# An environment-specific file takes precedence over the repository-level file.
_ENVIRONMENT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ENVIRONMENT_ROOT / ".env", override=False)
load_dotenv(_ENVIRONMENT_ROOT.parent / ".env", override=False)


# ============================================================================
# Enums for constrained values
# ============================================================================

# ============================================================================
# Structured output vocabulary -- BEHAVIOR-1K
# ============================================================================
# This is one third of M5's "three-piece" change (PORTING_PLAN 5): the Pydantic
# schema, the L2 controllers and action_outcome.effects have to agree. Structured
# output *constrains* the model, so whatever is not expressible here is not
# emittable at all -- and anything expressible but unknown to L2 becomes a
# wasted decision. The verbs below are exactly
# coop2.cognitive.action.behavior_action.BEHAVIOR_ACTION_TO_PRIMITIVE plus the
# two communication actions.
#
# Targets are free-form strings, deliberately not enums: the legal set is
# scene-dependent and is handed to the model each turn as `target_hints`. An
# enum would freeze one scene's objects into the schema. They are BDDL instance
# ids (`apple.n.01_1`), which is also how the goal predicates are written.

_TARGET = "Entity id exactly as shown in Current Reachable Targets, e.g. 'apple.n.01_1'. Never invent one."


class NavigateToAction(BaseModel):
    """Drive the base to a standable pose near the target."""

    action_type: Literal["navigate_to"] = "navigate_to"
    target: str = Field(description=_TARGET)


class GraspAction(BaseModel):
    """Pick the target up. Requires being within reach and both hands empty."""

    action_type: Literal["grasp"] = "grasp"
    target: str = Field(description=_TARGET)


class PlaceOnTopAction(BaseModel):
    """Put what you are holding on top of the target."""

    action_type: Literal["place_on_top"] = "place_on_top"
    target: str = Field(description=_TARGET)


class PlaceInsideAction(BaseModel):
    """Put what you are holding inside the target."""

    action_type: Literal["place_inside"] = "place_inside"
    target: str = Field(description=_TARGET)


class ReleaseAction(BaseModel):
    """Drop what you are holding where you stand."""

    action_type: Literal["release"] = "release"


class OpenAction(BaseModel):
    action_type: Literal["open"] = "open"
    target: str = Field(description=_TARGET)


class CloseAction(BaseModel):
    action_type: Literal["close"] = "close"
    target: str = Field(description=_TARGET)


class ToggleOnAction(BaseModel):
    action_type: Literal["toggle_on"] = "toggle_on"
    target: str = Field(description=_TARGET)


class ToggleOffAction(BaseModel):
    action_type: Literal["toggle_off"] = "toggle_off"
    target: str = Field(description=_TARGET)


class WaitAction(BaseModel):
    """Hold position, letting time pass so a teammate can finish.

    Costs real ticks, not zero: the world only advances while some agent has a
    primitive running, so a free wait would stop it rather than yield it.
    """

    action_type: Literal["wait"] = "wait"
    ticks: int = Field(
        default=200,
        ge=1,
        le=600,
        description="How long to hold, in simulation ticks. A teammate's "
                    "navigation takes roughly 300-500, so 200 is a short "
                    "pause and 600 is most of a long trip.",
    )


# Union type for all actions
LLMAction = Union[
    NavigateToAction,
    GraspAction,
    PlaceOnTopAction,
    PlaceInsideAction,
    ReleaseAction,
    OpenAction,
    CloseAction,
    ToggleOnAction,
    ToggleOffAction,
    WaitAction,
]


class Task(str, Enum):
    """Goal predicates a plan can aim at.

    Replaces crafter's achievement list (collect_wood, make_stone_pickaxe...).
    These are BDDL predicate tokens, so a task specification is the same shape
    as the goal condition L1d evaluates and M9's check_goal will compare.
    """

    ONTOP = "ontop"
    INSIDE = "inside"
    OPEN = "open"
    CLOSED = "closed"
    TOGGLED_ON = "toggled_on"
    HOLDING = "holding"


class TaskSpecification(BaseModel):
    """What the plan is trying to make true.

    ``object_type`` / ``object_id`` are kept as field names because
    ``cognitive_agent._collect_task_args`` and the process logger read them by
    name, and the point of the port is to leave those files untouched. Here
    ``object_type`` carries the BDDL entity id and ``object_id`` is unused
    (always 1) -- an id like ``apple.n.01_1`` already encodes the instance
    index, so a separate integer would be a second, conflicting source of truth.
    """

    task: Task = Field(description="Predicate to make true")
    object_type: str = Field(description=_TARGET)
    object_id: int = Field(default=1, description="Unused; the entity id already carries the index")
    reference: Optional[str] = Field(
        default=None,
        description="Second argument for binary predicates, e.g. the surface for 'ontop'. Same id format.",
    )

    def __str__(self) -> str:
        if self.reference:
            return f"{self.task.value}({self.object_type}, {self.reference})"
        return f"{self.task.value}({self.object_type})"


class InterruptDecision(str, Enum):
    """Decision options when agent is interrupted."""
    RESUME = "resume"  # Continue with the current plan
    REPLAN = "replan"  # Generate a completely new plan


# ============================================================================
# Pydantic models for structured LLM output - Action Classes

# Union type for all actions


# ============================================================================
# Response Models
# ============================================================================

class LLMPlanResponse(BaseModel):
    """Structured response from LLM for plan generation."""
    task: TaskSpecification = Field(description="The task specification with type and optional target object")
    actions: List[LLMAction] = Field(description="List of actions to execute")
    reasoning: str = Field(description="Brief explanation of why this plan was chosen")


class LLMMessageResponse(BaseModel):
    """Structured response from LLM for message generation."""
    recipients: List[str] = Field(description="List of agent IDs to send message to")
    content: str = Field(description="Message content to send")
    reasoning: str = Field(description="Brief explanation of why sending this message")


class LLMInterruptResponse(BaseModel):
    """Structured response from LLM for interrupt handling."""
    decision: InterruptDecision = Field(
        description="Whether to resume the current plan or generate a new plan"
    )
    reasoning: str = Field(
        description="Brief explanation of why this decision was made based on the messages received"
    )
    # Optional new plan - only required if decision is REPLAN
    new_plan: Optional[LLMPlanResponse] = Field(
        default=None,
        description="The new plan to execute (required if decision is 'replan')"
    )


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """
    Wrapper for Azure OpenAI and OpenAI-compatible API calls.
    
    Supports these backends:
    - Azure OpenAI: Azure-hosted OpenAI models with structured output support
    - Foundry: Azure AI Foundry OpenAI-compatible endpoint
    - DeepSeek: DeepSeek OpenAI-compatible endpoint
    
    Takes messages and a response format, returns structured response.
    """
    
    # Backends that support OpenAI's structured output (beta.chat.completions.parse)
    STRUCTURED_OUTPUT_BACKENDS = {"azure"}
    JSON_MODE_BACKENDS = {"deepseek"}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2-chat",
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        backend: str = "azure",  # "azure", "foundry", or "deepseek"
        base_url: Optional[str] = None,
        verbose: bool = False,  # Print API calls and responses
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key (reads from env var if not provided)
            model: Model name to use
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure API version
            backend: Backend type ("azure", "foundry", or "deepseek")
            base_url: Custom base URL for OpenAI-compatible backends
            verbose: If True, print API calls and responses for debugging
        """
        self.model = model
        self.backend = backend.lower()
        self.verbose = verbose
        
        if self.backend == "azure":
            # Azure OpenAI
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            )
        elif self.backend == "deepseek":
            # DeepSeek via OpenAI-compatible API
            self.client = OpenAI(
                base_url=base_url or os.getenv("DEEPSEEK_ENDPOINT"),
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY")
            )
        elif self.backend == "foundry":
            # Azure AI Foundry via OpenAI-compatible API
            self.client = OpenAI(
                base_url=base_url or os.getenv("AZURE_FOUNDRY_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_FOUNDRY_API_KEY")
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'azure', 'foundry', or 'deepseek'.")
    
    @classmethod
    def from_env(
        cls,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = False,
    ) -> "LLMClient":
        """
        Create LLM client from environment variables.
        
        Args:
            backend: Override backend type. If None, auto-detect from env vars.
            model: Override model/deployment name. If None, use backend-specific env var.
            verbose: If True, print API calls and responses for debugging.
        
        Environment variables:
            - LLM_BACKEND: "azure", "foundry", or "deepseek"
            - AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_VERSION
            - AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_API_KEY, AZURE_FOUNDRY_MODEL
            - DEEPSEEK_ENDPOINT, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
        """
        # Determine backend
        if backend is None:
            backend = os.getenv("LLM_BACKEND", "").lower()
        
        # Auto-detect if not specified (prefer Azure)
        if not backend:
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                backend = "azure"
            elif os.getenv("AZURE_FOUNDRY_ENDPOINT") and os.getenv("AZURE_FOUNDRY_API_KEY"):
                backend = "foundry"
            elif os.getenv("DEEPSEEK_API_KEY"):
                backend = "deepseek"
            else:
                backend = "azure"  # Default to Azure
        
        if backend == "deepseek":
            return cls(
                backend="deepseek",
                base_url=os.getenv("DEEPSEEK_ENDPOINT"),
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model=model or os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2"),
                verbose=verbose,
            )
        if backend == "foundry":
            return cls(
                backend="foundry",
                base_url=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
                api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
                model=model or os.getenv("AZURE_FOUNDRY_MODEL", "Llama-4-Scout-17B-16E-Instruct"),
                verbose=verbose,
            )
        else:  # azure (default)
            return cls(
                backend="azure",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                model=model or os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                verbose=verbose,
            )
    
    def _parse_json_response(self, content: str, response_format: type):
        """
        Parse JSON response manually for backends without structured output support.
        
        Args:
            content: Raw response content (may contain JSON in markdown code blocks)
            response_format: Pydantic model to parse into
            
        Returns:
            Parsed Pydantic model instance
        """
        import json
        import re
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
        
        # Parse JSON and validate with Pydantic
        data = json.loads(json_str)
        return response_format.model_validate(data)
    
    def _get_json_schema_prompt(self, response_format: type) -> str:
        """
        Generate a prompt suffix with JSON schema for models without structured output.
        
        Args:
            response_format: Pydantic model to generate schema for
            
        Returns:
            Prompt string describing the expected JSON format
        """
        schema = response_format.model_json_schema()
        import json
        return f"\n\nYou MUST respond with valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```\nRespond ONLY with the JSON object, no other text."

    @staticmethod
    def _is_guard_filter_error(error: Exception) -> bool:
        """Return True for Azure/OpenAI content-filter guardrail failures."""
        parts = [
            str(error),
            str(getattr(error, "code", "")),
            str(getattr(error, "body", "")),
            str(getattr(error, "message", "")),
        ]
        text = " ".join(parts).lower()
        markers = (
            "content_filter",
            "content filter",
            "responsibleaipolicy",
            "policy violation",
            "filtered due to",
            "safety system",
        )
        return any(marker in text for marker in markers)
    
    def generate(
        self,
        messages: List[Dict],
        response_format: Optional[type] = None,
        temperature: float = 0.7,
        max_retries: int = 6,
        retry_delay: float = 10.0
    ) -> tuple:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Pydantic model for structured output (LLMPlanResponse, 
                           LLMMessageResponse, LLMInterruptResponse, or None for free-form)
            temperature: Sampling temperature (ignored for models that don't support it)
            max_retries: Maximum number of retries on rate limit errors
            retry_delay: Seconds to wait before retrying (default: 30)
            
        Returns:
            Tuple of (response, usage_dict)
            - response: Parsed Pydantic model if response_format provided, else string
            - usage_dict: token counts plus latency/retry telemetry
        """
        # Some models (like gpt-5.2-chat) don't support temperature parameter
        # Check if model supports temperature, otherwise omit it
        supports_temperature = "gpt-5" not in self.model.lower()
        
        # Verbose logging: print API call info
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[LLM API CALL] Backend: {self.backend}, Model: {self.model}")
            print(f"  Response format: {response_format.__name__ if response_format else 'None (free-form)'}")
            print(f"  Temperature: {temperature if supports_temperature else 'N/A (model default)'}")
            print(f"  Messages ({len(messages)}):")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]
                print(f"    [{i}] {role}: {content}{'...' if len(msg.get('content', '')) > 200 else ''}")
            print(f"{'='*60}")
        
        raw_content = None  # For verbose logging
        call_start = time.monotonic()
        retry_count = 0
        rate_limit_retry_count = 0
        api_error_retry_count = 0
        
        # Build optional kwargs (only include temperature if supported)
        temp_kwargs = {"temperature": temperature} if supports_temperature else {}
        
        # Retry loop for rate limit errors
        for attempt in range(max_retries + 1):
            try:
                if response_format is not None and self.backend in self.STRUCTURED_OUTPUT_BACKENDS:
                    # Use OpenAI's structured output (beta.chat.completions.parse)
                    raw_response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=response_format,
                        **temp_kwargs
                    )
                    response = raw_response.choices[0].message.parsed
                    raw_content = str(response)
                elif response_format is not None:
                    # For OpenAI-compatible backends without structured output,
                    # add the JSON schema to the last user message and parse manually.
                    messages_with_schema = messages.copy()
                    if messages_with_schema and messages_with_schema[-1]["role"] == "user":
                        messages_with_schema[-1] = messages_with_schema[-1].copy()
                        messages_with_schema[-1]["content"] += self._get_json_schema_prompt(response_format)

                    response_kwargs = {}
                    if self.backend in self.JSON_MODE_BACKENDS:
                        response_kwargs["response_format"] = {"type": "json_object"}

                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages_with_schema,
                        **response_kwargs,
                        **temp_kwargs
                    )
                    raw_content = raw_response.choices[0].message.content
                    response = self._parse_json_response(raw_content, response_format)
                else:
                    # Free-form output
                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **temp_kwargs
                    )
                    response = raw_response.choices[0].message.content
                    raw_content = response
                
                # Success - break out of retry loop
                break
                
            except RateLimitError as e:
                if attempt < max_retries:
                    retry_count += 1
                    rate_limit_retry_count += 1
                    print(f"\n[RATE LIMIT] Hit rate limit. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"\n[RATE LIMIT] Max retries ({max_retries}) exceeded. Raising error.")
                    raise
            except APIError as e:
                # Azure may return 429 as APIError instead of RateLimitError
                if hasattr(e, 'status_code') and e.status_code == 429:
                    if attempt < max_retries:
                        retry_count += 1
                        rate_limit_retry_count += 1
                        api_error_retry_count += 1
                        print(f"\n[RATE LIMIT] Azure 429 error. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        print(f"\n[RATE LIMIT] Max retries ({max_retries}) exceeded. Raising error.")
                        raise
                else:
                    if self._is_guard_filter_error(e):
                        print(f"\n[GUARD FILTER] LLM request blocked by content filter/policy: {e}")
                    raise
        
        finish_reason = None
        guard_filter_count = 0
        if getattr(raw_response, "choices", None):
            finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
            if str(finish_reason).lower() == "content_filter":
                guard_filter_count = 1
                print("\n[GUARD FILTER] LLM response was filtered by content policy.")

        response_usage = getattr(raw_response, "usage", None)
        prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(response_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": time.monotonic() - call_start,
            "retry_count": retry_count,
            "rate_limit_retry_count": rate_limit_retry_count,
            "api_error_retry_count": api_error_retry_count,
            "guard_filter_count": guard_filter_count,
            "finish_reason": None if finish_reason is None else str(finish_reason),
        }
        
        # Verbose logging: print response
        if self.verbose:
            print(f"\n[LLM RESPONSE] Backend: {self.backend}")
            print(f"  Usage: {usage}")
            print(f"  Raw content ({len(raw_content) if raw_content else 0} chars):")
            if raw_content:
                # Print first 500 chars
                print(f"    {raw_content[:500]}{'...' if len(raw_content) > 500 else ''}")
            print(f"{'='*60}\n")
        
        return response, usage
    
    def generate_plan(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate a plan response."""
        return self.generate(messages, response_format=LLMPlanResponse, temperature=temperature)
    
    def generate_message(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate a message response."""
        return self.generate(messages, response_format=LLMMessageResponse, temperature=temperature)
    
    def generate_interrupt_decision(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate an interrupt decision response."""
        return self.generate(messages, response_format=LLMInterruptResponse, temperature=temperature)

    def generate_team_plan(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """One call that plans for every robot on a team."""
        return self.generate(messages, response_format=LLMTeamPlanResponse, temperature=temperature)

    def generate_team_interrupt_decision(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """One call that decides resume-or-replan for every robot on a team."""
        return self.generate(messages, response_format=LLMTeamInterruptResponse, temperature=temperature)


class TeamAgentPlan(BaseModel):
    """One team member's plan, inside a single team-wide response.

    Same three fields as :class:`LLMPlanResponse` plus the agent it is for, so
    the team response is N of these rather than a nested map -- a list keeps the
    JSON schema simple enough for structured output, and the agent_id travels
    *with* the plan rather than as a key that can drift from it.
    """

    agent_id: str = Field(description="Which robot this plan is for; must be one of the team's ids")
    task: TaskSpecification = Field(description="The task specification with type and optional target object")
    actions: List[LLMAction] = Field(description="List of actions for this robot to execute")
    reasoning: str = Field(description="Brief explanation of why this robot is doing this")


class LLMTeamPlanResponse(BaseModel):
    """One call, one plan per robot on the team.

    The team brain sees every member's observation at once, so the interesting
    field is ``reasoning``: it is where the allocation is justified, and the only
    place the division of labour is visible before the robots act on it.
    """

    plans: List[TeamAgentPlan] = Field(description="Exactly one plan per robot on the team")
    reasoning: str = Field(description="Why the work is divided between the robots this way")


class TeamAgentInterruptDecision(BaseModel):
    """Resume or replan, decided per robot after the team was interrupted."""

    agent_id: str = Field(description="Which robot this decision is for")
    decision: InterruptDecision = Field(
        description="Whether this robot resumes its current plan or gets a new one"
    )
    reasoning: str = Field(description="Why, given the message and this robot's own progress")
    new_plan: Optional[LLMPlanResponse] = Field(
        default=None,
        description="Required when decision is 'replan'; ignored when it is 'resume'",
    )


class LLMTeamInterruptResponse(BaseModel):
    """A message interrupts the whole team; the brain answers for each member.

    Per-agent rather than team-wide on purpose: a message that changes what one
    robot should do usually leaves the others' plans perfectly good, and making
    the whole team replan would throw away work the message never contradicted.
    """

    decisions: List[TeamAgentInterruptDecision] = Field(
        description="Exactly one decision per robot on the team"
    )
    reasoning: str = Field(description="What the message means for the team as a whole")


# ============================================================================
# Utility functions
# ============================================================================

def load_env_file(env_path: str = ".env"):
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to .env file (will also check project root)
    """
    # Try current directory first
    if not os.path.exists(env_path):
        # Try to find .env in project root (relative to this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(project_root, ".env")
    
    if not os.path.exists(env_path):
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)


# Auto-load .env file on import
load_env_file()
