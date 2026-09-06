"""
Base LLM Agent implementation.

Provides a reusable Agent that uses LLM for:
- Plan generation based on observations
- Interrupt handling decisions
- Message generation for inter-agent communication
"""

import json
import threading
from typing import List, Any, Optional, Dict, Callable

from .agent import Agent
from .llm_client import LLMClient, InterruptDecision
from .cognitive_agent import (
    build_plan_prompt,
    build_observation_prompt,
    build_interrupt_prompt,
    extract_status,
    extract_position,
    extract_facing,
    extract_visible_area,
    parse_plan_response,
    parse_interrupt_response,
    apply_repair_plan_recommendation,
)
from ..plan.plan import SymbolicPlan, SymbolicAction


class BaseLLMAgent(Agent):
    """
    Agent that uses real LLM for plan generation and communication.

    Uses OpenAI/Azure OpenAI API for:
    - Generating plans based on observations
    - Deciding whether to resume or replan on interrupts
    - Generating communication messages
    """
    _llm_print_lock = threading.Lock()
    
    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        should_broadcast: bool = True,
        max_memory_size: int = 10,
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        """
        Initialize LLM agent.
        
        Args:
            agent_id: Agent identifier
            llm_client: LLM client for API calls
            should_broadcast: Whether to send messages after plan generation
            max_memory_size: Number of recent events to keep in memory
            temperature: LLM sampling temperature
            verbose: Whether to print LLM inputs and outputs
        """
        super().__init__(agent_id, memory_size=max_memory_size)
        self.llm_client = llm_client
        self.should_broadcast = should_broadcast
        self.temperature = temperature
        self.verbose = verbose
        self.other_agents: List[str] = []  # Set externally
        
        # Token tracking
        self.total_tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        self.api_latency_seconds = 0.0
        self.max_api_latency_seconds = 0.0
        self.api_retries = 0
        self.api_rate_limit_retries = 0
        self.api_error_retries = 0
        self.llm_errors = 0
        self.guard_filter_events = 0
        
        # Environment info (set externally)
        self.coop_config: Optional[str] = None
        self.symbolic_view: Optional[str] = None
        self.target_hints: Optional[str] = None
    
    def observe(self, observation: Any, env_step: int):
        """Process observation from environment."""
        self.observation = observation
        self.env_step = env_step
    
    def set_env_info(
        self,
        coop_config: Optional[str] = None,
        symbolic_view: Optional[str] = None,
        target_hints: Optional[str] = None,
    ):
        """Set environment info for prompts."""
        self.coop_config = coop_config
        self.symbolic_view = symbolic_view
        self.target_hints = target_hints

    def _should_print_llm_io(self) -> bool:
        """Return True when either agent or client verbose mode wants prompt IO."""
        return bool(self.verbose or getattr(self.llm_client, "verbose", False))

    def _print_llm_messages(self, label: str, messages: List[Dict]):
        """Print the full input messages sent to the model for this agent."""
        with self._llm_print_lock:
            print(f"\n{'=' * 80}")
            print(f"LLM INPUT [{self.agent_id}] - {label}")
            print(f"{'=' * 80}")
            for index, message in enumerate(messages):
                role = str(message.get("role", "unknown")).upper()
                content = str(message.get("content", ""))
                print(f"\n--- MESSAGE {index}: {role} ---")
                print(content)
            print(f"{'=' * 80}\n")

    def _record_llm_usage(self, usage: Dict[str, Any]):
        """Record token/API usage from one LLM call."""
        self.api_calls += 1
        self.total_tokens_used += usage["total_tokens"]
        self.prompt_tokens += usage["prompt_tokens"]
        self.completion_tokens += usage["completion_tokens"]
        latency = float(usage.get("latency_seconds", 0.0) or 0.0)
        self.api_latency_seconds += latency
        self.max_api_latency_seconds = max(self.max_api_latency_seconds, latency)
        self.api_retries += int(usage.get("retry_count", 0) or 0)
        self.api_rate_limit_retries += int(usage.get("rate_limit_retry_count", 0) or 0)
        self.api_error_retries += int(usage.get("api_error_retry_count", 0) or 0)
        self.guard_filter_events += int(usage.get("guard_filter_count", 0) or 0)

    def _record_llm_error(self, error: Exception):
        """Record an LLM-call failure for run-level monitoring."""
        self.llm_errors += 1
        if self._is_guard_filter_error(error):
            self.guard_filter_events += 1

    @staticmethod
    def _is_guard_filter_error(error: Exception) -> bool:
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

    def _any_waiting_agent_not_ready(
        self,
        agents_by_id: Dict[str, Any],
        wait_for: Optional[List[str]] = None,
    ) -> bool:
        """Return True if any agent this agent waits for has not committed yet."""
        wait_ids = self.wait_for if wait_for is None else wait_for
        if not wait_ids or not agents_by_id:
            return False
        return any(
            agents_by_id.get(agent_id) and not agents_by_id[agent_id].ready
            for agent_id in wait_ids
        )

    def _finalize_generated_plan(
        self,
        plan: SymbolicPlan,
        repair_messages: Optional[List[Dict]] = None,
    ) -> SymbolicPlan:
        """Apply shared post-processing to plans generated by LLM topology agents."""
        return apply_repair_plan_recommendation(
            plan,
            self.agent_id,
            repair_messages,
        )

    def _build_plan_prompt_messages(
        self,
        system_prompt: str,
        agent_names: List[str],
        messages: Optional[List[Dict]] = None,
        user_prefix: str = "",
        coop_config_override: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the standard topology plan prompt with optional role-local context."""
        coop_config = self.coop_config if coop_config_override is None else coop_config_override
        obs_prompt = build_observation_prompt(
            env_step=self.env_step,
            agent_id=self.agent_id,
            agent_names=agent_names,
            status=extract_status(self.observation),
            position=extract_position(self.observation),
            facing=extract_facing(self.observation),
            visible_area=extract_visible_area(self.observation),
            messages=messages,
            memory=self.memory.get_events(),
            coop_config=coop_config,
            symbolic_view=self.symbolic_view,
            target_hints=self.target_hints,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prefix + obs_prompt},
        ]

    def _generate_plan_from_messages(
        self,
        prompt_messages: List[Dict],
        repair_messages: Optional[List[Dict]] = None,
        label: str = "Plan Generation",
        verbose_prefix: str = "Calling LLM for plan...",
    ) -> SymbolicPlan:
        """Call the LLM, parse a symbolic plan, and apply shared post-processing."""
        if self._should_print_llm_io():
            self._print_llm_messages(label, prompt_messages)

        if self.verbose:
            print(f"  [{self.agent_id}] {verbose_prefix}")

        try:
            response, usage = self.llm_client.generate_plan(
                messages=prompt_messages,
                temperature=self.temperature,
            )
            self._record_llm_usage(usage)

            plan = parse_plan_response(
                llm_response=response,
                agent_id=self.agent_id,
                env_step=self.env_step,
                plan_id=self.plan_count + 1,
            )
            plan = self._finalize_generated_plan(plan, repair_messages)
            if self.verbose:
                print(f"  [{self.agent_id}] Plan: {plan.specification}")
            return plan
        except Exception as e:
            self._record_llm_error(e)
            print(f"  [{self.agent_id}] LLM error: {e}")
            return self._generate_fallback_plan()

    def _generate_text_from_messages(
        self,
        messages: List[Dict],
        fallback: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Call the LLM for a plain text message and record usage consistently."""
        try:
            response, usage = self.llm_client.generate(
                messages=messages,
                response_format=None,
                temperature=self.temperature if temperature is None else temperature,
            )
            self._record_llm_usage(usage)
            return str(response)
        except Exception as e:
            self._record_llm_error(e)
            return fallback

    def _handle_coop2_repair_interrupt(
        self,
        plan_generator: Optional[Callable[..., SymbolicPlan]] = None,
        messages: Optional[List[Dict]] = None,
    ) -> bool:
        """Consume a COOP2 repair interrupt and regenerate a plan with its context."""
        if messages is None:
            pending_messages = self.get_messages(clear_buffer=False)
            if not self.has_coop2_repair_request(pending_messages):
                return False
            messages = self.get_messages(clear_buffer=True)
        elif not self.has_coop2_repair_request(messages):
            return False

        if self.verbose:
            print(f"  [{self.agent_id}] COOP2 repair request received: replanning")

        if plan_generator is None:
            self.generate_plan(messages=messages)
        else:
            self.plan = plan_generator(messages=messages)
        return True

    def generate_plan(self, messages: Optional[List[Dict]] = None) -> SymbolicPlan:
        """
        Generate a plan using LLM based on current observation.
        
        Returns:
            SymbolicPlan: Generated plan
        """
        repair_messages = messages
        # Build prompt for LLM
        prompt_messages = build_plan_prompt(
            observation=self.observation,
            env_step=self.env_step,
            agent_id=self.agent_id,
            messages=repair_messages,
            memory=self.memory.get_events(),
            coop_config=self.coop_config,
            symbolic_view=self.symbolic_view,
            target_hints=self.target_hints,
        )
        
        self.plan = self._generate_plan_from_messages(
            prompt_messages=prompt_messages,
            repair_messages=repair_messages,
            label="Plan Generation",
            verbose_prefix="Calling LLM for plan generation...",
        )
        return self.plan
    
    def _generate_fallback_plan(self) -> SymbolicPlan:
        """Generate a simple fallback plan when LLM fails."""
        return SymbolicPlan(
            specification="Fallback exploration",
            actions=[
                SymbolicAction("move", {"direction": "left", "num_steps": 2}),
                SymbolicAction("collect", {"target": "wood"}),
            ],
            plan_id=self.plan_count + 1,
            agent_id=self.agent_id,
            created_at_step=self.env_step
        )

    def describe_repair_intention(
        self,
        repair_context: Dict[str, Any],
        previous_statements: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate one concise statement for the ordered COOP2 repair round."""
        previous_statements = previous_statements or []
        plan_lines = []
        if self.plan is not None:
            plan_lines.append(f"Current plan #{self.plan.plan_id}: {self.plan.specification}")
            for index, action in enumerate(
                self.plan.actions[self.plan.current_action_index:],
                start=self.plan.current_action_index + 1,
            ):
                plan_lines.append(f"  {index}. {action}")
        else:
            plan_lines.append("Current plan: none")

        user_prompt = "\n".join([
            "You are participating in a COOP2 repair channel.",
            "Agents speak once in ascending agent-id order. Later agents can see earlier statements.",
            "State only what you intend to do or change for the predicted failure.",
            "Be concrete and concise; prefer the shared target/timing implied by the repair context.",
            "",
            f"You are: {self.agent_id}",
            f"Environment step: {self.env_step}",
            "",
            "Predicted repair context:",
            json.dumps(repair_context, indent=2, default=str),
            "",
            "Current reachable targets:",
            self.target_hints or "unknown",
            "",
            "Previous repair statements:",
            json.dumps(previous_statements, indent=2, default=str),
            "",
            "Your current plan:",
            "\n".join(plan_lines),
            "",
            "Write your repair intention in 1-3 sentences.",
        ])
        messages = [
            {
                "role": "system",
                "content": self._repair_intention_system_prompt(),
            },
            {"role": "user", "content": user_prompt},
        ]

        try:
            response, usage = self.llm_client.generate(
                messages=messages,
                response_format=None,
                temperature=self.temperature,
            )
            self._record_llm_usage(usage)
            return str(response).strip()
        except Exception as e:
            self._record_llm_error(e)
            if self.verbose:
                print(f"  [{self.agent_id}] LLM repair intention error: {e}")
            return super().describe_repair_intention(
                repair_context=repair_context,
                previous_statements=previous_statements,
            )

    def _repair_intention_system_prompt(self) -> str:
        """Use the topology role prompt when a topology subclass provides one."""
        role_prompt = None
        get_system_prompt = getattr(self, "_get_system_prompt", None)
        if callable(get_system_prompt):
            try:
                role_prompt = get_system_prompt()
            except Exception:
                role_prompt = None

        repair_rules = "\n".join([
            "COOP2 repair-channel response rules:",
            "- Keep the role and communication structure from the prompt above.",
            "- State only your repair intention for the current predicted failure.",
            "- Do not assign yourself or others a new permanent role.",
        ])
        if role_prompt:
            return f"{role_prompt}\n\n{repair_rules}"
        return "You are a cooperative planning agent.\n\n" + repair_rules
    
    def handle_interrupt(self):
        """
        Handle interrupt by asking LLM to decide whether to resume or replan.
        
        Provides LLM with:
        - Current observation
        - Memory of recent events
        - Current plan and its execution status
        - Messages that triggered the interrupt
        """
        messages = self.get_messages(clear_buffer=True)
        
        if not messages:
            return

        if self.has_coop2_repair_request(messages):
            self._handle_coop2_repair_interrupt(messages=messages)
            return

        if self.verbose:
            print(f"\n[{self.agent_id}] Received {len(messages)} message(s):")
            for msg in messages:
                print(f"  From {msg['sender']}: {msg['content']}")
        
        # Always use LLM to decide
        if self.verbose:
            print(f"  [{self.agent_id}] Asking LLM for interrupt decision...")
        
        try:
            interrupt_messages = build_interrupt_prompt(
                observation=self.observation,
                env_step=self.env_step,
                agent_id=self.agent_id,
                current_plan=self.plan,
                received_messages=messages,
                memory=self.memory.get_events(),
                coop_config=self.coop_config,
                symbolic_view=self.symbolic_view,
                target_hints=self.target_hints,
            )
            
            if self._should_print_llm_io():
                self._print_llm_messages("Interrupt Decision", interrupt_messages)
            
            response, usage = self.llm_client.generate_interrupt_decision(
                messages=interrupt_messages,
                temperature=self.temperature
            )
            
            self._record_llm_usage(usage)
            
            if self.verbose:
                print(f"    LLM reasoning: {response.reasoning}")
                print(f"    Tokens used: {usage['total_tokens']}")
            
            decision, new_plan = parse_interrupt_response(
                llm_response=response,
                agent_id=self.agent_id,
                env_step=self.env_step,
                plan_id=self.plan_count + 1
            )
            
            if decision == InterruptDecision.RESUME:
                if self.verbose:
                    print(f"  [{self.agent_id}] LLM decided: RESUME current plan")
            else:
                if self.verbose:
                    print(f"  [{self.agent_id}] LLM decided: REPLAN")
                if new_plan:
                    self.plan = new_plan
                    if self.verbose:
                        print(f"    New plan: {self.plan.specification}")
                        print(f"    Actions: {[str(a) for a in self.plan.actions]}")
                else:
                    # LLM said replan but didn't provide plan, generate one
                    if self.verbose:
                        print(f"    Generating new plan...")
                    self.generate_plan()
                    
        except Exception as e:
            self._record_llm_error(e)
            print(f"  [{self.agent_id}] LLM interrupt error: {e}")
            print(f"  [{self.agent_id}] Falling back to resume")
            import traceback
            traceback.print_exc()
    
    def handle_reasoning(self):
        """
        Handle reasoning state - generate plan and optionally broadcast.
        """
        # Generate plan using LLM
        self.generate_plan()
        
        # Optionally broadcast to other agents
        if self.should_broadcast and self.message_broker and self.other_agents:
            try:
                content = f"I'm starting plan: {self.plan.specification}"
                self.send_message(
                    recipients=self.other_agents,
                    content=content,
                    metadata={'plan_id': self.plan.plan_id, 'step': self.env_step}
                )
                if self.verbose:
                    print(f"  [{self.agent_id}] Broadcasted plan to {self.other_agents}")
            except Exception as e:
                print(f"  [{self.agent_id}] Broadcast failed: {e}")
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        # Keep token counts across resets for statistics
    
    def get_usage_stats(self) -> dict:
        """Get LLM usage statistics."""
        return {
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens_used,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'api_latency_seconds': self.api_latency_seconds,
            'avg_api_latency_seconds': (
                self.api_latency_seconds / self.api_calls if self.api_calls else 0.0
            ),
            'max_api_latency_seconds': self.max_api_latency_seconds,
            'api_retries': self.api_retries,
            'api_rate_limit_retries': self.api_rate_limit_retries,
            'api_error_retries': self.api_error_retries,
            'llm_errors': self.llm_errors,
            'guard_filter_events': self.guard_filter_events,
        }
