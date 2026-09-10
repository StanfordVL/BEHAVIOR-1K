"""
LLM-powered Centralized Topology.

Same structure as centralized.py but extends BaseLLMAgent directly.
Role information is injected into prompts.

Decision Flow:
    Leader:
        1. wait_for: []
        2. send_to: [all followers] with planning request
        3. wait_for_response: [all followers]
        4. generate plan (LLM)
    
    Follower:
        1. wait_for: [leader]
        2. send_to: [leader] (status/proposal response)
        3. generate plan (LLM)
"""

import json
from typing import List, Dict, Optional

from coop2.cognitive.agent import LLMClient, InterruptDecision
from coop2.cognitive.agent.base_llm_agent import BaseLLMAgent
from coop2.cognitive.agent.cognitive_agent import extract_position, extract_status
from coop2.cognitive.agent.prompts import build_system_prompt
from coop2.cognitive.plan import SymbolicPlan, SymbolicPlanStatus
from .centralized_flow import CentralizedFollowerFlow, CentralizedLeaderFlow


# Role descriptions injected into prompts
LEADER_ROLE = """
## Your Role: LEADER
Ask followers for local status/proposals, wait for their responses, then commit
a team plan using your observation plus those responses.
"""

FOLLOWER_ROLE = """
## Your Role: FOLLOWER
Reply to the leader with concise local status and one feasible proposal, then
execute a local plan that supports the team objective from your observation.
"""


class LLMLeaderAgent(CentralizedLeaderFlow, BaseLLMAgent):
    """
    LLM-powered Leader agent for centralized topology.
    
    Decision Flow:
        wait_for: []
        send_to: [all followers]
        wait_for_response: [all followers]
        then: generate plan via LLM
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient, follower_ids: List[str],
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        
        # Decision Flow Configuration (same as centralized.py)
        self.wait_for = []
        self.send_to = follower_ids
        self.wait_for_response = follower_ids
        
        self.expected_responses = set()
        self.follower_responses: List[str] = []
        self._system_prompt = None
        self.centralized_response_timeout_seconds = 30.0
    
    def _get_system_prompt(self) -> str:
        """Build system prompt with leader role injected."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + "\n\n" + LEADER_ROLE.strip()
        return self._system_prompt
    
    def _format_planning_request(self) -> str:
        """Request follower status before the leader commits a plan."""
        return (
            f"Leader planning request from {self.agent_id}: report current "
            "position/inventory, one useful visible target or missing prerequisite, "
            "and your next action proposal. Keep it concise."
        )

    def _follower_response_context(self) -> str:
        """Format collected follower responses for the leader planning prompt."""
        if not self.follower_responses:
            return ""
        return "## Follower Status Responses\n" + "\n".join(self.follower_responses) + "\n\n"

    def _build_leader_broadcast(self):
        return self._format_planning_request(), {
            'type': 'leader_broadcast',
            'interrupts_execution': True,
        }
    
    def _generate_plan_with_role(self, messages: Optional[List[Dict]] = None) -> SymbolicPlan:
        """Generate plan via LLM with role context."""
        repair_messages = messages
        prompt_messages = self._build_plan_prompt_messages(
            system_prompt=self._get_system_prompt(),
            agent_names=[self.agent_id] + list(self.send_to),
            messages=repair_messages,
            user_prefix=self._follower_response_context(),
        )
        return self._generate_plan_from_messages(
            prompt_messages=prompt_messages,
            repair_messages=repair_messages,
            label="Centralized Leader Plan Generation",
            verbose_prefix="LEADER calling LLM for plan...",
        )

    def _generate_leader_plan_after_responses(self) -> SymbolicPlan:
        return self._generate_plan_with_role()

    def _handle_leader_response_timeout(self, expected_responses: set[str]) -> None:
        print(f"  [{self.agent_id}] Warning: timeout waiting for {expected_responses}")

    def _on_leader_broadcast_sent(self, content):
        if self.verbose:
            print(f"  [{self.agent_id}] Sent: {str(content)[:60]}...")

    def _on_follower_response_received(self, sender: str) -> None:
        if self.verbose:
            print(f"  [{self.agent_id}] Got response from {sender}")

    def _on_all_follower_responses_received(self) -> None:
        if self.verbose:
            print(f"  [{self.agent_id}] All follower responses received")
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Leader replans on interrupt."""
        if self._handle_coop2_repair_interrupt(self._generate_plan_with_role):
            return
        # Ask before discarding. Replanning unconditionally was upstream's
        # behaviour and works in a grid world where an action is one step; here
        # a NAVIGATE_TO runs for hundreds of ticks, so a message that arrives
        # mid-trip used to throw away all the travel already paid for. On
        # RESUME nothing is touched: the primitive keeps running with its
        # accrued delay, and the plan continues from where it was.
        if self.decide_interrupt() is InterruptDecision.RESUME:
            return
        self._execute_flow()


class LLMFollowerAgent(CentralizedFollowerFlow, BaseLLMAgent):
    """
    LLM-powered Follower agent for centralized topology.
    
    Decision Flow:
        wait_for: [leader]
        send_to: [leader] (response)
        then: generate plan via LLM
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient, leader_id: str,
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        
        # Decision Flow Configuration (same as centralized.py)
        self.wait_for = [leader_id]
        self.send_to = [leader_id]
        
        self.leader_id = leader_id
        self.leader_agent: Optional['LLMLeaderAgent'] = None
        self.team_agent_ids: List[str] = [leader_id, agent_id]
        self.last_leader_request = None
        self._system_prompt = None
    
    def _get_system_prompt(self) -> str:
        """Build system prompt with follower role injected."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + "\n\n" + FOLLOWER_ROLE.strip()
        return self._system_prompt

    def _current_plan_summary(self) -> str:
        """Summarize the follower's committed plan for centralized status replies."""
        if self.plan is None:
            return "Current plan: none."
        remaining = self.plan.actions[self.plan.current_action_index:]
        action_text = "; ".join(str(action) for action in remaining[:4])
        if len(remaining) > 4:
            action_text += f"; ... +{len(remaining) - 4} more"
        if not action_text:
            action_text = "no remaining actions"
        return (
            f"Current plan #{self.plan.plan_id}: {self.plan.specification}; "
            f"status={self.plan.status.value}; remaining={action_text}."
        )
    
    def _generate_message(self, request: str) -> str:
        """Generate a concise status/proposal response for the leader."""
        status = extract_status(self.observation)
        position = extract_position(self.observation)
        user_prompt = "\n".join([
            "The leader is collecting follower status before committing a centralized plan.",
            "Reply with 2-4 concise sentences, not JSON.",
            "Include current position/status, current plan, one useful target or prerequisite, and whether you intend to resume or revise.",
            "",
            "Leader request:",
            request or "Report local status and a feasible next task.",
            "",
            self._current_plan_summary(),
            "",
            "Your current position:",
            json.dumps(position, default=str),
            "",
            "Your current status/inventory:",
            json.dumps(status, default=str),
            "",
            "Current reachable targets:",
            self.target_hints or "unknown",
            "",
            "Recent memory:",
            json.dumps(self.memory.get_events()[-3:], default=str),
        ])
        messages = [
            {
                "role": "system",
                "content": (
                    self._get_system_prompt()
                    + "\n\nFor this response, write only the message to send back to the leader."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        return self._generate_text_from_messages(
            messages=messages,
            fallback=(
                f"Status from {self.agent_id}: I will choose a feasible local "
                "prerequisite or reachable target from my current observation."
            ),
        )

    def _build_follower_response(self):
        return self._generate_message(self.last_leader_request or ""), {'type': 'follower_response'}
    
    def _generate_plan_with_role(self, messages: Optional[List[Dict]] = None) -> SymbolicPlan:
        """Generate plan via LLM with role context and leader's planning request."""
        repair_messages = messages
        request_ctx = ""
        if self.last_leader_request:
            request_ctx = f"## Leader Planning Request\n{self.last_leader_request}\n\n"
        
        prompt_messages = self._build_plan_prompt_messages(
            system_prompt=self._get_system_prompt(),
            agent_names=self.team_agent_ids,
            messages=repair_messages,
            user_prefix=request_ctx,
        )
        return self._generate_plan_from_messages(
            prompt_messages=prompt_messages,
            repair_messages=repair_messages,
            label="Centralized Follower Plan Generation",
            verbose_prefix="FOLLOWER calling LLM for plan...",
        )

    def _handle_leader_messages(self, messages: List[Dict]) -> None:
        for msg in messages:
            if msg['sender'] == self.leader_id:
                self.last_leader_request = msg['content']

    def _generate_follower_plan_after_response(self) -> SymbolicPlan:
        return self._generate_plan_with_role()

    def _has_active_plan_to_resume(self) -> bool:
        if self.plan is None:
            return False
        return self.plan.status in {
            SymbolicPlanStatus.PENDING,
            SymbolicPlanStatus.EXECUTING,
        }

    def _respond_to_leader_request(self, messages: List[Dict]) -> bool:
        leader_messages = [msg for msg in messages if msg.get('sender') == self.leader_id]
        self._handle_leader_messages(leader_messages)
        if not leader_messages:
            return False
        if self.send_to and self.message_broker is not None and self._leader_is_not_ready():
            content, metadata = self._build_follower_response()
            self.send_message(
                recipients=self.send_to,
                content=content,
                metadata=metadata,
            )
            self._on_follower_response_sent(content)
        return True

    def _on_follower_response_sent(self, content) -> None:
        if self.verbose:
            print(f"  [{self.agent_id}] Sent response: {str(content)[:60]}...")
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Reply to leader interrupts, then either resume or revise."""
        if self._handle_coop2_repair_interrupt(self._generate_plan_with_role):
            return
        messages = self.get_messages(clear_buffer=True)
        handled_leader_request = self._respond_to_leader_request(messages)
        if handled_leader_request and self._has_active_plan_to_resume():
            return
        if handled_leader_request or self.needs_new_plan():
            self.plan = self._generate_plan_with_role()


def create_llm_centralized_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True
) -> Dict[str, BaseLLMAgent]:
    """
    Create LLM-powered agents for centralized topology (1 leader, n-1 followers).
    """
    if n_agents < 2:
        raise ValueError("Centralized topology requires at least 2 agents")
    
    leader_id = "agent_0"
    follower_ids = [f"agent_{i}" for i in range(1, n_agents)]
    
    agents = {}
    leader = LLMLeaderAgent(leader_id, llm_client, follower_ids, temperature, verbose)
    agents[leader_id] = leader
    
    for fid in follower_ids:
        follower = LLMFollowerAgent(fid, llm_client, leader_id, temperature, verbose)
        follower.leader_agent = leader
        follower.team_agent_ids = [leader_id] + follower_ids
        agents[fid] = follower
    
    return agents
