"""
LLM-powered Individual Topology.

Same structure as individual.py but extends BaseLLMAgent directly.
No communication, independent decision-making via LLM.

Decision Flow:
    Each agent independently:
        1. wait_for: [] (no waiting)
        2. send_to: [] (no sending)
        3. generate plan via LLM
"""

from typing import Dict, List, Optional

from coop2.cognitive.agent import LLMClient
from coop2.cognitive.agent.base_llm_agent import BaseLLMAgent
from coop2.cognitive.agent.prompts import build_system_prompt
from coop2.cognitive.plan import SymbolicPlan


# Role description
INDIVIDUAL_ROLE = """
## Your Role: INDEPENDENT AGENT
Plan from your own observation during normal execution. Use COOP2 repair context
only when it is provided.
"""


class LLMIndividualAgent(BaseLLMAgent):
    """
    LLM-powered agent for individual topology - makes decisions independently.
    
    Decision Flow:
        wait_for: []
        send_to: []
        then: generate plan via LLM
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        temperature: float = 0.7,
        verbose: bool = True,
        goal_instruction: str = "",
    ):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        
        # Decision Flow Configuration (same as individual.py)
        self.wait_for = []
        self.send_to = []
        self._system_prompt = None
        self.goal_instruction = goal_instruction
        self.team_agent_ids: List[str] = [agent_id]
    
    def _get_system_prompt(self) -> str:
        """Build system prompt with individual role."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + INDIVIDUAL_ROLE
        return self._system_prompt
    
    def _generate_plan_with_role(self, messages: Optional[List[Dict]] = None) -> SymbolicPlan:
        """Generate plan via LLM."""
        repair_messages = messages
        coop_config = self.coop_config
        if self.goal_instruction:
            goal_text = f"GLOBAL OBJECTIVE: {self.goal_instruction}"
            coop_config = f"{goal_text}\n\n{coop_config}" if coop_config else goal_text
        prompt_messages = self._build_plan_prompt_messages(
            system_prompt=self._get_system_prompt(),
            agent_names=self.team_agent_ids,
            messages=repair_messages,
            coop_config_override=coop_config,
        )
        return self._generate_plan_from_messages(
            prompt_messages=prompt_messages,
            repair_messages=repair_messages,
            label="Individual Plan Generation",
            verbose_prefix="Calling LLM for plan...",
        )
    
    def _execute_flow(self) -> SymbolicPlan:
        """
        Execute decision flow (same structure as individual.py):
        1. No waiting
        2. Clear any stray messages
        3. No sending
        4. Generate plan via LLM
        """
        # Clear any stray messages
        self.get_messages(clear_buffer=True)
        
        # Generate plan via LLM
        self.plan = self._generate_plan_with_role()
        return self.plan
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Individual agents replan only when COOP2 repair requests it."""
        if not self._handle_coop2_repair_interrupt(self._generate_plan_with_role):
            self.get_messages(clear_buffer=True)


def create_llm_individual_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True,
    goal_instruction: str = "",
) -> Dict[str, LLMIndividualAgent]:
    """
    Create LLM-powered agents for individual topology (no communication).
    """
    agents = {}
    team_agent_ids = [f"agent_{i}" for i in range(n_agents)]
    for i in range(n_agents):
        agent_id = team_agent_ids[i]
        agents[agent_id] = LLMIndividualAgent(
            agent_id,
            llm_client,
            temperature,
            verbose,
            goal_instruction=goal_instruction,
        )
        agents[agent_id].team_agent_ids = team_agent_ids
    return agents
