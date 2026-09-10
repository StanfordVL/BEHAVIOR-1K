"""
LLM-powered Broadcast Chain topology.

The communication pattern is a fixed order in which each agent broadcasts to
all agents that follow it.
Each agent broadcasts to ALL following agents.

Decision Flow:
    Agent 0: plan → send (first speaker)
    Agent 1+: wait → plan → send
    
    Each agent:
        1. wait_for: [previous_agent] (empty for agent 0)
        2. generate plan via LLM
        3. send_to: [all following agents] with current plan proposal
"""

import time
from typing import Dict, List, Optional

from coop2.cognitive.agent import LLMClient, InterruptDecision
from coop2.cognitive.agent.base_llm_agent import BaseLLMAgent
from coop2.cognitive.agent.prompts import build_system_prompt
from coop2.cognitive.plan import SymbolicPlan


#: Does a relay interrupt an agent that is mid-primitive?
#:
#: Yes, and it has to: an agent that merely buffers the message never runs
#: handle_interrupt, so the wave would die at the first busy agent instead of
#: reaching the end of the chain.
RELAY_INTERRUPTS_EXECUTION = True

#: Seconds an agent will wait for its predecessor before giving up.
#:
#: Upstream's loop has no cap -- its only exit is `_any_waiting_agent_not_ready`
#: going False. That is the right guard and it is kept, but here the world is
#: frozen while any agent is not ready, so a wait that never ends does not slow
#: the run down, it stops it. A wave hop costs one model round trip, measured at
#: 3.9 s, so a nine-agent wave is ~35 s.
TURN_WAIT_TIMEOUT = 120.0


def get_broadcast_chain_role(speaker_order: int, n_agents: int) -> str:
    """Describe an agent's position in the Broadcast Chain."""
    if speaker_order == 0:
        return """
## Your Role: FIRST SPEAKER
Speak first and broadcast your current plan proposal to later speakers.
"""
    elif speaker_order == n_agents - 1:
        return """
## Your Role: FINAL SPEAKER
Speak last after seeing previous proposals, then commit your own plan.
"""
    else:
        return f"""
## Your Role: SPEAKER {speaker_order + 1} of {n_agents}
Read previous proposals, commit your plan, and broadcast it to later speakers.
"""


class LLMBroadcastChainAgent(BaseLLMAgent):
    """
    LLM-powered agent for Broadcast Chain.
    
    Decision Flow:
        wait_for: [previous_agent_id] (empty for first speaker)
        send_to: [all following_agent_ids]
        then: generate plan via LLM
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient, speaker_order: int,
                 following_agent_ids: List[str], previous_agent_id: str,
                 n_agents: int, temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.speaker_order = speaker_order
        self.n_agents = n_agents
        
        # Decision flow configuration.
        self.wait_for = [previous_agent_id] if previous_agent_id else []
        self.send_to = following_agent_ids
        
        self._last_flow_step = None
        self._all_agents: Dict[str, 'LLMBroadcastChainAgent'] = {}
        self._system_prompt = None
        self.broadcast_history: List[str] = []
    
    def _get_system_prompt(self) -> str:
        """Build the system prompt for this position in the chain."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + get_broadcast_chain_role(self.speaker_order, self.n_agents)
        return self._system_prompt
    
    def _generate_message(self, context: str) -> str:
        """Create a deterministic contribution from the current plan."""
        return context

    def _format_current_plan_contribution(self) -> str:
        """Summarize the generated plan for later speakers."""
        if self.plan is None:
            return (
                f"[{self.agent_id}] I do not yet have a plan; choose from the "
                "current observations and score objective."
            )
        action_text = "; ".join(
            f"{index}. {action}"
            for index, action in enumerate(self.plan.actions[:4], start=1)
        )
        if len(self.plan.actions) > 4:
            action_text += f"; ... +{len(self.plan.actions) - 4} more"
        return (
            f"[{self.agent_id}] Proposed plan: {self.plan.specification}. "
            f"Actions: {action_text}. Later speakers should consider this plan, "
            "then support, adapt, or choose a different plan from their own observation."
        )
    
    def _drain_into_history(self) -> List[Dict]:
        """Take everything buffered and record it for the next plan prompt."""
        messages = self.get_messages(clear_buffer=True)
        for msg in messages:
            self.broadcast_history.append(f"[{msg['sender']}]: {msg['content']}")
        return messages

    def _wait_for_previous_speaker(self) -> None:
        """Block until my predecessor has spoken, or until it commits without me.

        This is upstream's `_execute_flow` step 1, lifted out so the interrupt
        path can share it. It is the whole ordering mechanism of the chain and
        it gives three properties at once, which is why upstream needs nothing
        else: an agent does not think out of turn (no model call happens here),
        it does not speak out of turn (it has not reached step 4), and it stays
        in **I** while it waits, because `create_agent_thread` calls
        `set_ready()` only once the handler returns. One wave is therefore one
        message per agent, in chain order.

        The exit condition is the predecessor becoming *ready*: whatever it was
        going to say, it has said, so nothing is coming and the world must not
        be held for it.
        """
        if not (self.wait_for and self._any_waiting_agent_not_ready(self._all_agents)):
            return
        deadline = time.monotonic() + TURN_WAIT_TIMEOUT
        while not self.wait_for_messages_from(self.wait_for):
            if not self._any_waiting_agent_not_ready(self._all_agents):
                break
            if time.monotonic() >= deadline:
                print(
                    f"  [{self.agent_id}] waited {TURN_WAIT_TIMEOUT:.0f}s for "
                    f"{self.wait_for} and the wave never arrived; releasing"
                )
                break
            time.sleep(0.05)

    def _format_resume_contribution(self, messages: List[Dict]) -> str:
        """What a receiver says downstream when it keeps the plan it had."""
        senders = ", ".join(sorted({msg["sender"] for msg in messages})) or "a teammate"
        spec = self.plan.specification if self.plan is not None else "no plan"
        return (
            f"[{self.agent_id}] Heard {senders} and am continuing with my current "
            f"plan: {spec}. Treat it as committed, not as a fresh proposal."
        )

    def _broadcast(self, content: str, kind: str, interrupts: bool) -> None:
        """Send `content` to every agent after this one in the chain."""
        if not self.send_to or self.message_broker is None:
            return
        self.send_message(
            recipients=self.send_to,
            content=content,
            metadata={
                'type': 'broadcast_chain',
                'chain_message': kind,
                'speaker_order': self.speaker_order,
                'interrupts_execution': interrupts,
            },
        )
        if self.verbose:
            print(f"  [{self.agent_id}] Broadcast ({kind}) to {self.send_to}: {content[:50]}...")

    def _generate_plan_with_role(self, messages: Optional[List[Dict]] = None) -> SymbolicPlan:
        """Generate a plan with the earlier Broadcast Chain messages."""
        repair_messages = messages
        broadcast_context = ""
        if self.broadcast_history:
            broadcast_context = "## Earlier Proposals\n" + "\n".join(self.broadcast_history) + "\n\n"
        
        prompt_messages = self._build_plan_prompt_messages(
            system_prompt=self._get_system_prompt(),
            agent_names=[f"agent_{i}" for i in range(self.n_agents)],
            messages=repair_messages,
            user_prefix=broadcast_context,
        )
        return self._generate_plan_from_messages(
            prompt_messages=prompt_messages,
            repair_messages=repair_messages,
            label="Broadcast Chain Plan Generation",
            verbose_prefix="BROADCAST CHAIN calling LLM for plan...",
        )
    
    def _execute_flow(self) -> SymbolicPlan:
        """
        Execute the Broadcast Chain decision flow:
        1. Wait for message from previous agent (if not first)
        2. Generate a current plan using earlier proposals
        3. Broadcast that current plan to following agents
        """
        # Step 1: Wait for previous agent (only if they're not ready)
        self._wait_for_previous_speaker()

        # Step 2: Collect messages from earlier speakers.
        self._drain_into_history()
        
        # Step 3: Generate plan via LLM before broadcasting so downstream
        # speakers see the current proposal, not a stale previous plan.
        self._last_flow_step = self.env_step
        self.plan = self._generate_plan_with_role()

        # Step 4: Broadcast to all following agents
        self._broadcast(
            self._generate_message(self._format_current_plan_contribution()),
            kind='proposal',
            interrupts=True,
        )

        return self.plan
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Wait for the wave, then decide resume/replan, then pass it on.

        Upstream is one line here -- `self._execute_flow()` -- and that is the
        whole ordering discipline: step 1 waits for the predecessor, so an agent
        woken by a proposal aimed at someone above it neither thinks nor speaks
        until its turn comes. Our only departure is that we ask the model
        whether to resume before discarding a plan (upstream always replans,
        which is fine in a grid world where an action is one step and wrong here
        where a NAVIGATE_TO runs for hundreds of ticks). That decision has to
        happen after the wait, not before: deciding while the predecessor is
        still speaking is deciding on information that is about to change.
        """
        if self._handle_coop2_repair_interrupt(self._generate_plan_with_role):
            return

        self._wait_for_previous_speaker()

        # Peek, do not drain: `wait_for_messages_from` reads `buffer_senders`,
        # which `get_messages(clear_buffer=True)` wipes, so an early drain here
        # would leave _execute_flow's own step 1 unable to tell that the
        # predecessor had spoken.
        messages = self.get_messages(clear_buffer=False)
        if not messages:
            return

        if self.decide_interrupt(messages=messages) is InterruptDecision.RESUME:
            # Speak anyway. A receiver that keeps its plan is telling the agents
            # below it the one thing they cannot otherwise learn -- that this
            # target is committed and will not be reconsidered -- which is what
            # a later speaker needs in order to pick a different apple. With
            # only REPLAN speaking, 81 % of messages died where they landed.
            self._drain_into_history()
            self._broadcast(
                self._format_resume_contribution(messages),
                kind='resume_ack',
                interrupts=RELAY_INTERRUPTS_EXECUTION,
            )
            return
        self._execute_flow()

    def reset(self):
        """Reset agent state."""
        super().reset()
        self._last_flow_step = None
        self.broadcast_history = []


def create_llm_broadcast_chain_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True
) -> Dict[str, LLMBroadcastChainAgent]:
    """
    Create LLM-powered agents for Broadcast Chain.
    
    Broadcast order: Agent 0 speaks first, then Agent 1, ..., Agent n-1 speaks last.
    Each agent broadcasts to ALL agents that come after them.
    """
    agents = {}
    
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        following_ids = [f"agent_{j}" for j in range(i + 1, n_agents)]
        previous_id = f"agent_{i - 1}" if i > 0 else None
        
        agents[agent_id] = LLMBroadcastChainAgent(
            agent_id, llm_client, i, following_ids, previous_id, n_agents, temperature, verbose
        )
    
    # Set references to all agents
    for agent in agents.values():
        agent._all_agents = agents
    
    return agents
