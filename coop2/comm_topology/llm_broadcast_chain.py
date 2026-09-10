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


#: One pass down the chain is a **wave**, and the invariant is that each agent
#: emits at most one message per wave, in chain order.
#:
#: That invariant is the whole reason the relay is affordable, and it is easy
#: to lose. "Speak whenever a message arrives" loses it immediately: agent_1
#: broadcasts to 2..6, all five of them speak, and agent_3 speaks twice -- once
#: for agent_1's message and again for agent_2's -- so one wave compounds into
#: many. Gating the relay on the *immediate predecessor* gives exactly the
#: intended shape instead, because only one agent is ever the successor of the
#: agent that just spoke:
#:
#:     agent_1 speaks -> agent_2 relays -> agent_3 relays -> ... -> agent_5 relays
#:
#: N-1 relays, one per agent, in order, and a relay never starts a second wave.
#: Everyone below still *receives* the original broadcast and still puts it in
#: broadcast_history; what the gate removes is the duplicated re-fan-out -- and
#: the duplicated *thinking*, which is the expensive half: an agent whose turn
#: has not come does not consult the model at all.
#:
#: A relay interrupts, and has to: an agent mid-primitive that merely buffers
#: the message never runs handle_interrupt, so the wave would die at the first
#: busy agent instead of reaching the end of the chain.
RELAY_INTERRUPTS_EXECUTION = True

#: Seconds an agent will hold in I waiting for the wave to reach it before
#: giving up and releasing.
#:
#: The hold's real guard is `_any_waiting_agent_not_ready`: an agent waits only
#: while its predecessor is still uncommitted, which is the same condition
#: `_execute_flow` step 1 uses for the opening round. This cap is the belt for
#: that pair of braces -- the world is frozen while any agent is in I, so a hold
#: that never ends does not slow the run down, it stops it. A wave hop costs one
#: model round trip, measured at 3.9 s, so a nine-agent wave is ~35 s.
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
        """Take everything buffered and record it, whether or not it is my turn.

        Out of turn is not out of the loop: a relay plans having read everything
        said above it, which is the property the chain exists for.
        """
        messages = self.get_messages(clear_buffer=True)
        for msg in messages:
            self.broadcast_history.append(f"[{msg['sender']}]: {msg['content']}")
        return messages

    def _await_my_turn(self, pending: List[Dict]) -> List[Dict]:
        """Stay in I until my predecessor speaks, or until it commits without me.

        Being woken by a proposal aimed at the agent above me *is* an interrupt
        -- I have been told something and cannot act on it yet -- so the agent
        holds here rather than bouncing back to W and being woken again by the
        next hop. Before this, agent_5 in a six-agent wave logged five separate
        waiting spans at one env_step, one per agent above it; now that is one
        interrupted span, which is what actually happened.

        Blocking here is what keeps the state at I: `create_agent_thread` calls
        `set_ready()` only once `handle_interrupt` returns.
        """
        deadline = time.monotonic() + TURN_WAIT_TIMEOUT
        while not self._predecessor_spoke(pending):
            # Predecessor has committed: whatever it was going to say, it has
            # said. Nothing is coming, so do not hold the world for it.
            if not self._any_waiting_agent_not_ready(self._all_agents):
                break
            if time.monotonic() >= deadline:
                print(
                    f"  [{self.agent_id}] waited {TURN_WAIT_TIMEOUT:.0f}s for "
                    f"{self.wait_for} and the wave never arrived; releasing"
                )
                break
            time.sleep(0.05)
            pending.extend(self._drain_into_history())
        return pending

    def _predecessor_spoke(self, messages: List[Dict]) -> bool:
        """Has my immediate predecessor spoken in this batch -- is it my turn?

        This gates **thinking as well as speaking**, and the thinking half is
        the one that costs. `_execute_flow` already waits for the predecessor
        before planning, so the opening round is ordered; the interrupt path had
        no such wait, so mid-episode agent_3 would reconsider the moment
        agent_1 spoke and then reconsider again when agent_2 did -- two LLM
        round trips for one wave, from a proposal that was not addressed to it
        in the first place. Measured on the 6-agent run: 53 % of deliveries land
        on an agent that is not the next speaker.

        Not its turn does not mean not its business: the message is still
        recorded, and it is read when the agent's turn does come.
        """
        if not self.wait_for:
            return False  # the first speaker has no predecessor to wait for
        predecessor = self.wait_for[0]
        return any(msg.get("sender") == predecessor for msg in messages)

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
        if self.wait_for and self._any_waiting_agent_not_ready(self._all_agents):
            while not self.wait_for_messages_from(self.wait_for):
                if not self._any_waiting_agent_not_ready(self._all_agents):
                    break
                time.sleep(0.05)
        
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
        """Decide resume/replan, then pass the news down the chain either way."""
        # Read the buffer ONCE. decide_interrupt() clears it when it reads it
        # itself, and _execute_flow's own get_messages() then came back empty --
        # so a replan was generated without the proposal that triggered it ever
        # reaching broadcast_history. In a topology whose whole point is that
        # later speakers see earlier proposals, that is the wrong way round.
        messages = self._drain_into_history()
        if not messages:
            return

        if self._handle_coop2_repair_interrupt(self._generate_plan_with_role, messages=messages):
            return

        # Hold in I until the wave reaches me, rather than returning to W and
        # being woken again by every remaining hop. Still not one LLM call
        # while waiting: reconsidering on a proposal aimed at the agent above me
        # is duplicated thinking, not just duplicated traffic.
        messages = self._await_my_turn(messages)
        if not self._predecessor_spoke(messages):
            return

        # My turn. Ask before discarding: replanning unconditionally was
        # upstream's behaviour and works in a grid world where an action is one
        # step, but here a NAVIGATE_TO runs for hundreds of ticks, so a message
        # arriving mid-trip used to throw away all the travel already paid for.
        # On RESUME nothing is touched -- the primitive keeps running with its
        # accrued delay and the plan continues from where it was -- but the
        # agent still speaks, because a receiver that stays the course is
        # telling the agents below it the one thing they cannot otherwise
        # learn: this target is committed and will not be reconsidered. Before,
        # only REPLAN spoke, so 81 % of messages died where they landed.
        if self.decide_interrupt(messages=messages) is InterruptDecision.RESUME:
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
