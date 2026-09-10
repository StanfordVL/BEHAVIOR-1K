"""CPU-only regression test for the broadcast-chain topology's interrupt path.

No Isaac, no GPU, no LLM: the agent's LLM client and message broker are both
stubbed, so what is pinned here is the protocol -- who speaks, to whom, and
what the receiver knows when it plans.

Three claims:

  1. A receiver broadcasts downstream on **both** branches. RESUME used to
     return silently, so a message died at whichever agent received it and the
     chain's information stopped propagating one hop in.
  2. The messages that caused the interrupt reach ``broadcast_history`` on both
     branches. They did not on REPLAN: ``decide_interrupt`` cleared the buffer
     and ``_execute_flow``'s own read then came back empty, so the replan was
     generated without the proposal that triggered it.
  3. The acknowledgement does not interrupt an executing agent, while a
     proposal does -- the difference that keeps the chain's traffic linear
     instead of doubling at every position.

Run:
    python feasibility_verify/test_broadcast_chain_stubbed.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent import InterruptDecision
from coop2.comm_topology import llm_broadcast_chain as chain_module
from coop2.comm_topology.llm_broadcast_chain import (
    LLMBroadcastChainAgent,
    create_llm_broadcast_chain_topology,
)


def ok(message: str) -> None:
    print(f"  ok: {message}")


class FakeBroker:
    """Records sends; never delivers. Delivery is messages.py's job, not ours."""

    def __init__(self) -> None:
        self.sent = []


class FakeLLMClient:
    model = "stub"


def make_agent(speaker_order: int, n_agents: int, decision: InterruptDecision):
    """One chain agent with the LLM and the broker replaced."""
    agent = LLMBroadcastChainAgent(
        agent_id=f"agent_{speaker_order}",
        llm_client=FakeLLMClient(),
        speaker_order=speaker_order,
        following_agent_ids=[f"agent_{j}" for j in range(speaker_order + 1, n_agents)],
        previous_agent_id=f"agent_{speaker_order - 1}" if speaker_order else None,
        n_agents=n_agents,
        verbose=False,
    )
    agent.message_broker = FakeBroker()
    agent.plan = SimplePlan("ontop(apple.n.01_3, coffee_table.n.01_1)")

    # decide_interrupt is one LLM round trip; pin its answer instead of faking
    # a model. _execute_flow is the REPLAN branch and would call the model too.
    agent.decide_interrupt = lambda messages=None: decision
    agent.replanned = False

    def fake_execute_flow():
        agent.replanned = True
        agent._broadcast(
            agent._format_current_plan_contribution(), kind="proposal", interrupts=True
        )
        return agent.plan

    agent._execute_flow = fake_execute_flow

    sent = []

    def fake_send_message(recipients, content, metadata=None):
        sent.append({"recipients": list(recipients), "content": content,
                     "metadata": dict(metadata or {})})
        return sent[-1]

    agent.send_message = fake_send_message
    agent.sent = sent
    return agent


class SimplePlan:
    def __init__(self, specification: str) -> None:
        self.specification = specification
        self.actions = ["navigate_to(apple.n.01_3)", "grasp(apple.n.01_3)"]


def inbound(sender: str, content: str) -> dict:
    return {"sender": sender, "content": content, "recipients": [], "timestamp": 0.0}


def main() -> int:
    print("test: the chain speaks downstream only, and agent N-1 speaks to nobody")
    topology = create_llm_broadcast_chain_topology(9, FakeLLMClient(), verbose=False)
    assert topology["agent_0"].send_to == [f"agent_{j}" for j in range(1, 9)]
    assert topology["agent_3"].send_to == [f"agent_{j}" for j in range(4, 9)]
    assert topology["agent_8"].send_to == [], "the last speaker has nobody below it"
    assert topology["agent_0"].wait_for == []
    assert topology["agent_5"].wait_for == ["agent_4"]
    ok("send_to is strictly downstream, wait_for is the immediate predecessor")

    print("\ntest: RESUME still broadcasts downstream")
    agent = make_agent(speaker_order=3, n_agents=9, decision=InterruptDecision.RESUME)
    agent.message_buffer = [inbound("agent_1", "[agent_1] Proposed plan: ontop(apple.n.01_1, ...)")]
    agent.buffer_senders = {"agent_1": True}
    agent.handle_interrupt()
    assert not agent.replanned, "RESUME must not regenerate the plan"
    assert len(agent.sent) == 1, f"RESUME sent {len(agent.sent)} messages, expected 1"
    message = agent.sent[0]
    assert message["recipients"] == [f"agent_{j}" for j in range(4, 9)]
    assert message["metadata"]["chain_message"] == "resume_ack"
    assert agent.plan.specification in message["content"], "say which plan is being kept"
    ok("a receiver that keeps its plan tells the agents below it so")

    print("\ntest: RESUME does not interrupt an executing agent, a proposal does")
    assert message["metadata"]["interrupts_execution"] is chain_module.ACK_INTERRUPTS_EXECUTION
    assert chain_module.ACK_INTERRUPTS_EXECUTION is False, (
        "an interrupting ack doubles the traffic at every chain position"
    )
    replanner = make_agent(speaker_order=3, n_agents=9, decision=InterruptDecision.REPLAN)
    replanner.message_buffer = [inbound("agent_0", "[agent_0] Proposed plan: ontop(apple.n.01_9, ...)")]
    replanner.buffer_senders = {"agent_0": True}
    replanner.handle_interrupt()
    assert replanner.replanned, "REPLAN must regenerate the plan"
    assert len(replanner.sent) == 1
    assert replanner.sent[0]["metadata"]["interrupts_execution"] is True
    assert replanner.sent[0]["metadata"]["chain_message"] == "proposal"
    ok("proposal interrupts, acknowledgement buffers")

    print("\ntest: both branches plan and speak having read the triggering message")
    for decision, label in ((InterruptDecision.RESUME, "RESUME"),
                            (InterruptDecision.REPLAN, "REPLAN")):
        subject = make_agent(speaker_order=2, n_agents=9, decision=decision)
        subject.message_buffer = [inbound("agent_0", "[agent_0] Proposed plan: ontop(apple.n.01_7, ...)")]
        subject.buffer_senders = {"agent_0": True}
        subject.handle_interrupt()
        assert any("apple.n.01_7" in entry for entry in subject.broadcast_history), (
            f"{label}: the message that caused the interrupt never reached broadcast_history"
        )
        assert subject.message_buffer == [], f"{label}: the buffer must be drained once"
    ok("the triggering proposal is in broadcast_history on RESUME and on REPLAN")

    print("\ntest: an empty buffer is not an interrupt")
    quiet = make_agent(speaker_order=4, n_agents=9, decision=InterruptDecision.REPLAN)
    quiet.message_buffer = []
    quiet.buffer_senders = {}
    quiet.handle_interrupt()
    assert quiet.sent == [] and not quiet.replanned
    ok("no messages, no plan churn and no broadcast")

    print("\ntest: the last speaker has nobody to acknowledge to")
    last = make_agent(speaker_order=8, n_agents=9, decision=InterruptDecision.RESUME)
    last.message_buffer = [inbound("agent_7", "[agent_7] Proposed plan: ontop(apple.n.01_2, ...)")]
    last.buffer_senders = {"agent_7": True}
    last.handle_interrupt()
    assert last.sent == [], "agent_8 has an empty send_to; _broadcast must be a no-op"
    assert any("apple.n.01_2" in entry for entry in last.broadcast_history)
    ok("agent_8 still reads, and sends nothing")

    print("\ntest: the real broker buffers the ack and stops an executing agent for a proposal")
    # Everything above stubs send_message, so it pins what the topology *asks*
    # for. This one goes through the real MessageBroker, which is what decides
    # whether an agent mid-primitive is actually stopped -- if the metadata key
    # were wrong, every test above would still pass and the chain would either
    # stall or diverge.
    from coop2.cognitive.agent.agent import AgentState
    from coop2.cognitive.messages import MessageBroker

    sender = make_agent(speaker_order=0, n_agents=3, decision=InterruptDecision.RESUME)
    middle = make_agent(speaker_order=1, n_agents=3, decision=InterruptDecision.RESUME)
    for agent in (sender, middle):
        agent.message_buffer = []
        agent.buffer_senders = {}
    broker = MessageBroker({"agent_0": sender, "agent_1": middle})

    middle._set_state(AgentState.X, timestamp=0.0, env_step=0)
    broker.send_message(
        sender_id="agent_0", recipients=["agent_1"], content="ack",
        metadata={"type": "broadcast_chain", "chain_message": "resume_ack",
                  "interrupts_execution": chain_module.ACK_INTERRUPTS_EXECUTION},
        timestamp=0.0, env_step=0,
    )
    assert len(middle.message_buffer) == 1, "the ack must still be delivered"
    assert middle.state is AgentState.X, "the ack must not stop an executing agent"

    middle._set_state(AgentState.X, timestamp=0.0, env_step=0)
    broker.send_message(
        sender_id="agent_0", recipients=["agent_1"], content="proposal",
        metadata={"type": "broadcast_chain", "chain_message": "proposal",
                  "interrupts_execution": True},
        timestamp=0.0, env_step=0,
    )
    assert middle.state is AgentState.I, "a proposal must stop an executing agent"
    assert len(middle.message_buffer) == 2, "both messages are there to be read"
    ok("delivered either way; only the proposal interrupts")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
