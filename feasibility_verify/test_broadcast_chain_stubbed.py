"""CPU-only regression test for the broadcast-chain topology's interrupt path.

No Isaac, no GPU, no LLM: the agent's LLM client and message broker are both
stubbed, so what is pinned here is the protocol -- who speaks, to whom, and
what the receiver knows when it plans.

Four claims:

  1. **One agent relays per step of a wave, in chain order.** A receiver speaks
     only when the batch contains its immediate predecessor's message. Relaying
     on any received message compounds: agent_1 broadcasts to 2..6, all five
     speak, and agent_3 speaks twice -- once for agent_1 and again for agent_2.
  2. A relay happens on **both** branches. RESUME used to return silently, so a
     message died at whichever agent received it and the chain's information
     stopped propagating one hop in.
  3. The messages that caused the interrupt reach ``broadcast_history`` on both
     branches -- and at every agent, relay or not. They did not on REPLAN:
     ``decide_interrupt`` cleared the buffer and ``_execute_flow``'s own read
     then came back empty, so the replan was generated without the proposal
     that triggered it.
  4. A relay interrupts, because an agent mid-primitive that merely buffers the
     message never runs handle_interrupt and the wave would die there.

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

    # decide_interrupt is one LLM round trip; pin its answer instead of faking a
    # model, and count the calls -- the point of the turn gate is that a
    # bystander makes none. _execute_flow is the REPLAN branch, also a call.
    agent.decisions = 0

    def fake_decide_interrupt(messages=None):
        agent.decisions += 1
        return decision

    agent.decide_interrupt = fake_decide_interrupt
    agent.replanned = False

    def fake_execute_flow():
        # Stands in for the LLM call in step 3, plus step 4's broadcast.
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


def deliver(agent, messages) -> None:
    """Put messages in an agent's buffer the way the broker would."""
    agent.message_buffer = list(messages)
    agent.buffer_senders = {msg["sender"]: True for msg in messages}


def main() -> int:
    print("test: the chain speaks downstream only, and agent N-1 speaks to nobody")
    topology = create_llm_broadcast_chain_topology(9, FakeLLMClient(), verbose=False)
    assert topology["agent_0"].send_to == [f"agent_{j}" for j in range(1, 9)]
    assert topology["agent_3"].send_to == [f"agent_{j}" for j in range(4, 9)]
    assert topology["agent_8"].send_to == [], "the last speaker has nobody below it"
    assert topology["agent_0"].wait_for == []
    assert topology["agent_5"].wait_for == ["agent_4"]
    ok("send_to is strictly downstream, wait_for is the immediate predecessor")

    print("\ntest: only the immediate successor relays, on either branch")
    for decision, kind in ((InterruptDecision.RESUME, "resume_ack"),
                           (InterruptDecision.REPLAN, "proposal")):
        relay = make_agent(speaker_order=3, n_agents=9, decision=decision)
        deliver(relay, [inbound("agent_2", "[agent_2] Proposed plan: ontop(apple.n.01_1, ...)")])
        relay.handle_interrupt()
        assert len(relay.sent) == 1, f"{kind}: expected exactly one relay, got {len(relay.sent)}"
        assert relay.sent[0]["recipients"] == [f"agent_{j}" for j in range(4, 9)]
        assert relay.sent[0]["metadata"]["chain_message"] == kind
        assert (relay.replanned is (decision is InterruptDecision.REPLAN))

        assert relay.decisions == 1, "the relay consults the model exactly once"

        bystander = make_agent(speaker_order=3, n_agents=9, decision=decision)
        deliver(bystander, [inbound("agent_0", "[agent_0] Proposed plan: ontop(apple.n.01_9, ...)")])
        bystander.handle_interrupt()
        assert bystander.sent == [], (
            f"{kind}: agent_3 relayed for agent_0, whose successor is agent_1 -- "
            "that is the compounding this gate exists to stop"
        )
        # And it does not think either. _execute_flow already waits for the
        # predecessor before planning, so the opening round is ordered; without
        # the same gate here, agent_3 reconsidered when agent_1 spoke and then
        # again when agent_2 did -- two round trips for one wave.
        assert bystander.decisions == 0, (
            f"{kind}: a bystander consulted the model for a proposal aimed at the agent above it"
        )
        assert not bystander.replanned
        # Silent, but not deaf.
        assert any("apple.n.01_9" in entry for entry in bystander.broadcast_history)
    ok("out of turn: no message, no model call -- but the proposal is still recorded")

    print("\ntest: one wave = one message per agent, in order")
    # agent_0 opens; every later agent receives it, and thereafter each agent
    # receives whatever its predecessors sent. Replay that faithfully.
    n = 6
    team = {i: make_agent(i, n, InterruptDecision.RESUME) for i in range(n)}
    for agent in team.values():
        deliver(agent, [])
    spoke = []
    pending = [{"sender": "agent_0", "recipients": [f"agent_{j}" for j in range(1, n)],
                "content": "[agent_0] Proposed plan: ontop(apple.n.01_1, ...)", "timestamp": 0.0}]
    while pending:
        message = pending.pop(0)
        for name in message["recipients"]:
            receiver = team[int(name.split("_")[1])]
            deliver(receiver, [message])
            receiver.handle_interrupt()
            while receiver.sent:
                out = receiver.sent.pop(0)
                spoke.append(name)
                pending.append({"sender": name, "recipients": out["recipients"],
                                "content": out["content"], "timestamp": 0.0})
    # 1..n-2: the last agent is the wave's relay too, but its send_to is empty,
    # so the wave ends there rather than at a rule.
    assert spoke == [f"agent_{i}" for i in range(1, n - 1)], (
        f"a wave should be relayed once by each agent in order, got {spoke}"
    )
    assert len(spoke) == len(set(spoke)), f"an agent spoke twice in one wave: {spoke}"
    ok("agent_0 speaks, then 1, 2, 3, 4 in order -- each exactly once, and 5 has nobody")

    print("\ntest: a relay interrupts, or the wave dies at the first busy agent")
    assert chain_module.RELAY_INTERRUPTS_EXECUTION is True
    relay = make_agent(speaker_order=3, n_agents=9, decision=InterruptDecision.RESUME)
    deliver(relay, [inbound("agent_2", "[agent_2] Proposed plan: ontop(apple.n.01_5, ...)")])
    relay.handle_interrupt()
    assert relay.sent[0]["metadata"]["interrupts_execution"] is True
    assert relay.plan.specification in relay.sent[0]["content"], "say which plan is being kept"
    ok("the relay stops the next agent, so it actually runs handle_interrupt")

    print("\ntest: both branches plan and speak having read the triggering message")
    for decision, label in ((InterruptDecision.RESUME, "RESUME"),
                            (InterruptDecision.REPLAN, "REPLAN")):
        subject = make_agent(speaker_order=2, n_agents=9, decision=decision)
        deliver(subject, [inbound("agent_1", "[agent_1] Proposed plan: ontop(apple.n.01_7, ...)")])
        subject.handle_interrupt()
        assert any("apple.n.01_7" in entry for entry in subject.broadcast_history), (
            f"{label}: the message that caused the interrupt never reached broadcast_history"
        )
        assert subject.message_buffer == [], f"{label}: the buffer must be drained once"
    ok("the triggering proposal is in broadcast_history on RESUME and on REPLAN")

    print("\ntest: an empty buffer is not an interrupt")
    quiet = make_agent(speaker_order=4, n_agents=9, decision=InterruptDecision.REPLAN)
    deliver(quiet, [])
    quiet.handle_interrupt()
    assert quiet.sent == [] and not quiet.replanned
    ok("no messages, no plan churn and no broadcast")

    print("\ntest: the last speaker has nobody to acknowledge to")
    last = make_agent(speaker_order=8, n_agents=9, decision=InterruptDecision.RESUME)
    deliver(last, [inbound("agent_7", "[agent_7] Proposed plan: ontop(apple.n.01_2, ...)")])
    last.handle_interrupt()
    assert last.sent == [], "agent_8 has an empty send_to; _broadcast must be a no-op"
    assert any("apple.n.01_2" in entry for entry in last.broadcast_history)
    ok("agent_8 still reads, and sends nothing")

    print("\ntest: the real broker stops an executing agent for a relay")
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
        deliver(agent, [])
    broker = MessageBroker({"agent_0": sender, "agent_1": middle})

    for kind in ("resume_ack", "proposal"):
        middle._set_state(AgentState.X, timestamp=0.0, env_step=0)
        broker.send_message(
            sender_id="agent_0", recipients=["agent_1"], content=kind,
            metadata={"type": "broadcast_chain", "chain_message": kind,
                      "interrupts_execution": chain_module.RELAY_INTERRUPTS_EXECUTION},
            timestamp=0.0, env_step=0,
        )
        assert middle.state is AgentState.I, (
            f"a {kind} relay must stop an executing agent, or the wave dies here"
        )
    assert len(middle.message_buffer) == 2, "both messages are there to be read"
    ok("either kind of relay stops the next agent")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
