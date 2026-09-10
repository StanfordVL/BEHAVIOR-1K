"""CPU-only regression test for the broadcast-chain topology's interrupt path.

No Isaac, no GPU, no LLM: the agent's model client and message broker are both
stubbed, so what is pinned here is the protocol -- who speaks, to whom, and what
the speaker knows when it plans.

Nearly all of it is downstream of one mechanism: upstream's ``_execute_flow``
step 1, the wait for the previous speaker. Upstream's ``handle_interrupt`` is
literally ``self._execute_flow()``, so that wait has always governed the
interrupt path too; this port had lost it there, and the ordering with it.

  1. **The wait is the ordering discipline.** An agent woken by a proposal aimed
     at someone above it neither thinks nor speaks: it blocks, which keeps it in
     I, until its predecessor speaks. One wave is one message per agent in chain
     order -- not one per message received, which compounds (agent_0 broadcasts
     to 1..5, all five speak, agent_2 speaks again for agent_1's message...).
  2. It releases when the predecessor commits without ever speaking to it, so a
     wait cannot deadlock a barrier that freezes the world while it holds.
  3. A relay happens on **both** branches. RESUME used to return silently, so a
     message died at whichever agent received it.
  4. Everything heard while waiting reaches ``broadcast_history``, so an agent
     plans having read every proposal it sat through.
  5. A relay interrupts, because an agent mid-primitive that merely buffers the
     message never runs handle_interrupt and the wave would die there.

Run:
    python feasibility_verify/test_broadcast_chain_stubbed.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent import InterruptDecision
from coop2.comm_topology import llm_broadcast_chain as chain_module
from coop2.comm_topology.llm_broadcast_chain import (
    LLMBroadcastChainAgent,
    create_llm_broadcast_chain_topology,
)


def ok(message: str) -> None:
    print(f"  ok: {message}")


class FakeLLMClient:
    model = "stub"


class SimplePlan:
    def __init__(self, specification: str) -> None:
        self.specification = specification
        self.actions = ["navigate_to(apple.n.01_3)", "grasp(apple.n.01_3)"]


def make_agent(speaker_order: int, n_agents: int, decision: InterruptDecision):
    """One chain agent with the model and the broker replaced.

    ``decisions`` counts model consultations: the point of the wait is that an
    agent out of turn makes none, which no assertion on messages alone can see.
    """
    agent = LLMBroadcastChainAgent(
        agent_id=f"agent_{speaker_order}",
        llm_client=FakeLLMClient(),
        speaker_order=speaker_order,
        following_agent_ids=[f"agent_{j}" for j in range(speaker_order + 1, n_agents)],
        previous_agent_id=f"agent_{speaker_order - 1}" if speaker_order else None,
        n_agents=n_agents,
        verbose=False,
    )
    # _broadcast refuses to send with no broker, so one has to be present even
    # though send_message below is what actually records the call.
    agent.message_broker = object()
    agent.plan = SimplePlan("ontop(apple.n.01_3, coffee_table.n.01_1)")
    agent.decisions = 0
    agent.replanned = False

    def fake_decide_interrupt(messages=None):
        agent.decisions += 1
        return decision

    def fake_execute_flow():
        # Stands in for step 3's model call, keeping step 1's wait and step 4.
        agent._wait_for_previous_speaker()
        agent._drain_into_history()
        agent.replanned = True
        agent._broadcast(
            agent._format_current_plan_contribution(), kind="proposal", interrupts=True
        )
        return agent.plan

    sent = []

    def fake_send_message(recipients, content, metadata=None):
        sent.append({"recipients": list(recipients), "content": content,
                     "metadata": dict(metadata or {})})
        return sent[-1]

    agent.decide_interrupt = fake_decide_interrupt
    agent._execute_flow = fake_execute_flow
    agent.send_message = fake_send_message
    agent.sent = sent
    return agent


def inbound(sender: str, content: str) -> dict:
    return {"sender": sender, "content": content, "recipients": [], "timestamp": 0.0}


def deliver(agent, messages) -> None:
    """Put messages in an agent's buffer the way the broker would."""
    agent.message_buffer = list(messages)
    agent.buffer_senders = {msg["sender"]: True for msg in messages}


def main() -> int:
    print("test: the chain speaks downstream only, and the last speaker to nobody")
    topology = create_llm_broadcast_chain_topology(9, FakeLLMClient(), verbose=False)
    assert topology["agent_0"].send_to == [f"agent_{j}" for j in range(1, 9)]
    assert topology["agent_3"].send_to == [f"agent_{j}" for j in range(4, 9)]
    assert topology["agent_8"].send_to == [], "the last speaker has nobody below it"
    assert topology["agent_0"].wait_for == []
    assert topology["agent_5"].wait_for == ["agent_4"]
    ok("send_to is strictly downstream, wait_for is the immediate predecessor")

    print("\ntest: woken out of turn, the agent holds in I -- no message, no model call")
    for decision, kind in ((InterruptDecision.RESUME, "resume_ack"),
                           (InterruptDecision.REPLAN, "proposal")):
        holder = make_agent(speaker_order=3, n_agents=6, decision=decision)
        predecessor = make_agent(speaker_order=2, n_agents=6, decision=decision)
        predecessor.ready = False                      # still to speak
        holder._all_agents = {"agent_2": predecessor}
        deliver(holder, [inbound("agent_0", "[agent_0] Proposed plan: ontop(apple.n.01_6, ...)")])

        thread = threading.Thread(target=holder.handle_interrupt, daemon=True)
        thread.start()
        thread.join(timeout=0.4)
        assert thread.is_alive(), f"{kind}: released instead of holding for its predecessor"
        assert holder.sent == [] and holder.decisions == 0, (
            f"{kind}: acted on a proposal aimed at the agent above it"
        )

        deliver(holder, [inbound("agent_2", "[agent_2] Proposed plan: ontop(apple.n.01_3, ...)")])
        thread.join(timeout=5.0)
        assert not thread.is_alive(), f"{kind}: never woke when its predecessor spoke"
        assert holder.decisions == 1, f"{kind}: exactly one consultation, on its turn"
        assert len(holder.sent) == 1, f"{kind}: exactly one relay"
        assert holder.sent[0]["recipients"] == [f"agent_{j}" for j in range(4, 6)]
        assert holder.sent[0]["metadata"]["chain_message"] == kind
        assert any("apple.n.01_3" in e for e in holder.broadcast_history), (
            f"{kind}: the message that freed it must be in front of it when it plans"
        )
    ok("one hold, then one decision and one relay when the turn comes")

    print("\ntest: the wait releases if the predecessor commits without speaking")
    # Upstream proceeds here rather than swallowing the interrupt: nobody above
    # is going to speak, so the agent acts on what it already has.
    committed = make_agent(speaker_order=2, n_agents=6, decision=InterruptDecision.RESUME)
    committed.ready = True
    stranded = make_agent(speaker_order=3, n_agents=6, decision=InterruptDecision.RESUME)
    stranded._all_agents = {"agent_2": committed}
    deliver(stranded, [inbound("agent_0", "[agent_0] Proposed plan: ontop(apple.n.01_9, ...)")])
    started = time.monotonic()
    stranded.handle_interrupt()
    assert time.monotonic() - started < 1.0, "a committed predecessor must not be waited on"
    assert stranded.decisions == 1 and len(stranded.sent) == 1
    ok("no wave coming, no deadlock, and the interrupt is not swallowed")

    print("\ntest: one wave = one message per agent, in order")
    # An agent already in I is not re-entered -- `interrupt()` is a no-op from I
    # and `create_agent_thread` runs one handler -- so every agent below the
    # speaker handles the wave exactly once, blocking inside until its turn.
    # The replay therefore appends to buffers as relays happen and calls each
    # handler once, in chain order.
    n = 6
    team = {i: make_agent(i, n, InterruptDecision.RESUME) for i in range(n)}
    for agent in team.values():
        deliver(agent, [])
        agent.ready = False                            # all woken together
        agent._all_agents = {a: team[int(a.split("_")[1])] for a in agent.wait_for}
    team[0].ready = True                               # agent_0 has spoken

    def push(agent, message):
        agent.message_buffer.append(message)
        agent.buffer_senders[message["sender"]] = True

    opening = {"sender": "agent_0", "recipients": [f"agent_{j}" for j in range(1, n)],
               "content": "[agent_0] Proposed plan: ontop(apple.n.01_1, ...)", "timestamp": 0.0}
    for name in opening["recipients"]:
        push(team[int(name.split("_")[1])], opening)

    spoke = []
    for i in range(1, n):
        receiver = team[i]
        receiver.handle_interrupt()
        for out in receiver.sent:
            spoke.append(f"agent_{i}")
            relay = {"sender": f"agent_{i}", "recipients": out["recipients"],
                     "content": out["content"], "timestamp": 0.0}
            for name in out["recipients"]:
                push(team[int(name.split("_")[1])], relay)
        receiver.ready = True                          # spoken; its successor may go

    # 1..n-2 relay; agent_5 is the wave's relay too but its send_to is empty, so
    # the wave ends there rather than at a rule.
    assert spoke == [f"agent_{i}" for i in range(1, n - 1)], (
        f"a wave should be relayed once by each agent in order, got {spoke}"
    )
    assert len(spoke) == len(set(spoke)), f"an agent spoke twice in one wave: {spoke}"
    # Each relay planned having read everything said above it, not just the
    # message that freed it.
    assert any("agent_0" in e for e in team[4].broadcast_history)
    assert any("agent_3" in e for e in team[4].broadcast_history)
    ok("agent_0 speaks, then 1, 2, 3, 4 in order -- each exactly once")

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
    assert quiet.sent == [] and not quiet.replanned and quiet.decisions == 0
    ok("no messages, no plan churn and no broadcast")

    print("\ntest: the last speaker has nobody to relay to")
    last = make_agent(speaker_order=8, n_agents=9, decision=InterruptDecision.RESUME)
    deliver(last, [inbound("agent_7", "[agent_7] Proposed plan: ontop(apple.n.01_2, ...)")])
    last.handle_interrupt()
    assert last.sent == [], "agent_8 has an empty send_to; _broadcast must be a no-op"
    assert any("apple.n.01_2" in entry for entry in last.broadcast_history)
    ok("agent_8 still reads, and sends nothing")

    print("\ntest: the real broker stops an executing agent for a relay")
    # Everything above stubs send_message, so it pins what the topology *asks*
    # for. This goes through the real MessageBroker, which is what decides
    # whether an agent mid-primitive is actually stopped -- if the metadata key
    # were wrong every test above would still pass and the wave would die at
    # the first busy agent.
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
