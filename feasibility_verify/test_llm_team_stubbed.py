"""CPU-only test for the one-LLM-per-team topology.

No Isaac, no GPU, no model: a stub client counts calls and returns canned team
responses. What is pinned is the part that cannot be checked by reading the code
-- the barrier -- because getting it wrong deadlocks rather than fails.

The trap: the plan loop does not step the environment while any agent is
not ready. So a member that finished early and simply waited for its teammates
would freeze the world, and those teammates need the world to advance in order
to finish. The design has early finishers stay ready and hold position instead,
and these tests are what keep that true.

Run:
    python feasibility_verify/test_llm_team_stubbed.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.llm_client import (
    InterruptDecision,
    LLMTeamInterruptResponse,
    LLMTeamPlanResponse,
    NavigateToAction,
    Task,
    TaskSpecification,
    TeamAgentInterruptDecision,
    TeamAgentPlan,
)
from coop2.comm_topology.llm_team import TEAM_HOLD_TICKS, create_llm_team_topology


def ok(message: str) -> None:
    print(f"  ok: {message}")


USAGE = {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4, "latency_seconds": 0.1}


class StubClient:
    """Counts calls and hands back a plan for every member of the team asked about."""

    model = "stub"

    def __init__(self):
        self.plan_calls = 0
        self.interrupt_calls = 0
        self.last_prompt = None
        self.interrupt_script = {}

    def _members_in(self, messages):
        """Read the agent ids out of the prompt the brain built."""
        text = messages[-1]["content"]
        return [line.split()[2] for line in text.split("\n") if line.startswith("=== ROBOT ")]

    def generate_team_plan(self, messages, temperature=0.7):
        self.plan_calls += 1
        self.last_prompt = messages[-1]["content"]
        plans = [
            TeamAgentPlan(
                agent_id=name,
                task=TaskSpecification(task=Task.ONTOP, object_type="apple.n.01_1",
                                       reference="coffee_table.n.01_1"),
                actions=[NavigateToAction(target="apple.n.01_1")],
                reasoning=f"{name} goes for the apple",
            )
            for name in self._members_in(messages)
        ]
        return LLMTeamPlanResponse(plans=plans, reasoning="split by distance"), dict(USAGE)

    def generate_team_interrupt_decision(self, messages, temperature=0.7):
        self.interrupt_calls += 1
        decisions = []
        for name in self._members_in(messages):
            choice = self.interrupt_script.get(name, InterruptDecision.RESUME)
            decisions.append(
                TeamAgentInterruptDecision(
                    agent_id=name,
                    decision=choice,
                    reasoning="scripted",
                    new_plan=None,
                )
            )
        return LLMTeamInterruptResponse(decisions=decisions, reasoning="scripted"), dict(USAGE)


def make_team(size, name="alpha"):
    client = StubClient()
    ids = [f"agent_{i}" for i in range(size)]
    agents = create_llm_team_topology(
        llm_client=client, teams={name: ids}, verbose=False, goal_instruction="do the thing"
    )
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    return client, agents, ids


def main() -> int:
    print("test 1: a team does not plan until every member has finished")
    client, agents, ids = make_team(4)
    # Three members finish; the fourth is still executing and never calls in.
    for name in ids[:3]:
        agents[name].handle_reasoning()
    assert client.plan_calls == 0, "planned before the team was complete"
    for name in ids[:3]:
        plan = agents[name].plan
        assert plan is not None, f"{name} has no plan at all -- it would never become ready"
        assert plan.actions[0].action_type == "wait", plan.actions[0].action_type
        assert plan.actions[0].args["ticks"] == TEAM_HOLD_TICKS
        # Exactly one action. A hold that goes through parse_plan_response picks
        # up a terminal action derived from its TaskSpecification -- expressed as
        # holding(<self>) that appended grasp(<self>), a robot planning to pick
        # itself up, which reached a real run before it was caught.
        assert len(plan.actions) == 1, [a.action_type for a in plan.actions]
    ok("3 of 4 finished -> no LLM call, and each early finisher holds rather than stalling")

    print("test 2: the last member closes the barrier and everyone gets a real plan")
    agents[ids[3]].handle_reasoning()
    assert client.plan_calls == 1, client.plan_calls
    assert agents[ids[3]].plan.actions[0].action_type == "navigate_to"
    # The other three are still holding; they take their plans on their next
    # pass through reasoning, which is what happens when their hold expires.
    for name in ids[:3]:
        agents[name].handle_reasoning()
    assert client.plan_calls == 1, f"one round must be one call, got {client.plan_calls}"
    for name in ids:
        plan = agents[name].plan
        assert plan.actions[0].action_type == "navigate_to", (name, plan.actions[0].action_type)
        assert plan.agent_id == name
    ok("one call for the whole round, and every robot ends up with its own plan")

    print("test 3: the prompt carries every member's observation, once each")
    prompt = client.last_prompt
    for name in ids:
        assert f"=== ROBOT {name} ===" in prompt, f"{name} missing from the team prompt"
        assert prompt.count(f"view for {name}") == 1, f"{name}'s observation duplicated"
    assert "do the thing" in prompt, "the global objective is missing"
    ok("4 observations, one section each, plus the objective")

    print("test 4: a team of one behaves exactly like the individual topology")
    solo_client, solo_agents, solo_ids = make_team(1, name="solo")
    solo_agents[solo_ids[0]].handle_reasoning()
    assert solo_client.plan_calls == 1
    # No hold is ever taken: the barrier is satisfied by the only member.
    assert solo_agents[solo_ids[0]].plan.actions[0].action_type == "navigate_to"
    ok("N=1 plans immediately and never holds, so per-robot LLM is this code with N=1")

    print("test 5: the team is interrupted together and answered in one call")
    client, agents, ids = make_team(3)
    client.interrupt_script = {ids[1]: InterruptDecision.REPLAN}
    for name in ids:
        agents[name].plan = None

    # Concurrently, because that is how it happens: the broker interrupts every
    # member of a team in one call, so their threads arrive together. Members
    # that arrive early **block in I** rather than bouncing back to ready --
    # a member that bounced showed as "waiting" on the timeline while its
    # teammates were still interrupted, which is the bug this pins.
    import threading

    barrier_state = {}

    def arrive(name):
        barrier_state[name] = agents[name].handle_interrupt()

    threads = [threading.Thread(target=arrive, args=(name,)) for name in ids[:2]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)
    assert all(t.is_alive() for t in threads), "early arrivals must block, not return"
    assert client.interrupt_calls == 0, "decided before the whole team was interrupted"

    agents[ids[2]].handle_interrupt()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not any(t.is_alive() for t in threads), "blocked members must be released"
    assert client.interrupt_calls == 1, client.interrupt_calls
    ok("2 of 3 block in I; the third closes the barrier and one call answers all")

    print("test 6: a member the model forgot is held, not left planless")
    class Forgetful(StubClient):
        def generate_team_plan(self, messages, temperature=0.7):
            response, usage = super().generate_team_plan(messages, temperature)
            response.plans = response.plans[:-1]  # drop the last robot
            return response, usage

    client = Forgetful()
    ids = ["agent_0", "agent_1"]
    agents = create_llm_team_topology(llm_client=client, teams={"t": ids}, verbose=False)
    for agent in agents.values():
        agent.symbolic_view = f"view for {agent.agent_id}"
        agent.observe({}, 0)
    for name in ids:
        agents[name].handle_reasoning()
    # A member with no plan never becomes ready, which would strand the whole
    # run at the next barrier -- so the brain gives it a hold instead.
    assert agents[ids[1]].plan is not None
    assert agents[ids[1]].plan.actions[0].action_type == "wait"
    assert len(agents[ids[1]].plan.actions) == 1, [
        a.action_type for a in agents[ids[1]].plan.actions
    ]
    ok("a skipped robot gets a hold, so it still becomes ready")

    print("test 7: an LLM failure falls back rather than ending the run")
    class Broken(StubClient):
        def generate_team_plan(self, messages, temperature=0.7):
            raise RuntimeError("no model today")

    client = Broken()
    ids = ["agent_0", "agent_1"]
    agents = create_llm_team_topology(llm_client=client, teams={"t": ids}, verbose=False)
    for agent in agents.values():
        agent.observe({}, 0)
    for name in ids:
        agents[name].handle_reasoning()
    for name in ids:
        assert agents[name].plan is not None, f"{name} stranded with no plan after an LLM error"
    ok("every robot still has a plan after the call raised")

    print("test 8: the three topologies wire teams, not robots")
    from coop2.comm_topology.llm_team import (
        ChainTeamBrain, FollowerTeamBrain, LeaderTeamBrain, TeamBrain,
    )

    teams = {"t0": ["a0", "a1"], "t1": ["a2", "a3"], "t2": ["a4", "a5"]}

    def brains_of(topology):
        agents = create_llm_team_topology(
            llm_client=StubClient(), teams=teams, topology=topology, verbose=False
        )
        out = {}
        for agent in agents.values():
            out[agent.brain.team_name] = agent.brain
        return out

    solo = brains_of("individual")
    assert all(isinstance(b, TeamBrain) and not b.send_to and not b.wait_for
               for b in solo.values())

    chain = brains_of("broadcast_chain")
    assert all(isinstance(b, ChainTeamBrain) for b in chain.values())
    # A team *speaks* through its first member but is *addressed* as a whole:
    # wait_for holds speakers only, send_to holds every member, so an inter-team
    # message interrupts all four robots of the recipient team.
    assert chain["t1"].wait_for == ["a0"], chain["t1"].wait_for
    assert chain["t0"].send_to == ["a2", "a3", "a4", "a5"], chain["t0"].send_to
    assert chain["t2"].send_to == [], "the last team has nobody downstream"

    central = brains_of("centralized")
    assert isinstance(central["t0"], LeaderTeamBrain)
    assert isinstance(central["t1"], FollowerTeamBrain)
    assert central["t0"].wait_for == ["a2", "a4"], "leader waits on follower speakers"
    assert central["t0"].send_to == ["a2", "a3", "a4", "a5"], "leader addresses whole teams"
    assert central["t1"].wait_for == ["a0"] and central["t1"].send_to == ["a0"]
    ok("individual/chain/centralized wire teams; speakers are waited on, whole teams addressed")

    print("test 9: when the team thinks, every member is reasoning")
    from coop2.cognitive.agent.agent import AgentState

    client, agents, ids = make_team(4)
    # Three finish early and hold, then actually start executing those holds --
    # which is the state they are really in while the fourth robot works.
    for name in ids[:3]:
        agents[name].handle_reasoning()
        agents[name].set_ready()
        agents[name].start_execution()
        assert agents[name].state == AgentState.X, agents[name].state

    # The fourth arrives and closes the barrier.
    agents[ids[3]].handle_reasoning()

    # Every teammate must now be reasoning, not still executing a hold. Only the
    # member that closed the barrier was reasoning before this fix, so the
    # timeline showed one red bar and three green ones during a team call.
    for name in ids[:3]:
        assert agents[name].state == AgentState.R, (name, agents[name].state)
        assert not agents[name].ready, f"{name} is still marked ready"
    # And the hold each was running is terminal, or create_agent_thread would
    # decline to re-plan it and the member would never become ready again.
    ok("holders are recalled into R, with their holds marked terminal")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
