"""One LLM for a whole team of robots -- the unit every topology is built from.

A *team* has one brain: it sees every member's observation in a single prompt and
answers with one plan per robot, so the division of labour inside a team is made
once, with all of it visible, instead of emerging from N agents that cannot see
each other's intentions.

This is not a fourth topology. It is the **unit** the three existing ones are
expressed in: individual / broadcast_chain / centralized describe how teams talk
to *each other*, while inside every team it is always one LLM driving four
robots. Set the team size to 1 and each topology collapses to its old
one-LLM-per-robot behaviour, which is what makes this a generalisation rather
than a replacement.

A team speaks through one member, its **spokesagent** (the first), because the
message broker is keyed by agent id and the ordering primitives the chain and the
leader already use take agent ids. But a team is *addressed* as a whole: messages
go to every member of the recipient team, so the interrupt reaches all of them.

Three rules follow from that, and each one costs something:

**The team plans together.** Members return to reasoning as a group, only once
every one of them has finished its plan. That is the requirement, and it has a
trap in it: the plan loop does not step the environment while any agent is
not ready (``run_*.py``: ``while not all(agent.ready)``). So a member that
finished early and simply waited would freeze the world, and its teammates --
who need the world to advance to finish *their* plans -- would never finish. It
would deadlock on the first uneven round, every time.

So a member that finishes early stays **ready** and holds position instead: it
takes a short ``wait`` plan and keeps taking one until the team is complete. The
world keeps advancing, the early finisher costs real ticks doing nothing, and
that idle time is the true price of a joint decision point rather than something
hidden. :class:`TeamBrain` tracks who is done through those holds
(``_awaiting``), so a member idling for three holds is still "waiting to plan",
not "planning again".

**Messages interrupt the whole team.** A message addressed to any member
interrupts all of them, and the brain answers for each in one call: resume or
replan, per robot. Per robot rather than team-wide because a message that
changes what one robot should do usually leaves the others' plans perfectly
good, and making everyone replan would throw away work the message never
contradicted. There is no deadlock here -- an interrupt arrives at every member
at once, so the barrier closes immediately.

**One LLM call per round, not N.** The brain is shared, and the runner starts a
thread per agent, so several members reach it at the same instant. The first one
through the lock makes the call; the rest read its result. Without that, four
threads would each issue the same team-wide prompt and three answers would be
thrown away.

A team of one is the degenerate case and it behaves exactly like the individual
topology: the barrier is satisfied immediately, no holds are ever taken, and the
prompt contains one observation. That is why ``team_config`` gives an unteamed
robot a team of its own -- one-LLM-per-robot is this code with N=1, not a second
code path.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from coop2.cognitive.agent import LLMClient
from coop2.cognitive.agent.base_llm_agent import BaseLLMAgent
from coop2.cognitive.agent.cognitive_agent import parse_plan_response
from coop2.cognitive.agent.llm_client import InterruptDecision
from coop2.cognitive.agent.prompts import build_system_prompt
from coop2.cognitive.action.action import SymbolicAction
from coop2.cognitive.plan import SymbolicPlan

__all__ = [
    "ChainTeamBrain",
    "FollowerTeamBrain",
    "LLMTeamAgent",
    "LeaderTeamBrain",
    "TEAM_BRAIN_ROLES",
    "TeamBrain",
    "create_llm_team_topology",
]


TEAM_ROLE = """
## Your Role: TEAM CONTROLLER
You control {n} robots at once. You are given every robot's own observation and
you answer with one plan per robot, in the same call.

- Divide the work. Two robots sent to the same object waste one of them: the
  loser burns the whole trip and its grasp fails with OBJECT_CLAIMED.
- Each robot only sees the room it is standing in, and the ids it may use are
  the ones listed under that robot's own "You can do:". A target one robot can
  see is not necessarily reachable by another.
- The robots start their plans together and you are not asked again until every
  one of them has finished. A robot that finishes early holds position and does
  nothing useful, so plans of wildly different lengths waste the short ones.
"""

#: Ticks a member holds for while it waits for the rest of the team. Short
#: relative to a navigate (~100 ticks of settle plus 60/m) so the team regroups
#: soon after the last member lands, rather than overshooting by a long wait.
TEAM_HOLD_TICKS = 60

#: How long a member blocks in I waiting for its teammates to be interrupted
#: too. Generous because it should never be reached: the broker interrupts every
#: member of a team in the same call, so they arrive within milliseconds.
INTERRUPT_BARRIER_TIMEOUT = 30.0


class TeamBrain:
    """The shared LLM behind one team, and the barrier its members meet at.

    Holds no simulation state: members push their own observations in and pull
    their plan out. Every public method is safe to call from the per-agent
    threads the runners spawn.
    """

    def __init__(
        self,
        team_name: str,
        member_ids: List[str],
        llm_client: LLMClient,
        temperature: float = 0.7,
        verbose: bool = True,
        goal_instruction: str = "",
    ):
        self.team_name = team_name
        self.member_ids = list(member_ids)
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self.goal_instruction = goal_instruction

        self.members: Dict[str, "LLMTeamAgent"] = {}
        self._lock = threading.RLock()
        #: Members that have finished their plan and are waiting for the team.
        #: Set when a member enters reasoning, cleared when it takes its new
        #: plan -- so it stays set across the holds a member takes meanwhile.
        self._awaiting: set = set()
        #: Plans produced by the last team call, drained by their owners.
        self._pending_plans: Dict[str, SymbolicPlan] = {}
        self._pending_decisions: Dict[str, Any] = {}
        self._interrupted: set = set()
        #: Set once a round's decisions exist, so members blocked in I wake up.
        self._decided = threading.Event()
        self.rounds = 0
        self.holds = 0
        #: Topology wiring, in agent ids. ``wait_for`` holds the *spokesagents*
        #: of the teams this one waits on; ``send_to`` holds **every member** of
        #: the teams it addresses, so a message interrupts a whole team.
        self.wait_for: List[str] = []
        self.send_to: List[str] = []
        #: Every agent in the run, for the readiness check the ordering uses.
        self.all_agents: Dict[str, Any] = {}
        #: What this team heard since it last planned, folded into its prompt.
        self._heard: List[Dict] = []

    @property
    def speaker(self) -> Optional["LLMTeamAgent"]:
        """The member that speaks for the team."""
        for name in self.member_ids:
            if name in self.members:
                return self.members[name]
        return None

    # -- topology hooks ----------------------------------------------------

    def before_plan(self) -> None:
        """Communication that must happen before the team plans. Default: none."""

    def after_plan(self) -> None:
        """Communication that follows from the plan. Default: none."""

    def _collect_heard(self) -> None:
        """Drain every member's inbox into the team's shared record."""
        for name in self.member_ids:
            member = self.members.get(name)
            if member is None:
                continue
            for message in member.get_messages(clear_buffer=True) or []:
                self._heard.append(message)

    def _heard_block(self) -> str:
        if not self._heard:
            return ""
        lines = ["\nWHAT THE OTHER TEAMS SAID:"]
        for message in self._heard:
            lines.append(f"  From {message.get('sender', 'unknown')}: {message.get('content', '')}")
        return "\n".join(lines)

    def _say(self, content: str, kind: str, interrupts: bool) -> None:
        """Send @content to every member of the teams this one addresses."""
        speaker = self.speaker
        if not self.send_to or speaker is None or speaker.message_broker is None:
            return
        speaker.send_message(
            recipients=list(self.send_to),
            content=content,
            metadata={"type": kind, "team": self.team_name, "interrupts_execution": interrupts},
        )
        if self.verbose:
            print(f"  [{self.team_name}] -> {len(self.send_to)} agents ({kind}): {content[:60]}...")

    def _await_speakers(self, timeout: float = 30.0) -> None:
        """Block until the teams this one waits on have spoken, or committed."""
        speaker = self.speaker
        if speaker is None or not self.wait_for:
            return
        if not speaker._any_waiting_agent_not_ready(self.all_agents, self.wait_for):
            return
        deadline = time.monotonic() + timeout
        while not speaker.wait_for_messages_from(self.wait_for):
            if not speaker._any_waiting_agent_not_ready(self.all_agents, self.wait_for):
                break
            if time.monotonic() >= deadline:
                print(f"  [{self.team_name}] waited {timeout:.0f}s for {self.wait_for}; releasing")
                break
            time.sleep(0.05)

    def _plan_summary(self, plans: Dict[str, SymbolicPlan]) -> str:
        """One line per robot, for telling another team what this one will do."""
        parts = [f"{name}: {plan.specification}" for name, plan in sorted(plans.items())]
        return f"[{self.team_name}] " + "; ".join(parts)

    # -- registration ------------------------------------------------------

    def register(self, agent: "LLMTeamAgent") -> None:
        self.members[agent.agent_id] = agent

    @property
    def size(self) -> int:
        return len(self.member_ids)

    # -- planning ----------------------------------------------------------

    def request_plan(self, agent_id: str) -> Optional[SymbolicPlan]:
        """The team's plan for @agent_id, or None if the team is not ready yet.

        None means "hold and ask again": teammates are still executing, and the
        caller must stay ready so the world keeps advancing for them.
        """
        with self._lock:
            if agent_id in self._pending_plans:
                self._awaiting.discard(agent_id)
                return self._pending_plans.pop(agent_id)

            self._awaiting.add(agent_id)
            if not self._awaiting.issuperset(self.member_ids):
                return None

            # Every member is done. Before anything else, pull the teammates
            # that are still holding out of their holds and into R, so the whole
            # team is *reasoning* while the one call happens -- and so that
            # anything this team says to another team is said by a team that has
            # already stopped acting. Without this only the member that closed
            # the barrier showed as reasoning and the rest kept executing.
            self._recall_holders(except_id=agent_id)

            # One call for the whole team; whichever thread got here first does
            # it and the others take from the result. The topology's
            # communication brackets that call: whoever this team waits on has
            # to have spoken before it plans, and whatever it tells other teams
            # follows from the plan it just made.
            self.before_plan()
            self._collect_heard()
            self._pending_plans = self._generate_team_plans()
            self.rounds += 1
            self.after_plan()
            self._heard = []
            self._awaiting.discard(agent_id)
            return self._pending_plans.pop(agent_id, None)


    def _recall_holders(self, except_id: str) -> None:
        """Bring every holding teammate into R for the duration of the call.

        A member waiting for the team is *executing* a hold, which is what keeps
        the world moving while its teammates finish. Once the barrier closes
        nobody needs the world any more, so the holds are ended here and the
        whole team reasons together.

        Ending a hold takes two steps, and one alone is not enough.
        ``set_unready`` puts the member in R, but ``create_agent_thread`` only
        calls ``handle_reasoning`` when ``needs_new_plan()`` is also true -- and
        that asks the *plan* whether it is finished. A member left in R with a
        live hold plan would answer no, return early, never become ready, and
        hang the run. So the hold is marked terminal first.

        The stale ``wait`` primitive it leaves in the engine is not a leak: the
        wrapper aborts it when the replacement plan is committed
        (``_reset_symbolic_action_state``), which is the same path an
        interrupt-and-replan already takes.
        """
        from coop2.cognitive.plan.plan import SymbolicPlanStatus  # noqa: PLC0415

        for name in self.member_ids:
            if name == except_id:
                continue
            member = self.members.get(name)
            if member is None:
                continue
            plan = member.plan
            if plan is not None and str(plan.specification).startswith("wait_for_team"):
                plan.status = SymbolicPlanStatus.INTERRUPTED
            # 'plan_terminated' is exactly what happened -- the hold above was
            # just ended -- and it is the reason that transitions X -> R, which
            # is the state the member should be in while the team thinks. A
            # member already in W or R is left alone: it is not holding.
            from coop2.cognitive.agent.agent import AgentState  # noqa: PLC0415

            if member.state == AgentState.X:
                member.set_unready(reason="plan_terminated")

    def note_hold(self) -> None:
        with self._lock:
            self.holds += 1

    def _generate_team_plans(self) -> Dict[str, SymbolicPlan]:
        """One LLM call, one plan per member. Never raises."""
        members = [self.members[name] for name in self.member_ids if name in self.members]
        if not members:
            return {}
        anchor = members[0]
        prompt = self._build_team_prompt(members)

        if anchor._should_print_llm_io():
            anchor._print_llm_messages(f"Team Plan Generation [{self.team_name}]", prompt)
        if self.verbose:
            print(f"  [{self.team_name}] Calling LLM for {len(members)} plans...")

        try:
            response, usage = self.llm_client.generate_team_plan(
                messages=prompt, temperature=self.temperature
            )
            anchor._record_llm_usage(
                usage, f"Team Plan Generation [{self.team_name}]", prompt, response
            )
        except Exception as error:  # noqa: BLE001 - a dead LLM must not end the run
            anchor._record_llm_error(error)
            print(f"  [{self.team_name}] LLM error: {error}")
            return {name: self.members[name]._generate_fallback_plan() for name in self.member_ids
                    if name in self.members}

        by_agent = {plan.agent_id: plan for plan in response.plans}
        plans: Dict[str, SymbolicPlan] = {}
        for member in members:
            entry = by_agent.get(member.agent_id)
            if entry is None:
                # The model skipped a robot. Hold it rather than leaving it with
                # no plan at all, which would strand the whole team at the next
                # barrier: a member with no plan never becomes ready.
                print(f"  [{self.team_name}] no plan returned for {member.agent_id}; holding it")
                plans[member.agent_id] = member.build_hold_plan()
                continue
            single = _as_plan_response(entry)
            plan = parse_plan_response(
                llm_response=single,
                agent_id=member.agent_id,
                env_step=member.env_step,
                plan_id=member.plan_count + 1,
            )
            plans[member.agent_id] = member._finalize_generated_plan(plan)
            if self.verbose:
                print(f"  [{member.agent_id}] Plan: {plan.specification}")
        if self.verbose and getattr(response, "reasoning", ""):
            print(f"  [{self.team_name}] allocation: {response.reasoning}")
        return plans

    # -- interrupts --------------------------------------------------------

    def request_interrupt_decision(self, agent_id: str, messages: List[Dict]) -> Optional[Any]:
        """Resume/replan for @agent_id, or None while teammates have not arrived.

        Unlike planning there is no hold: a message interrupts every member at
        once, so the barrier closes on the same tick and nothing needs the world
        to advance in the meantime.
        """
        with self._lock:
            if agent_id in self._pending_decisions:
                self._interrupted.discard(agent_id)
                return self._pending_decisions.pop(agent_id)

            if not self._interrupted:
                # First arrival of a new round: nobody has decided yet.
                self._decided.clear()
            self._interrupted.add(agent_id)
            complete = self._interrupted.issuperset(self.member_ids)
            if complete:
                self._pending_decisions = self._decide_interrupts(messages)
                self._decided.set()
                self._interrupted.discard(agent_id)
                return self._pending_decisions.pop(agent_id, None)

        # Not everyone has arrived. **Block here**, staying in I, rather than
        # returning and being marked ready again: a member that bounced straight
        # back to W showed as "waiting" on the timeline while its teammates were
        # still interrupted, and the team is supposed to decide together. This
        # cannot deadlock the way the planning barrier could -- the message
        # interrupted every member at once, so they are all on their way here,
        # and nothing needs the world to advance in the meantime.
        if self._decided.wait(timeout=INTERRUPT_BARRIER_TIMEOUT):
            with self._lock:
                self._interrupted.discard(agent_id)
                return self._pending_decisions.pop(agent_id, None)
        print(f"  [{self.team_name}] waited {INTERRUPT_BARRIER_TIMEOUT:.0f}s for the team "
              f"to be interrupted and it never completed; resuming")
        with self._lock:
            self._interrupted.discard(agent_id)
        return None

    def _decide_interrupts(self, messages: List[Dict]) -> Dict[str, Any]:
        members = [self.members[name] for name in self.member_ids if name in self.members]
        if not members:
            return {}
        anchor = members[0]
        prompt = self._build_interrupt_prompt(members, messages)

        if anchor._should_print_llm_io():
            anchor._print_llm_messages(f"Team Interrupt [{self.team_name}]", prompt)
        try:
            response, usage = self.llm_client.generate_team_interrupt_decision(
                messages=prompt, temperature=self.temperature
            )
            anchor._record_llm_usage(
                usage, f"Team Interrupt [{self.team_name}]", prompt, response
            )
        except Exception as error:  # noqa: BLE001
            anchor._record_llm_error(error)
            print(f"  [{self.team_name}] interrupt LLM error: {error}; everyone resumes")
            return {name: (InterruptDecision.RESUME, None) for name in self.member_ids}

        decisions: Dict[str, Any] = {}
        by_agent = {d.agent_id: d for d in response.decisions}
        for member in members:
            entry = by_agent.get(member.agent_id)
            if entry is None or entry.decision == InterruptDecision.RESUME:
                decisions[member.agent_id] = (InterruptDecision.RESUME, None)
                continue
            plan = None
            if entry.new_plan is not None:
                plan = parse_plan_response(
                    llm_response=entry.new_plan,
                    agent_id=member.agent_id,
                    env_step=member.env_step,
                    plan_id=member.plan_count + 1,
                )
                plan = member._finalize_generated_plan(plan)
            decisions[member.agent_id] = (InterruptDecision.REPLAN, plan)
        if self.verbose:
            summary = ", ".join(
                f"{name}={decisions[name][0].value}" for name in decisions
            )
            print(f"  [{self.team_name}] interrupt: {summary}")
        return decisions

    # -- prompts -----------------------------------------------------------

    def _system_prompt(self, anchor: "LLMTeamAgent") -> str:
        base = build_system_prompt(self.team_name, max_actions=6, include_env_description=True)
        return base + "\n\n" + TEAM_ROLE.format(n=self.size).strip()

    def _member_block(self, member: "LLMTeamAgent") -> str:
        """One robot's section of the team prompt.

        The per-agent observation is reused verbatim rather than re-rendered:
        it is the same text a single-agent topology would send, so a team prompt
        differs from N individual prompts only by being concatenated.
        """
        lines = [f"=== ROBOT {member.agent_id} ==="]
        if member.symbolic_view:
            lines.append(member.symbolic_view)
        elif member.target_hints:
            lines.append(member.target_hints)
        else:
            lines.append("(no observation yet)")
        plan = member.plan
        if plan is not None:
            lines.append(f"Its last plan: {plan.specification} [{plan.status.value}]")
        return "\n".join(lines)

    def _build_team_prompt(self, members: List["LLMTeamAgent"]) -> List[Dict[str, str]]:
        anchor = members[0]
        parts = [f"=== STEP {anchor.env_step} ==="]
        if self.goal_instruction:
            parts.append(f"\nGLOBAL OBJECTIVE: {self.goal_instruction}")
        parts.append(f"\nTEAM {self.team_name} ({len(members)} robots): "
                     f"{', '.join(m.agent_id for m in members)}")
        for member in members:
            parts.append("\n" + self._member_block(member))
        heard = self._heard_block()
        if heard:
            parts.append(heard)
        parts.append(
            f"\nReturn exactly {len(members)} plans, one per robot, using each "
            "robot's own ids. Say in `reasoning` how you divided the work."
        )
        return [
            {"role": "system", "content": self._system_prompt(anchor)},
            {"role": "user", "content": "\n".join(parts)},
        ]

    def _build_interrupt_prompt(
        self, members: List["LLMTeamAgent"], messages: List[Dict]
    ) -> List[Dict[str, str]]:
        anchor = members[0]
        parts = [f"=== STEP {anchor.env_step} ===", "\nThe team was interrupted by a message."]
        if self.goal_instruction:
            parts.append(f"\nGLOBAL OBJECTIVE: {self.goal_instruction}")
        parts.append("\nMESSAGES:")
        for message in messages or []:
            sender = message.get("sender", "unknown")
            parts.append(f"  From {sender}: {message.get('content', '')}")
        for member in members:
            parts.append("\n" + self._member_block(member))
        parts.append(
            "\nFor each robot decide 'resume' or 'replan'. Resume unless the "
            "message actually contradicts what that robot is doing -- replanning "
            "a robot the message did not concern throws away work it has already "
            "paid for. A 'replan' decision must carry its new_plan."
        )
        return [
            {"role": "system", "content": self._system_prompt(anchor)},
            {"role": "user", "content": "\n".join(parts)},
        ]



class ChainTeamBrain(TeamBrain):
    """Teams speak in order; each broadcasts what it committed to the later ones.

    The ordering mechanism is the predecessor becoming *ready*: whatever it was
    going to say it has said, so nothing more is coming and the world must not be
    held for it. Same rule the per-robot chain uses, lifted a level -- what is
    ordered now is teams, and each message carries a whole team's allocation
    rather than one robot's intention.
    """

    def before_plan(self) -> None:
        self._await_speakers()

    def after_plan(self) -> None:
        if self._pending_plans:
            self._say(self._plan_summary(self._pending_plans), "broadcast_chain", interrupts=True)


class LeaderTeamBrain(TeamBrain):
    """Asks every follower team what it is doing, then plans for its own robots.

    The request interrupts the follower teams, which is the point: a follower
    that is mid-plan has to answer with where it actually is, not with what it
    intended several hundred ticks ago.
    """

    def before_plan(self) -> None:
        self._say(
            f"[{self.team_name}] Leader planning request: report each robot's position, "
            "what it holds, one useful target it can reach, and what it proposes to do next. "
            "Keep it short.",
            "leader_broadcast",
            interrupts=True,
        )
        self._await_speakers()


class FollowerTeamBrain(TeamBrain):
    """Waits for the leader's request, answers for its robots, then plans.

    The answer is assembled from the team's own state rather than generated:
    it is a status report, and spending an LLM call to paraphrase facts the
    brain already has would double this topology's cost for nothing.
    """

    def before_plan(self) -> None:
        self._await_speakers()
        self._say(self._status_report(), "follower_response", interrupts=False)

    def _status_report(self) -> str:
        parts = []
        for name in self.member_ids:
            member = self.members.get(name)
            if member is None:
                continue
            plan = member.plan
            spec = plan.specification if plan is not None else "no plan"
            parts.append(f"{name} was doing {spec}")
        return f"[{self.team_name}] " + "; ".join(parts)


#: Which brain each topology uses for which team. `individual` is the plain
#: TeamBrain: teams never address each other, so the only coordination in the
#: run is the one that happens *inside* each team.
TEAM_BRAIN_ROLES = {
    "individual": "no team talks to any other",
    "broadcast_chain": "teams speak in order, each broadcasting to the later ones",
    "centralized": "the first team leads; the rest report to it",
}


def _as_plan_response(entry: Any):
    """A ``TeamAgentPlan`` viewed as the single-agent response parse expects."""
    from coop2.cognitive.agent.llm_client import LLMPlanResponse  # noqa: PLC0415

    return LLMPlanResponse(task=entry.task, actions=entry.actions, reasoning=entry.reasoning)


class LLMTeamAgent(BaseLLMAgent):
    """One robot, whose thinking is done by the team's shared brain."""

    def __init__(
        self,
        agent_id: str,
        brain: TeamBrain,
        temperature: float = 0.7,
        verbose: bool = True,
        goal_instruction: str = "",
    ):
        super().__init__(agent_id, brain.llm_client, temperature=temperature, verbose=verbose)
        self.brain = brain
        self.wait_for = []
        self.send_to = []
        self.goal_instruction = goal_instruction
        self.team_agent_ids: List[str] = list(brain.member_ids)
        brain.register(self)

    # -- the two FSM hooks -------------------------------------------------

    def handle_reasoning(self):
        """Take the team's plan, or hold position until the team is complete."""
        self.get_messages(clear_buffer=True)
        plan = self.brain.request_plan(self.agent_id)
        if plan is None:
            # Teammates are still executing. Staying not-ready here would freeze
            # the world and they would never finish -- see the module docstring.
            self.brain.note_hold()
            if self.verbose:
                print(f"  [{self.agent_id}] holding {TEAM_HOLD_TICKS} ticks for the team")
            plan = self.build_hold_plan()
        self.plan = plan
        return self.plan

    def handle_interrupt(self):
        """Resume or replan, as the brain decided for this robot."""
        messages = self.get_messages(clear_buffer=True)
        decision = self.brain.request_interrupt_decision(self.agent_id, messages)
        if decision is None:
            # Teammates have not reached the barrier yet. Resume for now; the
            # message stays handled because whoever closes the barrier answers
            # for the whole team.
            return
        choice, new_plan = decision
        if choice == InterruptDecision.REPLAN and new_plan is not None:
            self.plan = new_plan

    # -- helpers -----------------------------------------------------------

    def build_hold_plan(self) -> SymbolicPlan:
        """A one-action ``wait`` plan, so idling costs ticks like anything else.

        Built directly rather than through ``parse_plan_response``. That helper
        ends with ``_ensure_task_terminal_action``, which appends an action to
        match the plan's TaskSpecification -- correct for a real goal, wrong
        here: a hold has no goal, and expressing it as ``holding(<self>)`` made
        the helper append ``grasp(<self>)``, i.e. a robot planning to pick
        *itself* up. Seen in a real run before this was fixed.
        """
        return SymbolicPlan(
            specification=f"wait_for_team({self.brain.team_name})",
            actions=[SymbolicAction(action_type="wait", args={"ticks": TEAM_HOLD_TICKS})],
            plan_id=self.plan_count + 1,
            agent_id=self.agent_id,
            created_at_step=self.env_step,
        )


def create_llm_team_topology(
    llm_client: LLMClient,
    teams: Dict[str, List[str]],
    topology: str = "individual",
    temperature: float = 0.7,
    verbose: bool = True,
    goal_instruction: str = "",
) -> Dict[str, LLMTeamAgent]:
    """One brain per team, one agent per robot, wired for @topology.

    The topology decides how teams address **each other**; inside every team it
    is always one LLM for all its robots. With teams of one this reproduces the
    old per-robot behaviour of the same topology, which is why there is no
    separate code path for it.

    Args:
        teams: ``{team name: [agent ids]}``, straight from the layout. Order
            matters: it is the chain's speaking order, and the first team leads
            under `centralized`.
    """
    if topology not in TEAM_BRAIN_ROLES:
        raise ValueError(f"unknown topology {topology!r}; have {sorted(TEAM_BRAIN_ROLES)}")

    names = list(teams)
    brains: Dict[str, TeamBrain] = {}
    for index, team_name in enumerate(names):
        if topology == "broadcast_chain":
            factory = ChainTeamBrain
        elif topology == "centralized":
            factory = LeaderTeamBrain if index == 0 else FollowerTeamBrain
        else:
            factory = TeamBrain
        brains[team_name] = factory(
            team_name=team_name,
            member_ids=list(teams[team_name]),
            llm_client=llm_client,
            temperature=temperature,
            verbose=verbose,
            goal_instruction=goal_instruction,
        )

    agents: Dict[str, LLMTeamAgent] = {}
    for team_name in names:
        for agent_id in teams[team_name]:
            agents[agent_id] = LLMTeamAgent(
                agent_id,
                brains[team_name],
                temperature=temperature,
                verbose=verbose,
                goal_instruction=goal_instruction,
            )

    # Wiring, in agent ids: a team is *addressed* as a whole (every member, so
    # the interrupt reaches all of it) but *speaks* through its first member,
    # because the broker and the ordering primitives are keyed by agent id.
    def speaker_of(team_name: str) -> str:
        return teams[team_name][0]

    def members_of(team_names) -> List[str]:
        return [agent_id for name in team_names for agent_id in teams[name]]

    if topology == "broadcast_chain":
        for index, team_name in enumerate(names):
            brain = brains[team_name]
            brain.wait_for = [speaker_of(names[index - 1])] if index > 0 else []
            brain.send_to = members_of(names[index + 1:])
    elif topology == "centralized":
        leader, followers = names[0], names[1:]
        brains[leader].wait_for = [speaker_of(name) for name in followers]
        brains[leader].send_to = members_of(followers)
        for name in followers:
            brains[name].wait_for = [speaker_of(leader)]
            brains[name].send_to = [speaker_of(leader)]

    for brain in brains.values():
        brain.all_agents = agents
    for agent in agents.values():
        agent.wait_for = list(agent.brain.wait_for)
        agent.send_to = list(agent.brain.send_to)
    return agents
