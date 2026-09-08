"""CPU-only regression test for L2: symbolic action -> engine primitive.

Stubs OmniGibson and the engine. It pins the contract L3 depends on
(pending/success/failed/terminate_plan), the entity-id grounding, and the one
structural difference from crafter: ``execute()`` assigns a primitive and
returns immediately instead of producing a per-tick action string.

Run:
    python feasibility_verify/test_behavior_action_stubbed.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from enum import Enum


class FakePrimitiveSet:
    class _Member:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

    NAVIGATE_TO = _Member("NAVIGATE_TO")
    GRASP = _Member("GRASP")
    PLACE_ON_TOP = _Member("PLACE_ON_TOP")
    PLACE_INSIDE = _Member("PLACE_INSIDE")
    RELEASE = _Member("RELEASE")
    OPEN = _Member("OPEN")
    CLOSE = _Member("CLOSE")
    TOGGLE_ON = _Member("TOGGLE_ON")
    TOGGLE_OFF = _Member("TOGGLE_OFF")


class FakeOutcome:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


class FakeEngine:
    """Records assignments; never advances anything -- like the real one."""

    def __init__(self, reject=None):
        self.assigned = []
        self.active = set()
        self.reject = reject

    def assign(self, agent_id, primitive, target=None):
        self.assigned.append((agent_id, primitive.name, target))
        if self.reject is not None:
            return FakeOutcome(self.reject)
        self.active.add(agent_id)
        return None

    def has_active(self, agent_id):
        return agent_id in self.active

    def finish(self, agent_id):
        self.active.discard(agent_id)


class FakeWorldState:
    def __init__(self, ids):
        self._ids = dict(ids)  # scene name -> entity id


def _install_stubs():
    def stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    stub("torch", Tensor=object)
    stub("omnigibson")
    stub("omnigibson.action_primitives")
    stub(
        "omnigibson.action_primitives.symbolic_semantic_action_primitives",
        SymbolicSemanticActionPrimitiveSet=FakePrimitiveSet,
    )
    stub(
        "omnigibson.action_primitives.action_primitive_set_base",
        ActionPrimitiveError=type("ActionPrimitiveError", (Exception,), {}),
        ActionPrimitiveErrorGroup=type("ActionPrimitiveErrorGroup", (Exception,), {}),
    )
    stub(
        "omnigibson.action_primitives.starter_semantic_action_primitives",
        StarterSemanticActionPrimitives=object,
        StarterSemanticActionPrimitiveSet=FakePrimitiveSet,
    )
    stub("omnigibson.robots", Robot=object)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    """Register the coop2 packages with real __path__ so relative imports work.

    The modules under test are pure Python; only OmniGibson needs stubbing.
    Giving the packages their real paths lets ``action.py``'s ``from ..constants``
    resolve normally instead of needing a stub per transitive import.
    """
    for name, relative in (
        ("coop2", "coop2"),
        ("coop2.behavior_env", "coop2/behavior_env"),
        ("coop2.cognitive", "coop2/cognitive"),
        ("coop2.cognitive.action", "coop2/cognitive/action"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [os.path.join(ROOT, relative)]
        sys.modules[name] = package
    _load("coop2.behavior_env.primitive_engine", "coop2/behavior_env/primitive_engine.py")
    _load("coop2.cognitive.action.action", "coop2/cognitive/action/action.py")
    return _load("coop2.cognitive.action.behavior_action", "coop2/cognitive/action/behavior_action.py")


def main() -> int:
    _install_stubs()
    module = _load_modules()
    Executor = module.BehaviorActionExecutor
    status_enum = sys.modules["coop2.cognitive.action.action"].SymbolicActionStatus
    reason_codes = sys.modules["coop2.behavior_env.primitive_engine"].ReasonCode

    def ok(message):
        print(f"  PASS {message}")

    world = FakeWorldState({"apple_agveuv_0": "apple#1", "table_xyz_3": "table#1"})

    print("test 1: the whole symbolic vocabulary maps onto primitives")
    missing = set(module.BEHAVIOR_ACTION_SCHEMA) - set(module.BEHAVIOR_ACTION_TO_PRIMITIVE) - set(
        module.COMMUNICATION_ACTIONS
    )
    assert not missing, missing
    assert set(module.BEHAVIOR_ACTION_TO_PRIMITIVE) == {
        "navigate_to", "grasp", "place_on_top", "place_inside", "release",
        "open", "close", "toggle_on", "toggle_off",
    }
    ok("9 primitives + wait/share, and every schema entry is reachable")

    print("test 2: execute() assigns and returns without advancing anything")
    engine = FakeEngine()
    executor = Executor("agent_0", engine=engine, world_state=world)
    returned = executor.execute("grasp", target="apple#1", current_step=7)
    assert returned == "GRASP", returned
    assert engine.assigned == [("agent_0", "GRASP", "apple_agveuv_0")], engine.assigned
    ok("grasp(apple#1) -> engine.assign(agent_0, GRASP, apple_agveuv_0)")

    print("test 3: entity ids are grounded to scene names")
    # This is the whole reason L2 exists: the LLM never sees apple_agveuv_0 and
    # the engine never sees apple#1.
    assert executor.resolve_target("table#1") == "table_xyz_3"
    assert executor.resolve_target("apple_agveuv_0") == "apple_agveuv_0", "scene names pass through"
    assert executor.resolve_target(None) is None
    ok("apple#1 -> apple_agveuv_0; unknown/None pass through unchanged")

    print("test 4: an in-flight primitive reports pending")
    assert executor.check_termination_condition() == {"status": "pending"}
    ok("engine.has_active -> pending, no outcome invented")

    print("test 5: a terminal success completes the record")
    engine.finish("agent_0")
    result = executor.check_termination_condition(
        action_outcome={"status": "success", "reason_code": "OK", "reason": ""}
    )
    assert result["status"] == "success", result
    assert executor.current_symbolic_action.status == status_enum.SUCCESS
    assert executor.current_symbolic_action.end_step == 7
    ok("status success, ActionRecord completed at the right step")

    print("test 6: failure carries the reason through to the plan layer")
    engine = FakeEngine()
    executor = Executor("agent_0", engine=engine, world_state=world)
    executor.execute("grasp", target="apple#1")
    engine.finish("agent_0")
    result = executor.check_termination_condition(
        action_outcome={
            "status": "failed",
            "reason_code": "OBJECT_CLAIMED",
            "reason": "apple_0 is currently held by agent_1",
        }
    )
    assert result["status"] == "failed"
    assert "held by agent_1" in result["failure_reason"]
    # Contention terminates the plan and returns the agent to reasoning. Both
    # codes are individually recoverable, but the plan that produced them was
    # written against a world that has since contradicted it -- and reasoning is
    # the only stage where the agent can negotiate or retarget.
    assert result["terminate_plan"] is True, "contention must send the agent back to reasoning"
    ok("OBJECT_CLAIMED -> failed, reason preserved, plan terminated")

    print("test 7: every terminating code sends the agent back to reasoning")
    for code in ("PLANNING", "SAMPLING", "TIMEOUT", "INVALID_TARGET", "CRASHED",
                 "OBJECT_CLAIMED", "TOO_FAR"):
        engine = FakeEngine()
        executor = Executor("agent_0", engine=engine, world_state=world)
        executor.execute("navigate_to", target="apple#1")
        engine.finish("agent_0")
        result = executor.check_termination_condition(
            action_outcome={"status": "failed", "reason_code": code, "reason": code}
        )
        assert result["terminate_plan"] is True, code
    for code in ("OBJECT_CLAIMED", "TOO_FAR"):
        assert code in reason_codes.TERMINATES_PLAN, code
    # These stay non-terminating: they are failures of one action under a plan
    # the world has not contradicted, so the next action is still meaningful.
    for code in ("PRE_CONDITION", "POST_CONDITION", "EXECUTION"):
        assert code not in reason_codes.TERMINATES_PLAN, code
    ok("7 codes terminate incl. both contention codes; PRE/POST_CONDITION and EXECUTION do not")

    print("test 8: assign() rejecting up front surfaces as a failure, not a hang")
    engine = FakeEngine(reject={"status": "failed", "reason_code": "INVALID_TARGET", "reason": "no such object"})
    executor = Executor("agent_0", engine=engine, world_state=world)
    executor.execute("grasp", target="ghost#9")
    result = executor.check_termination_condition()
    assert result["status"] == "failed" and result["terminate_plan"] is True, result
    ok("unknown target fails immediately with terminate_plan")

    print("test 9: wait needs no primitive; share is not an action at all")
    engine = FakeEngine()
    executor = Executor("agent_0", engine=engine, world_state=world)
    assert executor.execute("wait") == "wait"
    assert engine.assigned == [], "communication must not touch the engine"
    assert executor.check_termination_condition()["status"] == "success"

    # share was a text message that completed instantly and never failed, so a
    # plan made of shares scored as a success while touching nothing: one
    # broadcast run issued 395 of them against 30 navigate_to. Agent-to-agent
    # text goes through the MessageBroker, which the topologies drive; it is
    # not a plan action, and asking for it must fail like any unknown verb.
    engine = FakeEngine()
    executor = Executor("agent_0", engine=engine, world_state=world)
    assert executor.execute("share", target_agent="agent_1", message="hi") == "invalid"
    assert engine.assigned == []
    result = executor.check_termination_condition()
    assert result["status"] == "failed", result
    ok("wait completes without a primitive; share is rejected as an unknown verb")

    print("test 10: an unknown action fails loudly instead of silently no-oping")
    # crafter's executor returned "noop" for anything unknown, which turns an
    # LLM hallucinating a verb into an invisible wasted step.
    engine = FakeEngine()
    executor = Executor("agent_0", engine=engine, world_state=world)
    assert executor.execute("teleport", target="apple#1") == "invalid"
    result = executor.check_termination_condition()
    assert result["status"] == "failed"
    assert "Valid actions" in result["failure_reason"], result["failure_reason"]
    ok("unknown verb -> failed with the legal vocabulary in the message")

    print("test: production PlanningEnvWrapper builds the BEHAVIOR wrapper")
    # This was wrong for the whole of M7 and no test caught it, because the
    # only place the swap happened was a monkeypatch inside the GPU harness.
    # Crafter's base class turns symbolic actions into integer action ids; the
    # facade drops them (0 is falsy) and every plan sits in "executing" until
    # the episode ends -- no exception, no failed action, just a wasted run.
    from coop2.cognitive.plan import plan_env_wrapper
    from coop2.cognitive.action.behavior_env_wrapper import BehaviorSymbolicEnvWrapper

    assert plan_env_wrapper.SymbolicEnvWrapper is BehaviorSymbolicEnvWrapper, (
        f"PlanningEnvWrapper would build {plan_env_wrapper.SymbolicEnvWrapper.__name__}"
    )
    ok("PlanningEnvWrapper is wired to BehaviorSymbolicEnvWrapper")

    print("test 13: only a BDDL activity can end an episode early")
    # `terminated` used to be hardcoded False for every agent, and a stand-in
    # check ("every agent holds an apple") was added while BDDL was not wired
    # up. That stand-in is gone: with BDDL deciding, a second authority can
    # only disagree, and a run without an activity should be visibly unbounded
    # rather than quietly ending on a proxy.
    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv

    class Compiled:
        def __init__(self, met):
            self.met = met
            self.calls = 0
        def check_goal(self, _evaluate):
            self.calls += 1
            return self.met, {"satisfied": [0] if self.met else [], "unsatisfied": []}

    class Task:
        def __init__(self, met):
            self.compiled_task = Compiled(met)
        def _evaluate_predicate(self, *_):
            return True

    def make(activity, met):
        env = object.__new__(CooperativeBehaviorEnv)
        env.bddl_activity = activity
        env.goal_reached_at = None
        env.env = type("E", (), {"task": Task(met)})()
        env.engine = type("Eng", (), {"env_step": 11})()
        return env

    # No activity: nothing can end the episode, whatever the world looks like.
    no_activity = make(None, met=True)
    assert no_activity._goal_reached() is False, "without an activity nothing terminates"
    assert no_activity.goal_reached_at is None

    unmet = make("coop_two_apples_pomaria", met=False)
    assert unmet._goal_reached() is False
    assert unmet.goal_reached_at is None
    assert unmet.env.task.compiled_task.calls == 1, "check_goal must actually be consulted"

    met = make("coop_two_apples_pomaria", met=True)
    assert met._goal_reached() is True
    assert met.goal_reached_at == 11, "the step it happened on is recorded"
    # Sticky, and without re-consulting the task.
    before = met.env.task.compiled_task.calls
    assert met._goal_reached() is True
    assert met.env.task.compiled_task.calls == before
    ok("check_goal is the only authority; no activity means no early termination")

    print("test: a zero-padded instance suffix still resolves")
    # Models pad the BDDL instance index: three runs produced apple.n.01_01 and
    # coffee_table.n.01_01 for _1. One such typo cost a whole plan -- agent_0
    # grasped its apple, died on navigate_to(coffee_table.n.01_01), and spent
    # the remaining 1400 steps recovering. _01 and _1 cannot name different
    # instances, so rejecting it buys nothing.
    from coop2.cognitive.action.behavior_action import _normalise_instance_id

    assert _normalise_instance_id("apple.n.01_01") == "apple.n.01_1"
    assert _normalise_instance_id("apple.n.01_1") == "apple.n.01_1"
    assert _normalise_instance_id("apple.n.01_012") == "apple.n.01_12"
    # The synset's own digits must survive: only the trailing index is touched.
    assert _normalise_instance_id("apple.n.01_01").startswith("apple.n.01_")
    assert _normalise_instance_id("no_digits_here") is None
    assert _normalise_instance_id("nounderscore") is None
    assert _normalise_instance_id(None) is None

    # The shared fixture above still uses the pre-BDDL "apple#1" ids; production
    # ids are BDDL instance names, which is where the padding happens.
    bddl_world = FakeWorldState({
        "apple_omzprq_0": "apple.n.01_1",
        "coffee_table_gpkbiw_0": "coffee_table.n.01_1",
    })
    executor = Executor("agent_0", engine=FakeEngine(), world_state=bddl_world)
    assert executor.resolve_target("apple.n.01_1") == "apple_omzprq_0"
    assert executor.resolve_target("apple.n.01_01") == "apple_omzprq_0", "padded must resolve too"
    assert executor.resolve_target("coffee_table.n.01_01") == "coffee_table_gpkbiw_0", \
        "the id that actually cost a plan in the recorded run"
    assert executor.resolve_target("apple.n.01_9") == "apple.n.01_9", \
        "an id that matches nothing must pass through unchanged"
    ok("apple.n.01_01 resolves to the same object as apple.n.01_1")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
