"""CPU-only regression test for L1d: cooperative task tracking.

The metric containers are carried over from ma_crafter verbatim, so what needs
testing is (a) that the copy is intact -- a dataclass that lost its decorator
still imports fine and then silently accepts no arguments -- and (b) that the
BEHAVIOR tracker fills the exact paths ``compute_metrics.py`` reads by name.

Run:
    python feasibility_verify/test_cooperative_tasks_stubbed.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types


class FakeEntity:
    def __init__(self, entity_id, name, position, is_robot=False, states=None):
        self.entity_id = entity_id
        self.name = name
        self.position = position
        self.is_robot = is_robot
        self.states = states or {}
        self.abilities = []
        self.is_fixed = False
        self.held_by = None
        self.rooms = ["kitchen_0"]

    @property
    def room(self):
        return self.rooms[0] if self.rooms else None


class FakeFact:
    def __init__(self, predicate, args, value=True):
        self.predicate = predicate
        self.args = args
        self.value = value


class FakeWorld:
    """Stands in for BehaviorWorldState: entities, facts, token mapping."""

    def __init__(self, entities, facts):
        self._entities = {e.entity_id: e for e in entities}
        self._facts = list(facts)

    def entities(self):
        return dict(self._entities)

    def facts(self, entities=None):
        return list(self._facts)

    def predicate_token(self, name):
        return {"Open": "open", "ToggledOn": "toggled_on"}.get(str(name), str(name))

    def relation_holds(self, token, source_id, target_id):
        """One binary relation, evaluated on demand.

        Production evaluates the predicate it was asked about rather than
        enumerating every true relation in the scene; the stub mirrors that
        contract so the test exercises the path production takes.
        """
        for fact in self._facts:
            if (fact.predicate, *fact.args) == (token, source_id, target_id):
                return bool(fact.value)
        return False


def _load():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # matplotlib is imported lazily by plot_metrics_timeline only.
    spec = importlib.util.spec_from_file_location(
        "coop2_cooperative_tasks", os.path.join(root, "coop2/behavior_env/cooperative_tasks.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def main() -> int:
    m = _load()

    def ok(message):
        print(f"  PASS {message}")

    print("test 1: the copied dataclasses are really dataclasses")
    # A @dataclass that lost its decorator in transit still imports and then
    # raises "takes no arguments" the first time it is constructed. This caught
    # exactly that for CapabilityChange / StepMetrics / StepTaskSummary.
    for name in ("CapabilityChange", "StepMetrics", "StepTaskSummary"):
        cls = getattr(m, name)
        assert hasattr(cls, "__dataclass_fields__"), f"{name} lost its @dataclass decorator"
    assert m.StepMetrics(env_step=3).env_step == 3
    ok("CapabilityChange / StepMetrics / StepTaskSummary all constructible")

    print("test 2: StepMetrics.to_dict fills the paths compute_metrics reads")
    d = m.StepMetrics(env_step=1).to_dict()
    changes = d["constraint_changes"]
    for constraint in ("spatial", "temporal", "dependency"):
        assert set(changes[constraint]) >= {"improved", "worsened", "improved_tasks", "worsened_tasks"}
    assert set(d["capability_changes"]) >= {"increased", "decreased"}
    ok("constraint_changes.{spatial,temporal,dependency}.{improved,worsened}_tasks + capability_changes")

    print("test 3: spatial counts agents within the threshold")
    apple = FakeEntity("apple.n.01_1", "apple_0", (0.0, 0.0, 0.05))
    near = FakeEntity("agent.n.01_1", "agent_0", (1.0, 0.0, 0.0), is_robot=True)
    far = FakeEntity("agent.n.01_2", "agent_1", (9.0, 0.0, 0.0), is_robot=True)
    world = FakeWorld([apple, near, far], [])
    task = m.BehaviorTaskState(
        task_id="t1", target_id="apple.n.01_1",
        goal=("ontop", "apple.n.01_1", "table.n.02_1"), required_agents=2, distance_threshold=2.0,
    )
    tracker = m.CoopTaskTracker(world, [task])
    metrics = tracker.step(env_step=1)
    assert task.agents_nearby == ["agent_0"], task.agents_nearby
    assert task.spatial_count == 1 and metrics.spatial_satisfied == 0, "1 < required 2"
    ok("agent at 1.0 m counted, agent at 9.0 m not; spatial unsatisfied at 1/2")

    print("test 4: temporal comes from decisions, not from world state")
    # "acting simultaneously" is invisible in the world; only the decisions show
    # it, which is why step() takes an acting map.
    metrics = tracker.step(env_step=2, acting={"agent_0": "t1", "agent_1": "t1"})
    assert task.temporal_count == 2 and metrics.temporal_satisfied == 1
    assert metrics.temporal_improved == 1 and "t1" in metrics.temporal_improved_tasks
    ok("2 agents acting on t1 -> temporal satisfied and recorded as improved")

    print("test 5: improvement and worsening are diffed against the last step")
    metrics = tracker.step(env_step=3, acting={})
    assert metrics.temporal_worsened == 1 and "t1" in metrics.temporal_worsened_tasks
    assert metrics.temporal_improved == 0
    ok("temporal 2 -> 0 recorded as worsened, not just as a lower count")

    print("test 6: the goal is evaluated from the fact list, in BDDL tokens")
    world._facts = [FakeFact("ontop", ("apple.n.01_1", "table.n.02_1"))]
    tracker.step(env_step=4)
    assert task.satisfied and task.status == m.TaskStatus.COMPLETED
    assert tracker.summary() == {"total": 1, "satisfied": 1, "score": 1.0}
    ok("ontop(apple.n.01_1, table.n.02_1) satisfies the goal")

    print("test 7: dependency is a precondition predicate, not a held tool")
    fridge = FakeEntity("electric_refrigerator.n.01_1", "fridge_0", (0.5, 0.0, 0.5),
                        states={"Open": False})
    world2 = FakeWorld([apple, near, fridge], [])
    gated = m.BehaviorTaskState(
        task_id="t2", target_id="apple.n.01_1",
        goal=("ontop", "apple.n.01_1", "table.n.02_1"),
        precondition=("open", "electric_refrigerator.n.01_1"),
    )
    tracker2 = m.CoopTaskTracker(world2, [gated])
    tracker2.step(env_step=1)
    assert gated.dependency_met is False, "fridge shut -> dependency unmet"
    fridge.states["Open"] = True
    metrics = tracker2.step(env_step=2)
    assert gated.dependency_met is True
    assert metrics.dependency_improved == 1 and "t2" in metrics.dependency_improved_tasks
    ok("open(fridge) False -> True recorded as a dependency improvement")

    print("test 8: the per-step record is what plan_log_saver writes out")
    history = tracker2.get_task_states_history()
    assert len(history) == 2
    record = history[-1]
    assert set(record) >= {"env_step", "tasks", "metrics"}
    assert record["metrics"]["constraint_changes"]["dependency"]["improved_tasks"] == ["t2"]
    assert m.convert_to_serializable(record) is not None
    ok("task_states records carry env_step/tasks/metrics and survive serialisation")

    print("test 9: get_history() snapshots satisfy plan_log_saver's contract")
    # plan_log_saver diffs these by attribute AND calls .to_dict() on them,
    # so dict-shaped values crash it at save time -- after the episode has
    # already run. It also snapshots per step, so the values must be copies:
    # storing the live task objects makes every step show the final state.
    summaries = tracker2.get_history()
    assert len(summaries) == 2
    for summary in summaries:
        for task in summary.tasks.values():
            for attr in ("spatial_count", "temporal_count", "dependency_met", "status"):
                assert hasattr(task, attr), f"saver reads .{attr} by attribute"
            assert isinstance(task.to_dict(), dict)
        assert isinstance(summary.to_dict(), dict)
    assert summaries[0].tasks["t2"].dependency_met is False, "step 1 snapshot must not show step 2's value"
    assert summaries[1].tasks["t2"].dependency_met is True
    ok("per-step snapshots are attribute-readable copies, not dicts or live references")

    print("test 10: the post-episode output functions actually execute")
    # These three are carried over verbatim and only ever ran at the very end
    # of a GPU episode, so a missing lazy import in one of them (mpatches) cost
    # a full 10-minute run to surface. Execute them here instead.
    import json
    import tempfile

    import matplotlib
    matplotlib.use("Agg")

    with tempfile.TemporaryDirectory() as tmp:
        plot_path = os.path.join(tmp, "metrics_timeline.png")
        m.plot_metrics_timeline(tracker2.metrics_history, output_path=plot_path, show=False)
        assert os.path.getsize(plot_path) > 0, "plot_metrics_timeline wrote nothing"

        states_path = os.path.join(tmp, "task_states.json")
        m.save_task_states_log(tracker2.get_task_states_history(), states_path)
        written = json.load(open(states_path))
        # compute_metrics.compute_constraint_metrics reads exactly these keys.
        assert written and set(written[0]) >= {"step", "task_states", "metrics"}
        assert written[-1]["metrics"]["constraint_changes"]["dependency"]["improved_tasks"] == ["t2"]

        cap_path = os.path.join(tmp, "capability_log.json")
        m.save_capability_log(tracker2.capability_history, cap_path)
        assert os.path.getsize(cap_path) > 0
    ok("plot_metrics_timeline / save_task_states_log / save_capability_log all run")

    print("test 11: ('holding', obj) reads held_by, not the object's state dict")
    # The default task set is ("holding", "apple.n.01_1"). Routing it through
    # the state dict -- which carries only Open/ToggledOn -- made it return
    # False forever: an agent held the apple for a thousand steps while every
    # snapshot said satisfied=False, so the run had no visible success at all.
    held_apple = FakeEntity("apple.n.01_1", "apple_0", (0.0, 0.0, 0.9))
    free_apple = FakeEntity("apple.n.01_2", "apple_1", (0.5, 0.0, 0.9))
    robot = FakeEntity("agent_0", "agent_0", (0.2, 0.0, 0.0), is_robot=True)
    held_apple.held_by = "agent_0"

    world3 = FakeWorld([held_apple, free_apple, robot], [])
    task = m.BehaviorTaskState(
        task_id="t_held", target_id="apple.n.01_1", goal=("holding", "apple.n.01_1")
    )
    other = m.BehaviorTaskState(
        task_id="t_free", target_id="apple.n.01_2", goal=("holding", "apple.n.01_2")
    )
    tracker3 = m.CoopTaskTracker(world3, [task, other])
    tracker3.step(env_step=1)
    assert task.satisfied is True, "an apple in a gripper satisfies holding()"
    assert other.satisfied is False, "an apple on the floor does not"

    held_apple.held_by = None
    tracker3.step(env_step=2)
    assert task.satisfied is False, "releasing it must un-satisfy the task"
    ok("holding() follows held_by in both directions")

    print("test 12: the first snapshot has no predecessor, so it needs a baseline")
    # _diff skips a task whose previous state is unknown, and the tracker only
    # samples when a primitive terminates -- hundreds of ticks in, with the
    # agents already standing at their targets. Without a snapshot at step 0
    # the nobody-near -> someone-near transition is never counted, and C+ reads
    # 0 for an episode in which both agents reached an object.
    far = FakeEntity("apple.n.01_9", "apple_9", (50.0, 50.0, 0.9))
    walker = FakeEntity("agent_0", "agent_0", (0.0, 0.0, 0.0), is_robot=True)
    world4 = FakeWorld([far, walker], [])
    task4 = m.BehaviorTaskState(
        task_id="t_far", target_id="apple.n.01_9", goal=("holding", "apple.n.01_9")
    )
    tracker4 = m.CoopTaskTracker(world4, [task4])

    baseline = tracker4.step(env_step=0)          # nobody near
    assert baseline.spatial_improved == 0, "a baseline cannot improve on nothing"
    assert task4.spatial_count == 0

    walker.position = (50.0, 50.5, 0.0)           # walked up to it
    after = tracker4.step(env_step=400)
    assert task4.spatial_count == 1
    assert after.spatial_improved == 1, "approaching must count as an improvement"
    assert "t_far" in after.spatial_improved_tasks
    ok("a step-0 baseline makes the first approach visible to C+")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
