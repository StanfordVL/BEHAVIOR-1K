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

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
