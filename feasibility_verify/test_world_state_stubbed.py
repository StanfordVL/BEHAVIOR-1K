"""CPU-only regression test for L1a/L1b: world model and text observation.

Stubs OmniGibson entirely -- no Isaac, no GPU, no scene load. It pins the
behaviours that make the observation usable by an LLM, and the two room-membership
traps that would otherwise be invisible until a cup is carried somewhere.

Run:
    python feasibility_verify/test_world_state_stubbed.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types


class FakeVector(list):
    def __getitem__(self, index):
        value = list.__getitem__(self, index)
        return FakeVector(value) if isinstance(index, slice) else value


def _install_stubs():
    def stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    stub("torch", Tensor=FakeVector, tensor=lambda d, dtype=None: FakeVector(d))
    stub("omnigibson")
    stub("omnigibson.scene_graphs")
    stub("omnigibson.scene_graphs.graph_builder", SceneGraphBuilder=FakeSceneGraphBuilder)


class FakeSceneGraphBuilder:
    """Records the kwargs it was constructed with; serves a fixed graph."""

    last_kwargs = None

    def __init__(self, **kwargs):
        FakeSceneGraphBuilder.last_kwargs = kwargs
        self.started = False
        self.steps = 0
        self.graph = None

    def start(self, scene):
        self.started = True

    def step(self, scene):
        self.steps += 1

    def get_scene_graph(self):
        return self.graph


class FakeGraph:
    def __init__(self, nodes=None, edges=None):
        self._nodes = nodes or {}
        self._edges = edges or []

    @property
    def nodes(self):
        return self._nodes

    def edges(self, data=False):
        return list(self._edges) if data else [(a, b) for a, b, _ in self._edges]

    def __contains__(self, item):
        return item in self._nodes


class FakeSegMap:
    """x < 0 is kitchen_0, x >= 0 is living_room_0; |x| > 8 is off-map."""

    def get_room_instance_by_point(self, xy):
        x = float(xy[0])
        if abs(x) > 8:
            return None
        return "kitchen_0" if x < 0 else "living_room_0"


class FakeObject:
    def __init__(self, name, category, position, in_rooms=None, abilities=None, visual_only=False):
        self.name = name
        self.category = category
        self._position = FakeVector(position)
        self.in_rooms = list(in_rooms or [])
        self.abilities = list(abilities or [])
        self.visual_only = visual_only

    def get_position_orientation(self):
        return self._position, None

    def set_position(self, position):
        self._position = FakeVector(position)


class FakeRobot(FakeObject):
    def __init__(self, name, position):
        super().__init__(name, "robot", position)
        self._ag_obj_in_hand = {"left": None}


class FakeScene:
    def __init__(self, objects, fixed=(), seg_map=None):
        self.objects = list(objects)
        self.fixed_objects = {o.name: o for o in fixed}
        self.seg_map = seg_map
        self._seg_map = seg_map


class FakeEnv:
    def __init__(self, scene, robots):
        self.scene = scene
        self.robots = list(robots)


def _load(module_name, relative_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(root, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    for name in ("coop2", "coop2.behavior_env"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    ws = _load("coop2.behavior_env.world_state", "coop2/behavior_env/world_state.py")
    sv = _load("coop2.behavior_env.symbolic_view", "coop2/behavior_env/symbolic_view.py")
    return ws, sv


def main() -> int:
    _install_stubs()
    ws, sv = _load_modules()

    def ok(message):
        print(f"  PASS {message}")

    seg = FakeSegMap()
    # A fixed counter annotated as kitchen, and a cup that starts in the kitchen.
    counter = FakeObject("counter_xyz_0", "countertop", [-2.0, 0.0, 0.9], in_rooms=["kitchen_0"])
    cup = FakeObject("cup_abc_0", "cup", [-1.0, 0.0, 0.9], in_rooms=["kitchen_0"])
    cup2 = FakeObject("cup_abc_1", "cup", [-1.5, 0.5, 0.9], in_rooms=["kitchen_0"])
    ghost = FakeObject("marker_0", "marker", [1.0, 0.0, 0.0], visual_only=True)
    alice = FakeRobot("agent_0", [-1.2, 0.0, 0.0])
    bob = FakeRobot("agent_1", [2.0, 0.0, 0.0])
    scene = FakeScene([counter, cup, cup2, ghost, alice, bob], fixed=[counter], seg_map=seg)
    env = FakeEnv(scene, [alice, bob])

    print("test 1: the scene graph builder is configured for multi-agent")
    world = ws.BehaviorWorldState(env)
    world.start()
    kwargs = FakeSceneGraphBuilder.last_kwargs
    assert kwargs["full_obs"] is True, kwargs
    assert kwargs["robot_names"] == ["agent_0", "agent_1"], kwargs
    assert kwargs["egocentric"] is False
    ok("full_obs=True and every robot name passed (else teammates vanish / FOVs intersect)")

    print("test 2: type-local ids are readable and stable")
    world.step()
    entities = world.entities()
    ids = {e.name: e.entity_id for e in entities.values()}
    assert ids["cup_abc_0"] == "cup#1" and ids["cup_abc_1"] == "cup#2", ids
    cup.set_position([3.0, 0.0, 0.9])
    world.step()
    assert world.entities()["cup#1"].name == "cup_abc_0"
    ok("cup#1 / cup#2 assigned, and cup#1 still means the same object after it moves")

    print("test 3: visual_only objects are excluded")
    assert not any(e.name == "marker_0" for e in world.entities().values())
    ok("marker_0 (visual_only) never reaches the observation")

    print("test 4: fixed furniture keeps in_rooms; movables are point-queried")
    entities = world.entities()
    assert entities["countertop#1"].rooms == ["kitchen_0"], entities["countertop#1"].rooms
    # cup#1 was carried to x=+3, so its stale in_rooms says kitchen but the
    # live query must say living_room. This is the trap: in_rooms is written at
    # scene load and never updated.
    assert cup.in_rooms == ["kitchen_0"], "precondition: the annotation is stale"
    assert entities["cup#1"].rooms == ["living_room_0"], entities["cup#1"].rooms
    ok("stale in_rooms overridden for the moved cup; fixed counter keeps its annotation")

    print("test 5: held_by is a cross-robot view")
    alice._ag_obj_in_hand["left"] = cup2
    world.step()
    assert world.entities()["cup#2"].held_by == "agent_0"
    ok("cup#2 reports held by agent_0, read from every robot's _ag_obj_in_hand")

    print("test 6: an agent sees its own room, and remembers rooms it has visited")
    obs = world.observation_for("agent_1")  # bob at x=+2 -> living_room_0
    assert obs.room == "living_room_0", obs.room
    names = {e.name for e in obs.entities.values()}
    assert "cup_abc_0" in names, "the cup was carried into bob's room"
    assert "counter_xyz_0" not in names, "the kitchen counter is not visible from the living room"
    bob.set_position([-3.0, 0.0, 0.0])
    world.step()
    obs = world.observation_for("agent_1")
    assert obs.room == "kitchen_0"
    assert "counter_xyz_0" in {e.name for e in obs.entities.values()}
    bob.set_position([2.0, 0.0, 0.0])
    world.step()
    obs = world.observation_for("agent_1")
    assert "counter_xyz_0" in {e.name for e in obs.entities.values()}, "kitchen was seen once, so it is remembered"
    obs_forget = world.observation_for("agent_1", include_seen_rooms=False)
    assert "counter_xyz_0" not in {e.name for e in obs_forget.entities.values()}
    ok("current room always; visited rooms retained unless include_seen_rooms=False")

    print("test 7: relations are translated into entity ids")
    world._graph = FakeGraph(
        nodes={cup: {"states": {"Open": False}}, counter: {"states": {}}},
        edges=[(cup, counter, {"states": [("OnTop", True)]})],
    )
    entities = world.entities()
    facts = world.facts(entities)
    assert len(facts) == 1 and facts[0].predicate == "OnTop"
    assert facts[0].args == ("cup#1", "countertop#1"), facts[0].args
    ok(f"edge rendered as {facts[0]}")

    print("test 8: target_hints is the LLM's list of legal actions")
    obs = world.observation_for("agent_0")
    hints = {(h.primitive, h.target_id) for h in sv.target_hints(obs)}
    assert ("release", "cup#2") in hints, "agent_0 holds cup#2"
    assert ("grasp", "cup#2") not in hints, "cannot grasp what you already hold"
    assert ("place_on_top", "countertop#1") in hints, "holding something -> can place it"
    ok("release/place offered while holding; grasp withheld")

    print("test 9: an object held by a teammate is surfaced as blocked, not hidden")
    bob.set_position([-1.0, 0.0, 0.0])
    bob._ag_obj_in_hand["left"] = counter  # stand-in for "teammate holds it"
    world.step()
    obs = world.observation_for("agent_0")
    blocked = [h for h in sv.target_hints(obs) if h.primitive == "blocked"]
    assert blocked and blocked[0].target_id == "countertop#1", blocked
    assert "agent_1" in blocked[0].note
    ok(f"reported as: {blocked[0].note}")

    print("test 10: out-of-range targets keep navigate_to and lose the rest")
    alice._ag_obj_in_hand["left"] = None
    # Far, but in agent_0's own room: at x=+7 it would be in a room agent_0 has
    # never visited, so it would be absent from the observation entirely --
    # which is correct behaviour, but tests visibility rather than range.
    cup.set_position([-7.0, 0.0, 0.9])
    world.step()
    obs = world.observation_for("agent_0")
    near = {(h.primitive, h.target_id) for h in sv.target_hints(obs, interaction_radius=1.5)}
    assert ("navigate_to", "cup#1") in near, "must stay reachable or the LLM cannot discover the fix"
    assert ("grasp", "cup#1") not in near
    ok("far cup offers navigate_to only")

    print("test 11: the rendered view groups by room and lists actions")
    text = sv.render_symbolic_view(obs, interaction_radius=1.5)
    assert "you are agent_0" in text and "kitchen_0" in text
    assert "You can do:" in text and "cup#1" in text
    assert "cup_abc_0" not in text.split("You can do:")[0].split("Relations:")[0], "raw names must not leak into the entity list"
    ok("header, per-room grouping, and an action list all present")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
