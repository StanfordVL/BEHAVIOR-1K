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
        self.arm_names = ["left"]
        self._ag_obj_in_hand = {"left": None}

    def is_grasping(self, arm="default", candidate_obj=None):
        """Mirrors ManipulationRobot.is_grasping, which the code now calls
        instead of reading the private _ag_obj_in_hand."""
        arm = "left" if arm == "default" else arm
        held = self._ag_obj_in_hand.get(arm)
        return held is not None if candidate_obj is None else held is candidate_obj


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
    cup = FakeObject("apple_abc_0", "apple", [-1.0, 0.0, 0.9], in_rooms=["kitchen_0"])
    cup2 = FakeObject("apple_abc_1", "apple", [-1.5, 0.5, 0.9], in_rooms=["kitchen_0"])
    ghost = FakeObject("marker_0", "marker", [1.0, 0.0, 0.0], visual_only=True)
    alice = FakeRobot("agent_0", [-1.2, 0.0, 0.0])
    bob = FakeRobot("agent_1", [2.0, 0.0, 0.0])
    scene = FakeScene([counter, cup, cup2, ghost, alice, bob], fixed=[counter], seg_map=seg)
    env = FakeEnv(scene, [alice, bob])

    print("test 1: the scene graph builder is off by default, correct when asked for")
    # Upstream's builder scans every ordered pair in the scene against every
    # relative boolean state: 1.59M get_value calls and 14.9 s per refresh on
    # a 654-object scene, against 44 ms for a physics tick. Relations are
    # computed scoped to the shown entities instead, so the builder is opt-in.
    FakeSceneGraphBuilder.last_kwargs = None
    default_world = ws.BehaviorWorldState(env)
    default_world.start()
    assert FakeSceneGraphBuilder.last_kwargs is None, "builder must not be built by default"
    assert default_world._graph is None

    world = ws.BehaviorWorldState(env, use_scene_graph=True)
    world.start()
    kwargs = FakeSceneGraphBuilder.last_kwargs
    assert kwargs["full_obs"] is True, kwargs
    assert kwargs["robot_names"] == ["agent_0", "agent_1"], kwargs
    assert kwargs["egocentric"] is False
    ok("builder opt-in; when on, full_obs=True and every robot name is passed")

    print("test 2: type-local ids are readable and stable")
    world.step()
    entities = world.entities()
    ids = {e.name: e.entity_id for e in entities.values()}
    # BDDL instance naming: synset + _index, the same strings an activity
    # definition and its goal predicates use.
    assert ids["apple_abc_0"] == "apple.n.01_1", ids
    assert ids["apple_abc_1"] == "apple.n.01_2", ids
    cup.set_position([3.0, 0.0, 0.9])
    world.step()
    assert world.entities()["apple.n.01_1"].name == "apple_abc_0"
    ok("apple.n.01_1 / _2 assigned from the BDDL taxonomy, stable after the object moves")

    print("test 3: visual_only objects are excluded")
    assert not any(e.name == "marker_0" for e in world.entities().values())
    ok("marker_0 (visual_only) never reaches the observation")

    print("test 4: fixed furniture keeps in_rooms; movables are point-queried")
    entities = world.entities()
    assert entities["countertop.n.01_1"].rooms == ["kitchen_0"], entities["countertop.n.01_1"].rooms
    # cup#1 was carried to x=+3, so its stale in_rooms says kitchen but the
    # live query must say living_room. This is the trap: in_rooms is written at
    # scene load and never updated.
    assert cup.in_rooms == ["kitchen_0"], "precondition: the annotation is stale"
    assert entities["apple.n.01_1"].rooms == ["living_room_0"], entities["apple.n.01_1"].rooms
    ok("stale in_rooms overridden for the moved cup; fixed counter keeps its annotation")

    print("test 5: held_by is a cross-robot view")
    alice._ag_obj_in_hand["left"] = cup2
    world.step()
    assert world.entities()["apple.n.01_2"].held_by == "agent_0"
    ok("apple.n.01_2 reports held by agent_0, via the public is_grasping API")

    print("test 6: an agent sees its own room, and remembers rooms it has visited")
    obs = world.observation_for("agent_1")  # bob at x=+2 -> living_room_0
    assert obs.room == "living_room_0", obs.room
    names = {e.name for e in obs.entities.values()}
    assert "apple_abc_0" in names, "the cup was carried into bob's room"
    assert "counter_xyz_0" not in names, "the kitchen counter is not visible from the living room"
    bob.set_position([-3.0, 0.0, 0.0])
    world.step()
    obs = world.observation_for("agent_1")
    assert obs.room == "kitchen_0"
    assert "counter_xyz_0" in {e.name for e in obs.entities.values()}
    bob.set_position([2.0, 0.0, 0.0])
    world.step()
    # Default is the CURRENT room only. An observation that accumulates every
    # room ever visited grows without bound and stops describing where the
    # agent is; the current room is also the scope COOP2's spatial constraint
    # is defined on.
    obs = world.observation_for("agent_1")
    assert "counter_xyz_0" not in {e.name for e in obs.entities.values()}, "left the kitchen -> not shown"
    obs_remember = world.observation_for("agent_1", include_seen_rooms=True)
    assert "counter_xyz_0" in {e.name for e in obs_remember.entities.values()}, "memory still available opt-in"
    ok("current room by default; visited rooms only with include_seen_rooms=True")

    print("test 7: relations are translated into entity ids")
    world._graph = FakeGraph(
        nodes={cup: {"states": {"Open": False}}, counter: {"states": {}}},
        edges=[(cup, counter, {"states": [("OnTop", True)]})],
    )
    entities = world.entities()
    facts = world.facts(entities)
    assert len(facts) == 1
    assert facts[0].args == ("apple.n.01_1", "countertop.n.01_1"), facts[0].args
    # omnigibson.utils.bddl_utils is not importable under the stubs, so the
    # label falls through unchanged -- that fallback is the behaviour on any
    # scene where the mapping cannot be built.
    assert facts[0].predicate == "OnTop", facts[0].predicate
    ok(f"edge rendered as {facts[0]} (token mapping unavailable -> label kept)")

    print("test 7b: with the mapping available, edges carry BDDL tokens")
    # Seeded rather than imported: the real table comes from inverting
    # bddl_utils.PREDICATE_TO_STATE, which needs a live object_states import.
    # The pairs below are the real ones, including the two that do NOT match by
    # resemblance and would silently pass a name-equality check.
    ws.BehaviorWorldState._token_cache = {
        "OnTop": "ontop", "Inside": "inside", "Heated": "hot",
        "AttachedTo": "attached", "ToggledOn": "toggled_on",
    }
    try:
        assert world.predicate_token("OnTop") == "ontop"
        assert world.predicate_token("Heated") == "hot", "Hot is object_states.Heated, not Heated"
        assert world.predicate_token("AttachedTo") == "attached"
        assert world.predicate_token("Unmapped") == "Unmapped", "unknown labels pass through"
        world._graph = FakeGraph(
            nodes={cup: {"states": {}}, counter: {"states": {}}},
            edges=[(cup, counter, {"states": [("OnTop", True)]})],
        )
        fact = world.facts(world.entities())[0]
        assert str(fact) == "ontop(apple.n.01_1, countertop.n.01_1)", str(fact)
    finally:
        ws.BehaviorWorldState._token_cache = None
    ok(f"{fact} -- character for character what an activity definition writes")

    print("test 8: target_hints is the LLM's list of legal actions")
    obs = world.observation_for("agent_0")
    hints = {(h.primitive, h.target_id) for h in sv.target_hints(obs)}
    assert ("release", "apple.n.01_2") in hints, "agent_0 holds apple.n.01_2"
    assert ("grasp", "apple.n.01_2") not in hints, "cannot grasp what you already hold"
    assert ("place_on_top", "countertop.n.01_1") in hints, "holding something -> can place it"
    ok("release/place offered while holding; grasp withheld")

    print("test 9: an object held by a teammate is surfaced as blocked, not hidden")
    bob.set_position([-1.0, 0.0, 0.0])
    bob._ag_obj_in_hand["left"] = counter  # stand-in for "teammate holds it"
    world.step()
    obs = world.observation_for("agent_0")
    blocked = [h for h in sv.target_hints(obs) if h.primitive == "blocked"]
    assert blocked and blocked[0].target_id == "countertop.n.01_1", blocked
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
    assert ("navigate_to", "apple.n.01_1") in near, "must stay reachable or the LLM cannot discover the fix"
    assert ("grasp", "apple.n.01_1") not in near
    ok("far cup offers navigate_to only")

    print("test 11: the rendered view groups by room and lists actions")
    text = sv.render_symbolic_view(obs, interaction_radius=1.5)
    assert "you are agent_0" in text and "kitchen_0" in text
    assert "You can do:" in text and "apple.n.01_1" in text
    assert "apple_abc_0" not in text.split("You can do:")[0].split("Relations:")[0], "raw names must not leak into the entity list"
    ok("header, per-room grouping, and an action list all present")

    print("test 12: interaction_radius may be resolved per entity")
    # The gate (interaction_radius_for) is per-object: a table's radius exceeds
    # an apple's. Passing one scalar taken from a probe object -- in practice
    # the smallest one -- labelled every larger object "too far" while the gate
    # would have allowed it, contradicting what the system prompt promises.
    obs = world.observation_for("agent_0")
    distances = {}
    me = obs.entities[sv._agent_entity_id(obs)]
    for eid, e in obs.entities.items():
        if not e.is_robot:
            distances[eid] = ((e.position[0] - me.position[0]) ** 2
                              + (e.position[1] - me.position[1]) ** 2) ** 0.5
    far_id = max(distances, key=distances.get)
    far_d = distances[far_id]

    tight = {(h.primitive, h.target_id) for h in sv.target_hints(obs, interaction_radius=far_d / 2)}
    assert ("navigate_to", far_id) in tight
    assert not any(p == "grasp" and t == far_id for p, t in tight), "scalar radius must exclude it"

    # Same observation, but this entity alone gets a radius large enough.
    def per_entity(entity, _far=far_id, _d=far_d):
        return _d * 2 if entity.entity_id == _far else _d / 2

    loose = {(h.primitive, h.target_id) for h in sv.target_hints(obs, interaction_radius=per_entity)}
    assert ("navigate_to", far_id) in loose
    assert any(t == far_id and p != "navigate_to" for p, t in loose), \
        "a per-entity radius must bring the far object's other verbs back"
    ok("callable radius is applied per entity, scalar still works")

    print("test 13: a light switch is not a surface")
    # "fixed and not structural" accepted electric switches, downlights and
    # sliding doors as places to put things. Placement then asked OmniGibson to
    # sample an apple onto a light switch, and one refusal takes 113 seconds.
    class Ent:
        def __init__(self, entity_id, abilities=(), is_fixed=True):
            self.entity_id, self.abilities, self.is_fixed = entity_id, list(abilities), is_fixed

    supports = ["shelf.n.01_1", "breakfast_table.n.01_1", "countertop.n.01_1", "bookcase.n.01_1"]
    not_supports = ["switch.n.01_1", "room_light.n.01_1", "door.n.01_1", "painting.n.01_1"]
    for entity_id in supports:
        assert sv.is_support_surface(Ent(entity_id)), entity_id
        assert sv.is_receptacle(Ent(entity_id)), entity_id
    for entity_id in not_supports:
        assert not sv.is_support_surface(Ent(entity_id)), entity_id
        assert not sv.is_receptacle(Ent(entity_id)), entity_id
    # A fridge is not a support by ancestry but is still a receptacle.
    fridge = Ent("electric_refrigerator.n.01_1", abilities=["fillable", "openable"])
    assert not sv.is_support_surface(fridge)
    assert sv.is_receptacle(fridge)
    ok("supports by taxonomy ancestry; switches/lights/doors rejected; fridge still a receptacle")

    print("test 14: out-of-range objects are marked unreachable, not just verb-less")
    # An out-of-range line used to differ from an in-range one only by what was
    # missing: "navigate_to [5.7 m away]" against "grasp, navigate_to". One
    # recorded run had an agent plan grasp-then-place on an object five metres
    # away, so the state is now named.
    obs = world.observation_for("agent_0")
    me = obs.entities[sv._agent_entity_id(obs)]
    distances = {
        eid: ((e.position[0] - me.position[0]) ** 2 + (e.position[1] - me.position[1]) ** 2) ** 0.5
        for eid, e in obs.entities.items() if not e.is_robot
    }
    far_id = max(distances, key=distances.get)

    tight = sv.target_hints(obs, interaction_radius=distances[far_id] / 2)
    far_hints = [h for h in tight if h.target_id == far_id]
    primitives = {h.primitive for h in far_hints}
    assert "unreachable" in primitives, primitives
    assert "navigate_to" in primitives
    assert primitives == {"unreachable", "navigate_to"}, "no manipulation verbs on an unreachable target"
    unreachable_note = next(h.note for h in far_hints if h.primitive == "unreachable")
    assert "too far" in unreachable_note and "navigate_to first" in unreachable_note, unreachable_note
    # The rendered line keeps the shorter distance note; both say the same thing.
    shown = next(h.note for h in far_hints if h.note)
    assert "m away" in shown, shown

    # And the rendered line leads with the status word, not with a verb.
    text = sv.render_symbolic_view(obs, interaction_radius=distances[far_id] / 2)
    line = next(l for l in text.splitlines() if l.strip().startswith(far_id + ":"))
    assert line.split(":", 1)[1].strip().startswith("unreachable"), line

    # A reachable object must not be marked.
    loose = sv.target_hints(obs, interaction_radius=distances[far_id] * 2)
    assert not any(h.primitive == "unreachable" for h in loose), "in-range must not be marked"
    ok(f"{far_id} reads 'unreachable, navigate_to'; in-range targets are unmarked")

    print("test 15: BDDL object_scope bindings win over enumeration order")
    # entity_id_for numbers instances in scene order, so coffee_table.n.01_1
    # was whichever table the scene listed first while the activity's goal
    # meant the one its sampler bound -- a different table in the same room.
    # The agent placed both apples on the id it was shown and check_goal kept
    # saying unmet: an unwinnable task that reads as an agent failure.
    first = FakeObject("table_gcollb_0", "coffee_table", [0.0, 0.0, 0.4])
    second = FakeObject("table_gpkbiw_0", "coffee_table", [2.0, 0.0, 0.4])
    scene2 = FakeScene([first, second, alice], fixed=[first, second], seg_map=seg)
    env2 = FakeEnv(scene2, [alice])
    scoped = ws.BehaviorWorldState(env2)
    scoped.start()

    # Enumeration order alone would give the first table _1.
    plain = ws.BehaviorWorldState(env2)
    plain.start()
    assert plain.entity_id_for(first).endswith("_1"), plain.entity_id_for(first)

    class Task:
        object_scope = {"coffee_table.n.01_1": second}

    assert scoped.adopt_task_scope(Task()) == 1
    assert scoped.entity_id_for(second) == "coffee_table.n.01_1", scoped.entity_id_for(second)
    # And the unbound table must not be handed the id the task already used.
    other = scoped.entity_id_for(first)
    assert other != "coffee_table.n.01_1", other
    ok(f"the bound table keeps coffee_table.n.01_1; the other becomes {other}")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
