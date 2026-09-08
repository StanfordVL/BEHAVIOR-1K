"""CPU-only regression test for the symbolic contention layer.

Stubs OmniGibson (and torch) entirely, so it runs anywhere -- no Isaac, no GPU,
no scene load. It pins the three rules that put resource competition back into
the distance-blind, holder-blind symbolic primitive set:

    * an interaction radius, so acting on an object requires navigating to it;
    * distance-proportional travel ticks charged *before* the teleport;
    * cross-robot claims, so grasping a teammate's object fails loudly instead
      of silently teleporting it out of their hand and double-jointing it.

It also pins the two traps that make these rules interact badly if configured
independently: the radius must cover the navigation sampler's upper bound, and
the travel distance must be measured before the base moves.

Run:
    python feasibility_verify/test_symbolic_contention_stubbed.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
import sys
import types


# ---------------------------------------------------------------------------
# Stubs: a minimal torch, then the OmniGibson surface the modules import
# ---------------------------------------------------------------------------


class FakeVector(list):
    """Enough of a torch tensor for the sampler: indexing, slicing, item()."""

    def __getitem__(self, index):
        value = list.__getitem__(self, index)
        return FakeVector(value) if isinstance(index, slice) else value

    def item(self):
        assert len(self) == 1, self
        return self[0]

    def __float__(self):
        assert len(self) == 1, self
        return float(self[0])


def _install_torch_stub():
    module = types.ModuleType("torch")
    module.tensor = lambda data, dtype=None: FakeVector(float(x) for x in data)
    module.rand = lambda n: FakeVector(random.random() for _ in range(n))
    module.mean = lambda vector: FakeVector([sum(vector) / len(vector)])
    module.norm = lambda vector: FakeVector([math.sqrt(sum(float(v) ** 2 for v in vector))])
    module.clone = lambda x: x
    module.float32 = "float32"
    module.Tensor = FakeVector
    sys.modules["torch"] = module


class FakeActionPrimitiveError(ValueError):
    class Reason:
        PRE_CONDITION_ERROR = type("_Member", (), {"name": "PRE_CONDITION_ERROR"})()
        PLANNING_ERROR = type("_Member", (), {"name": "PLANNING_ERROR"})()

    def __init__(self, reason, message, metadata=None):
        self.reason = reason
        self.metadata = metadata or {}
        self.message = message
        super().__init__(f"{reason.name}: {message}")


class FakeActionPrimitiveErrorGroup(ValueError):
    def __init__(self, exceptions):
        self.exceptions = list(exceptions)
        super().__init__("; ".join(str(e) for e in self.exceptions))


class FakePrimitiveSet:
    class _Member:
        def __init__(self, name):
            self.name = name

    NAVIGATE_TO = _Member("NAVIGATE_TO")
    GRASP = _Member("GRASP")


class FakeSymbolicPrimitives:
    """Stands in for SymbolicSemanticActionPrimitives.

    Reproduces the upstream behaviour the contention layer is wrapping: every
    primitive mutates the scene on its *first* next() and only then yields
    settle actions, and none of them look at distance or at who else is
    holding the object.
    """

    def __init__(self, env, robot, **kwargs):
        self.env = env
        self.robot = robot
        self.arm = "left"
        self._motion_generator = None
        self.log = []

    def _postprocess_action(self, action):
        return action

    def _settle_robot(self):
        for _ in range(3):
            yield "settle"

    def _navigate_to_pose(self, pose_2d):
        # Upstream teleports before yielding anything at all.
        self.robot.position = FakeVector([float(pose_2d[0]), float(pose_2d[1]), 0.0])
        self.log.append(("navigate", float(pose_2d[0]), float(pose_2d[1])))
        yield from self._settle_robot()

    def _grasp(self, obj):
        # Upstream: teleport the object to the eef and establish the joint,
        # guarded only by *this* robot's own hand being empty.
        obj.position = FakeVector(list(self.robot.position))
        self.robot._ag_obj_in_hand[self.arm] = obj
        self.log.append(("grasp", obj.name))
        yield from self._settle_robot()

    def _place_with_predicate(self, obj, predicate, *args, **kwargs):
        self.robot._ag_obj_in_hand[self.arm] = None
        self.log.append(("place", obj.name))
        yield from self._settle_robot()

    def _open_or_close(self, obj, should_open):
        obj.open = should_open
        self.log.append(("open_or_close", obj.name, should_open))
        yield from self._settle_robot()

    def _toggle(self, obj, value):
        obj.toggled = value
        self.log.append(("toggle", obj.name, value))
        yield from self._settle_robot()


def _install_omnigibson_stubs():
    def stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    stub("omnigibson")
    stub("omnigibson.action_primitives")
    stub(
        "omnigibson.action_primitives.action_primitive_set_base",
        ActionPrimitiveError=FakeActionPrimitiveError,
        ActionPrimitiveErrorGroup=FakeActionPrimitiveErrorGroup,
    )
    stub(
        "omnigibson.action_primitives.symbolic_semantic_action_primitives",
        SymbolicSemanticActionPrimitives=FakeSymbolicPrimitives,
    )
    # primitive_engine only needs these to import; test 11 exercises the pure
    # ReasonCode mapping, not the engine loop (that is the engine's own test).
    stub(
        "omnigibson.action_primitives.starter_semantic_action_primitives",
        StarterSemanticActionPrimitives=object,
        StarterSemanticActionPrimitiveSet=FakePrimitiveSet,
    )
    stub("omnigibson.robots", Robot=object)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(module_name, relative_path):
    """Load one coop2 module under a stub-friendly alias.

    symbolic_contention imports symbolic_navigation by absolute package path,
    so that name has to be registered before it is executed.
    """
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    package = types.ModuleType("coop2")
    package.__path__ = []
    sys.modules["coop2"] = package
    sub = types.ModuleType("coop2.behavior_env")
    sub.__path__ = []
    sys.modules["coop2.behavior_env"] = sub
    _load("coop2.behavior_env.primitive_engine", "coop2/behavior_env/primitive_engine.py")
    _load("coop2.behavior_env.symbolic_navigation", "coop2/behavior_env/symbolic_navigation.py")
    return _load("coop2.behavior_env.symbolic_contention", "coop2/behavior_env/symbolic_contention.py")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSegMap:
    """'kitchen_0' is the disc of radius 3 around the origin; else 'hall_0'."""

    def get_room_instance_by_point(self, xy):
        radius = math.hypot(float(xy[0]), float(xy[1]))
        return "kitchen_0" if radius <= 3.0 else "hall_0"


class FakeScene:
    def __init__(self, seg_map=None):
        self.seg_map = seg_map
        self.trav_map = None  # no trav map -> _is_traversable is a no-op
        self.robots = []


class FakeRobot:
    def __init__(self, scene, name, position=(0.0, 0.0, 0.0)):
        self.scene = scene
        self.name = name
        self.position = FakeVector(position)
        self.arm_names = ["left"]
        self.arm_workspace_range = {"left": FakeVector([0.5, 1.0])}
        # norm([0.8, 0.8]) / 2 == 0.5657 circumscribed radius.
        self.reset_joint_pos_aabb_extent = FakeVector([0.8, 0.8, 1.2])
        self._ag_obj_in_hand = {"left": None}
        scene.robots.append(self)

    def is_grasping(self, arm="default", candidate_obj=None):
        """Mirrors ManipulationRobot.is_grasping, which the code now calls
        instead of reading the private _ag_obj_in_hand."""
        arm = "left" if arm == "default" else arm
        held = self._ag_obj_in_hand.get(arm)
        return held is not None if candidate_obj is None else held is candidate_obj

    def get_position_orientation(self):
        return self.position, None

    def get_joint_positions(self):
        return FakeVector([0.0])

    def q_to_action(self, q):
        return "hold"


class FakeObject:
    def __init__(self, name, position, in_rooms=None, aabb_extent=(0.05, 0.05, 0.05)):
        self.name = name
        self.position = FakeVector(position)
        self.in_rooms = list(in_rooms or [])
        self.aabb_extent = FakeVector(aabb_extent)
        self.open = False
        self.toggled = False

    def get_position_orientation(self):
        return self.position, None


def expect_error(generator, code):
    """Drive a primitive generator and assert it fails with ``code`` at tick 0."""
    try:
        next(generator)
    except FakeActionPrimitiveError as error:
        assert error.metadata.get("reason_code") == code, error.metadata
        assert error.reason.name == "PRE_CONDITION_ERROR", error.reason
        return error
    raise AssertionError(f"expected {code}, but the primitive started running")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def main() -> int:
    _install_torch_stub()
    _install_omnigibson_stubs()
    module = _load_modules()
    Contentious = module.ContentiousSymbolicActionPrimitives

    def ok(message):
        print(f"  PASS {message}")

    def fresh(radius=None, **kwargs):
        """A scene with two robots at the origin and a controller for each."""
        scene = FakeScene(FakeSegMap())
        alice = FakeRobot(scene, "agent_0", (0.0, 0.0, 0.0))
        bob = FakeRobot(scene, "agent_1", (0.0, 0.0, 0.0))
        make = lambda robot: Contentious(None, robot, interaction_radius=radius, **kwargs)
        return scene, alice, bob, make(alice), make(bob)

    print("test 1: the interaction radius is derived per object from the sampler's range")
    # A single global radius cannot serve both an apple and a table: the table's
    # clearance alone exceeds a sane apple radius, so an agent that navigated
    # successfully to a table would still be told TOO_FAR, forever.
    default = Contentious(None, FakeRobot(FakeScene(), "r"))
    apple = FakeObject("apple_0", [0.0, 0.0, 0.5])
    table = FakeObject("table_0", [0.0, 0.0, 0.4], aabb_extent=(1.6, 0.9, 0.7))
    apple_radius = default.interaction_radius_for(apple)
    table_radius = default.interaction_radius_for(table)
    assert table_radius > apple_radius, (table_radius, apple_radius)
    for obj, radius in ((apple, apple_radius), (table, table_radius)):
        # Anything NAVIGATE_TO can produce must be inside the radius, or the
        # agent loops navigate -> TOO_FAR -> navigate.
        assert radius >= default.sampling_range_for(obj)[1], (radius, obj.name)
    fixed = Contentious(None, FakeRobot(FakeScene(), "r2"), interaction_radius=1.0)
    assert fixed.interaction_radius_for(table) == 1.0
    ok(f"apple radius {apple_radius:.2f} m < table radius {table_radius:.2f} m; explicit override honoured")

    print("test 2: grasping across the room fails with TOO_FAR and moves nothing")
    _, alice, _, ctrl_a, _ = fresh()
    far_cup = FakeObject("cup_0", [5.0, 0.0, 0.5])
    before = list(far_cup.position)
    error = expect_error(ctrl_a._grasp(far_cup), "TOO_FAR")
    assert list(far_cup.position) == before, far_cup.position
    assert alice._ag_obj_in_hand["left"] is None
    assert error.metadata["distance"] == 5.0, error.metadata
    assert ctrl_a.log == [], ctrl_a.log
    ok(f"5.0 m grasp rejected at 0 ticks, object untouched ({error.message[:44]}...)")

    print("test 3: after navigating, the same grasp succeeds")
    ticks = list(ctrl_a._navigate_to_obj(far_cup))
    assert ctrl_a.log[0][0] == "navigate", ctrl_a.log
    assert ctrl_a.distance_to(far_cup) <= ctrl_a.interaction_radius_for(far_cup), ctrl_a.distance_to(far_cup)
    assert list(ctrl_a._grasp(far_cup)) == ["settle"] * 3
    assert alice._ag_obj_in_hand["left"] is far_cup
    ok(f"navigate ({len(ticks)} ticks) then grasp -> in hand")

    print("test 4: grasping a teammate's object fails with OBJECT_CLAIMED")
    _, alice, bob, ctrl_a, ctrl_b = fresh()
    cup = FakeObject("cup_0", [0.5, 0.0, 0.5])
    assert list(ctrl_a._grasp(cup)) == ["settle"] * 3
    assert alice._ag_obj_in_hand["left"] is cup
    held_at = list(cup.position)
    error = expect_error(ctrl_b._grasp(cup), "OBJECT_CLAIMED")
    # The upstream primitive would have teleported it and made a second joint.
    assert list(cup.position) == held_at, cup.position
    assert bob._ag_obj_in_hand["left"] is None
    assert alice._ag_obj_in_hand["left"] is cup, "the holder must keep it"
    assert "agent_0" in error.message, error.message
    ok("second grasp rejected; object stays with agent_0, no double joint")

    print("test 5: the holder can still act on what it holds")
    assert ctrl_a.holder_of(cup) is alice
    assert list(ctrl_a._place_with_predicate(FakeObject("table_0", [0.6, 0.0, 0.4]), "OnTop")) == ["settle"] * 3
    assert alice._ag_obj_in_hand["left"] is None
    assert ctrl_b.holder_of(cup) is None, "released -> claim cleared"
    ok("holder_of() is a live cross-robot view, and self-claims do not block")

    print("test 6: travel ticks are proportional to distance and precede the teleport")
    _, alice, _, ctrl_a, _ = fresh(travel_ticks_per_meter=10.0)
    steps = list(ctrl_a._navigate_to_pose([3.0, 4.0]))  # 5 m from the origin
    assert steps[:50] == ["hold"] * 50, steps[:5]
    assert steps[50:] == ["settle"] * 3, steps[50:]
    assert ctrl_a.travel_ticks(5.0) == 50 and ctrl_a.travel_ticks(1.0) == 10
    ok("5 m -> 50 hold ticks, then the teleport + settle (not the reverse)")

    print("test 7: distance is measured before the base moves")
    # If the distance were read after the teleport it would be 0 every time,
    # which is the whole reason the padding lives in _navigate_to_pose.
    _, alice, _, ctrl_a, _ = fresh(travel_ticks_per_meter=10.0)
    alice.position = FakeVector([10.0, 0.0, 0.0])
    holds = [s for s in ctrl_a._navigate_to_pose([0.0, 0.0]) if s == "hold"]
    assert len(holds) == 100, len(holds)
    ok("travelling from 10 m away costs 100 ticks, not 0")

    print("test 8: travel ticks are clamped")
    _, _, _, ctrl_a, _ = fresh(travel_ticks_per_meter=10.0, travel_ticks_range=(5, 30))
    assert ctrl_a.travel_ticks(0.0) == 5, "a floor, so navigating in place is not free"
    assert ctrl_a.travel_ticks(100.0) == 30, "a ceiling, so one navigate cannot eat the budget"
    ok("clamped to (5, 30)")

    print("test 9: place / open / toggle are gated on the same radius")
    _, alice, _, ctrl_a, _ = fresh()
    fridge = FakeObject("fridge_0", [6.0, 0.0, 1.0])
    alice._ag_obj_in_hand["left"] = FakeObject("cup_0", [0.0, 0.0, 0.5])
    expect_error(ctrl_a._place_with_predicate(fridge, "Inside"), "TOO_FAR")
    expect_error(ctrl_a._open_or_close(fridge, True), "TOO_FAR")
    expect_error(ctrl_a._toggle(fridge, True), "TOO_FAR")
    assert fridge.open is False and fridge.toggled is False
    ok("PLACE_INSIDE / OPEN / TOGGLE_ON all rejected at 6.0 m, state unchanged")

    print("test 10: every gate is individually switchable")
    _, _, _, ctrl_a, _ = fresh(gated_primitives={module.GATE_GRASP})
    loose_fridge = FakeObject("fridge_0", [6.0, 0.0, 1.0])
    assert list(ctrl_a._open_or_close(loose_fridge, True)) == ["settle"] * 3
    expect_error(ctrl_a._grasp(loose_fridge), "TOO_FAR")
    _, _, bob2, ctrl_a2, ctrl_b2 = fresh(enforce_claims=False)
    shared = FakeObject("cup_0", [0.5, 0.0, 0.5])
    list(ctrl_a2._grasp(shared))
    list(ctrl_b2._grasp(shared))  # the upstream corruption, back on request
    assert bob2._ag_obj_in_hand["left"] is shared
    ok("gated_primitives and enforce_claims=False both honoured")

    print("test 11: the engine maps these onto their own reason codes")
    ReasonCode = sys.modules["coop2.behavior_env.primitive_engine"].ReasonCode

    claimed = FakeActionPrimitiveError(
        FakeActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "held", {"reason_code": "OBJECT_CLAIMED"}
    )
    plain = FakeActionPrimitiveError(FakeActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "hand full")
    assert ReasonCode.from_primitive_error(claimed) == ReasonCode.OBJECT_CLAIMED
    assert ReasonCode.from_primitive_error(plain) == ReasonCode.PRE_CONDITION
    # Both terminate: the agent goes back to reasoning, where it can negotiate
    # for the contested object or pick a different target.
    assert ReasonCode.OBJECT_CLAIMED in ReasonCode.TERMINATES_PLAN
    assert ReasonCode.TOO_FAR in ReasonCode.TERMINATES_PLAN
    ok("metadata['reason_code'] wins over the enum; both terminate the plan")

    print("test: grasping what you already hold fails instead of no-oping")
    # Upstream re-grasps happily -- fifty settle ticks, reports success, changes
    # nothing. An agent that had achieved its goal kept proposing the same
    # grasp and kept being told it worked, nineteen times in one episode, so a
    # no-op scored as a success both wasted its decisions and inflated Y_plan.
    _, alice, bob, ctrl_a, ctrl_b = fresh()
    mine = FakeObject("cup_1", [0.5, 0.0, 0.5])
    assert list(ctrl_a._grasp(mine)) == ["settle"] * 3
    assert alice._ag_obj_in_hand["left"] is mine

    error = expect_error(ctrl_a._grasp(mine), "ALREADY_HELD")
    assert "already holding" in error.message.lower(), error.message
    assert alice._ag_obj_in_hand["left"] is mine, "the failure must not drop it"
    # A teammate reaching for it still gets the contention code, not this one.
    expect_error(ctrl_b._grasp(mine), "OBJECT_CLAIMED")
    ok("re-grasping your own object raises ALREADY_HELD; a teammate still gets OBJECT_CLAIMED")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
