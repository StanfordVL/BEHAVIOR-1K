"""CPU-only regression test for the cuRobo-free symbolic NAVIGATE_TO.

Stubs OmniGibson (and torch) entirely, so it runs anywhere -- no Isaac, no GPU,
no scene load. It checks the sampler's geometry and, most importantly, the two
upstream defects this override exists to work around:

    * the inherited sampler dereferences ``self._motion_generator``, which the
      symbolic constructor never builds (``AttributeError: 'NoneType' object
      has no attribute 'update_obstacles'``);
    * ``_navigate_to_obj`` forwards ``skip_obstacle_update`` into the symbolic
      ``_navigate_to_pose(self, pose_2d)``, which does not accept it
      (``TypeError``).

Run:
    python feasibility_verify/test_symbolic_navigation_stubbed.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
import sys
import types


# ---------------------------------------------------------------------------
# Stubs: a minimal torch, then the OmniGibson surface the module imports
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
    # Identity on purpose: FakeVector(x) iterates x, and FakeTravMap answers
    # __getitem__ for any index, so the legacy iteration protocol would spin
    # forever building an infinite list.
    module.clone = lambda x: x
    module.float32 = "float32"
    module.Tensor = FakeVector
    sys.modules["torch"] = module


class FakeActionPrimitiveError(ValueError):
    class Reason:
        PLANNING_ERROR = type("_Member", (), {"name": "PLANNING_ERROR"})()

    def __init__(self, reason, message, metadata=None):
        self.reason = reason
        self.metadata = metadata or {}
        super().__init__(f"{reason.name}: {message}")


class FakeSymbolicPrimitives:
    """Stands in for SymbolicSemanticActionPrimitives.

    Note the ``_navigate_to_pose`` signature: it takes **no**
    ``skip_obstacle_update``, faithfully reproducing the upstream override that
    the inherited ``_navigate_to_obj`` breaks on.
    """

    def __init__(self, env, robot, **kwargs):
        self.env = env
        self.robot = robot
        self.arm = "left"
        self.teleported_to = None
        # The whole point: the symbolic constructor skips cuRobo.
        self._motion_generator = None

    def _navigate_to_pose(self, pose_2d):
        self.teleported_to = pose_2d
        for _ in range(3):
            yield "settle"


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
    )
    stub(
        "omnigibson.action_primitives.symbolic_semantic_action_primitives",
        SymbolicSemanticActionPrimitives=FakeSymbolicPrimitives,
    )


def _load_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coop2",
        "behavior_env",
        "symbolic_navigation.py",
    )
    spec = importlib.util.spec_from_file_location("coop2_symbolic_navigation_stubbed", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSegMap:
    """'kitchen_0' is the disc of radius 3 around the origin; else 'hall_0'."""

    def get_room_instance_by_point(self, xy):
        radius = math.hypot(float(xy[0]), float(xy[1]))
        return "kitchen_0" if radius <= 3.0 else "hall_0"


class FakeTravMap:
    """Everything inside |x|,|y| < 4 m is standable; outside is not.

    world_to_map/_erode_trav_map/floor_map mirror the real TraversableMap
    surface the sampler touches.
    """

    def __init__(self, half_extent=4.0):
        self.half_extent = half_extent
        self.floor_map = [self]
        self.shape = (100, 100)

    def _erode_trav_map(self, floor_map, robot=None):
        return self

    def world_to_map(self, xy):
        # Centre the 100x100 grid on the origin at 0.1 m per cell.
        return FakeVector([float(xy[0]) / 0.1 + 50, float(xy[1]) / 0.1 + 50])

    def __getitem__(self, row):
        return _TravRow(self, row)


class _TravRow:
    def __init__(self, trav_map, row):
        self.trav_map = trav_map
        self.row = row

    def __getitem__(self, col):
        x = (self.row - 50) * 0.1
        y = (col - 50) * 0.1
        inside = abs(x) < self.trav_map.half_extent and abs(y) < self.trav_map.half_extent
        return 255 if inside else 0


class FakeScene:
    def __init__(self, seg_map=None, trav_map=None):
        self.seg_map = seg_map
        self.trav_map = trav_map
        self.robots = []


class FakeRobot:
    def __init__(self, scene, name="agent_0", position=(0.0, 0.0, 0.0)):
        self.scene = scene
        self.name = name
        self.position = FakeVector(position)
        self.arm_workspace_range = {"left": FakeVector([0.5, 1.0])}
        # norm([0.8, 0.8]) / 2 == 0.5657 -> a tidy circumscribed radius.
        self.reset_joint_pos_aabb_extent = FakeVector([0.8, 0.8, 1.2])
        scene.robots.append(self)

    def get_position_orientation(self):
        return self.position, None


class FakeObject:
    def __init__(self, name, position, in_rooms=None, aabb_extent=(0.05, 0.05, 0.05)):
        self.name = name
        self._position = FakeVector(position)
        self.in_rooms = list(in_rooms or [])
        self.aabb_extent = FakeVector(aabb_extent)

    def get_position_orientation(self):
        return self._position, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def main() -> int:
    _install_torch_stub()
    _install_omnigibson_stubs()
    module = _load_module()
    Navigable = module.NavigableSymbolicActionPrimitives

    def ok(message):
        print(f"  PASS {message}")

    scene = FakeScene(FakeSegMap())
    robot = FakeRobot(scene)
    controller = Navigable(None, robot, require_traversable=False)
    apple = FakeObject("apple_0", [1.0, 0.5, 0.4])  # no in_rooms -> point query

    print("test 1: samples inside the target's room, within the object-relative range")
    lo, hi = controller.sampling_range_for(apple)
    for _ in range(200):
        pose = controller._sample_pose_near_object(apple)
        assert pose is not None
        distance = math.hypot(float(pose[0]) - 1.0, float(pose[1]) - 0.5)
        assert lo - 1e-6 <= distance <= hi + 1e-6, (distance, lo, hi)
        assert scene.seg_map.get_room_instance_by_point(pose[:2]) == "kitchen_0"
    ok(f"200 samples in kitchen_0, all within [{lo:.2f}, {hi:.2f}] m")

    print("test 1b: the lower bound keeps the robot off the target")
    # The old range started at 0.0, so the robot could stand ON a floor object.
    # Measured consequence: symbolic grasp then teleported the apple into a
    # robot standing on it and the settle shook it loose (POST_CONDITION).
    assert lo >= controller.robot_radius, (lo, controller.robot_radius)
    table = FakeObject("table_0", [1.0, 0.5, 0.4], aabb_extent=(1.6, 0.9, 0.7))
    table_lo, table_hi = controller.sampling_range_for(table)
    assert table_lo > lo, (table_lo, lo)
    assert table_hi > hi, (table_hi, hi)
    ok(f"apple lo={lo:.2f} m, table lo={table_lo:.2f} m -- clearance scales with the object")

    print("test 2: yaw faces the target (upstream formula: yaw + pi - mean(workspace))")
    pose = controller._sample_pose_near_object(apple)
    sampling_yaw = math.atan2(float(pose[1]) - 0.5, float(pose[0]) - 1.0)
    expected = sampling_yaw + math.pi - 0.75
    delta = ((float(pose[2]) - expected + math.pi) % (2 * math.pi)) - math.pi
    assert abs(delta) < 1e-4, (float(pose[2]), expected)
    ok("yaw == sampling_yaw + pi - mean(arm_workspace_range)")

    print("test 3: explicit in_rooms wins over the live point query")
    far = FakeObject("far_apple", [10.0, 0.0, 0.4], in_rooms=["hall_0"])
    pose = controller._sample_pose_near_object(far)
    assert pose is not None
    assert scene.seg_map.get_room_instance_by_point(pose[:2]) == "hall_0"
    ok("in_rooms=['hall_0'] respected")

    print("test 4: unreachable room -> None -> PLANNING_ERROR, not a crash")
    impossible = FakeObject("ghost", [10.0, 0.0, 0.4], in_rooms=["kitchen_0"])
    assert controller._sample_pose_near_object(impossible) is None
    generator = controller._navigate_to_obj(impossible)
    try:
        next(generator)
        raise AssertionError("expected PLANNING_ERROR")
    except FakeActionPrimitiveError as error:
        assert error.reason.name == "PLANNING_ERROR", error.reason
        assert error.metadata["object"] == "ghost"
    ok("None from the sampler surfaces as PLANNING_ERROR with metadata")

    print("test 5: _navigate_to_obj must NOT forward skip_obstacle_update")
    controller.teleported_to = None
    steps = list(controller._navigate_to_obj(apple))
    assert steps == ["settle"] * 3, steps
    assert controller.teleported_to is not None
    ok("teleport + settle ran; no TypeError from the symbolic _navigate_to_pose signature")

    print("test 6: no cuRobo motion generator is ever touched")
    assert controller._motion_generator is None
    ok("_motion_generator stayed None throughout")

    print("test 7: eef_pose overrides the object position as the sampling centre")
    eef_pose = (FakeVector([-2.0, 0.0, 0.9]), None)
    pose = controller._sample_pose_near_object(apple, eef_pose=eef_pose)
    distance = math.hypot(float(pose[0]) + 2.0, float(pose[1]))
    assert distance <= controller.sampling_range_for(apple)[1] + 1e-6, distance
    ok("sampled around eef_pose, not around the object")

    print("test 8: plain Scene without a seg map -> accepts the first candidate")
    plain = Navigable(None, FakeRobot(FakeScene(seg_map=None)), require_traversable=False)
    assert plain._sample_pose_near_object(apple) is not None
    ok("works without a segmentation map")

    print("test 9: require_same_room=False bypasses the room filter")
    loose = Navigable(None, robot, require_same_room=False, require_traversable=False)
    assert loose._sample_pose_near_object(impossible) is not None
    ok("require_same_room=False returns a pose even for an out-of-room target")

    print("test 10: untraversable poses are rejected, not returned")
    # The whole reason this filter exists: without it the sampler returned the
    # first candidate whatever it was, and on Rs_int 96% of those put the robot
    # somewhere it does not fit -- it teleported into a wall and toppled.
    trav_scene = FakeScene(FakeSegMap(), trav_map=FakeTravMap(half_extent=4.0))
    trav_robot = FakeRobot(trav_scene)
    strict = Navigable(None, trav_robot, require_same_room=False)
    for _ in range(50):
        pose = strict._sample_pose_near_object(apple)
        assert pose is not None
        assert strict._is_traversable(pose[:2]), pose
    # A target sitting outside the traversable region has no standable pose at
    # all, and that must surface as None -> PLANNING_ERROR rather than a bad pose.
    marooned = FakeObject("marooned", [40.0, 40.0, 0.4])
    assert strict._sample_pose_near_object(marooned) is None
    ok("50 samples all traversable; a marooned target yields None")

    print("test 11: candidates too close to another robot are rejected")
    crowd_scene = FakeScene(FakeSegMap())
    mover = FakeRobot(crowd_scene, name="agent_0")
    parked = FakeRobot(crowd_scene, name="agent_1", position=(1.0, 0.5, 0.0))
    crowded = Navigable(None, mover, require_same_room=False, require_traversable=False)
    separation = crowded.robot_separation
    assert abs(separation - 2 * crowded.robot_radius) < 1e-9, separation
    for _ in range(50):
        pose = crowded._sample_pose_near_object(apple)
        if pose is None:
            continue
        gap = math.hypot(float(pose[0]) - 1.0, float(pose[1]) - 0.5)
        assert gap >= separation - 1e-6, (gap, separation)
    ok(f"no sample within {separation:.2f} m of agent_1 (2 x robot radius)")

    print("test 12: concurrent samplers must not all pick the same spot")
    # The TOCTOU race this registry exists for: every agent samples while every
    # other robot is still at its start pose metres away, so a
    # current-position-only check passes for all of them and they teleport into
    # each other. Measured at 0.64 m apart against a 1.24 m requirement, which
    # let one robot's assisted grasp latch onto another robot.
    race_scene = FakeScene(FakeSegMap())
    movers = [FakeRobot(race_scene, name=f"agent_{i}", position=(5.0 * i, 5.0 * i, 0.0)) for i in range(3)]
    registry = module.DestinationRegistry()
    controllers = [
        Navigable(None, robot, require_same_room=False, require_traversable=False, destinations=registry)
        for robot in movers
    ]
    contested = FakeObject("apple_0", [0.0, 0.0, 0.4])
    poses = []
    for controller in controllers:
        pose = controller._sample_pose_near_object(contested)  # nobody has moved yet
        assert pose is not None
        poses.append((float(pose[0]), float(pose[1])))
    separation = controllers[0].robot_separation
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            gap = math.hypot(poses[i][0] - poses[j][0], poses[i][1] - poses[j][1])
            assert gap >= separation - 1e-6, f"agents {i},{j} would overlap at {gap:.2f} m"
    assert len(registry) == 3
    ok(f"3 concurrent samples around one object, all >= {separation:.2f} m apart")

    print("test 13: without the registry the race reproduces")
    # Guards the guard: if this stops finding overlaps, test 12 has stopped
    # testing anything. Measured over many trials rather than one, because a
    # single random draw of three poses around an object sometimes happens to
    # be well separated -- a one-shot version of this assertion is flaky and
    # would eventually be "fixed" by deleting it.
    def closest_pair(use_registry):
        scene = FakeScene(FakeSegMap())
        robots = [FakeRobot(scene, name=f"r{i}", position=(5.0 * i, 5.0 * i, 0.0)) for i in range(3)]
        shared = module.DestinationRegistry() if use_registry else None
        picked = []
        for robot in robots:
            controller = Navigable(
                None, robot, require_same_room=False, require_traversable=False, destinations=shared
            )
            pose = controller._sample_pose_near_object(contested)
            if pose is None:
                return None
            picked.append((float(pose[0]), float(pose[1])))
        return min(
            math.hypot(picked[i][0] - picked[j][0], picked[i][1] - picked[j][1])
            for i in range(3)
            for j in range(i + 1, 3)
        )

    trials = 200
    unguarded = [closest_pair(False) for _ in range(trials)]
    guarded = [closest_pair(True) for _ in range(trials)]
    overlaps_without = sum(1 for d in unguarded if d is not None and d < separation)
    overlaps_with = sum(1 for d in guarded if d is not None and d < separation)
    assert overlaps_without > trials * 0.1, f"only {overlaps_without}/{trials} overlapped; race not reproduced"
    assert overlaps_with == 0, f"{overlaps_with}/{trials} overlapped despite reservations"
    ok(
        f"no registry: {overlaps_without}/{trials} trials overlap; "
        f"with registry: {overlaps_with}/{trials}"
    )

    print("test 14: separation is just-touching, not a planning margin")
    # Symbolic navigation teleports, so there is no path to keep clear -- only
    # bodies to keep from overlapping. Anything larger needlessly shrinks the
    # usable floor, which matters at N=9.
    assert crowded.robot_separation == 2 * crowded.robot_radius
    tight = Navigable(None, mover, robot_separation=0.1, require_traversable=False)
    assert tight.robot_separation == 0.1
    ok("default = 2 x radius, and it is overridable")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
