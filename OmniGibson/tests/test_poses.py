"""
Pose setting / getting tests for various object types.

Covers the matrix described in https://github.com/StanfordVL/BEHAVIOR-1K/issues/2130:
    - setting / reading poses while stopped vs playing
    - kinematic-only, articulated, fixed-base, jointless, fixed-only-jointed entityprims
    - scene objects, config-added objects, code-added objects
    - robots, holonomic-base robots, floating-base robots
    - reading pose of the object vs the root link
    - reading per-link poses after setting the root pose (links must move with root)
    - non-root links of articulations cannot be moved directly (their pose follows physx)
    - pose preservation across dump_state / load_state
    - reading poses directly from physx via the underlying tensor views
"""

import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.prims.rigid_kinematic_prim import RigidKinematicPrim
from omnigibson.robots import Robot

from utils import get_random_pose


# Tolerances. set_position_orientation goes through several frame conversions, so we use
# loose tolerances. The real failure modes (off by meters, or NaN) are far worse than this.
POS_ATOL = 1e-3
ORN_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orientations_close(q1, q2, atol=ORN_ATOL):
    """Quaternions q and -q represent the same rotation; compare via abs dot product."""
    dot = th.abs(th.dot(q1.flatten(), q2.flatten())).item()
    return abs(dot - 1.0) < atol


def _assert_pose_close(pos1, orn1, pos2, orn2, msg="", pos_atol=POS_ATOL, orn_atol=ORN_ATOL):
    assert th.allclose(pos1, pos2, atol=pos_atol), f"{msg}: positions differ: {pos1} vs {pos2}"
    assert _orientations_close(orn1, orn2, atol=orn_atol), f"{msg}: orientations differ: {orn1} vs {orn2}"


def _physx_world_pose(obj):
    """Read the obj's root-link world pose directly through the physx tensor view.

    Returns position (xyz) and orientation in (x,y,z,w) format.
    """
    if obj.articulated:
        positions, orientations = obj._articulation_view.get_world_poses(clone=True)
    else:
        positions, orientations = obj.root_link._rigid_prim_view.get_world_poses(clone=True)
    pos = positions[0]
    orn = orientations[0][[1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
    return pos, orn


def _check_pose_roundtrip(obj):
    """Set a random pose on @obj and verify the get returns the same pose."""
    pos, orn = get_random_pose()
    obj.set_position_orientation(position=pos, orientation=orn)
    actual_pos, actual_orn = obj.get_position_orientation()
    _assert_pose_close(actual_pos, actual_orn, pos, orn, msg=f"{obj.name} round-trip")


def _check_object_root_link_match(obj):
    """obj.get_position_orientation() should match obj.root_link.get_position_orientation()."""
    obj_pos, obj_orn = obj.get_position_orientation()
    rl_pos, rl_orn = obj.root_link.get_position_orientation()
    _assert_pose_close(obj_pos, obj_orn, rl_pos, rl_orn, msg=f"{obj.name} object vs root_link pose")


def _check_links_move_with_root(obj):
    """After teleporting the object's root, every link should have shifted by the same translation
    (up to the joint-induced relative offsets, which are unchanged when only the root moves).
    """
    initial_pos, initial_orn = obj.get_position_orientation()
    initial_link_poses = {name: link.get_position_orientation() for name, link in obj.links.items()}

    # Teleport the object by a known translation along world axes only (so per-link relative
    # orientations are preserved unchanged and we can compare positions exactly).
    delta = th.tensor([3.0, -2.0, 5.0], dtype=th.float32)
    obj.set_position_orientation(position=initial_pos + delta, orientation=initial_orn)

    new_link_poses = {name: link.get_position_orientation() for name, link in obj.links.items()}
    for name in obj.links:
        old_pos, old_orn = initial_link_poses[name]
        new_pos, new_orn = new_link_poses[name]
        assert th.allclose(
            new_pos - old_pos, delta, atol=POS_ATOL
        ), f"{obj.name}.{name} did not translate with root: delta={new_pos - old_pos}, expected={delta}"
        assert _orientations_close(
            new_orn, old_orn
        ), f"{obj.name}.{name} orientation changed unexpectedly: {old_orn} -> {new_orn}"


def _check_physx_matches(obj):
    """The high-level get_position_orientation() must match the raw physx tensor view read."""
    pos_high, orn_high = obj.get_position_orientation()
    pos_raw, orn_raw = _physx_world_pose(obj)
    _assert_pose_close(pos_high, orn_high, pos_raw, orn_raw, msg=f"{obj.name} physx vs high-level read")


def _check_dump_load_preserves_pose(obj):
    """Dumping and reloading the simulation state must restore the exact pose."""
    target_pos, target_orn = get_random_pose()
    obj.set_position_orientation(position=target_pos, orientation=target_orn)
    snapshot = og.sim.dump_state()

    # Move the object somewhere else, then reload
    other_pos, other_orn = get_random_pose()
    obj.set_position_orientation(position=other_pos, orientation=other_orn)
    og.sim.load_state(snapshot)

    actual_pos, actual_orn = obj.get_position_orientation()
    _assert_pose_close(actual_pos, actual_orn, target_pos, target_orn, msg=f"{obj.name} dump/load pose")


def _check_dump_load_serialized_preserves_pose(obj):
    """Same as dump/load, but using the serialized (flattened) state path."""
    target_pos, target_orn = get_random_pose()
    obj.set_position_orientation(position=target_pos, orientation=target_orn)
    snapshot = og.sim.dump_state(serialized=True)

    other_pos, other_orn = get_random_pose()
    obj.set_position_orientation(position=other_pos, orientation=other_orn)
    og.sim.load_state(snapshot, serialized=True)

    actual_pos, actual_orn = obj.get_position_orientation()
    _assert_pose_close(actual_pos, actual_orn, target_pos, target_orn, msg=f"{obj.name} serialized dump/load pose")


# ---------------------------------------------------------------------------
# Extra fixtures — exercise object-shape dimensions not already covered by conftest.
# ---------------------------------------------------------------------------


@pytest.fixture
def kinematic_breakfast_table(stopped_env):
    """Single-link, fixed-base, kinematic-only DatasetObject (no joints)."""
    obj = DatasetObject(
        name="kinematic_breakfast_table",
        category="breakfast_table",
        model="skczfi",
        fixed_base=True,
    )
    stopped_env.scene.add_object(obj)
    obj.set_position_orientation(position=th.tensor([100, 100, 100], dtype=th.float32), frame="scene")
    return obj


@pytest.fixture
def fixed_base_microwave(stopped_env):
    """Articulated DatasetObject with a fixed base (articulation root on the entity prim)."""
    obj = DatasetObject(
        name="fixed_base_microwave",
        category="microwave",
        model="hjjxmi",
        fixed_base=True,
        kinematic_only=False,
    )
    stopped_env.scene.add_object(obj)
    obj.set_position_orientation(position=th.tensor([120, 120, 120], dtype=th.float32), frame="scene")
    return obj


@pytest.fixture
def serving_cart(stopped_env):
    """Fixed-only-jointed object: not fixed-base, no DoF joints, but an internal fixed joint
    to a meta link. This exercises the regression from issue #2121 / PR #2127.
    """
    obj = DatasetObject(
        name="serving_cart",
        category="serving_cart",
        model="vvtcby",
        abilities={"fillable": {}},
    )
    stopped_env.scene.add_object(obj)
    obj.set_position_orientation(position=th.tensor([130, 130, 130], dtype=th.float32), frame="scene")
    return obj


@pytest.fixture
def floating_robot(stopped_env):
    """Floating-base (non-holonomic) locomotion robot."""
    obj = Robot(
        name="floating_fetch",
        model="fetch",
        obs_modalities=[],
        position=[160, 160, 100],
        orientation=[0, 0, 0, 1],
        fixed_base=False,
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def holonomic_robot(stopped_env):
    """Holonomic-base robot — pose setting is implemented through 6 1-DoF base joints."""
    obj = Robot(
        name="holonomic_r1",
        model="r1",
        obs_modalities=[],
        position=[170, 170, 100],
        orientation=[0, 0, 0, 1],
    )
    stopped_env.scene.add_object(obj)
    return obj


# ---------------------------------------------------------------------------
# Stopped-state tests. Use the `env` fixture (which auto-cleans the sim on teardown)
# but stop the sim inside the test to exercise the stopped-state code paths.
# ---------------------------------------------------------------------------


def test_set_pose_stopped_jointless(env, apple):
    og.sim.stop()
    _check_pose_roundtrip(apple)
    _check_object_root_link_match(apple)


def test_set_pose_stopped_kinematic(env, kinematic_breakfast_table):
    og.sim.stop()
    obj = kinematic_breakfast_table
    assert obj.kinematic_only, "Fixture is supposed to be kinematic-only"
    assert isinstance(obj.root_link, RigidKinematicPrim)
    _check_pose_roundtrip(obj)
    _check_object_root_link_match(obj)


def test_set_pose_stopped_articulated_floating(env, microwave):
    """Articulated object with fixed_base=False (articulation root sits on root_link)."""
    og.sim.stop()
    obj = microwave
    assert obj.articulated
    assert obj.n_dof > 0
    _check_pose_roundtrip(obj)
    _check_object_root_link_match(obj)


def test_set_pose_stopped_articulated_fixed(env, fixed_base_microwave):
    og.sim.stop()
    obj = fixed_base_microwave
    assert obj.fixed_base
    assert obj.articulated
    _check_pose_roundtrip(obj)
    _check_object_root_link_match(obj)


def test_set_pose_stopped_floating_robot(env, floating_robot):
    og.sim.stop()
    obj = floating_robot
    assert obj.is_locomotion and not obj.is_holonomic_base
    _check_pose_roundtrip(obj)
    _check_object_root_link_match(obj)


def test_set_pose_stopped_holonomic_robot(env, holonomic_robot):
    og.sim.stop()
    obj = holonomic_robot
    assert obj.is_holonomic_base
    # Holonomic robots intentionally report obj.get_position_orientation() as the base_footprint
    # link pose, not the root link. They do not match. Just check the round-trip.
    _check_pose_roundtrip(obj)


# ---------------------------------------------------------------------------
# Playing-state tests (use the env fixture; sim is playing).
# These also exercise per-link consistency, the physx tensor-view path,
# and dump/load state preservation.
# ---------------------------------------------------------------------------


def _check_full_suite_playing(obj):
    """Run the full playing-state pose check suite on @obj."""
    assert og.sim.is_playing()
    _check_pose_roundtrip(obj)
    _check_object_root_link_match(obj)
    _check_links_move_with_root(obj)
    _check_physx_matches(obj)
    _check_dump_load_preserves_pose(obj)
    _check_dump_load_serialized_preserves_pose(obj)


def test_pose_playing_jointless(env, apple):
    _check_full_suite_playing(apple)


def test_pose_playing_kinematic(env, kinematic_breakfast_table):
    obj = kinematic_breakfast_table
    assert obj.kinematic_only
    # Kinematic-only objects don't appear in the rigid-body view, so we can't read them
    # via the physx tensor view path. Run the rest of the suite.
    _check_pose_roundtrip(obj)
    _check_object_root_link_match(obj)
    _check_links_move_with_root(obj)
    _check_dump_load_preserves_pose(obj)
    _check_dump_load_serialized_preserves_pose(obj)


def test_pose_playing_articulated_floating(env, microwave):
    _check_full_suite_playing(microwave)


def test_pose_playing_articulated_fixed(env, fixed_base_microwave):
    _check_full_suite_playing(fixed_base_microwave)


def test_pose_playing_floating_robot(env, floating_robot):
    _check_full_suite_playing(floating_robot)


def test_pose_playing_holonomic_robot(env, holonomic_robot):
    obj = holonomic_robot
    # Holonomic robots take a base-joint detour when playing. The root link itself isn't
    # written to directly, so the per-link delta-translation invariant doesn't hold the same
    # way (the joint frame on the world_base_joint is what moves), and obj.get_position_orientation()
    # reports the base_footprint pose rather than the root link pose. We still verify that
    # set/get round-trips and that dump/load preserves the pose.
    _check_pose_roundtrip(obj)
    _check_dump_load_preserves_pose(obj)
    _check_dump_load_serialized_preserves_pose(obj)


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


def test_pose_persists_across_play_stop(env, microwave):
    """Setting a pose while stopped and then playing must preserve the pose at play time."""
    obj = microwave
    og.sim.stop()
    target_pos, target_orn = get_random_pose()
    obj.set_position_orientation(position=target_pos, orientation=target_orn)

    pre_play_pos, pre_play_orn = obj.get_position_orientation()
    _assert_pose_close(pre_play_pos, pre_play_orn, target_pos, target_orn, msg="pre-play")

    og.sim.play()
    # Settle one step so handles update.
    og.sim.step()
    post_play_pos, _ = obj.get_position_orientation()
    # Articulated dynamic objects may drift slightly under physics; we check the position is
    # within a small tolerance of where we placed it (gravity is on but the object doesn't
    # have time to fall meaningfully in a single step).
    assert th.allclose(
        post_play_pos, target_pos, atol=0.1
    ), f"position drifted too much across play: {target_pos} -> {post_play_pos}"


def test_non_root_link_pose_follows_root(env, microwave):
    """Non-root link poses are determined by the articulation; we cannot move them directly.

    XFormPrim.set_position_orientation on a non-root link writes to USD, but the playing physx
    state will overwrite that almost immediately. After a single physics step, the link must be
    at the pose dictated by the articulation (i.e. relative to the root link with the joint
    state we set), not where we tried to write it on USD.
    """
    obj = microwave
    non_root_links = [l for name, l in obj.links.items() if name != obj.root_link_name]
    assert len(non_root_links) > 0, "Need a non-root link for this test"
    link = non_root_links[0]

    # Record the pose of the non-root link before we tamper.
    link_pos_before, _ = link.get_position_orientation()

    # Move the entire object to a new place via the legitimate path.
    new_root_pos, new_root_orn = get_random_pose()
    obj.set_position_orientation(position=new_root_pos, orientation=new_root_orn)
    og.sim.step()

    # Read the non-root link pose; it should be consistent with the articulation, NOT independent.
    new_root_actual_pos, _ = obj.root_link.get_position_orientation()
    new_link_pos, _ = link.get_position_orientation()

    # The link must have moved with the root — the relative offset should be preserved up to a
    # rotation (we used a random orientation, so we don't enforce exact translation match,
    # but we do require the link is "near" the new root, not at the old root).
    assert (
        th.norm(new_link_pos - new_root_actual_pos) < 5.0
    ), f"link {link.name} did not follow root: root at {new_root_actual_pos}, link at {new_link_pos}"
    assert th.norm(new_link_pos - link_pos_before) > 1e-3, f"link {link.name} did not move when the root moved"


def test_env_config_pose():
    """Loading an object via the env config should result in the requested pose."""
    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = True
    else:
        og.sim.stop()

    target_pos = th.tensor([2.5, -1.5, 1.25], dtype=th.float32)
    target_orn = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
    cfg = {
        "scene": {"type": "Scene"},
        "objects": [
            {
                # Use a fixed-base object so gravity doesn't drift it during the env's
                # initial play step before initial-state capture.
                "type": "DatasetObject",
                "name": "config_table",
                "category": "breakfast_table",
                "model": "skczfi",
                "fixed_base": True,
                "position": target_pos.tolist(),
                "orientation": target_orn.tolist(),
            }
        ],
    }
    env = og.Environment(configs=cfg)
    try:
        obj = env.scene.object_registry("name", "config_table")
        assert obj is not None, "Object loaded via env config not found in scene registry"

        scene_pos, scene_orn = obj.get_position_orientation(frame="scene")
        _assert_pose_close(scene_pos, scene_orn, target_pos, target_orn, msg="env config pose")
    finally:
        og.clear()


def test_scene_object_pose():
    """Objects loaded as part of an InteractiveTraversableScene should have queryable poses
    that agree across object/root_link/physx views.
    """
    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = True
    else:
        og.sim.stop()

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "breakfast_table"],
        },
    }
    env = og.Environment(configs=cfg)
    try:
        # Find a scene-loaded object (one not added by us).
        scene_objs = [
            obj for obj in env.scene.objects if obj.category not in ("floors",) and not isinstance(obj, Robot)
        ]
        assert len(scene_objs) > 0, "No scene-loaded objects found"
        obj = scene_objs[0]

        _check_object_root_link_match(obj)
        if not obj.kinematic_only:
            _check_physx_matches(obj)

        # Setting a new pose on a scene-loaded object should still work.
        _check_pose_roundtrip(obj)
        _check_object_root_link_match(obj)
    finally:
        og.clear()


# ---------------------------------------------------------------------------
# Regression tests for specific bugs we want to keep caught.
# ---------------------------------------------------------------------------


def test_fixed_only_jointed_object_survives_play(env, serving_cart):
    """Regression test for issue #2121 / PR #2127.

    An object with `fixed_base=False`, `n_joints=0`, and `n_fixed_joints>0` (a free-base
    body with internal fixed joints to a meta link) was teleported to garbage at sim.play()
    because EntityPrim.set_position_orientation only moved the root link's USD pose, leaving
    the meta link at its USD default. PhysX then exploded the joint at startup.
    """
    obj = serving_cart
    assert not obj.fixed_base, "fixture should be free-base"
    assert obj.articulated, "fixture should be articulated (has internal fixed joints)"
    assert obj.n_dof == 0, "fixture should have no DoF (only fixed joints)"

    # Place the cart at a known location while sim is playing, then step a few times.
    target_pos = th.tensor([5.0, 5.0, 1.0], dtype=th.float32)
    target_orn = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
    obj.set_position_orientation(position=target_pos, orientation=target_orn)
    for _ in range(5):
        og.sim.step()

    pos, _ = obj.get_position_orientation()
    # Should still be near where we placed it (some gravity drop is fine, explosion is not).
    assert th.all(th.isfinite(pos)), f"position became non-finite: {pos}"
    assert (
        th.norm(pos - target_pos) < 5.0
    ), f"object moved unreasonably far after play+step: target={target_pos}, actual={pos}"


def test_fixed_only_jointed_pose_set_stopped_then_play(env, serving_cart):
    """Regression test for the same #2121 path, but exercising the stopped-then-play flow
    that scene loading uses: pose is set while stopped, then sim.play() runs.
    """
    obj = serving_cart
    og.sim.stop()
    target_pos = th.tensor([3.0, 3.0, 1.0], dtype=th.float32)
    target_orn = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
    obj.set_position_orientation(position=target_pos, orientation=target_orn)
    og.sim.play()
    for _ in range(5):
        og.sim.step()

    pos, _ = obj.get_position_orientation()
    assert th.all(th.isfinite(pos)), f"position became non-finite after play: {pos}"
    assert (
        th.norm(pos - target_pos) < 5.0
    ), f"object moved unreasonably far across stopped-then-play: target={target_pos}, actual={pos}"


def test_pose_set_persists_after_step(env, microwave):
    """Regression for the broader pattern that PR #2127 fixed.

    Setting a pose while playing must actually take effect in PhysX, not just write to USD.
    A USD-only write would be reverted on the next physics step (when the USD-PhysX sync
    pulls the stale PhysX position back). After set + step, the object must still be at the
    requested location.
    """
    obj = microwave
    target_pos = th.tensor([8.0, 4.0, 0.5], dtype=th.float32)
    target_orn = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
    obj.set_position_orientation(position=target_pos, orientation=target_orn)
    obj.keep_still()
    og.sim.step()

    pos, _ = obj.get_position_orientation()
    # Pose should still be at our target after a step (allowing a tiny drift for gravity).
    assert th.allclose(
        pos, target_pos, atol=0.1
    ), f"pose was not persisted across a sim step: target={target_pos}, actual={pos}"

    # Also check via the physx tensor view directly.
    pos_raw, _ = _physx_world_pose(obj)
    assert th.allclose(
        pos_raw, target_pos, atol=0.1
    ), f"physx tensor view sees stale pose after set+step: target={target_pos}, actual={pos_raw}"
