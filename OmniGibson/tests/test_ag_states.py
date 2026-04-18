"""
Tests for assisted-grasping (AG) state serialization.

Covers:
  - dict roundtrip  (dump_state / load_state)
  - tensor roundtrip (dump_state serialized=True / load_state serialized=True)
  - backwards compatibility: tensors that predate AG serialization (no magic sentinel) load cleanly
  - expected-value check: AG block in serialized tensor matches known hand-crafted params
"""

import pytest
import torch as th

import omnigibson as og
from omnigibson.robots.robot import _AG_MAGIC, _AG_STATE_SIZE
from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_ag_grasp(robot, target_obj):
    """
    Directly create an AG joint with deterministic, hand-crafted constraint params.
    Returns (arm_name, constraint_params) — constraint_params still holds the live object ref.
    """
    arm = robot.arm_names[0]
    target_link_name = sorted(target_obj.links.keys())[0]
    constraint_params = {
        "target_obj": target_obj,
        "target_link_name": target_link_name,
        "parent_frame_pos": th.tensor([0.1, 0.0, 0.0]),
        "parent_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]),
        "child_frame_pos": th.tensor([0.0, 0.0, 0.05]),
        "child_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]),
        "joint_type": "FixedJoint",
    }
    robot._create_assisted_grasp_joint(arm, constraint_params)
    return arm, constraint_params


def _assert_ag_restored(robot, arm, original_params, target_obj):
    restored = robot._ag_obj_constraint_params[arm]
    assert restored is not None, "AG constraint was not restored"
    assert restored["target_obj"] is target_obj
    assert restored["target_link_name"] == original_params["target_link_name"]
    assert th.allclose(restored["parent_frame_pos"], original_params["parent_frame_pos"], atol=1e-5)
    assert th.allclose(restored["parent_frame_orn"], original_params["parent_frame_orn"], atol=1e-5)
    assert th.allclose(restored["child_frame_pos"], original_params["child_frame_pos"], atol=1e-5)
    assert th.allclose(restored["child_frame_orn"], original_params["child_frame_orn"], atol=1e-5)
    assert restored["joint_type"] == original_params["joint_type"]


# ---------------------------------------------------------------------------
# Dict-path roundtrip tests
# ---------------------------------------------------------------------------


def test_ag_dict_no_grasp(env, assisted_robot):
    """No active grasp: dump/load leaves all arms ungrasped."""
    state = og.sim.dump_state()
    og.sim.load_state(state)
    for arm in assisted_robot.arm_names:
        assert assisted_robot._ag_obj_constraint_params[arm] is None


def test_ag_dict_with_grasp(env, assisted_robot, apple):
    """Active grasp: dump/load fully restores the AG constraint."""
    arm, original_params = _force_ag_grasp(assisted_robot, apple)

    state = og.sim.dump_state()
    assisted_robot.release_grasp_immediately(arm=arm)
    assert assisted_robot._ag_obj_constraint_params[arm] is None
    # Step to flush the prim deletion and rebuild physics handles; without this,
    # SynchronizeToFabric() in editing_usd.__exit__ fires USD notices while the
    # guard is inactive and leaves a deferred error that breaks the next load_state.
    og.sim.step()

    og.sim.load_state(state)
    _assert_ag_restored(assisted_robot, arm, original_params, apple)


# ---------------------------------------------------------------------------
# Tensor-path roundtrip tests  (serialized=True)
# ---------------------------------------------------------------------------


def test_ag_tensor_no_grasp(env, assisted_robot):
    """No active grasp: serialized dump/load leaves all arms ungrasped."""
    state = og.sim.dump_state(serialized=True)
    og.sim.load_state(state, serialized=True)
    for arm in assisted_robot.arm_names:
        assert assisted_robot._ag_obj_constraint_params[arm] is None


def test_ag_tensor_with_grasp(env, assisted_robot, apple):
    """Active grasp: serialized dump/load fully restores the AG constraint."""
    arm, original_params = _force_ag_grasp(assisted_robot, apple)

    state = og.sim.dump_state(serialized=True)
    assisted_robot.release_grasp_immediately(arm=arm)
    assert assisted_robot._ag_obj_constraint_params[arm] is None
    og.sim.step()  # flush prim deletion; see test_ag_dict_with_grasp for explanation

    og.sim.load_state(state, serialized=True)
    _assert_ag_restored(assisted_robot, arm, original_params, apple)


# ---------------------------------------------------------------------------
# Backwards-compatibility test
# ---------------------------------------------------------------------------


def test_ag_tensor_backwards_compat(env, assisted_robot, apple):
    """
    A tensor that predates AG serialization (magic sentinel absent) deserializes
    cleanly with no grasp restored, without corrupting the byte count.

    We simulate an old recording by serializing with a grasp active, then
    stripping the AG block (magic + per-arm data) from the robot's portion of
    the tensor, and re-running deserialize directly.
    """
    arm, _ = _force_ag_grasp(assisted_robot, apple)

    robot_state = assisted_robot._dump_state()
    full_tensor = assisted_robot.serialize(robot_state)

    # Locate and remove the AG tail: magic (1) + n_arms * _AG_STATE_SIZE
    n_ag = 1 + _AG_STATE_SIZE * len(assisted_robot.arm_names)
    assert full_tensor[-n_ag].item() == _AG_MAGIC, "Expected AG magic at tail of serialized robot state"
    old_tensor = full_tensor[:-n_ag]

    state_dict, idx = assisted_robot.deserialize(old_tensor)

    # All bytes consumed (no under/over-read)
    assert idx == len(old_tensor), f"deserialize consumed {idx} of {len(old_tensor)} elements"
    # No grasp restored
    assert state_dict.get("ag_obj_constraint_params", {}) == {}


# ---------------------------------------------------------------------------
# Multi-env roundtrip tests  (two different robot models)
# ---------------------------------------------------------------------------


def _run_ag_roundtrip(robot_model, target_obj_name="apple"):
    """
    Spin up a fresh env with the given robot in assisted mode, force a grasp,
    and verify dict + tensor roundtrips both restore it correctly.
    Called inline (not as a fixture) so each invocation gets a clean sim.
    """
    og.clear()

    config = {
        "scene": {"type": "Scene"},
        "robots": [
            {
                "model": robot_model,
                "grasping_mode": "assisted",
                "obs_modalities": [],
                "position": [150, 150, 100],
                "orientation": [0, 0, 0, 1],
            }
        ],
        "objects": [
            {
                "type": "DatasetObject",
                "name": target_obj_name,
                "category": "apple",
                "model": "agveuv",
                "position": [152, 150, 100],
            }
        ],
    }
    env = og.Environment(configs=config)
    og.sim.play()
    og.sim.step()

    robot = env.robots[0]
    obj = env.scene.object_registry("name", target_obj_name)
    arm, original_params = _force_ag_grasp(robot, obj)

    # Dict roundtrip
    state = og.sim.dump_state()
    robot.release_grasp_immediately(arm=arm)
    og.sim.step()
    og.sim.load_state(state)
    _assert_ag_restored(robot, arm, original_params, obj)

    # Tensor roundtrip
    robot.release_grasp_immediately(arm=arm)
    og.sim.step()
    robot._create_assisted_grasp_joint(arm, {**original_params, "target_obj": obj})
    state_t = og.sim.dump_state(serialized=True)
    robot.release_grasp_immediately(arm=arm)
    og.sim.step()
    og.sim.load_state(state_t, serialized=True)
    _assert_ag_restored(robot, arm, original_params, obj)

    og.clear()


@pytest.mark.parametrize("robot_model", ["fetch", "franka"])
def test_ag_roundtrip_multi_env(robot_model):
    """AG state roundtrips correctly across different robot models."""
    _run_ag_roundtrip(robot_model)


# ---------------------------------------------------------------------------
# Expected-value check for the serialized AG block
# ---------------------------------------------------------------------------


def test_ag_serialized_block(env, assisted_robot, apple):
    """
    The AG block in the serialized tensor matches expected values built from
    the same hand-crafted params used in _force_ag_grasp.  All values are
    deterministic: frame params are literal constants, UUID is derived from
    the object name, link index is 0 (first sorted link).
    """
    _force_ag_grasp(assisted_robot, apple)
    robot_state = assisted_robot._dump_state()
    serialized = assisted_robot.serialize(robot_state)

    # Extract the AG tail: magic sentinel (1) + n_arms * _AG_STATE_SIZE
    n_ag = 1 + _AG_STATE_SIZE * len(assisted_robot.arm_names)
    ag_block = serialized[-n_ag:]

    expected = th.tensor(
        [
            _AG_MAGIC,
            # arm 0 — matches the params in _force_ag_grasp
            1.0,  # has_grasp
            float(apple.uuid),  # obj_uuid  (deterministic hash of name "apple")
            0.0,  # link_idx  (sorted index 0 = first link alphabetically)
            0.1,
            0.0,
            0.0,  # parent_frame_pos
            0.0,
            0.0,
            0.0,
            1.0,  # parent_frame_orn
            0.0,
            0.0,
            0.05,  # child_frame_pos
            0.0,
            0.0,
            0.0,
            1.0,  # child_frame_orn
            0.0,  # joint_type: FixedJoint
        ]
    )

    assert th.allclose(
        ag_block, expected, atol=1e-6
    ), f"AG block mismatch.\n  actual:   {ag_block.tolist()}\n  expected: {expected.tolist()}"
