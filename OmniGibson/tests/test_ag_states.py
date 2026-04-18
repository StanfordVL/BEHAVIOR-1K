"""
Tests for assisted-grasping (AG) state serialization.

Covers:
  - dict roundtrip  (dump_state / load_state)
  - tensor roundtrip (dump_state serialized=True / load_state serialized=True)
  - single-arm grasps (left and right tested individually) and both-arms grasps
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

# Deterministic frame params — different per arm so an accidental swap would be detectable.
_LEFT_FRAME = {
    "parent_frame_pos": th.tensor([0.1, 0.0, 0.0]),
    "parent_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]),
    "child_frame_pos": th.tensor([0.0, 0.0, 0.05]),
    "child_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]),
    "joint_type": "FixedJoint",
}
_RIGHT_FRAME = {
    "parent_frame_pos": th.tensor([0.2, 0.0, 0.0]),
    "parent_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]),
    "child_frame_pos": th.tensor([0.0, 0.0, 0.07]),
    "child_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]),
    "joint_type": "FixedJoint",
}

# Per-block element count in the serialized tensor: magic + arm_idx + payload
_AG_BLOCK_LEN = 2 + _AG_STATE_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_ag_grasp(robot, arm, target_obj, frame_params):
    """Create an AG joint on `arm` with deterministic hand-crafted params."""
    target_link_name = sorted(target_obj.links.keys())[0]
    constraint_params = {
        "target_obj": target_obj,
        "target_link_name": target_link_name,
        **frame_params,
    }
    robot._create_assisted_grasp_joint(arm, constraint_params)
    return constraint_params


def _assert_ag_restored(robot, arm, original_params, target_obj):
    restored = robot._ag_obj_constraint_params[arm]
    assert restored is not None, f"AG constraint was not restored on arm {arm}"
    assert restored["target_obj"] is target_obj
    assert restored["target_link_name"] == original_params["target_link_name"]
    assert th.allclose(restored["parent_frame_pos"], original_params["parent_frame_pos"], atol=1e-5)
    assert th.allclose(restored["parent_frame_orn"], original_params["parent_frame_orn"], atol=1e-5)
    assert th.allclose(restored["child_frame_pos"], original_params["child_frame_pos"], atol=1e-5)
    assert th.allclose(restored["child_frame_orn"], original_params["child_frame_orn"], atol=1e-5)
    assert restored["joint_type"] == original_params["joint_type"]


def _release_and_flush(robot, *arms):
    for arm in arms:
        robot.release_grasp_immediately(arm=arm)
        assert robot._ag_obj_constraint_params[arm] is None
    # Step to flush the prim deletion and rebuild physics handles; without this,
    # SynchronizeToFabric() in editing_usd.__exit__ fires USD notices while the
    # guard is inactive and leaves a deferred error that breaks the next load_state.
    og.sim.step()


# ---------------------------------------------------------------------------
# Dict-path roundtrip tests
# ---------------------------------------------------------------------------


def test_ag_dict_no_grasp(env, assisted_robot):
    """No active grasp: dump/load leaves all arms ungrasped."""
    state = og.sim.dump_state()
    og.sim.load_state(state)
    for arm in assisted_robot.arm_names:
        assert assisted_robot._ag_obj_constraint_params[arm] is None


@pytest.mark.parametrize("arm_name,frame_params", [("left", _LEFT_FRAME), ("right", _RIGHT_FRAME)])
def test_ag_dict_single_arm(env, assisted_robot, apple, arm_name, frame_params):
    """dict roundtrip for a single-arm grasp, tested individually for each arm."""
    params = _force_ag_grasp(assisted_robot, arm_name, apple, frame_params)

    state = og.sim.dump_state()
    _release_and_flush(assisted_robot, arm_name)

    og.sim.load_state(state)
    _assert_ag_restored(assisted_robot, arm_name, params, apple)
    other_arm = "right" if arm_name == "left" else "left"
    assert assisted_robot._ag_obj_constraint_params[other_arm] is None


def test_ag_dict_both_arms(env, assisted_robot, apple, bowl):
    """dict roundtrip with both arms grasping different objects."""
    left_params = _force_ag_grasp(assisted_robot, "left", apple, _LEFT_FRAME)
    right_params = _force_ag_grasp(assisted_robot, "right", bowl, _RIGHT_FRAME)

    state = og.sim.dump_state()
    _release_and_flush(assisted_robot, "left", "right")

    og.sim.load_state(state)
    _assert_ag_restored(assisted_robot, "left", left_params, apple)
    _assert_ag_restored(assisted_robot, "right", right_params, bowl)


# ---------------------------------------------------------------------------
# Tensor-path roundtrip tests (serialized=True)
# ---------------------------------------------------------------------------


def test_ag_tensor_no_grasp(env, assisted_robot):
    """No active grasp: serialized dump/load leaves all arms ungrasped."""
    state = og.sim.dump_state(serialized=True)
    og.sim.load_state(state, serialized=True)
    for arm in assisted_robot.arm_names:
        assert assisted_robot._ag_obj_constraint_params[arm] is None


@pytest.mark.parametrize("arm_name,frame_params", [("left", _LEFT_FRAME), ("right", _RIGHT_FRAME)])
def test_ag_tensor_single_arm(env, assisted_robot, apple, arm_name, frame_params):
    """Tensor roundtrip for a single-arm grasp, tested individually for each arm."""
    params = _force_ag_grasp(assisted_robot, arm_name, apple, frame_params)

    state = og.sim.dump_state(serialized=True)
    _release_and_flush(assisted_robot, arm_name)

    og.sim.load_state(state, serialized=True)
    _assert_ag_restored(assisted_robot, arm_name, params, apple)
    other_arm = "right" if arm_name == "left" else "left"
    assert assisted_robot._ag_obj_constraint_params[other_arm] is None


def test_ag_tensor_both_arms(env, assisted_robot, apple, bowl):
    """Tensor roundtrip with both arms grasping different objects."""
    left_params = _force_ag_grasp(assisted_robot, "left", apple, _LEFT_FRAME)
    right_params = _force_ag_grasp(assisted_robot, "right", bowl, _RIGHT_FRAME)

    state = og.sim.dump_state(serialized=True)
    _release_and_flush(assisted_robot, "left", "right")

    og.sim.load_state(state, serialized=True)
    _assert_ag_restored(assisted_robot, "left", left_params, apple)
    _assert_ag_restored(assisted_robot, "right", right_params, bowl)


# ---------------------------------------------------------------------------
# Backwards-compatibility test
# ---------------------------------------------------------------------------


def test_ag_tensor_backwards_compat(env, assisted_robot, apple):
    """
    A tensor that predates AG serialization (no magic sentinel) deserializes cleanly
    with no grasp restored and no under/over-read.

    We simulate an old recording by serializing with a grasp active, then stripping
    the AG block at the tail, and re-running deserialize directly.
    """
    _force_ag_grasp(assisted_robot, "left", apple, _LEFT_FRAME)

    robot_state = assisted_robot._dump_state()
    full_tensor = assisted_robot.serialize(robot_state)

    # One grasping arm = one block = magic + arm_idx + _AG_STATE_SIZE payload
    assert full_tensor[-_AG_BLOCK_LEN].item() == _AG_MAGIC, "Expected AG magic at tail of serialized state"
    old_tensor = full_tensor[:-_AG_BLOCK_LEN]

    state_dict, idx = assisted_robot.deserialize(old_tensor)

    assert idx == len(old_tensor), f"deserialize consumed {idx} of {len(old_tensor)} elements"
    assert state_dict.get("ag_obj_constraint_params", {}) == {}


# ---------------------------------------------------------------------------
# Expected-value checks for the serialized AG block layout
# ---------------------------------------------------------------------------


def test_ag_serialized_block_right_arm_only(env, assisted_robot, apple):
    """
    A right-arm-only grasp produces exactly one block whose arm_idx == 1.
    This validates that arm_idx sits immediately after _AG_MAGIC and that no
    zero-padding is emitted for the non-grasping (left) arm.
    """
    _force_ag_grasp(assisted_robot, "right", apple, _RIGHT_FRAME)
    serialized = assisted_robot.serialize(assisted_robot._dump_state())

    ag_block = serialized[-_AG_BLOCK_LEN:]
    expected = th.tensor(
        [
            _AG_MAGIC,
            1.0,  # arm_idx — "right" is index 1 in r1pro's arm_names
            float(apple.uuid),
            0.0,  # link_idx (first sorted link)
            0.2,
            0.0,
            0.0,  # parent_frame_pos
            0.0,
            0.0,
            0.0,
            1.0,  # parent_frame_orn
            0.0,
            0.0,
            0.07,  # child_frame_pos
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


def test_ag_serialized_block_both_arms_layout(env, assisted_robot, apple, bowl):
    """
    Both-arms grasp emits two sequential magic-prefixed blocks in arm_names order
    (r1pro: left then right), each containing the correct arm_idx and uuid.
    """
    _force_ag_grasp(assisted_robot, "left", apple, _LEFT_FRAME)
    _force_ag_grasp(assisted_robot, "right", bowl, _RIGHT_FRAME)
    serialized = assisted_robot.serialize(assisted_robot._dump_state())

    blocks = serialized[-2 * _AG_BLOCK_LEN :]

    # Left block
    assert blocks[0].item() == _AG_MAGIC
    assert blocks[1].item() == 0.0  # arm_idx for "left"
    assert blocks[2].item() == float(apple.uuid)

    # Right block
    assert blocks[_AG_BLOCK_LEN].item() == _AG_MAGIC
    assert blocks[_AG_BLOCK_LEN + 1].item() == 1.0  # arm_idx for "right"
    assert blocks[_AG_BLOCK_LEN + 2].item() == float(bowl.uuid)


# ---------------------------------------------------------------------------
# Multi-env cross-model sanity check
# ---------------------------------------------------------------------------


def _run_ag_roundtrip(robot_model):
    """Spin up a fresh env with the given robot and verify dict+tensor roundtrip."""
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
                "name": "apple",
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
    obj = env.scene.object_registry("name", "apple")
    arm = robot.arm_names[0]
    params = _force_ag_grasp(robot, arm, obj, _LEFT_FRAME)

    state = og.sim.dump_state()
    robot.release_grasp_immediately(arm=arm)
    og.sim.step()
    og.sim.load_state(state)
    _assert_ag_restored(robot, arm, params, obj)

    robot.release_grasp_immediately(arm=arm)
    og.sim.step()
    robot._create_assisted_grasp_joint(arm, {**params, "target_obj": obj})
    state_t = og.sim.dump_state(serialized=True)
    robot.release_grasp_immediately(arm=arm)
    og.sim.step()
    og.sim.load_state(state_t, serialized=True)
    _assert_ag_restored(robot, arm, params, obj)

    og.clear()


@pytest.mark.parametrize("robot_model", ["fetch", "franka"])
def test_ag_roundtrip_multi_env(robot_model):
    """AG state roundtrips correctly across different single-arm robot models."""
    _run_ag_roundtrip(robot_model)
