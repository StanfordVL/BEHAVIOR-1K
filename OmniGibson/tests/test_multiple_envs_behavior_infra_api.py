"""
test_multiple_envs_behavior_infra_api.py
========================================
Multi-environment infrastructure tests using BehaviorTask with R1Pro.

Covers:
  Section 1 – Environment construction
  Section 2 – Step and reset
"""

import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.tasks.behavior_task import BehaviorTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"
SCENE_MODEL = "house_double_floor_lower"

# Test counter for progress tracking
_test_counter = {"current": 0, "total": 6}


def _progress(test_name):
    """Print progress for the current test."""
    _test_counter["current"] += 1
    n = _test_counter["current"]
    total = _test_counter["total"]
    print(f"\n{'='*60}")
    print(f"[{n}/{total}] RUNNING: {test_name}")
    print(f"{'='*60}")


def _passed(test_name):
    """Print pass confirmation."""
    print(f"[PASSED] {test_name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _init_macros():
    """Set simulator macros (only once, before first Environment is created)."""
    if og.sim is None:
        gm.RENDER_VIEWER_CAMERA = False
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = False
    else:
        og.sim.stop()


def setup_behavior_environment(num_envs=NUM_ENVS, use_presampled_robot_pose=True):
    """Create an Environment with BehaviorTask and R1Pro."""
    _init_macros()
    cfg = {
        "env": {"num_envs": num_envs},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": SCENE_MODEL,
            "load_room_types": ["living_room", "kitchen"],
        },
        "robots": [{"model": "r1pro", "obs_modalities": []}],
        "task": {
            "type": "BehaviorTask",
            "activity_name": ACTIVITY_NAME,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "use_presampled_robot_pose": use_presampled_robot_pose,
            "termination_config": {"max_steps": 500},
            "reward_config": {"r_potential": 1.0},
        },
    }
    print(
        f"  Setting up BehaviorTask env: num_envs={num_envs}, robot=r1pro, "
        f"activity={ACTIVITY_NAME}, presampled_pose={use_presampled_robot_pose}"
    )
    env = og.Environment(configs=cfg)
    print("  BehaviorTask environment created successfully")
    return env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def behavior_env():
    """Build the BehaviorTask env once per module and tear it down at the end.

    Loading `house_double_floor_lower` x NUM_ENVS with a BehaviorTask is the
    dominant CI cost; sharing one env across the module's tests is what keeps
    the suite under the 30-minute timeout.
    """
    env = setup_behavior_environment()
    try:
        yield env
    finally:
        og.clear()


@pytest.fixture(autouse=True)
def _reset_behavior_env(request):
    """Restore a clean post-reset state before each test that uses behavior_env."""
    if "behavior_env" in request.fixturenames:
        request.getfixturevalue("behavior_env").reset()
    yield


# ===================================================================
#  Section 1 – Environment construction
# ===================================================================


class TestBehaviorEnvConstruction:
    """Basic BehaviorTask environment construction tests."""

    def test_behavior_env_construction(self, behavior_env):
        """BehaviorTask env with num_envs=2 creates correct structure."""
        _progress("TestBehaviorEnvConstruction::test_behavior_env_construction")
        env = behavior_env

        assert len(env.scenes) == NUM_ENVS
        assert env.num_envs == NUM_ENVS
        for scene in env.scenes:
            assert len(scene.robots) == 1
        assert isinstance(env.task, BehaviorTask)
        assert env.task.activity_name == ACTIVITY_NAME

        _passed("TestBehaviorEnvConstruction::test_behavior_env_construction")

    def test_scenes_spatially_separated(self, behavior_env):
        """BehaviorTask scenes occupy different spatial regions."""
        _progress("TestBehaviorEnvConstruction::test_scenes_spatially_separated")
        env = behavior_env

        scene_positions = [s.get_position_orientation()[0] for s in env.scenes]
        for i in range(len(scene_positions)):
            for j in range(i + 1, len(scene_positions)):
                dist = th.norm(scene_positions[i] - scene_positions[j])
                print(f"  Scene {i} <-> Scene {j} distance: {dist:.2f}")
                assert dist > 1.0, f"Scenes {i} and {j} are too close: {dist:.2f}"

        _passed("TestBehaviorEnvConstruction::test_scenes_spatially_separated")


# ===================================================================
#  Section 2 – Step and reset
# ===================================================================


class TestBehaviorStepAndReset:
    """step() / reset() contract tests with BehaviorTask."""

    def test_step_return_shapes(self, behavior_env):
        """step() returns tensors of shape (num_envs,) for rewards/terminateds/truncateds."""
        _progress("TestBehaviorStepAndReset::test_step_return_shapes")
        env = behavior_env

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        obs_list, rewards, terminateds, truncateds, infos = env.step(actions)

        print(f"  obs_list len={len(obs_list)}, rewards shape={rewards.shape}")
        assert isinstance(obs_list, list) and len(obs_list) == NUM_ENVS
        assert rewards.shape == (NUM_ENVS,)
        assert terminateds.shape == (NUM_ENVS,) and terminateds.dtype == th.bool
        assert truncateds.shape == (NUM_ENVS,) and truncateds.dtype == th.bool
        assert isinstance(infos, list) and len(infos) == NUM_ENVS

        _passed("TestBehaviorStepAndReset::test_step_return_shapes")

    def test_selective_reset(self, behavior_env):
        """Resetting env_indices=[1] only resets scene 1, leaving scene 0 unchanged."""
        _progress("TestBehaviorStepAndReset::test_selective_reset")
        env = behavior_env

        known_pos = th.tensor([1.0, 1.0, 0.5])
        env.scenes[0].robots[0].set_position_orientation(position=known_pos, frame="scene")
        og.sim.step()

        pos_before = env.scenes[0].robots[0].get_position_orientation(frame="scene")[0].clone()

        env.reset(env_indices=th.tensor([1]))

        pos_after = env.scenes[0].robots[0].get_position_orientation(frame="scene")[0]
        print(f"  pos_before={pos_before}, pos_after={pos_after}")
        # BehaviorTask scenes have many objects, so physics settling causes more drift than minimal scenes
        assert th.allclose(
            pos_before, pos_after, atol=0.15
        ), f"Scene 0 robot moved after resetting only scene 1: {pos_before} vs {pos_after}"

        _passed("TestBehaviorStepAndReset::test_selective_reset")

    def test_per_env_step_counters(self, behavior_env):
        """episode_steps is a (num_envs,) tensor that tracks steps independently."""
        _progress("TestBehaviorStepAndReset::test_per_env_step_counters")
        env = behavior_env

        assert env.episode_steps.shape == (NUM_ENVS,)
        assert (env.episode_steps == 0).all()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        env.step(actions)

        print(f"  episode_steps after 1 step: {env.episode_steps}")
        assert (env.episode_steps == 1).all()

        env.reset(env_indices=th.tensor([0]))
        print(f"  episode_steps after resetting env 0: {env.episode_steps}")
        assert env.episode_steps[0] == 0
        assert env.episode_steps[1] == 1

        _passed("TestBehaviorStepAndReset::test_per_env_step_counters")


# ===================================================================
#  Section 1b – Single-env construction (kept separate; needs num_envs=1)
# ===================================================================
#
# This test must run *after* every test that uses `behavior_env`, because it
# tears down the shared module-scope env to build a 1-env one. Pytest collects
# tests in source order, so keeping this class at the bottom of the file is
# what guarantees the correct ordering.


# TODO(vector): Clean up all of these multiple envs tests to not use the weird progress/passed/etc. helpers.
# Also read through everything to make sure they're not doing anything weird.
class TestBehaviorSingleEnv:
    """BehaviorTask with num_envs=1. Cannot share the module-scope 2-env fixture."""

    def test_single_env_behavior(self):
        """BehaviorTask with num_envs=1 works; scene property returns first scene."""
        _progress("TestBehaviorEnvConstruction::test_single_env_behavior")
        # Tear down the module-scope `behavior_env` (built with num_envs=2) before constructing
        # the num_envs=1 variant. _init_macros only stops the sim; without an explicit clear,
        # the new env's object-state machinery references prims from the previous scenes and
        # crashes during play() with `'NoneType' object has no attribute 'state_updated'`.
        og.clear()
        env = setup_behavior_environment(num_envs=1)
        env.reset()

        assert env.scene is env.scenes[0]
        assert len(env.scenes) == 1
        assert isinstance(env.task, BehaviorTask)

        action = th.from_numpy(env.scenes[0].robots[0].action_space.sample()).float().unsqueeze(0)
        obs_list, rewards, terminateds, truncateds, infos = env.step(action)

        assert rewards.shape == (1,)
        assert len(obs_list) == 1

        og.clear()
        _passed("TestBehaviorEnvConstruction::test_single_env_behavior")
