"""
test_multi_env_behavior_task.py
===============================
Multi-environment BehaviorTask-specific logic tests with R1Pro.

Covers BDDL object scope, goal conditions, potential reward, task observations,
presampled robot pose, goal status, activity attributes, and instructions.

Uses the ``picking_up_trash`` activity on ``house_double_floor_lower``.
Infrastructure tests live in ``test_multi_env_behavior_infra.py``.
"""

import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"
SCENE_MODEL = "house_double_floor_lower"

# Test counter for progress tracking
_test_counter = {"current": 0, "total": 9}


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
#  BehaviorTask-specific logic
# ===================================================================


# TODO(vector): Add the actual BEHAVIOR tests - start a task, move the objects to the goal state,
# check that the task is successful, and that the reward is correct (also that they are not complete
# before you move them etc.) - do this for various combinations of none/some/all scenes in the env
class TestBehaviorTaskLogic:
    """Tests for logic unique to BehaviorTask: BDDL scope, goal conditions,
    potential reward, task observations, presampled robot pose, and goal status."""

    def test_object_scope_per_env(self, behavior_env):
        """Each env has its own independent object scope."""
        _progress("TestBehaviorTaskLogic::test_object_scope_per_env")
        env = behavior_env

        for env_idx in range(NUM_ENVS):
            scope = env.task.object_scope[env_idx]
            assert scope is not None, f"object_scope[{env_idx}] is None"
            assert isinstance(scope, dict)
            assert "agent.n.01_1" in scope, f"agent not found in object_scope[{env_idx}]"
            # Agent entity should be this scene's robot (raw sim object after #2040 BDDLEntity removal)
            agent_entity = scope["agent.n.01_1"]
            assert agent_entity is not None
            print(f"  env {env_idx}: scope has {len(scope)} entries, agent={agent_entity.name}")

        # Scopes are independent dict objects
        assert env.task.object_scope[0] is not env.task.object_scope[1]

        _passed("TestBehaviorTaskLogic::test_object_scope_per_env")

    def test_goal_conditions_per_env(self, behavior_env):
        """Goal conditions / ground goal state options are shared (singular) across envs."""
        _progress("TestBehaviorTaskLogic::test_goal_conditions_per_env")
        env = behavior_env

        goals = env.task.activity_goal_conditions
        assert goals is not None, "activity_goal_conditions is None"
        assert len(goals) > 0, "activity_goal_conditions is empty"

        ggo = env.task.ground_goal_state_options
        assert ggo is not None, "ground_goal_state_options is None"
        assert isinstance(ggo, list)
        assert len(ggo) > 0, "ground_goal_state_options is empty"

        print(f"  shared: {len(goals)} goal conditions, {len(ggo)} ground goal state options")

        _passed("TestBehaviorTaskLogic::test_goal_conditions_per_env")

    def test_potential_reward_computation(self, behavior_env):
        """get_potential returns a finite float for each env."""
        _progress("TestBehaviorTaskLogic::test_potential_reward_computation")
        env = behavior_env

        for env_idx in range(NUM_ENVS):
            potential = env.task.get_potential(env, env_idx)
            assert isinstance(potential, float), f"get_potential({env_idx}) returned {type(potential)}, expected float"
            assert not th.isnan(th.tensor(potential)), f"get_potential({env_idx}) returned NaN"
            assert not th.isinf(th.tensor(potential)), f"get_potential({env_idx}) returned Inf"
            # Potential should be non-positive (negative success score)
            assert potential <= 0.0, f"get_potential({env_idx}) = {potential}, expected <= 0"
            print(f"  env {env_idx}: potential={potential:.4f}")

        _passed("TestBehaviorTaskLogic::test_potential_reward_computation")

    def test_task_obs_per_env(self, behavior_env):
        """Task observations are produced per env and contain expected keys."""
        _progress("TestBehaviorTaskLogic::test_task_obs_per_env")
        env = behavior_env

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        obs_list, rewards, terminateds, truncateds, infos = env.step(actions)

        # Verify each env has observations
        for env_idx in range(NUM_ENVS):
            obs = obs_list[env_idx]
            assert isinstance(obs, dict), f"obs_list[{env_idx}] is not a dict"
            # Task observations should be under "task" key
            assert "task" in obs, f"obs_list[{env_idx}] missing 'task' key, keys: {list(obs.keys())}"
            task_obs = obs["task"]
            assert isinstance(task_obs, dict), f"task obs for env {env_idx} is not a dict"
            print(f"  env {env_idx}: task obs keys={list(task_obs.keys())}")

        # Also test task.get_obs directly
        for env_idx in range(NUM_ENVS):
            task_obs = env.task.get_obs(env=env, env_idx=env_idx)
            assert isinstance(task_obs, dict)
            # Should contain "low_dim" with flattened object state observations
            if "low_dim" in task_obs:
                low_dim = task_obs["low_dim"]
                assert isinstance(low_dim, th.Tensor)
                assert low_dim.dim() == 1
                assert low_dim.shape[0] > 0
                print(f"  env {env_idx}: low_dim obs dim={low_dim.shape[0]}")

        _passed("TestBehaviorTaskLogic::test_task_obs_per_env")

    def test_presampled_robot_pose(self, behavior_env):
        """Robot is positioned at a presampled pose after reset (verifies case-insensitive lookup)."""
        _progress("TestBehaviorTaskLogic::test_presampled_robot_pose")
        env = behavior_env

        for env_idx in range(NUM_ENVS):
            robot = env.scenes[env_idx].robots[0]
            pos, ori = robot.get_position_orientation(frame="scene")
            print(f"  env {env_idx}: robot pos={pos}, ori={ori}")

            # Position should be finite and not at the origin (presampled poses are non-trivial)
            assert th.isfinite(pos).all(), f"Robot {env_idx} position has non-finite values"
            assert th.isfinite(ori).all(), f"Robot {env_idx} orientation has non-finite values"
            assert pos.norm() > 0.01, f"Robot {env_idx} at origin, expected presampled pose"

            # Orientation quaternion should be unit length
            ori_norm = ori.norm()
            assert th.allclose(
                ori_norm, th.tensor(1.0), atol=1e-2
            ), f"Robot {env_idx} orientation not unit quaternion: norm={ori_norm:.4f}"

        _passed("TestBehaviorTaskLogic::test_presampled_robot_pose")

    def test_goal_status_in_info(self, behavior_env):
        """Step info contains goal_status from PredicateGoal termination condition."""
        _progress("TestBehaviorTaskLogic::test_goal_status_in_info")
        env = behavior_env

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        obs_list, rewards, terminateds, truncateds, infos = env.step(actions)

        for env_idx in range(NUM_ENVS):
            info = infos[env_idx]
            assert "done" in info, f"info[{env_idx}] missing 'done' key"
            done_info = info["done"]
            assert "goal_status" in done_info, f"done info[{env_idx}] missing 'goal_status'"
            goal_status = done_info["goal_status"]
            assert "satisfied" in goal_status, f"goal_status[{env_idx}] missing 'satisfied'"
            assert "unsatisfied" in goal_status, f"goal_status[{env_idx}] missing 'unsatisfied'"
            assert isinstance(goal_status["satisfied"], list)
            assert isinstance(goal_status["unsatisfied"], list)
            n_sat = len(goal_status["satisfied"])
            n_unsat = len(goal_status["unsatisfied"])
            print(f"  env {env_idx}: satisfied={n_sat}, unsatisfied={n_unsat}")

            # Termination conditions should be reported per condition
            assert "termination_conditions" in done_info
            assert "timeout" in done_info["termination_conditions"]
            assert "predicate" in done_info["termination_conditions"]

        _passed("TestBehaviorTaskLogic::test_goal_status_in_info")

    def test_activity_attributes(self, behavior_env):
        """BehaviorTask has correct activity attributes after construction."""
        _progress("TestBehaviorTaskLogic::test_activity_attributes")
        env = behavior_env

        from omnigibson.utils.bddl_utils import BDDLSampler

        assert env.task.activity_name == ACTIVITY_NAME
        assert env.task.activity_definition_id == 0
        assert env.task.activity_instance_id == 0
        assert env.task.compiled_task is not None
        assert env.task.compiled_task.conditions is not None
        assert env.task.activity_initial_conditions is not None
        assert "BehaviorTask" in env.task.name
        assert ACTIVITY_NAME in env.task.name

        # Natural language conditions should be available
        assert env.task.activity_natural_language_initial_conditions is not None
        assert env.task.activity_natural_language_goal_conditions is not None

        # Symbolic compiled task is now singular (shared across envs), not a list of NUM_ENVS
        assert not isinstance(env.task.compiled_task, list)
        # object_scope stays per-env
        assert isinstance(env.task.object_scope, list)
        assert len(env.task.object_scope) == NUM_ENVS
        # Sampler is a single BDDLSampler bound to env 0
        assert isinstance(env.task.sampler, BDDLSampler)

        # Multi-env fixture must be in cache mode; online+multi-env is forbidden by _load's assert
        assert env.task.online_object_sampling is False

        print(f"  task name: {env.task.name}")
        print(f"  NL goal conditions: {env.task.activity_natural_language_goal_conditions}")

        _passed("TestBehaviorTaskLogic::test_activity_attributes")

    def test_show_instruction(self, behavior_env):
        """show_instruction returns valid instruction data per env."""
        _progress("TestBehaviorTaskLogic::test_show_instruction")
        env = behavior_env

        # Need a step for goal evaluation to populate goal_status
        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        env.step(actions)

        for env_idx in range(NUM_ENVS):
            instruction, color, objects = env.task.show_instruction(env_idx=env_idx)
            assert isinstance(instruction, str), f"instruction for env {env_idx} is not a string"
            assert len(instruction) > 0, f"instruction for env {env_idx} is empty"
            assert len(color) == 3, f"color for env {env_idx} should be RGB (3 values), got {len(color)}"
            assert isinstance(objects, list), f"objects for env {env_idx} is not a list"
            print(f"  env {env_idx}: instruction='{instruction}', color={color}, n_objects={len(objects)}")

        # Test iterate_instruction
        initial_idx = env.task.currently_viewed_index
        env.task.iterate_instruction()
        new_idx = env.task.currently_viewed_index
        total_conditions = len(env.task.compiled_task.conditions.parsed_goal_conditions)
        expected_idx = (initial_idx + 1) % total_conditions
        assert new_idx == expected_idx, f"iterate_instruction: expected idx {expected_idx}, got {new_idx}"
        print(f"  iterate_instruction: {initial_idx} -> {new_idx} (total={total_conditions})")

        _passed("TestBehaviorTaskLogic::test_show_instruction")


# ===================================================================
#  No-presample variant (kept separate; needs use_presampled_robot_pose=False)
# ===================================================================
#
# This test must run *after* every test that uses `behavior_env`, because it
# tears down the shared module-scope env to build one with a different config.
# Pytest collects in source order, so keeping this class at the bottom of the
# file is what guarantees the correct ordering.


class TestBehaviorTaskNoPresample:
    """BehaviorTask with use_presampled_robot_pose=False. Cannot share the module-scope env."""

    def test_no_presampled_robot_pose(self):
        """Robot has valid pose after reset without presampled pose."""
        _progress("TestBehaviorTaskLogic::test_no_presampled_robot_pose")
        # Tear down the module-scope `behavior_env` (built with use_presampled_robot_pose=True)
        # before constructing the variant. _init_macros only stops the sim; without an explicit
        # clear, the new env's object-state machinery references prims from the previous scenes
        # and crashes during play() with `'NoneType' object has no attribute 'state_updated'`.
        og.clear()
        env = setup_behavior_environment(use_presampled_robot_pose=False)
        env.reset()

        for env_idx in range(NUM_ENVS):
            robot = env.scenes[env_idx].robots[0]
            pos, ori = robot.get_position_orientation(frame="scene")
            print(f"  env {env_idx}: robot pos={pos}, ori={ori}")

            assert th.isfinite(pos).all(), f"Robot {env_idx} position has non-finite values"
            assert th.isfinite(ori).all(), f"Robot {env_idx} orientation has non-finite values"

            ori_norm = ori.norm()
            assert th.allclose(
                ori_norm, th.tensor(1.0), atol=1e-2
            ), f"Robot {env_idx} orientation not unit quaternion: norm={ori_norm:.4f}"

        og.clear()
        _passed("TestBehaviorTaskLogic::test_no_presampled_robot_pose")
