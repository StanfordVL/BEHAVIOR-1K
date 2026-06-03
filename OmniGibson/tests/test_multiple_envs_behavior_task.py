"""
test_multi_env_behavior_task.py
===============================
Multi-environment BehaviorTask-specific logic tests with R1Pro.

Covers BDDL object scope, potential reward, task observations, presampled robot
pose, activity attributes, instructions, and end-to-end goal completion (moving
objects into the goal state and verifying per-env success / reward / reset).

Uses the ``picking_up_trash`` activity on ``house_double_floor_lower``.
Infrastructure tests live in ``test_multi_env_behavior_infra.py``.
"""

import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Inside

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"
SCENE_MODEL = "house_double_floor_lower"
# Reward weight used in the task config; the completing env.step should earn ~R_POTENTIAL.
R_POTENTIAL = 1.0


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
            "reward_config": {"r_potential": R_POTENTIAL},
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
# Goal-completion helpers
# ---------------------------------------------------------------------------
def _goal_objects(env, env_idx):
    """Return ([can_of_soda, ...], ashcan) bound in @env_idx's BDDL object scope.

    picking_up_trash's goal is `forall can: inside(can, ashcan)`; these are the
    objects we move to drive the task to success.
    """
    scope = env.task.object_scope[env_idx]
    ashcans = [scope[k] for k in sorted(scope) if k.startswith("ashcan.n.01")]
    cans = [scope[k] for k in sorted(scope) if k.startswith("can__of__soda.n.01")]
    assert len(ashcans) == 1 and ashcans[0] is not None, f"ashcan not bound in scope[{env_idx}]"
    assert len(cans) >= 1 and all(c is not None for c in cans), f"cans not bound in scope[{env_idx}]"
    return cans, ashcans[0]


def _place_inside(can, ashcan, retries=3):
    """Place @can Inside @ashcan via kinematic sampling, retrying for flakiness.

    Inside.set_value rejection-samples a pose, settles physics, and verifies the
    object is still inside; it returns False on a failed sample. Retry a few times
    before giving up (sample_kinematics is known to be occasionally flaky).
    """
    for _ in range(retries):
        if can.states[Inside].set_value(ashcan, True):
            return True
    return False


def _zero_actions(env):
    """Build a (num_envs, action_dim) all-zeros action that holds the robots still.

    Zero (rather than random) actions keep the robot from disturbing cans we've
    already placed, so a step purely re-evaluates the goal/reward.
    """
    return th.stack(
        [th.zeros_like(th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float()) for i in range(NUM_ENVS)]
    )


def _random_actions(env):
    """Build a (num_envs, action_dim) random action sampled per robot."""
    return th.stack([th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)])


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


class TestBehaviorTaskLogic:
    """Tests for logic unique to BehaviorTask: BDDL scope, potential reward,
    task observations, presampled robot pose, activity attributes, instructions."""

    def test_object_scope_per_env(self, behavior_env):
        """Each env has its own independent object scope bound to its own scene's robot."""
        env = behavior_env

        for env_idx in range(NUM_ENVS):
            scope = env.task.object_scope[env_idx]
            assert scope is not None, f"object_scope[{env_idx}] is None"
            assert isinstance(scope, dict)
            assert "agent.n.01_1" in scope, f"agent not found in object_scope[{env_idx}]"
            # Agent entity should be *this* scene's robot (raw sim object after #2040 BDDLEntity removal),
            # not some other env's robot — this is what guarantees per-env predicate evaluation is isolated.
            agent_entity = scope["agent.n.01_1"]
            assert agent_entity is not None
            assert (
                agent_entity is env.scenes[env_idx].robots[0]
            ), f"agent in scope[{env_idx}] is not scene {env_idx}'s robot"
            print(f"  env {env_idx}: scope has {len(scope)} entries, agent={agent_entity.name}")

        # Scopes are independent dict objects
        assert env.task.object_scope[0] is not env.task.object_scope[1]

    def test_potential_reward_computation(self, behavior_env):
        """get_potential returns a finite float for each env."""
        env = behavior_env

        for env_idx in range(NUM_ENVS):
            potential = env.task.get_potential(env, env_idx)
            assert isinstance(potential, float), f"get_potential({env_idx}) returned {type(potential)}, expected float"
            assert not th.isnan(th.tensor(potential)), f"get_potential({env_idx}) returned NaN"
            assert not th.isinf(th.tensor(potential)), f"get_potential({env_idx}) returned Inf"
            # Potential should be non-positive (negative success score)
            assert potential <= 0.0, f"get_potential({env_idx}) = {potential}, expected <= 0"
            print(f"  env {env_idx}: potential={potential:.4f}")

    def test_task_obs_per_env(self, behavior_env):
        """task.get_obs produces a per-env low-dim observation vector.

        (The env.step()->obs "task" key plumbing is already covered by
        test_step_return_shapes in test_multiple_envs_behavior_infra_api.py; here
        we exercise the BehaviorTask-specific get_obs content directly.)
        """
        env = behavior_env

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

    def test_presampled_robot_pose(self, behavior_env):
        """Robot is positioned at a presampled pose after reset (verifies case-insensitive lookup)."""
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

    def test_activity_attributes(self, behavior_env):
        """BehaviorTask has correct activity attributes after construction."""
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

    def test_show_instruction(self, behavior_env):
        """show_instruction returns valid instruction data per env."""
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


# ===================================================================
#  End-to-end goal completion (move objects to goal state, verify success)
# ===================================================================
#
# picking_up_trash's goal is a single quantified condition,
#   forall ?can: inside(?can, ashcan),
# so get_potential / goal_status are binary at the goal-condition granularity:
# the one goal condition is unsatisfied (potential 0.0) until *every* can is
# inside, at which point it flips satisfied (potential -1.0). Per-can progress is
# therefore checked via the public states[Inside].get_value, not goal_status.


class TestBehaviorTaskGoalCompletion:
    """Drive the task to its goal state and verify success, reward, and reset,
    across the none / some / all and per-env-isolation combinations called for
    by the picking_up_trash goal."""

    def test_no_completion_under_random_actions(self, behavior_env):
        """With nothing moved to the goal, no env ever reports success (none combination)."""
        env = behavior_env

        for _ in range(10):
            _, _, terminateds, _, infos = env.step(_random_actions(env))
            assert not terminateds.any(), f"task terminated under random actions: {terminateds}"
            for env_idx in range(NUM_ENVS):
                gs = infos[env_idx]["done"]["goal_status"]
                assert len(gs["unsatisfied"]) > 0, f"env {env_idx} goal satisfied without placing cans"

    def test_single_env_completion_and_reward(self, behavior_env):
        """Completing only env 0 makes env 0 (and only env 0) succeed, with correct reward."""
        env = behavior_env

        cans, ashcan = _goal_objects(env, 0)
        for i, can in enumerate(cans):
            assert _place_inside(can, ashcan), f"failed to place can {i} inside ashcan (env 0)"

        # Stored potential is 0.0 from the pre-test reset; placing cans does not touch it,
        # so this completing step should earn reward == r_potential * (0 - (-1)) for env 0.
        _, rewards, terminateds, _, infos = env.step(_zero_actions(env))

        # env 0 succeeds, env 1 (untouched) does not
        assert terminateds[0], "env 0 not terminated after all cans placed"
        assert not terminateds[1], "env 1 terminated despite no cans placed"
        assert infos[0]["done"]["termination_conditions"]["predicate"]["done"]
        assert len(infos[0]["done"]["goal_status"]["unsatisfied"]) == 0, "env 0 goal still unsatisfied"
        assert len(infos[1]["done"]["goal_status"]["unsatisfied"]) > 0, "env 1 goal unexpectedly satisfied"

        # All cans actually register Inside in env 0
        assert all(can.states[Inside].get_value(ashcan) for can in cans), "not all cans report Inside in env 0"

        # Reward is the potential delta on the completing step (env 1 sees no change)
        print(f"  rewards={rewards}")
        assert abs(rewards[0].item() - R_POTENTIAL) < 1e-3, f"env 0 reward {rewards[0].item()} != {R_POTENTIAL}"
        assert abs(rewards[1].item()) < 1e-3, f"env 1 reward {rewards[1].item()} != 0"

        # Potential reflects full success in env 0 only
        assert abs(env.task.get_potential(env, 0) - (-1.0)) < 1e-6, "env 0 potential not -1.0 at success"
        assert env.task.get_potential(env, 1) > -1.0, "env 1 potential indicates success without placement"

    def test_all_envs_completion(self, behavior_env):
        """Completing every env makes every env succeed (all combination)."""
        env = behavior_env

        for env_idx in range(NUM_ENVS):
            cans, ashcan = _goal_objects(env, env_idx)
            for i, can in enumerate(cans):
                assert _place_inside(can, ashcan), f"failed to place can {i} inside ashcan (env {env_idx})"

        _, _, terminateds, _, infos = env.step(_zero_actions(env))

        assert terminateds.all(), f"not all envs terminated: {terminateds}"
        for env_idx in range(NUM_ENVS):
            assert infos[env_idx]["done"]["termination_conditions"]["predicate"]["done"]
            assert len(infos[env_idx]["done"]["goal_status"]["unsatisfied"]) == 0, f"env {env_idx} goal unsatisfied"

    def test_partial_progress_does_not_complete(self, behavior_env):
        """Placing all-but-one can leaves the goal unsatisfied (some combination)."""
        env = behavior_env

        cans, ashcan = _goal_objects(env, 0)
        for i, can in enumerate(cans[:-1]):
            assert _place_inside(can, ashcan), f"failed to place can {i} inside ashcan"

        _, rewards, terminateds, _, infos = env.step(_zero_actions(env))

        # Goal is a single forall: partial progress does not satisfy it
        assert not terminateds[0], "env 0 terminated with partial progress"
        gs = infos[0]["done"]["goal_status"]
        assert len(gs["satisfied"]) == 0, f"forall goal counted satisfied with partial progress: {gs}"
        assert len(gs["unsatisfied"]) > 0, "goal unexpectedly fully satisfied"

        # ...but per-can progress is real: exactly the placed cans report Inside
        n_inside = sum(bool(can.states[Inside].get_value(ashcan)) for can in cans)
        assert n_inside == len(cans) - 1, f"expected {len(cans) - 1} cans inside, got {n_inside}"

        # Binary potential => still 0.0, and no reward delta, until the last can goes in
        assert abs(env.task.get_potential(env, 0)) < 1e-6, "potential nonzero before full completion"
        assert abs(rewards[0].item()) < 1e-3, f"env 0 earned reward {rewards[0].item()} on partial progress"

    def test_selective_reset_clears_completion(self, behavior_env):
        """Resetting env 0 only clears its goal completion, leaving env 1 untouched."""
        env = behavior_env

        # Complete env 0
        cans, ashcan = _goal_objects(env, 0)
        for i, can in enumerate(cans):
            assert _place_inside(can, ashcan), f"failed to place can {i} inside ashcan"
        _, _, terminateds, _, _ = env.step(_zero_actions(env))
        assert terminateds[0], "env 0 should be complete before reset"

        # Reset only env 0; step once so kinematic caches refresh against the restored poses
        env.reset(env_indices=th.tensor([0]))
        _, _, terminateds2, _, _ = env.step(_zero_actions(env))

        assert not terminateds2[0], "env 0 still terminating after reset"
        assert abs(env.task.get_potential(env, 0)) < 1e-6, "potential not restored to baseline after reset"
        cans0, ashcan0 = _goal_objects(env, 0)
        assert not any(can.states[Inside].get_value(ashcan0) for can in cans0), "cans still inside ashcan after reset"


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
        """BehaviorTask constructs, resets, and steps without a presampled pose.

        (Pose validity/unit-quaternion checks are covered by
        TestBehaviorTaskLogic::test_presampled_robot_pose; this variant only needs
        to prove the use_presampled_robot_pose=False path doesn't crash.)
        """
        # Tear down the module-scope `behavior_env` (built with use_presampled_robot_pose=True)
        # before constructing the variant. _init_macros only stops the sim; without an explicit
        # clear, the new env's object-state machinery references prims from the previous scenes
        # and crashes during play() with `'NoneType' object has no attribute 'state_updated'`.
        og.clear()
        env = setup_behavior_environment(use_presampled_robot_pose=False)
        env.reset()
        env.step(_zero_actions(env))

        for env_idx in range(NUM_ENVS):
            pos, ori = env.scenes[env_idx].robots[0].get_position_orientation(frame="scene")
            assert th.isfinite(pos).all(), f"Robot {env_idx} position has non-finite values"
            assert th.isfinite(ori).all(), f"Robot {env_idx} orientation has non-finite values"
            print(f"  env {env_idx}: robot pos={pos}, ori={ori}")

        og.clear()
