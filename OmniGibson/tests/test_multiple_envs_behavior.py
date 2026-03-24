"""
test_multiple_envs_behavior.py
==============================
Integration tests for BehaviorTask with R1Pro in a multi-environment setup.

Covers:
  Section 1 – Environment construction
  Section 2 – Step and reset
  Section 3 – Task tensors (PotentialReward, Timeout, PredicateGoal)
  Section 4 – Scene coordinates
  Section 5 – Robot getter/setter
  Section 6 – BehaviorTask-specific logic (object scope, goal conditions,
              potential reward, task obs, robot pose, goal status, etc.)

All tests use R1Pro only.  BehaviorTask tests use the ``picking_up_trash``
activity on ``house_double_floor_lower``.
"""

import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.transform_utils import quat_multiply

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"
SCENE_MODEL = "house_double_floor_lower"

# Test counter for progress tracking
_test_counter = {"current": 0, "total": 23}


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
    print(f"  BehaviorTask environment created successfully")
    return env


# ===================================================================
#  Section 1 – Environment construction
# ===================================================================


class TestBehaviorEnvConstruction:
    """Basic BehaviorTask environment construction tests."""

    def test_behavior_env_construction(self):
        """BehaviorTask env with num_envs=2 creates correct structure."""
        _progress("TestBehaviorEnvConstruction::test_behavior_env_construction")
        env = setup_behavior_environment(num_envs=NUM_ENVS)

        assert len(env.scenes) == NUM_ENVS
        assert env.num_envs == NUM_ENVS
        for scene in env.scenes:
            assert len(scene.robots) == 1
        assert isinstance(env.task, BehaviorTask)
        assert env.task.activity_name == ACTIVITY_NAME

        og.clear()
        _passed("TestBehaviorEnvConstruction::test_behavior_env_construction")

    def test_single_env_behavior(self):
        """BehaviorTask with num_envs=1 works; scene property returns first scene."""
        _progress("TestBehaviorEnvConstruction::test_single_env_behavior")
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

    def test_scenes_spatially_separated(self):
        """BehaviorTask scenes occupy different spatial regions."""
        _progress("TestBehaviorEnvConstruction::test_scenes_spatially_separated")
        env = setup_behavior_environment(num_envs=NUM_ENVS)

        scene_positions = [s.get_position_orientation()[0] for s in env.scenes]
        for i in range(len(scene_positions)):
            for j in range(i + 1, len(scene_positions)):
                dist = th.norm(scene_positions[i] - scene_positions[j])
                print(f"  Scene {i} <-> Scene {j} distance: {dist:.2f}")
                assert dist > 1.0, f"Scenes {i} and {j} are too close: {dist:.2f}"

        og.clear()
        _passed("TestBehaviorEnvConstruction::test_scenes_spatially_separated")


# ===================================================================
#  Section 2 – Step and reset
# ===================================================================


class TestBehaviorStepAndReset:
    """step() / reset() contract tests with BehaviorTask."""

    def test_step_return_shapes(self):
        """step() returns tensors of shape (num_envs,) for rewards/terminateds/truncateds."""
        _progress("TestBehaviorStepAndReset::test_step_return_shapes")
        env = setup_behavior_environment()
        env.reset()

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

        og.clear()
        _passed("TestBehaviorStepAndReset::test_step_return_shapes")

    def test_selective_reset(self):
        """Resetting env_indices=[1] only resets scene 1, leaving scene 0 unchanged."""
        _progress("TestBehaviorStepAndReset::test_selective_reset")
        env = setup_behavior_environment()
        env.reset()

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

        og.clear()
        _passed("TestBehaviorStepAndReset::test_selective_reset")

    def test_per_env_step_counters(self):
        """episode_steps is a (num_envs,) tensor that tracks steps independently."""
        _progress("TestBehaviorStepAndReset::test_per_env_step_counters")
        env = setup_behavior_environment()
        env.reset()

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

        og.clear()
        _passed("TestBehaviorStepAndReset::test_per_env_step_counters")


# ===================================================================
#  Section 3 – Task / reward / termination tensor tests
# ===================================================================


class TestBehaviorTaskTensors:
    """Verify BehaviorTask step outputs have correct tensor shapes."""

    def test_task_step_tensors(self):
        """Task reward / done / success are (num_envs,) tensors."""
        _progress("TestBehaviorTaskTensors::test_task_step_tensors")
        env = setup_behavior_environment()
        env.reset()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        env.step(actions)

        print(f"  reward={env.task.reward}, done={env.task.done}, success={env.task.success}")
        assert env.task.reward.shape == (NUM_ENVS,)
        assert env.task.done.shape == (NUM_ENVS,)
        assert env.task.success.shape == (NUM_ENVS,)

        og.clear()
        _passed("TestBehaviorTaskTensors::test_task_step_tensors")

    def test_reward_tensor_returns(self):
        """PotentialReward returns (num_envs,) float tensor."""
        _progress("TestBehaviorTaskTensors::test_reward_tensor_returns")
        env = setup_behavior_environment()
        env.reset()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        env.step(actions)

        for rf_name, rf in env.task._reward_functions.items():
            print(f"  reward fn '{rf_name}': shape={rf._reward.shape}, values={rf._reward}")
            assert rf._reward.shape == (
                NUM_ENVS,
            ), f"Reward function '{rf_name}' _reward has wrong shape: {rf._reward.shape}"

        # BehaviorTask must have the potential reward
        assert "potential" in env.task._reward_functions
        assert isinstance(env.task._reward_functions["potential"], PotentialReward)

        og.clear()
        _passed("TestBehaviorTaskTensors::test_reward_tensor_returns")

    def test_termination_tensor_returns(self):
        """Timeout and PredicateGoal return (num_envs,) bool tensors."""
        _progress("TestBehaviorTaskTensors::test_termination_tensor_returns")
        env = setup_behavior_environment()
        env.reset()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        env.step(actions)

        for tc_name, tc in env.task._termination_conditions.items():
            print(f"  termination '{tc_name}': shape={tc._done.shape}, dtype={tc._done.dtype}, values={tc._done}")
            assert tc._done.shape == (
                NUM_ENVS,
            ), f"Termination condition '{tc_name}' _done has wrong shape: {tc._done.shape}"
            assert tc._done.dtype == th.bool

        # BehaviorTask must have timeout and predicate conditions
        assert "timeout" in env.task._termination_conditions
        assert "predicate" in env.task._termination_conditions
        assert isinstance(env.task._termination_conditions["timeout"], Timeout)
        assert isinstance(env.task._termination_conditions["predicate"], PredicateGoal)

        og.clear()
        _passed("TestBehaviorTaskTensors::test_termination_tensor_returns")


# ===================================================================
#  Section 4 – Scene coordinate system tests
# ===================================================================


class TestBehaviorSceneCoordinates:
    """Multi-scene position/orientation and state dump/load tests with BehaviorTask."""

    def test_dump_load_states(self):
        """Scene state can be saved and restored correctly with BehaviorTask."""
        _progress("TestBehaviorSceneCoordinates::test_dump_load_states")
        env = setup_behavior_environment()

        pose_0 = (th.tensor([1, 1, 1], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32))
        pose_1 = (th.tensor([0, 2, 1], dtype=th.float32), th.tensor([0, 0, 0.7071, 0.7071], dtype=th.float32))

        env.scenes[0].robots[0].set_position_orientation(*pose_0, frame="scene")
        env.scenes[1].robots[0].set_position_orientation(*pose_1, frame="scene")

        print("  Running 10 sim steps...")
        for _ in range(10):
            og.sim.step()

        initial_pos_0 = env.scenes[0].robots[0].get_position_orientation(frame="scene")
        initial_pos_1 = env.scenes[1].robots[0].get_position_orientation(frame="scene")

        print("  Saving states...")
        state_0 = env.scenes[0]._dump_state()
        state_1 = env.scenes[1]._dump_state()

        print("  Resetting env...")
        env.reset()

        print("  Loading states in different order...")
        env.scenes[1]._load_state(state_1)
        env.scenes[0]._load_state(state_0)

        post_pos_0 = env.scenes[0].robots[0].get_position_orientation(frame="scene")
        post_pos_1 = env.scenes[1].robots[0].get_position_orientation(frame="scene")

        print(f"  scene 0: initial={initial_pos_0[0]} -> post={post_pos_0[0]}")
        print(f"  scene 1: initial={initial_pos_1[0]} -> post={post_pos_1[0]}")

        assert th.allclose(initial_pos_0[0], post_pos_0[0], atol=1e-3)
        assert th.allclose(initial_pos_1[0], post_pos_1[0], atol=1e-3)
        assert th.allclose(initial_pos_0[1], post_pos_0[1], atol=1e-3)
        assert th.allclose(initial_pos_1[1], post_pos_1[1], atol=1e-3)

        og.clear()
        _passed("TestBehaviorSceneCoordinates::test_dump_load_states")

    def test_get_local_position(self):
        """Robot scene-frame position + scene origin equals world position."""
        _progress("TestBehaviorSceneCoordinates::test_get_local_position")
        env = setup_behavior_environment()

        robot_local = env.scenes[1].robots[0].get_position_orientation(frame="scene")[0]
        robot_global = env.scenes[1].robots[0].get_position_orientation()[0]
        scene_pos = env.scenes[1].get_position_orientation()[0]

        print(f"  local={robot_local}, global={robot_global}, scene_origin={scene_pos}")
        assert th.allclose(robot_global, scene_pos + robot_local, atol=1e-3)

        og.clear()
        _passed("TestBehaviorSceneCoordinates::test_get_local_position")

    def test_position_orientation_relative_to_scene(self):
        """set/get position in scene frame is consistent."""
        _progress("TestBehaviorSceneCoordinates::test_position_orientation_relative_to_scene")
        env = setup_behavior_environment()

        robot = env.scenes[1].robots[0]
        new_relative_pos = th.tensor([1.0, 2.0, 0.5])
        new_relative_ori = th.tensor([0, 0, 0.7071, 0.7071])

        robot.set_position_orientation(position=new_relative_pos, orientation=new_relative_ori, frame="scene")
        updated_pos, updated_ori = robot.get_position_orientation(frame="scene")

        print(f"  set relative pos={new_relative_pos}, got={updated_pos}")
        assert th.allclose(updated_pos, new_relative_pos, atol=1e-3)
        assert th.allclose(updated_ori, new_relative_ori, atol=1e-3)

        scene_pos, scene_ori = env.scenes[1].get_position_orientation()
        global_pos, global_ori = robot.get_position_orientation()
        expected_global_pos = scene_pos + updated_pos
        assert th.allclose(global_pos, expected_global_pos, atol=1e-3)
        expected_global_ori = quat_multiply(scene_ori, new_relative_ori)
        assert th.allclose(global_ori, expected_global_ori, atol=1e-3)

        og.clear()
        _passed("TestBehaviorSceneCoordinates::test_position_orientation_relative_to_scene")


# ===================================================================
#  Section 5 – Robot getter/setter tests (R1Pro only)
# ===================================================================


class TestBehaviorRobotGetterSetter:
    """Position/orientation getter and setter correctness for R1Pro with BehaviorTask."""

    def test_getter(self):
        """Position getter works in both world and scene frames."""
        _progress("TestBehaviorRobotGetterSetter::test_getter")
        env = setup_behavior_environment()

        # Scene 0 robot: world == scene (scene 0 at origin)
        robot0 = env.scenes[0].robots[0]
        r0_world_pos, r0_world_ori = robot0.get_position_orientation()
        r0_scene_pos, r0_scene_ori = robot0.get_position_orientation(frame="scene")

        print(f"  scene 0 robot: world_pos={r0_world_pos}, scene_pos={r0_scene_pos}")
        assert th.allclose(r0_world_pos, r0_scene_pos, atol=1e-3)
        assert th.allclose(r0_world_ori, r0_scene_ori, atol=1e-3)

        # Scene 1 robot: verify coordinate transform
        robot1 = env.scenes[1].robots[0]
        s1_pos, s1_ori = env.scenes[1].get_position_orientation()
        r1_world_pos, r1_world_ori = robot1.get_position_orientation()
        r1_scene_pos, r1_scene_ori = robot1.get_position_orientation(frame="scene")

        print(f"  scene 1 robot: world_pos={r1_world_pos}, scene_pos={r1_scene_pos}")
        combined_pos, combined_ori = T.pose_transform(s1_pos, s1_ori, r1_scene_pos, r1_scene_ori)
        assert th.allclose(r1_world_pos, combined_pos, atol=1e-3)
        assert th.allclose(r1_world_ori, combined_ori, atol=1e-3)

        og.clear()
        _passed("TestBehaviorRobotGetterSetter::test_getter")

    def test_setter(self):
        """Position setter works in both world and scene frames."""
        _progress("TestBehaviorRobotGetterSetter::test_setter")
        env = setup_behavior_environment()

        robot = env.scenes[1].robots[0]

        # Test world frame setter
        new_world_pos = th.tensor([1.0, 2.0, 0.5])
        new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
        robot.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

        got_world_pos, got_world_ori = robot.get_position_orientation()
        print(f"  set world pos={new_world_pos}, got={got_world_pos}")
        assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
        assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

        # Test scene frame setter
        new_scene_pos = th.tensor([0.5, 1.0, 0.25])
        new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
        robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

        got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
        print(f"  set scene pos={new_scene_pos}, got={got_scene_pos}")
        assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
        assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

        # Setting a different scene-frame pose should change world-frame result
        new_scene_pos2 = th.tensor([-1.0, -2.0, 0.1])
        new_scene_ori2 = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
        robot.set_position_orientation(position=new_scene_pos2, orientation=new_scene_ori2, frame="scene")

        got_world_pos2, _ = robot.get_position_orientation()
        assert not th.allclose(got_world_pos2, new_world_pos, atol=1e-3)

        og.clear()
        _passed("TestBehaviorRobotGetterSetter::test_setter")


# ===================================================================
#  Section 6 – BehaviorTask-specific logic
# ===================================================================


class TestBehaviorTaskLogic:
    """Tests for logic unique to BehaviorTask: BDDL scope, goal conditions,
    potential reward, task observations, presampled robot pose, and goal status."""

    def test_object_scope_per_env(self):
        """Each env has its own independent object scope."""
        _progress("TestBehaviorTaskLogic::test_object_scope_per_env")
        env = setup_behavior_environment()
        env.reset()

        for env_idx in range(NUM_ENVS):
            scope = env.task.object_scope[env_idx]
            assert scope is not None, f"object_scope[{env_idx}] is None"
            assert isinstance(scope, dict)
            assert "agent.n.01_1" in scope, f"agent not found in object_scope[{env_idx}]"
            # Agent entity should be a BDDLEntity wrapping this scene's robot
            agent_entity = scope["agent.n.01_1"]
            assert agent_entity is not None
            print(f"  env {env_idx}: scope has {len(scope)} entries, agent={agent_entity.name}")

        # Scopes are independent dict objects
        assert env.task.object_scope[0] is not env.task.object_scope[1]

        og.clear()
        _passed("TestBehaviorTaskLogic::test_object_scope_per_env")

    def test_goal_conditions_per_env(self):
        """Each env has its own goal conditions and ground goal state options."""
        _progress("TestBehaviorTaskLogic::test_goal_conditions_per_env")
        env = setup_behavior_environment()
        env.reset()

        for env_idx in range(NUM_ENVS):
            goals = env.task.activity_goal_conditions[env_idx]
            assert goals is not None, f"activity_goal_conditions[{env_idx}] is None"
            assert isinstance(goals, list)
            assert len(goals) > 0, f"activity_goal_conditions[{env_idx}] is empty"

            ggo = env.task.ground_goal_state_options[env_idx]
            assert ggo is not None, f"ground_goal_state_options[{env_idx}] is None"
            assert isinstance(ggo, list)
            assert len(ggo) > 0, f"ground_goal_state_options[{env_idx}] is empty"

            print(f"  env {env_idx}: {len(goals)} goal conditions, {len(ggo)} ground goal state options")

        og.clear()
        _passed("TestBehaviorTaskLogic::test_goal_conditions_per_env")

    def test_potential_reward_computation(self):
        """get_potential returns a finite float for each env."""
        _progress("TestBehaviorTaskLogic::test_potential_reward_computation")
        env = setup_behavior_environment()
        env.reset()

        for env_idx in range(NUM_ENVS):
            potential = env.task.get_potential(env, env_idx)
            assert isinstance(potential, float), f"get_potential({env_idx}) returned {type(potential)}, expected float"
            assert not th.isnan(th.tensor(potential)), f"get_potential({env_idx}) returned NaN"
            assert not th.isinf(th.tensor(potential)), f"get_potential({env_idx}) returned Inf"
            # Potential should be non-positive (negative success score)
            assert potential <= 0.0, f"get_potential({env_idx}) = {potential}, expected <= 0"
            print(f"  env {env_idx}: potential={potential:.4f}")

        og.clear()
        _passed("TestBehaviorTaskLogic::test_potential_reward_computation")

    def test_task_obs_per_env(self):
        """Task observations are produced per env and contain expected keys."""
        _progress("TestBehaviorTaskLogic::test_task_obs_per_env")
        env = setup_behavior_environment()
        env.reset()

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

        og.clear()
        _passed("TestBehaviorTaskLogic::test_task_obs_per_env")

    def test_presampled_robot_pose(self):
        """Robot is positioned at a presampled pose after reset (verifies case-insensitive lookup)."""
        _progress("TestBehaviorTaskLogic::test_presampled_robot_pose")
        env = setup_behavior_environment(use_presampled_robot_pose=True)
        env.reset()

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

        og.clear()
        _passed("TestBehaviorTaskLogic::test_presampled_robot_pose")

    def test_no_presampled_robot_pose(self):
        """Robot has valid pose after reset without presampled pose."""
        _progress("TestBehaviorTaskLogic::test_no_presampled_robot_pose")
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

    def test_goal_status_in_info(self):
        """Step info contains goal_status from PredicateGoal termination condition."""
        _progress("TestBehaviorTaskLogic::test_goal_status_in_info")
        env = setup_behavior_environment()
        env.reset()

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

        og.clear()
        _passed("TestBehaviorTaskLogic::test_goal_status_in_info")

    def test_activity_attributes(self):
        """BehaviorTask has correct activity attributes after construction."""
        _progress("TestBehaviorTaskLogic::test_activity_attributes")
        env = setup_behavior_environment()

        assert env.task.activity_name == ACTIVITY_NAME
        assert env.task.activity_definition_id == 0
        assert env.task.activity_instance_id == 0
        assert env.task.activity_conditions is not None
        assert env.task.activity_initial_conditions is not None
        assert "BehaviorTask" in env.task.name
        assert ACTIVITY_NAME in env.task.name

        # Natural language conditions should be available
        assert env.task.activity_natural_language_initial_conditions is not None
        assert env.task.activity_natural_language_goal_conditions is not None

        # Sampler should be initialized per env
        assert isinstance(env.task.sampler, list)
        assert len(env.task.sampler) == NUM_ENVS

        print(f"  task name: {env.task.name}")
        print(f"  NL goal conditions: {env.task.activity_natural_language_goal_conditions}")

        og.clear()
        _passed("TestBehaviorTaskLogic::test_activity_attributes")

    def test_show_instruction(self):
        """show_instruction returns valid instruction data per env."""
        _progress("TestBehaviorTaskLogic::test_show_instruction")
        env = setup_behavior_environment()
        env.reset()

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
        total_conditions = len(env.task.activity_conditions.parsed_goal_conditions)
        expected_idx = (initial_idx + 1) % total_conditions
        assert new_idx == expected_idx, f"iterate_instruction: expected idx {expected_idx}, got {new_idx}"
        print(f"  iterate_instruction: {initial_idx} -> {new_idx} (total={total_conditions})")

        og.clear()
        _passed("TestBehaviorTaskLogic::test_show_instruction")
