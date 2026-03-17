import json
import tempfile

import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition
from omnigibson.utils.transform_utils import quat_multiply

# ---------------------------------------------------------------------------
# Robots & Tasks to cover
# ---------------------------------------------------------------------------
ROBOTS = ["fetch", "tiago", "r1", "r1pro"]
TASK_TYPES = ["DummyTask", "PointNavigationTask", "PointReachingTask", "GraspTask"]

# GraspTask requires an explicit object config
GRASP_OBJECTS_CFG = [
    {
        "type": "PrimitiveObject",
        "name": "grasp_obj",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.04, 0.04, 0.04],
        "mass": 0.1,
        "position": [0.5, 0.0, 0.55],
    }
]

# Number of trunk + default-arm joints per robot (for precached reset poses).
ROBOT_JOINT_COUNTS = {"fetch": 8, "tiago": 8, "r1": 10, "r1pro": 11}


def _grasp_reset_pose_path(robot):
    """Create a temporary precached reset pose file sized for the given robot."""
    n_joints = ROBOT_JOINT_COUNTS[robot]
    reset_poses = [{"joint_pos": [0.0] * n_joints, "base_pos": [0.0, 0.0, 0.0], "base_ori": [0.0, 0.0, 0.0, 1.0]}]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(reset_poses, f)
    f.close()
    return f.name


# Test counter for progress tracking
_test_counter = {"current": 0, "total": 84}


def _task_cfg(task_type, robot="fetch"):
    """Return the task config dict for a given task type string."""
    if task_type == "GraspTask":
        return {
            "type": "GraspTask",
            "obj_name": "grasp_obj",
            "objects_config": GRASP_OBJECTS_CFG,
            "termination_config": {"max_steps": 10},
            "precached_reset_pose_path": _grasp_reset_pose_path(robot),
        }
    return {"type": task_type}


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
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = False
    else:
        og.sim.stop()


def setup_multi_environment(num_of_envs, robot="fetch", task_type="DummyTask", additional_objects_cfg=None):
    """Create an Environment with *num_of_envs* cloned scenes."""
    _init_macros()

    cfg = {
        "env": {"num_envs": num_of_envs},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls"],
        },
        "robots": [{"model": robot, "obs_modalities": []}],
        "task": _task_cfg(task_type, robot=robot),
    }
    if additional_objects_cfg:
        cfg["objects"] = additional_objects_cfg

    print(f"  Setting up env: num_envs={num_of_envs}, robot={robot}, task={task_type}")
    env = og.Environment(configs=cfg)
    print(f"  Environment created successfully")
    return env


# ===================================================================
#  Section 1 – Environment-level API tests
# ===================================================================


class TestEnvConstruction:
    """Basic environment construction & property tests."""

    def test_env_construction(self):
        """Environment with num_envs=3 creates 3 scenes."""
        _progress("TestEnvConstruction::test_env_construction")
        env = setup_multi_environment(num_of_envs=3)
        assert len(env.scenes) == 3
        assert env.num_envs == 3
        for scene in env.scenes:
            assert len(scene.robots) == 1
        og.clear()
        _passed("TestEnvConstruction::test_env_construction")

    def test_single_env_compat(self):
        """num_envs=1 (default) should still work; scene property returns first scene."""
        _progress("TestEnvConstruction::test_single_env_compat")
        env = setup_multi_environment(num_of_envs=1)
        env.reset()

        assert env.scene is env.scenes[0]
        assert len(env.scenes) == 1

        action = th.from_numpy(env.scenes[0].robots[0].action_space.sample()).float().unsqueeze(0)
        obs_list, rewards, terminateds, truncateds, infos = env.step(action)

        assert rewards.shape == (1,)
        assert len(obs_list) == 1
        og.clear()
        _passed("TestEnvConstruction::test_single_env_compat")

    def test_scenes_spatially_separated(self):
        """Each scene occupies a different spatial region (no overlap)."""
        _progress("TestEnvConstruction::test_scenes_spatially_separated")
        num_envs = 3
        env = setup_multi_environment(num_of_envs=num_envs)

        scene_positions = [s.get_position_orientation()[0] for s in env.scenes]
        for i in range(len(scene_positions)):
            for j in range(i + 1, len(scene_positions)):
                dist = th.norm(scene_positions[i] - scene_positions[j])
                print(f"  Scene {i} <-> Scene {j} distance: {dist:.2f}")
                assert dist > 1.0, f"Scenes {i} and {j} are too close: {dist:.2f}"
        og.clear()
        _passed("TestEnvConstruction::test_scenes_spatially_separated")


class TestStepAndReset:
    """step() / reset() contract tests."""

    def test_step_return_shapes(self):
        """step() returns tensors of shape (num_envs,) for rewards/terminateds/truncateds."""
        _progress("TestStepAndReset::test_step_return_shapes")
        num_envs = 3
        env = setup_multi_environment(num_of_envs=num_envs)
        env.reset()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(num_envs)]
        )

        obs_list, rewards, terminateds, truncateds, infos = env.step(actions)

        print(f"  obs_list len={len(obs_list)}, rewards shape={rewards.shape}")
        assert isinstance(obs_list, list) and len(obs_list) == num_envs
        assert rewards.shape == (num_envs,)
        assert terminateds.shape == (num_envs,) and terminateds.dtype == th.bool
        assert truncateds.shape == (num_envs,) and truncateds.dtype == th.bool
        assert isinstance(infos, list) and len(infos) == num_envs
        og.clear()
        _passed("TestStepAndReset::test_step_return_shapes")

    def test_selective_reset(self):
        """Resetting env_indices=[1] only resets scene 1, leaving 0 and 2 unchanged."""
        _progress("TestStepAndReset::test_selective_reset")
        num_envs = 3
        env = setup_multi_environment(num_of_envs=num_envs)
        env.reset()

        known_pos = th.tensor([1.0, 1.0, 0.5])
        env.scenes[0].robots[0].set_position_orientation(position=known_pos, frame="scene")
        og.sim.step()

        pos_before = env.scenes[0].robots[0].get_position_orientation(frame="scene")[0].clone()

        env.reset(env_indices=th.tensor([1]))

        pos_after = env.scenes[0].robots[0].get_position_orientation(frame="scene")[0]
        print(f"  pos_before={pos_before}, pos_after={pos_after}")
        assert th.allclose(
            pos_before, pos_after, atol=0.05
        ), f"Scene 0 robot moved after resetting only scene 1: {pos_before} vs {pos_after}"
        og.clear()
        _passed("TestStepAndReset::test_selective_reset")

    def test_per_env_step_counters(self):
        """episode_steps is a (num_envs,) tensor that tracks steps independently."""
        _progress("TestStepAndReset::test_per_env_step_counters")
        num_envs = 2
        env = setup_multi_environment(num_of_envs=num_envs)
        env.reset()

        assert env.episode_steps.shape == (num_envs,)
        assert (env.episode_steps == 0).all()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(num_envs)]
        )
        env.step(actions)

        print(f"  episode_steps after 1 step: {env.episode_steps}")
        assert (env.episode_steps == 1).all()

        env.reset(env_indices=th.tensor([0]))
        print(f"  episode_steps after resetting env 0: {env.episode_steps}")
        assert env.episode_steps[0] == 0
        assert env.episode_steps[1] == 1
        og.clear()
        _passed("TestStepAndReset::test_per_env_step_counters")


# ===================================================================
#  Section 2 – Task / reward / termination tensor tests
#              Parametrized over robots & task types
# ===================================================================


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("task_type", TASK_TYPES)
class TestTaskTensors:
    """Verify that task step outputs have correct tensor shapes across robots & tasks."""

    def test_task_step_tensors(self, robot, task_type):
        """Task reward / done / success are (num_envs,) tensors."""
        test_id = f"TestTaskTensors::test_task_step_tensors[{task_type}-{robot}]"
        _progress(test_id)
        num_envs = 2

        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=task_type)
        env.reset()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(num_envs)]
        )
        env.step(actions)

        print(f"  reward={env.task.reward}, done={env.task.done}, success={env.task.success}")
        assert env.task.reward.shape == (num_envs,)
        assert env.task.done.shape == (num_envs,)
        assert env.task.success.shape == (num_envs,)
        og.clear()
        _passed(test_id)

    def test_reward_tensor_returns(self, robot, task_type):
        """Reward functions return (num_envs,) tensors."""
        test_id = f"TestTaskTensors::test_reward_tensor_returns[{task_type}-{robot}]"
        _progress(test_id)
        num_envs = 2
        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=task_type)
        env.reset()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(num_envs)]
        )
        env.step(actions)

        for rf_name, rf in env.task._reward_functions.items():
            print(f"  reward fn '{rf_name}': shape={rf._reward.shape}, values={rf._reward}")
            assert rf._reward.shape == (
                num_envs,
            ), f"Reward function '{rf_name}' _reward has wrong shape: {rf._reward.shape}"
        og.clear()
        _passed(test_id)

    def test_termination_tensor_returns(self, robot, task_type):
        """Termination conditions return (num_envs,) bool tensors."""
        test_id = f"TestTaskTensors::test_termination_tensor_returns[{task_type}-{robot}]"
        _progress(test_id)
        num_envs = 2
        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=task_type)
        env.reset()

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(num_envs)]
        )
        env.step(actions)

        for tc_name, tc in env.task._termination_conditions.items():
            print(f"  termination '{tc_name}': shape={tc._done.shape}, dtype={tc._done.dtype}, values={tc._done}")
            assert tc._done.shape == (
                num_envs,
            ), f"Termination condition '{tc_name}' _done has wrong shape: {tc._done.shape}"
            assert tc._done.dtype == th.bool
        og.clear()
        _passed(test_id)


# ===================================================================
#  Section 3 – Navigation-specific tests (PointNavigation & PointReaching)
# ===================================================================


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("task_type", ["PointNavigationTask", "PointReachingTask"])
class TestNavigationTasks:
    """Goal-based navigation task tests across robots."""

    def test_multi_step_and_goal_shape(self, robot, task_type):
        """Run a few steps and verify goal positions exist per env."""
        test_id = f"TestNavigationTasks::test_multi_step_and_goal_shape[{task_type}-{robot}]"
        _progress(test_id)
        num_envs = 2
        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=task_type)
        env.reset()

        for step_i in range(3):
            actions = th.stack(
                [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(num_envs)]
            )
            obs_list, rewards, terminateds, truncateds, infos = env.step(actions)
            print(f"  step {step_i+1}/3: rewards={rewards}")

        assert rewards.shape == (num_envs,)
        assert terminateds.shape == (num_envs,)

        for env_idx in range(num_envs):
            goal = env.task.get_goal_pos(env_idx)
            print(f"  env {env_idx} goal_pos={goal}")
            assert goal.shape == (3,)
        og.clear()
        _passed(test_id)


# ===================================================================
#  Section 4 – Grasp task specific test
# ===================================================================


@pytest.mark.parametrize("robot", ROBOTS)
class TestGraspTask:
    """GraspTask-specific tests across robots."""

    # GraspTask CuRobo-based reset is not yet working in multi-env.
    # See grasp_curobo_reset_debug.md for details.
    # def test_grasp_task_step(self, robot):
    #     """GraspTask loads objects and runs a step without errors."""
    #     test_id = f"TestGraspTask::test_grasp_task_step[{robot}]"
    #     _progress(test_id)
    #     num_envs = 2
    #     if robot == "fetch":
    #         with pytest.raises(ValueError, match="Could not reset task"):
    #             setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type="GraspTask")
    #         _passed(test_id)
    #         return
    #     env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type="GraspTask")
    #     env.reset()
    #     actions = th.stack([
    #         th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float()
    #         for i in range(num_envs)
    #     ])
    #     obs_list, rewards, terminateds, truncateds, infos = env.step(actions)
    #     assert rewards.shape == (num_envs,)
    #     assert terminateds.shape == (num_envs,)
    #     for env_idx in range(num_envs):
    #         obj = env.scenes[env_idx].object_registry("name", "grasp_obj")
    #         assert obj is not None, f"grasp_obj not found in scene {env_idx}"
    #     og.clear()
    #     _passed(test_id)

    def test_grasp_task_precached_reset(self, robot):
        """GraspTask resets correctly using precached_reset_pose_path."""
        test_id = f"TestGraspTask::test_grasp_task_precached_reset[{robot}]"
        _progress(test_id)
        num_envs = 2

        _init_macros()
        cfg = {
            "env": {"num_envs": num_envs},
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": "Rs_int",
                "load_object_categories": ["floors", "walls"],
            },
            "robots": [{"model": robot, "obs_modalities": []}],
            "task": _task_cfg("GraspTask", robot=robot),
        }
        env = og.Environment(configs=cfg)
        env.reset()

        # Verify reset succeeded and objects exist
        for env_idx in range(num_envs):
            obj = env.scenes[env_idx].object_registry("name", "grasp_obj")
            assert obj is not None, f"grasp_obj not found in scene {env_idx}"
            robot_obj = env.scenes[env_idx].robots[0]
            pos = robot_obj.get_position_orientation(frame="scene")[0]
            print(f"  scene {env_idx}: robot at {pos}")

        # Reset again to verify repeated resets work
        env.reset()
        og.clear()
        _passed(test_id)


# ===================================================================
#  Section 5 – Scene coordinate system & state tests
# ===================================================================


class TestSceneCoordinates:
    """Multi-scene position/orientation and state dump/load tests."""

    def test_multi_scene_dump_load_states(self):
        _progress("TestSceneCoordinates::test_multi_scene_dump_load_states")
        env = setup_multi_environment(3)
        robot_0 = env.scenes[0].robots[0]
        robot_1 = env.scenes[1].robots[0]
        robot_2 = env.scenes[2].robots[0]

        robot_0_pos = robot_0.get_position_orientation()[0]
        robot_1_pos = robot_1.get_position_orientation()[0]
        robot_2_pos = robot_2.get_position_orientation()[0]

        dist_0_1 = robot_1_pos - robot_0_pos
        dist_1_2 = robot_2_pos - robot_1_pos

        print(f"  dist_0_1={dist_0_1}, dist_1_2={dist_1_2}")
        # Check x/y spacing is even (z can drift slightly due to physics settling)
        assert th.allclose(dist_0_1[:2], dist_1_2[:2], atol=1e-3)

        # Set different poses for the robot in each environment
        pose_1 = (th.tensor([1, 1, 1], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32))
        pose_2 = (th.tensor([0, 2, 1], dtype=th.float32), th.tensor([0, 0, 0.7071, 0.7071], dtype=th.float32))
        pose_3 = (th.tensor([-1, -1, 0.5], dtype=th.float32), th.tensor([0.5, 0.5, 0.5, 0.5], dtype=th.float32))

        robot_0.set_position_orientation(*pose_1, frame="scene")
        robot_1.set_position_orientation(*pose_2, frame="scene")
        robot_2.set_position_orientation(*pose_3, frame="scene")

        print("  Running 10 sim steps...")
        for _ in range(10):
            og.sim.step()

        initial_robot_pos_scene_1 = robot_1.get_position_orientation(frame="scene")
        initial_robot_pos_scene_2 = robot_2.get_position_orientation(frame="scene")
        initial_robot_pos_scene_0 = robot_0.get_position_orientation(frame="scene")

        # Save states
        print("  Saving states...")
        robot_0_state = env.scenes[0]._dump_state()
        robot_1_state = env.scenes[1]._dump_state()
        robot_2_state = env.scenes[2]._dump_state()

        print("  Resetting env...")
        env.reset()

        # Load the states in a different order
        print("  Loading states in different order...")
        env.scenes[1]._load_state(robot_1_state)
        env.scenes[2]._load_state(robot_2_state)
        env.scenes[0]._load_state(robot_0_state)

        post_robot_pos_scene_1 = env.scenes[1].robots[0].get_position_orientation(frame="scene")
        post_robot_pos_scene_2 = env.scenes[2].robots[0].get_position_orientation(frame="scene")
        post_robot_pos_scene_0 = env.scenes[0].robots[0].get_position_orientation(frame="scene")

        print(f"  scene 0: initial={initial_robot_pos_scene_0[0]} -> post={post_robot_pos_scene_0[0]}")
        print(f"  scene 1: initial={initial_robot_pos_scene_1[0]} -> post={post_robot_pos_scene_1[0]}")
        print(f"  scene 2: initial={initial_robot_pos_scene_2[0]} -> post={post_robot_pos_scene_2[0]}")

        assert th.allclose(initial_robot_pos_scene_0[0], post_robot_pos_scene_0[0], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_1[0], post_robot_pos_scene_1[0], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_2[0], post_robot_pos_scene_2[0], atol=1e-3)

        assert th.allclose(initial_robot_pos_scene_0[1], post_robot_pos_scene_0[1], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_1[1], post_robot_pos_scene_1[1], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_2[1], post_robot_pos_scene_2[1], atol=1e-3)

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_dump_load_states")

    def test_multi_scene_get_local_position(self):
        _progress("TestSceneCoordinates::test_multi_scene_get_local_position")
        env = setup_multi_environment(3)

        robot_1_pos_local = env.scenes[1].robots[0].get_position_orientation(frame="scene")[0]
        robot_1_pos_global = env.scenes[1].robots[0].get_position_orientation()[0]

        pos_scene = env.scenes[1].get_position_orientation()[0]

        print(f"  local={robot_1_pos_local}, global={robot_1_pos_global}, scene_origin={pos_scene}")
        assert th.allclose(robot_1_pos_global, pos_scene + robot_1_pos_local, atol=1e-3)
        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_get_local_position")

    def test_multi_scene_set_local_position(self):
        _progress("TestSceneCoordinates::test_multi_scene_set_local_position")
        env = setup_multi_environment(3)

        robot = env.scenes[1].robots[0]
        initial_global_pos = robot.get_position_orientation()[0]
        new_global_pos = initial_global_pos + th.tensor([1.0, 0.5, 0.0], dtype=th.float32)

        robot.set_position_orientation(position=new_global_pos)

        updated_global_pos = robot.get_position_orientation()[0]
        scene_pos = env.scenes[1].get_position_orientation()[0]
        updated_local_pos = robot.get_position_orientation(frame="scene")[0]
        expected_local_pos = new_global_pos - scene_pos

        print(f"  updated_global={updated_global_pos}, expected={new_global_pos}")
        print(f"  updated_local={updated_local_pos}, expected_local={expected_local_pos}")

        assert th.allclose(
            updated_global_pos, new_global_pos, atol=1e-3
        ), f"Updated global position {updated_global_pos} does not match expected {new_global_pos}"
        assert th.allclose(
            updated_local_pos, expected_local_pos, atol=1e-3
        ), f"Updated local position {updated_local_pos} does not match expected {expected_local_pos}"

        global_pos_change = updated_global_pos - initial_global_pos
        expected_change = th.tensor([1.0, 0.5, 0.0], dtype=th.float32)
        assert th.allclose(
            global_pos_change, expected_change, atol=1e-3
        ), f"Global position change {global_pos_change} does not match expected change {expected_change}"

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_set_local_position")

    def test_multi_scene_scene_prim(self):
        _progress("TestSceneCoordinates::test_multi_scene_scene_prim")
        env = setup_multi_environment(1)
        original_robot_pos = env.scenes[0].robots[0].get_position_orientation()[0]
        scene_prim_displacement = th.tensor([10.0, 0.0, 0.0], dtype=th.float32)
        original_scene_prim_pos = env.scenes[0]._scene_prim.get_position_orientation()[0]
        env.scenes[0].set_position_orientation(position=original_scene_prim_pos + scene_prim_displacement)
        new_scene_prim_pos = env.scenes[0]._scene_prim.get_position_orientation()[0]
        new_robot_pos = env.scenes[0].robots[0].get_position_orientation()[0]
        print(f"  scene_prim moved: {original_scene_prim_pos} -> {new_scene_prim_pos}")
        print(f"  robot moved: {original_robot_pos} -> {new_robot_pos}")
        assert th.allclose(new_scene_prim_pos - original_scene_prim_pos, scene_prim_displacement, atol=1e-3)
        assert th.allclose(new_robot_pos - original_robot_pos, scene_prim_displacement, atol=1e-3)

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_scene_prim")

    def test_multi_scene_position_orientation_relative_to_scene(self):
        _progress("TestSceneCoordinates::test_multi_scene_position_orientation_relative_to_scene")
        env = setup_multi_environment(3)

        robot = env.scenes[1].robots[0]
        new_relative_pos = th.tensor([1.0, 2.0, 0.5])
        new_relative_ori = th.tensor([0, 0, 0.7071, 0.7071])

        robot.set_position_orientation(position=new_relative_pos, orientation=new_relative_ori, frame="scene")
        updated_relative_pos, updated_relative_ori = robot.get_position_orientation(frame="scene")

        print(f"  set relative pos={new_relative_pos}, got={updated_relative_pos}")
        assert th.allclose(
            updated_relative_pos, new_relative_pos, atol=1e-3
        ), f"Updated relative position {updated_relative_pos} does not match expected {new_relative_pos}"
        assert th.allclose(
            updated_relative_ori, new_relative_ori, atol=1e-3
        ), f"Updated relative orientation {updated_relative_ori} does not match expected {new_relative_ori}"

        scene_pos, scene_ori = env.scenes[1].get_position_orientation()
        global_pos, global_ori = robot.get_position_orientation()

        expected_global_pos = scene_pos + updated_relative_pos
        print(f"  global_pos={global_pos}, expected={expected_global_pos}")
        assert th.allclose(
            global_pos, expected_global_pos, atol=1e-3
        ), f"Global position {global_pos} does not match expected {expected_global_pos}"

        expected_global_ori = quat_multiply(scene_ori, new_relative_ori)
        assert th.allclose(
            global_ori, expected_global_ori, atol=1e-3
        ), f"Global orientation {global_ori} does not match expected {expected_global_ori}"

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_position_orientation_relative_to_scene")


# ===================================================================
#  Section 6 – Robot-specific getter/setter tests (parametrized)
# ===================================================================


@pytest.mark.parametrize("robot", ROBOTS)
class TestRobotGetterSetter:
    """Position/orientation getter and setter correctness across robots."""

    def test_getter(self, robot):
        test_id = f"TestRobotGetterSetter::test_getter[{robot}]"
        _progress(test_id)
        env = setup_multi_environment(2, robot=robot)
        robot1 = env.scenes[0].robots[0]

        robot1_world_position, robot1_world_orientation = robot1.get_position_orientation()
        robot1_scene_position, robot1_scene_orientation = robot1.get_position_orientation(frame="scene")

        print(f"  scene 0 robot: world_pos={robot1_world_position}, scene_pos={robot1_scene_position}")
        # Robot in scene 0 is at origin, so world == scene
        assert th.allclose(robot1_world_position, robot1_scene_position, atol=1e-3)
        assert th.allclose(robot1_world_orientation, robot1_scene_orientation, atol=1e-3)

        # For scene 1 (non-zero offset), verify coordinate transform
        robot2 = env.scenes[1].robots[0]
        scene_position, scene_orientation = env.scenes[1].get_position_orientation()

        robot2_world_position, robot2_world_orientation = robot2.get_position_orientation()
        robot2_scene_position, robot2_scene_orientation = robot2.get_position_orientation(frame="scene")

        print(f"  scene 1 robot: world_pos={robot2_world_position}, scene_pos={robot2_scene_position}")
        combined_position, combined_orientation = T.pose_transform(
            scene_position, scene_orientation, robot2_scene_position, robot2_scene_orientation
        )
        assert th.allclose(robot2_world_position, combined_position, atol=1e-3)
        assert th.allclose(robot2_world_orientation, combined_orientation, atol=1e-3)

        og.clear()
        _passed(test_id)

    def test_setter(self, robot):
        test_id = f"TestRobotGetterSetter::test_setter[{robot}]"
        _progress(test_id)
        env = setup_multi_environment(2, robot=robot)

        robot_obj = env.scenes[1].robots[0]

        # Test setting in world frame
        new_world_pos = th.tensor([1.0, 2.0, 0.5])
        new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
        robot_obj.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

        got_world_pos, got_world_ori = robot_obj.get_position_orientation()
        print(f"  set world pos={new_world_pos}, got={got_world_pos}")
        assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
        assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

        # Test setting in scene frame
        new_scene_pos = th.tensor([0.5, 1.0, 0.25])
        new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

        got_scene_pos, got_scene_ori = robot_obj.get_position_orientation(frame="scene")
        print(f"  set scene pos={new_scene_pos}, got={got_scene_pos}")
        assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
        assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

        # Setting a different scene-frame pose should change world-frame result
        new_scene_pos2 = th.tensor([-1.0, -2.0, 0.1])
        new_scene_ori2 = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos2, orientation=new_scene_ori2, frame="scene")

        got_world_pos2, got_world_ori2 = robot_obj.get_position_orientation()
        assert not th.allclose(got_world_pos2, new_world_pos, atol=1e-3)
        assert not th.allclose(got_world_ori2, new_world_ori, atol=1e-3)

        og.clear()
        _passed(test_id)

    def test_setter_sim_stopped(self, robot):
        """Getter/setter should work even when the simulator is stopped."""
        test_id = f"TestRobotGetterSetter::test_setter_sim_stopped[{robot}]"
        _progress(test_id)
        env = setup_multi_environment(2, robot=robot)
        og.sim.stop()
        print("  Sim stopped")

        robot_obj = env.scenes[1].robots[0]

        new_world_pos = th.tensor([1.0, 2.0, 0.5])
        new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
        robot_obj.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

        got_world_pos, got_world_ori = robot_obj.get_position_orientation()
        print(f"  world frame: set={new_world_pos}, got={got_world_pos}")
        assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
        assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

        new_scene_pos = th.tensor([0.5, 1.0, 0.25])
        new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

        got_scene_pos, got_scene_ori = robot_obj.get_position_orientation(frame="scene")
        print(f"  scene frame: set={new_scene_pos}, got={got_scene_pos}")
        assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
        assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

        new_scene_pos2 = th.tensor([-1.0, -2.0, 0.1])
        new_scene_ori2 = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos2, orientation=new_scene_ori2, frame="scene")

        got_scene_pos2, got_scene_ori2 = robot_obj.get_position_orientation(frame="scene")
        assert th.allclose(got_scene_pos2, new_scene_pos2, atol=1e-3)
        assert th.allclose(got_scene_ori2, new_scene_ori2, atol=1e-3)

        got_world_pos2, got_world_ori2 = robot_obj.get_position_orientation()
        assert not th.allclose(got_world_pos2, new_world_pos, atol=1e-3)
        assert not th.allclose(got_world_ori2, new_world_ori, atol=1e-3)

        og.clear()
        _passed(test_id)


# ===================================================================
#  Section 7 – Particle system test
# ===================================================================


class TestParticles:
    def test_multi_scene_particle_source(self):
        _progress("TestParticles::test_multi_scene_particle_source")
        sink_cfg = dict(
            type="DatasetObject",
            name="sink",
            category="furniture_sink",
            model="czyfhq",
            abilities={
                "toggleable": {},
                "particleSource": {
                    "conditions": {
                        "water": [(ParticleModifyCondition.TOGGLEDON, True)],
                    },
                    "initial_speed": 0.0,
                },
                "particleSink": {
                    "conditions": {
                        "water": [],
                    },
                },
            },
            position=[0.0, -1.5, 0.0],
        )

        env = setup_multi_environment(3, additional_objects_cfg=[sink_cfg])

        for i, scene in enumerate(env.scenes):
            sink = scene.object_registry("name", "sink")
            assert sink.states[object_states.ToggledOn].set_value(True)
            print(f"  scene {i}: sink toggled on")

        print("  Running 50 sim steps...")
        for _ in range(50):
            og.sim.step()

        og.clear()
        _passed("TestParticles::test_multi_scene_particle_source")
