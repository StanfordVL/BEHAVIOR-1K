"""
test_multiple_envs_behavior_infra_scene.py
==========================================
Multi-environment infrastructure tests using BehaviorTask with R1Pro.

Covers:
  Section 3 – Task tensors (PotentialReward, Timeout, PredicateGoal)
  Section 4 – Scene coordinates
  Section 5 – Robot getter/setter
"""

import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.transform_utils import quat_multiply

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"
SCENE_MODEL = "house_double_floor_lower"


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
#  Section 3 – Task / reward / termination tensor tests
# ===================================================================


class TestBehaviorTaskTensors:
    """Verify BehaviorTask step outputs have correct tensor shapes."""

    def test_task_step_tensors(self, behavior_env):
        """Task reward / done / success are (num_envs,) tensors."""
        env = behavior_env

        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
        )
        env.step(actions)

        print(f"  reward={env.task.reward}, done={env.task.done}, success={env.task.success}")
        assert env.task.reward.shape == (NUM_ENVS,)
        assert env.task.done.shape == (NUM_ENVS,)
        assert env.task.success.shape == (NUM_ENVS,)

    def test_reward_tensor_returns(self, behavior_env):
        """PotentialReward returns (num_envs,) float tensor."""
        env = behavior_env

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

    def test_termination_tensor_returns(self, behavior_env):
        """Timeout and PredicateGoal return (num_envs,) bool tensors."""
        env = behavior_env

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


# ===================================================================
#  Section 4 – Scene coordinate system tests
# ===================================================================


class TestBehaviorSceneCoordinates:
    """Multi-scene position/orientation and state dump/load tests with BehaviorTask."""

    def test_dump_load_states(self, behavior_env):
        """Scene state can be saved and restored correctly with BehaviorTask."""
        env = behavior_env

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

    def test_get_local_position(self, behavior_env):
        """Robot scene-frame position + scene origin equals world position."""
        env = behavior_env

        robot_local = env.scenes[1].robots[0].get_position_orientation(frame="scene")[0]
        robot_global = env.scenes[1].robots[0].get_position_orientation()[0]
        scene_pos = env.scenes[1].get_position_orientation()[0]

        print(f"  local={robot_local}, global={robot_global}, scene_origin={scene_pos}")
        assert th.allclose(robot_global, scene_pos + robot_local, atol=1e-3)

    def test_position_orientation_relative_to_scene(self, behavior_env):
        """set/get position in scene frame is consistent."""
        env = behavior_env

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


# ===================================================================
#  Section 5 – Robot getter/setter tests (R1Pro only)
# ===================================================================


class TestBehaviorRobotGetterSetter:
    """Position/orientation getter and setter correctness for R1Pro with BehaviorTask."""

    def test_getter(self, behavior_env):
        """Position getter works in both world and scene frames."""
        env = behavior_env

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

    def test_setter(self, behavior_env):
        """Position setter works in both world and scene frames."""
        env = behavior_env

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
