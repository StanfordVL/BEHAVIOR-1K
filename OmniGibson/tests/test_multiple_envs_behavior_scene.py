"""
test_multiple_envs_behavior_scene.py
====================================
Multi-environment infrastructure tests using BehaviorTask with R1Pro.

Covers:
  - Task tensors (BehaviorTask's PotentialReward / Timeout / PredicateGoal wiring)
  - Scene state dump/load
  - Robot getter/setter

The env comes from the shared `behavior_env` fixture (conftest.py).

Deliberately NOT covered here: scene placement and frame conversions. Those are Scene/prim logic,
independent of the task, and are tested once in test_multiple_envs_dummy_scene.py. Only
dump/load state is kept on the BehaviorTask side, where the scene has far more objects.
"""

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout

from utils import (
    BEHAVIOR_NUM_ENVS as NUM_ENVS,
)

# ===================================================================
#  Task / reward / termination tensors
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
#  Scene state dump/load
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


# ===================================================================
#  Robot getter/setter (R1Pro only)
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
