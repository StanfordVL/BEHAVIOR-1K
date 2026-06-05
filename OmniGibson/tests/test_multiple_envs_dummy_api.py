import pytest
import torch as th

import omnigibson as og

from utils import (
    MULTI_ENV_ROBOTS,
    setup_multi_environment,
)

TASK_TYPE = "DummyTask"


# ===================================================================
#  Section 1 – Environment-level API tests
# ===================================================================


class TestEnvConstruction:
    """Basic environment construction & property tests."""

    def test_env_construction(self):
        """Environment with num_envs=3 creates 3 scenes."""
        env = setup_multi_environment(num_of_envs=3)
        assert len(env.scenes) == 3
        assert env.num_envs == 3
        for scene in env.scenes:
            assert len(scene.robots) == 1
        og.clear()

    def test_single_env_compat(self):
        """num_envs=1 (default) should still work; scene property returns first scene."""
        env = setup_multi_environment(num_of_envs=1)
        env.reset()

        assert env.scene is env.scenes[0]
        assert len(env.scenes) == 1

        action = th.from_numpy(env.scenes[0].robots[0].action_space.sample()).float().unsqueeze(0)
        obs_list, rewards, terminateds, truncateds, infos = env.step(action)

        assert rewards.shape == (1,)
        assert len(obs_list) == 1
        og.clear()

    def test_scenes_spatially_separated(self):
        """Each scene occupies a different spatial region (no overlap)."""
        num_envs = 3
        env = setup_multi_environment(num_of_envs=num_envs)

        scene_positions = [s.get_position_orientation()[0] for s in env.scenes]
        for i in range(len(scene_positions)):
            for j in range(i + 1, len(scene_positions)):
                dist = th.norm(scene_positions[i] - scene_positions[j])
                print(f"  Scene {i} <-> Scene {j} distance: {dist:.2f}")
                assert dist > 1.0, f"Scenes {i} and {j} are too close: {dist:.2f}"
        og.clear()


# ===================================================================
#  Section 2 – step() / reset() contract tests
# ===================================================================


class TestStepAndReset:
    """step() / reset() contract tests."""

    def test_step_return_shapes(self):
        """step() returns tensors of shape (num_envs,) for rewards/terminateds/truncateds."""
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

    def test_selective_reset(self):
        """Resetting env_indices=[1] only resets scene 1, leaving 0 and 2 unchanged."""
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

    def test_per_env_step_counters(self):
        """episode_steps is a (num_envs,) tensor that tracks steps independently."""
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


# ===================================================================
#  Section 3 – Task / reward / termination tensor tests (DummyTask)
# ===================================================================


@pytest.mark.parametrize("robot", MULTI_ENV_ROBOTS)
class TestTaskTensors:
    """Verify that DummyTask step outputs have correct tensor shapes across robots."""

    def test_task_step_tensors(self, robot):
        """Task reward / done / success are (num_envs,) tensors."""
        num_envs = 2

        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=TASK_TYPE)
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

    def test_reward_tensor_returns(self, robot):
        """Reward functions return (num_envs,) tensors."""
        num_envs = 2
        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=TASK_TYPE)
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

    def test_termination_tensor_returns(self, robot):
        """Termination conditions return (num_envs,) bool tensors."""
        num_envs = 2
        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=TASK_TYPE)
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
