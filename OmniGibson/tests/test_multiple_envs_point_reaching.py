import pytest
import torch as th

import omnigibson as og

from utils import (
    MULTI_ENV_ROBOTS,
    setup_multi_environment,
)

TASK_TYPE = "PointReachingTask"


# ===================================================================
#  Section 1 – Task / reward / termination tensor tests (PointReachingTask)
# ===================================================================


@pytest.mark.parametrize("robot", MULTI_ENV_ROBOTS)
class TestTaskTensors:
    """Verify that PointReachingTask step outputs have correct tensor shapes across robots."""

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


# ===================================================================
#  Section 2 – Navigation-specific tests (PointReachingTask)
# ===================================================================


@pytest.mark.parametrize("robot", MULTI_ENV_ROBOTS)
class TestNavigationTasks:
    """Goal-based PointReachingTask tests across robots."""

    def test_multi_step_and_goal_shape(self, robot):
        """Run a few steps and verify goal positions exist per env."""
        num_envs = 2
        env = setup_multi_environment(num_of_envs=num_envs, robot=robot, task_type=TASK_TYPE)
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
