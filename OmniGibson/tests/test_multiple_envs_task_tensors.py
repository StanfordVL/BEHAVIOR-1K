"""
test_multiple_envs_task_tensors.py
==================================
Per-env tensor contract shared by every task type: with num_envs=N, a task's reward / done /
success, each reward function's `_reward`, and each termination condition's `_done` must all be
(N,)-shaped tensors.

This lives in one place and is parametrized over MULTI_ENV_TASK_TYPES. It used to be copy-pasted
into test_multiple_envs_{dummy_api,grasp,point_navigation,point_reaching}.py -- four byte-identical
copies, each additionally parametrized over all four robots, for 48 test instances of what is a
single property of task_base. The assertions are pure shape checks, so the robot axis added no
coverage; per-robot multi-env construction is covered by
test_multiple_envs_dummy_scene.py::TestRobotGetterSetter and the task-specific suites.
"""

import pytest
import torch as th

from utils import MULTI_ENV_TASK_TYPES

NUM_ENVS = 2


@pytest.fixture(scope="module", params=MULTI_ENV_TASK_TYPES)
def task_type(request):
    """Task type under test. Module-scoped + parametrized so pytest groups all tests for one task
    together, letting `make_multi_env` build a single env per task type instead of one per test."""
    return request.param


def _step_once(env):
    actions = th.stack(
        [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(env.num_envs)]
    )
    env.step(actions)


class TestTaskTensors:
    """Task step outputs must be per-env tensors, for every task type."""

    def test_task_step_tensors(self, make_multi_env, task_type):
        """Task reward / done / success are (num_envs,) tensors."""
        env = make_multi_env(NUM_ENVS, task_type=task_type)
        _step_once(env)

        print(f"  [{task_type}] reward={env.task.reward}, done={env.task.done}, success={env.task.success}")
        assert env.task.reward.shape == (NUM_ENVS,)
        assert env.task.done.shape == (NUM_ENVS,)
        assert env.task.success.shape == (NUM_ENVS,)

    def test_reward_tensor_returns(self, make_multi_env, task_type):
        """Reward functions return (num_envs,) tensors."""
        env = make_multi_env(NUM_ENVS, task_type=task_type)
        _step_once(env)

        for rf_name, rf in env.task._reward_functions.items():
            print(f"  [{task_type}] reward fn '{rf_name}': shape={rf._reward.shape}, values={rf._reward}")
            assert rf._reward.shape == (
                NUM_ENVS,
            ), f"Reward function '{rf_name}' _reward has wrong shape: {rf._reward.shape}"

    def test_termination_tensor_returns(self, make_multi_env, task_type):
        """Termination conditions return (num_envs,) bool tensors."""
        env = make_multi_env(NUM_ENVS, task_type=task_type)
        _step_once(env)

        for tc_name, tc in env.task._termination_conditions.items():
            print(f"  [{task_type}] termination '{tc_name}': shape={tc._done.shape}, values={tc._done}")
            assert tc._done.shape == (
                NUM_ENVS,
            ), f"Termination condition '{tc_name}' _done has wrong shape: {tc._done.shape}"
            assert tc._done.dtype == th.bool
