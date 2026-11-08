"""
test_multiple_envs_point_tasks.py
=================================
Goal-based point tasks (PointNavigationTask, PointReachingTask) under num_envs > 1: stepping works
and each env exposes its own goal position.

Replaces test_multiple_envs_point_navigation.py and test_multiple_envs_point_reaching.py, which were
byte-identical apart from the task-type string. The per-env tensor-shape contract they also each
duplicated now lives in test_multiple_envs_task_tensors.py.
"""

import pytest
import torch as th


POINT_TASK_TYPES = ["PointNavigationTask", "PointReachingTask"]
NUM_ENVS = 2


@pytest.fixture(scope="module", params=POINT_TASK_TYPES)
def point_task_type(request):
    """Task type under test, module-scoped so pytest groups tests per task type."""
    return request.param


class TestPointTasks:
    """Goal-based point-task behavior across robots and task types."""

    def test_multi_step_and_goal_shape(self, make_multi_env, point_task_type, robot_model):
        """Run a few steps and verify goal positions exist per env."""
        env = make_multi_env(NUM_ENVS, robot=robot_model, task_type=point_task_type)

        for step_i in range(3):
            actions = th.stack(
                [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(NUM_ENVS)]
            )
            obs_list, rewards, terminateds, truncateds, infos = env.step(actions)
            print(f"  [{point_task_type}/{robot_model}] step {step_i + 1}/3: rewards={rewards}")

        assert rewards.shape == (NUM_ENVS,)
        assert terminateds.shape == (NUM_ENVS,)

        for env_idx in range(NUM_ENVS):
            goal = env.task.get_goal_pos(env_idx)
            print(f"  env {env_idx} goal_pos={goal}")
            assert goal.shape == (3,)
