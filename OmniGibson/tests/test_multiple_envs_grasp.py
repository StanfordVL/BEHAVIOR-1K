import pytest
import torch as th

import omnigibson as og

from utils import (
    MULTI_ENV_ROBOTS,
    _init_multi_env_macros,
    multi_env_task_cfg,
    setup_multi_environment,
)

TASK_TYPE = "GraspTask"


# ===================================================================
#  Section 1 – Task / reward / termination tensor tests (GraspTask)
# ===================================================================


@pytest.mark.parametrize("robot", MULTI_ENV_ROBOTS)
class TestTaskTensors:
    """Verify that GraspTask step outputs have correct tensor shapes across robots."""

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
#  Section 2 – Grasp task specific tests
# ===================================================================


@pytest.mark.parametrize("robot", MULTI_ENV_ROBOTS)
class TestGraspTask:
    """GraspTask-specific tests across robots."""

    def test_grasp_task_precached_reset(self, robot):
        """GraspTask resets correctly using precached_reset_pose_path."""
        num_envs = 2

        _init_multi_env_macros()
        cfg = {
            "env": {"num_envs": num_envs},
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": "Rs_int",
                "load_object_categories": ["floors", "walls"],
            },
            "robots": [{"model": robot, "obs_modalities": []}],
            "task": multi_env_task_cfg("GraspTask", robot=robot),
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
