"""
test_multiple_envs_grasp.py
===========================
GraspTask-specific behavior under num_envs > 1.

The per-env tensor-shape contract this file also used to cover (a byte-identical copy of
TestTaskTensors) now lives once in test_multiple_envs_task_tensors.py.
"""

NUM_ENVS = 2


class TestGraspTask:
    """GraspTask-specific tests across robots."""

    def test_grasp_task_precached_reset(self, make_multi_env, robot_model):
        """GraspTask resets correctly using precached_reset_pose_path."""
        env = make_multi_env(NUM_ENVS, robot=robot_model, task_type="GraspTask")

        # Verify reset succeeded and objects exist
        for env_idx in range(NUM_ENVS):
            obj = env.scenes[env_idx].object_registry("name", "grasp_obj")
            assert obj is not None, f"grasp_obj not found in scene {env_idx}"
            robot_obj = env.scenes[env_idx].robots[0]
            pos = robot_obj.get_position_orientation(frame="scene")[0]
            print(f"  scene {env_idx}: robot at {pos}")

        # Reset again to verify repeated resets work
        env.reset()
