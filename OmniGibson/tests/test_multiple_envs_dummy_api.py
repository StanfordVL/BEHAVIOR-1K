import torch as th

import omnigibson as og


TASK_TYPE = "DummyTask"

# Number of envs for the multi-env tests. 3 rather than 2 so that "only env 1 was reset" has a
# third, untouched env to compare against.
NUM_ENVS = 3

# Scene layout constant: scenes are laid out along +X, separated by their AABB extent plus this
# margin (see Scene._load_scene_prim_with_objects / simulator m.SCENE_MARGIN).
SCENE_MARGIN = 10.0


# ===================================================================
#  Environment construction & properties
# ===================================================================


class TestEnvConstruction:
    """Basic environment construction & property tests."""

    def test_env_construction(self, make_multi_env):
        """Environment with num_envs=NUM_ENVS creates that many scenes."""
        env = make_multi_env(NUM_ENVS)
        assert len(env.scenes) == NUM_ENVS
        assert env.num_envs == NUM_ENVS
        for scene in env.scenes:
            assert len(scene.robots) == 1

    def test_scenes_spatially_separated(self, make_multi_env):
        """Scenes tile along +X with at least SCENE_MARGIN between them, and every env places its
        robot identically within its own scene."""
        env = make_multi_env(NUM_ENVS)

        scene_positions = [s.get_position_orientation()[0] for s in env.scenes]

        # Scene 0 anchors the layout at the origin
        assert th.allclose(
            scene_positions[0], th.zeros(3), atol=1e-3
        ), f"Scene 0 should sit at the origin, got {scene_positions[0]}"

        # Scenes are tiled along +X only, in index order, with at least SCENE_MARGIN of clearance
        for i in range(1, NUM_ENVS):
            delta = scene_positions[i] - scene_positions[i - 1]
            print(f"  Scene {i - 1} -> {i} offset: {delta}")
            assert (
                delta[0] >= SCENE_MARGIN
            ), f"Scenes {i - 1} and {i} are only {delta[0]:.2f} apart in X (expected >= {SCENE_MARGIN})"
            assert th.allclose(
                delta[1:], th.zeros(2), atol=1e-3
            ), f"Scenes should only be offset along X, but scene {i} is offset by {delta} from {i - 1}"

        # The layout must not leak into the scenes: each robot sits at the same pose *within* its
        # own scene, and its world pose is exactly that local pose shifted by the scene origin.
        # This is what a per-env placement bug would break -- a pure distance check would not.
        ref_local_pos, ref_local_ori = env.scenes[0].robots[0].get_position_orientation(frame="scene")
        for i in range(NUM_ENVS):
            local_pos, local_ori = env.scenes[i].robots[0].get_position_orientation(frame="scene")
            world_pos, _ = env.scenes[i].robots[0].get_position_orientation()
            print(f"  Scene {i} robot: local={local_pos}, world={world_pos}")
            assert th.allclose(
                local_pos, ref_local_pos, atol=1e-3
            ), f"Env {i} robot scene-local position {local_pos} differs from env 0's {ref_local_pos}"
            assert th.allclose(
                local_ori, ref_local_ori, atol=1e-3
            ), f"Env {i} robot scene-local orientation {local_ori} differs from env 0's {ref_local_ori}"
            assert th.allclose(
                world_pos, scene_positions[i] + local_pos, atol=1e-3
            ), f"Env {i} robot world position {world_pos} is not its scene origin plus its local position"


# ===================================================================
#  step() / reset() contract
# ===================================================================


class TestStepAndReset:
    """step() / reset() contract tests."""

    def test_step_return_shapes(self, make_multi_env):
        """step() returns tensors of shape (num_envs,) for rewards/terminateds/truncateds."""
        env = make_multi_env(NUM_ENVS)

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

    def test_selective_reset(self, make_multi_env):
        """Resetting env_indices=[1] only resets scene 1, leaving 0 and 2 unchanged."""
        env = make_multi_env(NUM_ENVS)

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

    def test_per_env_step_counters(self, make_multi_env):
        """episode_steps is a (num_envs,) tensor that tracks steps independently."""
        env = make_multi_env(NUM_ENVS)

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


# ===================================================================
#  Single-env compatibility
# ===================================================================
#
# Placed last purely as an optimization: `make_multi_env` keeps one env alive and rebuilds whenever the
# requested configuration changes, so putting the lone num_envs=1 test after the NUM_ENVS ones
# avoids rebuilding them. Nothing here depends on ordering for correctness.


class TestSingleEnvCompat:
    """num_envs=1 must keep working through the vectorized code paths."""

    def test_single_env_compat(self, make_multi_env):
        """num_envs=1 (default) should still work; scene property returns first scene."""
        env = make_multi_env(1)

        assert env.scene is env.scenes[0]
        assert len(env.scenes) == 1

        action = th.from_numpy(env.scenes[0].robots[0].action_space.sample()).float().unsqueeze(0)
        obs_list, rewards, terminateds, truncateds, infos = env.step(action)

        assert rewards.shape == (1,)
        assert len(obs_list) == 1
