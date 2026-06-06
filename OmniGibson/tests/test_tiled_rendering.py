"""
Tests for tiled rendering in multi-env (vectorized) environments.

In multi-env mode (num_envs > 1), all per-env robot cameras are batched into a single tiled render
product (TiledVisionSensor), and per-env camera observations are slices of the batched buffer.
"""

import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor

# Deliberately use 5 envs: the tile grid is 3x2 with one empty tile, which exercises the
# non-square indexing math of the reshape_tiled_image warp kernel.
NUM_ENVS = 5

# Distance of the marker cube in front of each env's camera (distinct per env so that tile<->env
# misalignment is detectable via depth). NOTE: the fetch camera looks downward and its optical axis
# hits the floor at ~1.14m, so distances must stay below that for the marker to remain above the floor.
MARKER_DISTANCES = [0.4 + 0.15 * i for i in range(NUM_ENVS)]
MARKER_SIZE = 0.4
# Tile<->env misalignment shifts the measured depth by >= the 0.15 distance spacing, so a tolerance
# below that detects any tile shuffle while absorbing rendering noise
DEPTH_TOLERANCE = 0.07

MARKER_CFG = [
    {
        "type": "PrimitiveObject",
        "name": "marker",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "size": MARKER_SIZE,
        "visual_only": True,
        # Initially hidden far below the floor; tests teleport it in front of each env's camera
        "position": [0, 0, -50.0],
    }
]


def _init_macros():
    """Set simulator macros (only once, before the first Environment is created)."""
    if og.sim is None:
        gm.RENDER_VIEWER_CAMERA = False
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False
        gm.ENABLE_OBJECT_STATES = False
    else:
        og.sim.stop()


def _setup_vec_env(num_envs, obs_modalities, scene_cfg=None, objects_cfg=None, env_cfg_extra=None):
    _init_macros()
    cfg = {
        "env": {"num_envs": num_envs, **(env_cfg_extra or {})},
        "scene": scene_cfg
        or {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls"],
        },
        "robots": [{"model": "fetch", "obs_modalities": list(obs_modalities)}],
        "task": {"type": "DummyTask"},
    }
    if objects_cfg:
        cfg["objects"] = objects_cfg
    return og.Environment(configs=cfg)


def _camera_sensor_name(robot):
    return next(name for name, sensor in robot.sensors.items() if isinstance(sensor, VisionSensor))


def _place_markers(env, distances):
    """Teleport each scene's marker cube directly in front of that env's camera at the given distance."""
    for env_idx, scene in enumerate(env.scenes):
        robot = scene.robots[0]
        cam = robot.sensors[_camera_sensor_name(robot)]
        cam_pos, cam_quat = cam.get_position_orientation()
        # USD cameras look down their local -z axis
        cam_forward = T.quat2mat(cam_quat) @ th.tensor([0.0, 0.0, -1.0])
        marker = scene.object_registry("name", "marker")
        marker.set_position_orientation(position=cam_pos + cam_forward * distances[env_idx])
    # Step + render so that annotator buffers reflect the new marker poses
    og.sim.step()
    for _ in range(3):
        og.sim.render()


def _center_patch(img, half=8):
    h, w = img.shape[0], img.shape[1]
    return img[h // 2 - half : h // 2 + half, w // 2 - half : w // 2 + half]


def test_tiled_rendering_core():
    """Alignment, shapes/dtypes, proprio, per-sensor parity, obs space conformance and reset freshness."""
    try:
        env = _setup_vec_env(
            num_envs=NUM_ENVS,
            obs_modalities=["rgb", "depth_linear", "proprio"],
            objects_cfg=MARKER_CFG,
        )
        robot_name = env.scenes[0].robots[0].name
        cam_name = _camera_sensor_name(env.scenes[0].robots[0])

        # --- Tiled sensor is active and covers the camera with the requested modalities ---
        assert env._tiled_sensor is not None, "Tiled sensor should be created when num_envs > 1"
        assert cam_name in env._tiled_sensor.modalities
        assert {"rgb", "depth_linear"} == set(env._tiled_sensor.modalities[cam_name])

        # --- Knob plumbing: default number of re-renders on reset ---
        assert env._num_rerenders_on_reset == 3

        _place_markers(env, MARKER_DISTANCES)
        obs_list, info_list = env.get_obs()
        assert len(obs_list) == NUM_ENVS

        expected_center_depths = [d - MARKER_SIZE / 2 for d in MARKER_DISTANCES]
        for i in range(NUM_ENVS):
            cam_obs = obs_list[i][robot_name][cam_name]

            # --- Shapes / dtypes / device ---
            rgb = cam_obs["rgb"]
            depth = cam_obs["depth_linear"]
            assert rgb.shape == (128, 128, 4), f"env {i}: rgb shape {rgb.shape}"
            assert rgb.dtype == th.uint8
            assert rgb.is_cuda, "Tiled rgb should live on GPU"
            assert depth.shape == (128, 128), f"env {i}: depth shape {depth.shape}"
            assert depth.dtype == th.float32

            # --- Tile <-> env alignment: the marker cube of env i must appear in tile i at distance d_i ---
            center_depth = _center_patch(depth).median().item()
            assert abs(center_depth - expected_center_depths[i]) < DEPTH_TOLERANCE, (
                f"env {i}: center depth {center_depth:.3f} does not match expected "
                f"{expected_center_depths[i]:.3f} -- tile/env misalignment?"
            )
            # The marker is pure red: red must dominate green/blue at the image center
            center_rgb = _center_patch(rgb).float()
            red, green, blue = center_rgb[..., 0].mean(), center_rgb[..., 1].mean(), center_rgb[..., 2].mean()
            assert (
                red > green + 20 and red > blue + 20
            ), f"env {i}: expected red marker at center, got rgb=({red:.1f}, {green:.1f}, {blue:.1f})"

            # --- Proprio must be preserved in multi-env mode ---
            proprio = obs_list[i][robot_name]["proprio"]
            assert proprio.numel() > 0 and th.all(th.isfinite(proprio)), f"env {i}: invalid proprio"

        # --- Parity with per-sensor rendering. Depth is the strict geometric check; rgb is loose since
        # the tiled and per-sensor render products denoise/accumulate samples differently ---
        for i in range(NUM_ENVS):
            sensor = env.scenes[i].robots[0].sensors[cam_name]
            s_obs, _ = sensor.get_obs()
            tiled_rgb = obs_list[i][robot_name][cam_name]["rgb"].float().cpu()
            sensor_rgb = s_obs["rgb"].float().cpu()
            mean_diff = (tiled_rgb - sensor_rgb).abs().mean().item()
            assert mean_diff < 40.0, f"env {i}: tiled vs per-sensor rgb mean abs diff {mean_diff:.1f}"
            tiled_depth_center = _center_patch(obs_list[i][robot_name][cam_name]["depth_linear"].cpu()).median()
            sensor_depth_center = _center_patch(s_obs["depth_linear"].cpu()).median()
            assert abs(tiled_depth_center - sensor_depth_center) < 0.2, f"env {i}: tiled vs per-sensor depth mismatch"

        # --- Reset: returned observations must conform to the observation space (checked internally by
        # reset()) and, with num_rerenders_on_reset > 0, reflect the *reset* state (markers back below
        # the floor), not the pre-reset frame ---
        pre_reset_center_depth = _center_patch(obs_list[0][robot_name][cam_name]["depth_linear"]).median().item()
        reset_obs_list, _ = env.reset()
        assert len(reset_obs_list) == NUM_ENVS
        post_reset_center_depth = _center_patch(reset_obs_list[0][robot_name][cam_name]["depth_linear"]).median().item()
        assert post_reset_center_depth > pre_reset_center_depth + 0.3, (
            f"reset() returned a stale frame: center depth {post_reset_center_depth:.3f} still matches the "
            f"pre-reset marker at {pre_reset_center_depth:.3f}"
        )

        # --- step() returns tiled observations as well ---
        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(env.num_envs)]
        )
        obs_list, _, _, _, _ = env.step(actions)
        assert obs_list[0][robot_name][cam_name]["rgb"].shape == (128, 128, 4)
    finally:
        og.clear()


def test_tiled_rendering_empty_scene():
    """Multi-env layout of object-less scenes must produce finite, increasing scene offsets (AABB fix)."""
    try:
        env = _setup_vec_env(
            num_envs=3,
            obs_modalities=["rgb"],
            scene_cfg={"type": "Scene", "use_floor_plane": True},
            env_cfg_extra={"num_rerenders_on_reset": 2},
        )

        # --- Knob plumbing: custom number of re-renders on reset ---
        assert env._num_rerenders_on_reset == 2

        # --- Scene offsets must be finite and strictly increasing along x ---
        xs = [scene.get_position_orientation()[0][0].item() for scene in env.scenes]
        assert all(th.isfinite(th.tensor(xs))), f"Non-finite scene positions: {xs}"
        assert xs[0] == pytest.approx(0.0, abs=1e-3)
        assert xs[1] > xs[0] and xs[2] > xs[1], f"Scene positions not increasing: {xs}"

        # --- Robots must be placed apart accordingly ---
        robot_xs = [scene.robots[0].get_position_orientation()[0][0].item() for scene in env.scenes]
        assert robot_xs[1] > robot_xs[0] and robot_xs[2] > robot_xs[1], f"Robot positions not increasing: {robot_xs}"

        # --- Stepping and tiled observations work in empty scenes ---
        robot_name = env.scenes[0].robots[0].name
        cam_name = _camera_sensor_name(env.scenes[0].robots[0])
        actions = th.stack(
            [th.from_numpy(env.scenes[i].robots[0].action_space.sample()).float() for i in range(env.num_envs)]
        )
        obs_list, _, _, _, _ = env.step(actions)
        for i in range(3):
            rgb = obs_list[i][robot_name][cam_name]["rgb"]
            assert rgb.shape == (128, 128, 4) and rgb.dtype == th.uint8
    finally:
        og.clear()


def test_tiled_rendering_unsupported_modality():
    """Modalities that cannot be tiled (e.g. pointcloud) must fail at construction with a clear error.

    NOTE: this test must remain LAST in the module. It deliberately leaves a partially-constructed
    environment behind (og.clear() cannot recover from a failed construction), which is reclaimed
    at process exit.
    """
    with pytest.raises(ValueError, match="Unsupported modality"):
        _setup_vec_env(num_envs=2, obs_modalities=["rgb", "pointcloud"])
