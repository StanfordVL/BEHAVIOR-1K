import h5py
import json
import tempfile
from unittest.mock import MagicMock

import torch as th
import os

import omnigibson as og
from omnigibson.envs import HDF5CollectionWrapper, HDF5PlaybackWrapper, LeRobotPlaybackWrapper, LeRobotDataWrapper
from omnigibson.envs.hdf5_data_wrapper import HDF5DataWrapper
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Helpers for unit tests
# ---------------------------------------------------------------------------


class _MockGymSpace:
    def __init__(self, shape):
        self.shape = shape


class _MinimalHDF5Wrapper(HDF5DataWrapper):
    """Concrete subclass that bypasses Environment setup for unit testing."""

    def __init__(self, hdf5_file, compression=None):
        self.wrapped_obj = None
        self.hdf5_file = hdf5_file
        self.compression = compression or {}
        self.traj_count = 0
        self.step_count = 0
        self.current_traj_history = []
        self.flush_every_n_traj = 10
        self.only_successes = False
        self.max_state_size = 0
        self.current_transitions = dict()
        self.checkpoint_rollback_trajs = None


# ---------------------------------------------------------------------------
# Unit tests – HDF5DataWrapper._process_traj_to_hdf5
# ---------------------------------------------------------------------------


def test_hdf5_process_traj_to_hdf5():
    fd, path = tempfile.mkstemp(".hdf5", dir=og.tempdir)
    os.close(fd)

    with h5py.File(path, "w") as f:
        wrapper = _MinimalHDF5Wrapper(hdf5_file=f)
        traj_data = [
            {"obs": {"proprio": th.zeros(2)}},
            {"action": th.tensor([1.0, 2.0]), "reward": 0.5, "terminated": False},
            {"action": th.tensor([3.0, 4.0]), "reward": 1.0, "terminated": True},
        ]
        grp = wrapper._process_traj_to_hdf5(traj_data, "demo_0")

        assert "action" in grp
        assert grp["action"].shape == (2, 2)
        assert th.allclose(th.tensor(grp["action"][()]), th.tensor([[1.0, 2.0], [3.0, 4.0]]))

        assert "reward" in grp
        assert grp["reward"].shape == (2,)
        assert th.allclose(th.tensor(grp["reward"][()]), th.tensor([0.5, 1.0]))

        assert "terminated" in grp
        assert grp["terminated"].shape == (2,)
        assert grp.attrs["num_samples"] == 2


def test_hdf5_process_traj_nested_obs():
    fd, path = tempfile.mkstemp(".hdf5", dir=og.tempdir)
    os.close(fd)

    with h5py.File(path, "w") as f:
        wrapper = _MinimalHDF5Wrapper(hdf5_file=f)
        traj_data = [
            {"obs": {"rgb": th.full((3, 4, 4), -1.0), "proprio": th.full((5,), -1.0)}},
            {"obs": {"rgb": th.zeros(3, 4, 4), "proprio": th.ones(5)}, "action": th.zeros(2)},
            {"obs": {"rgb": th.ones(3, 4, 4), "proprio": th.zeros(5)}, "action": th.ones(2)},
        ]
        grp = wrapper._process_traj_to_hdf5(traj_data, "demo_0", nested_keys=["obs"])

        assert "obs" in grp
        assert grp["obs"]["rgb"].shape == (2, 3, 4, 4)
        assert grp["obs"]["proprio"].shape == (2, 5)
        assert grp["action"].shape == (2, 2)


def test_hdf5_add_metadata_roundtrip():
    fd, path = tempfile.mkstemp(".hdf5", dir=og.tempdir)
    os.close(fd)

    with h5py.File(path, "w") as f:
        wrapper = _MinimalHDF5Wrapper(hdf5_file=f)
        data_grp = f.create_group("data")

        dict_data = {"key1": [1, 2, 3], "key2": "value"}
        wrapper.add_metadata(data_grp, "config", dict_data)

        raw = data_grp.attrs["config"]
        round_tripped = json.loads(raw)
        assert round_tripped == dict_data

        wrapper.add_metadata(data_grp, "episode_count", 42)
        assert data_grp.attrs["episode_count"] == 42


def test_hdf5_close_dataset_attrs():
    fd, path = tempfile.mkstemp(".hdf5", dir=og.tempdir)
    os.close(fd)

    with h5py.File(path, "w") as f:
        wrapper = _MinimalHDF5Wrapper(hdf5_file=f)
        f.create_group("data")
        wrapper.traj_count = 7
        wrapper.step_count = 100
        wrapper.close_dataset()

    with h5py.File(path, "r") as f:
        assert f["data"].attrs["n_episodes"] == 7
        assert f["data"].attrs["n_steps"] == 100


def test_hdf5_traj_compression():
    fd, path = tempfile.mkstemp(".hdf5", dir=og.tempdir)
    os.close(fd)

    compression = {"compression": "gzip", "compression_opts": 9}
    with h5py.File(path, "w") as f:
        wrapper = _MinimalHDF5Wrapper(hdf5_file=f, compression=compression)
        traj_data = [
            {"obs": {"proprio": th.zeros(10)}},
            {"action": th.randn(10), "reward": th.tensor(0.1)},
            {"action": th.randn(10), "reward": th.tensor(0.2)},
        ]
        grp = wrapper._process_traj_to_hdf5(traj_data, "demo_0")

        assert grp["action"].compression == "gzip"
        assert grp["reward"].compression == "gzip"
        expected_actions = th.stack([d["action"] for d in traj_data if "action" in d], dim=0)
        assert th.allclose(th.tensor(grp["action"][()]), expected_actions)


# ---------------------------------------------------------------------------
# Unit tests – LeRobotDataWrapper.get_lerobot_obs_mapping
# ---------------------------------------------------------------------------


def test_lerobot_obs_mapping_proprio():
    mock_env = MagicMock()
    mock_env.observation_space = {
        "robot::fetch:proprio::proprio": _MockGymSpace((12,)),
    }

    mapping, features = LeRobotDataWrapper.get_lerobot_obs_mapping(mock_env, fps=30)

    assert "robot::fetch:proprio::proprio" in mapping
    assert mapping["robot::fetch:proprio::proprio"] == "observation.state"
    assert "observation.state" in features
    assert features["observation.state"]["dtype"] == "float32"
    assert features["observation.state"]["shape"] == (12,)


def test_lerobot_obs_mapping_rgb():
    mock_env = MagicMock()
    mock_env.observation_space = {
        "robot::fetch:eef_link:Camera:0::rgb": _MockGymSpace((128, 128, 4)),
    }

    mapping, features = LeRobotDataWrapper.get_lerobot_obs_mapping(mock_env, fps=30)

    assert mapping["robot::fetch:eef_link:Camera:0::rgb"] == "observation.rgb.eef_link_camera_0"
    key = "observation.rgb.eef_link_camera_0"
    assert key in features
    assert features[key]["dtype"] == "video"
    assert features[key]["shape"] == (128, 128, 3)


def test_lerobot_obs_mapping_depth():
    mock_env = MagicMock()
    mock_env.observation_space = {
        "robot::fetch:eef_link:Camera:0::depth_linear": _MockGymSpace((128, 128)),
    }

    mapping, features = LeRobotDataWrapper.get_lerobot_obs_mapping(mock_env, fps=30)

    assert "observation.depth_linear.eef_link_camera_0" in features
    assert features["observation.depth_linear.eef_link_camera_0"]["dtype"] == "video"
    assert features["observation.depth_linear.eef_link_camera_0"]["shape"] == (128, 128, 1)

    tf_key = "observation.robot2cam_pose.eef_link_camera_0"
    assert tf_key in features
    assert features[tf_key]["shape"] == (7,)


# ---------------------------------------------------------------------------
# Unit tests – DataWrapper.should_save_current_episode / flush_current_traj
# ---------------------------------------------------------------------------


def test_should_save_current_episode():
    mock_env = MagicMock()
    mock_env.task.success = False

    wrapper = _MinimalHDF5Wrapper(hdf5_file=MagicMock())
    wrapper.env = mock_env
    wrapper.only_successes = True
    wrapper.current_traj_history = []
    # Failed episode with no initial-obs-only exception should not be saved.
    assert not wrapper.should_save_current_episode
    # Initial observation only should NOT be saved.
    wrapper.current_traj_history = [{"obs": {"proprio": th.zeros(2)}}]
    assert not wrapper.should_save_current_episode
    # Failed episode with trajectory data beyond initial obs should not be saved when only_successes=True.
    wrapper.current_traj_history = [
        {"obs": {"proprio": th.zeros(2)}},
        {"action": th.zeros(2), "reward": 0.0, "terminated": False, "truncated": False},
    ]
    assert not wrapper.should_save_current_episode
    wrapper.only_successes = False
    assert wrapper.should_save_current_episode
    # Even when only_successes=False, obs-only trajectories are still excluded from saving.
    wrapper.current_traj_history = [{"obs": {"proprio": th.zeros(2)}}]
    assert not wrapper.should_save_current_episode
    wrapper.only_successes = True
    wrapper.current_traj_history = [
        {"obs": {"proprio": th.zeros(2)}},
        {"action": th.zeros(2), "reward": 0.0, "terminated": True, "truncated": False},
    ]
    mock_env.task.success = True
    assert wrapper.should_save_current_episode


def test_flush_current_traj_counter():
    fd, path = tempfile.mkstemp(".hdf5", dir=og.tempdir)
    os.close(fd)

    with h5py.File(path, "w") as f:
        wrapper = _MinimalHDF5Wrapper(hdf5_file=f)
        mock_env = MagicMock()
        mock_env.task.success = False
        wrapper.env = mock_env
        wrapper.only_successes = False
        f.create_group("data")

        wrapper.current_traj_history = [
            {"obs": {"proprio": th.zeros(2)}},
            {"action": th.tensor([1.0, 2.0])},
            {"action": th.tensor([3.0, 4.0])},
        ]
        assert wrapper.traj_count == 0
        wrapper.flush_current_traj()
        assert wrapper.traj_count == 1
        assert len(wrapper.current_traj_history) == 0


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


def _get_test_cfg():
    return {
        "env": {
            "external_sensors": [],
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "breakfast_table"],
        },
        "robots": [
            {
                "model": "fetch",
                "name": "fetch",
                "obs_modalities": [],
                "fixed_base": False,
            }
        ],
        "task": {
            "type": "BehaviorTask",
            "activity_name": "laying_wood_floors",
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        },
    }


def _ensure_sim_stopped():
    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = False
    else:
        og.sim.stop()


def _get_playback_configs(img_h, img_w, cfg):
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "depth_linear"],
            "sensor_kwargs": {
                "image_height": img_h,
                "image_width": img_w,
            },
        },
    }
    external_sensors_config = [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": f"/controllable__{cfg['robots'][0]['model'].lower()}__{cfg['robots'][0]['name']}/root_link/external_sensor0",
            "modalities": ["rgb", "depth_linear"],
            "sensor_kwargs": {
                "image_height": img_h,
                "image_width": img_w,
            },
            "position": th.tensor([-0.26549, -0.30288, 1.0 + 0.861], dtype=th.float32),
            "orientation": th.tensor([0.36165891, -0.24745751, -0.50752921, 0.74187715], dtype=th.float32),
        },
    ]
    return robot_sensor_config, external_sensors_config


# ---------------------------------------------------------------------------
# Integration tests – data collection
# ---------------------------------------------------------------------------


def test_collection_basic():
    cfg = _get_test_cfg()
    _ensure_sim_stopped()

    _, collect_hdf5_path = tempfile.mkstemp("test_collection_basic.hdf5", dir=og.tempdir)

    env = og.Environment(configs=cfg)
    env = HDF5CollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        obj_attr_keys=["scale", "visible"],
    )

    for _ in range(2):
        env.reset()
        for _ in range(3):
            env.step(env.robots[0].action_space.sample())

    env.save_data()

    with h5py.File(collect_hdf5_path, "r") as f:
        assert f["data"].attrs["n_episodes"] == 2
        assert f["data"].attrs["n_steps"] > 0
        assert "demo_0" in f["data"]
        assert "demo_1" in f["data"]

    og.clear()


def test_collection_with_transitions():
    cfg = _get_test_cfg()
    _ensure_sim_stopped()

    _, collect_hdf5_path = tempfile.mkstemp("test_collection_transitions.hdf5", dir=og.tempdir)

    env = og.Environment(configs=cfg)
    env = HDF5CollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
    )

    env.reset()
    for _ in range(2):
        env.step(env.robots[0].action_space.sample())

    obj = DatasetObject(name="banana", category="banana")
    env.scene.add_object(obj)
    obj.set_position(th.ones(3, dtype=th.float32) * 10.0)

    for _ in range(2):
        env.step(env.robots[0].action_space.sample())

    env.scene.remove_object(obj)

    for _ in range(2):
        env.step(env.robots[0].action_space.sample())

    water = env.scene.get_system("water")
    water.generate_particles(positions=th.rand(5, 3, dtype=th.float32) * 10.0)

    for _ in range(2):
        env.step(env.robots[0].action_space.sample())

    env.scene.clear_system("water")

    for _ in range(2):
        env.step(env.robots[0].action_space.sample())

    env.save_data()

    with h5py.File(collect_hdf5_path, "r") as f:
        demo_grp = f["data"]["demo_0"]
        transitions = json.loads(demo_grp.attrs["transitions"])
        assert len(transitions) > 0

    og.clear()


def test_collection_checkpoint_rollback():
    cfg = _get_test_cfg()
    _ensure_sim_stopped()

    _, collect_hdf5_path = tempfile.mkstemp("test_collection_ckpt.hdf5", dir=og.tempdir)

    env = og.Environment(configs=cfg)
    env = HDF5CollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
    )

    env.reset()
    for _ in range(2):
        env.step(env.robots[0].action_space.sample())

    env.update_checkpoint()
    robot_eef_state = {arm: env.robots[0].get_eef_position(arm=arm) for arm in env.robots[0].arm_names}

    env.step(env.robots[0].action_space.sample())

    water = env.scene.get_system("water")
    water.generate_particles(positions=th.rand(10, 3, dtype=th.float32) * 10.0)
    for _ in range(2):
        env.step(env.robots[0].action_space.sample())

    env.rollback_to_checkpoint()

    assert "water" not in env.scene.active_systems

    for arm, pos in robot_eef_state.items():
        assert th.all(th.isclose(pos, env.robots[0].get_eef_position(arm=arm))).item()

    for _ in range(2):
        env.step(env.robots[0].action_space.sample())
    env.save_data()

    with h5py.File(collect_hdf5_path, "r") as f:
        assert f["data"].attrs["n_episodes"] == 1
        assert "demo_0" in f["data"]

    og.clear()


# ---------------------------------------------------------------------------
# Integration tests – playback
# ---------------------------------------------------------------------------


def test_hdf5_playback_and_dataset():
    cfg = _get_test_cfg()
    img_h, img_w = 128, 128
    _ensure_sim_stopped()

    _, collect_hdf5_path = tempfile.mkstemp("test_hdf5_playback_collect.hdf5", dir=og.tempdir)
    _, playback_hdf5_path = tempfile.mkstemp("test_hdf5_playback_output.hdf5", dir=og.tempdir)

    # --- Collect data ---
    env = og.Environment(configs=cfg)
    env = HDF5CollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
    )

    for _ in range(2):
        env.reset()
        for _ in range(3):
            env.step(env.robots[0].action_space.sample())

    env.save_data()

    # --- Playback ---
    og.clear(
        physics_dt=0.001,
        rendering_dt=0.001,
        sim_step_dt=0.001,
    )

    robot_sensor_config, external_sensors_config = _get_playback_configs(img_h, img_w, cfg)
    obs_modalities = ["proprio", "rgb", "depth_linear"]

    env = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=collect_hdf5_path,
        output_path=playback_hdf5_path,
        robot_obs_modalities=obs_modalities,
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=1,
        only_successes=False,
    )

    obs, info = env.reset()
    for mod in obs_modalities:
        found_obs_key = False
        for obs_key in obs.keys():
            if mod in obs_key.split("::")[-1]:
                found_obs_key = True
                break
        assert found_obs_key, f"Failed to find obs modality: {mod} in observation keys: {tuple(obs.keys())}"

    env.playback_dataset(record_data=True)
    env.save_data()

    with h5py.File(playback_hdf5_path, "r") as f:
        assert f["data"].attrs["n_episodes"] == 2
        assert "demo_0" in f["data"]
        assert "demo_1" in f["data"]
        assert "omnigibson_git_hash" in f["data"].attrs
        assert f["data"].attrs["omnigibson_git_hash"] is not None

        n_episodes = f["data"].attrs["n_episodes"]
        assert n_episodes == 2, f"Expected 2 episodes, got {n_episodes}"

        for demo_idx in range(n_episodes):
            demo_grp = f["data"][f"demo_{demo_idx}"]

            if "observation" in demo_grp:
                obs_grp = demo_grp["observation"]
                for obs_key in obs_grp.keys():
                    obs_data = obs_grp[obs_key]
                    if isinstance(obs_data, h5py.Dataset):
                        assert obs_data.shape[0] > 0, f"Demo {demo_idx} observation {obs_key} has no data"

                        if "rgb" in obs_key:
                            expected_shape = (obs_data.shape[0], 3, img_h, img_w)
                            assert (
                                obs_data.shape == expected_shape
                            ), f"Demo {demo_idx} obs {obs_key}: expected {expected_shape}, got {obs_data.shape}"
                            if (
                                "external_sensor0" not in obs_key
                                and "eef_link_camera_0" not in obs_key
                                and "eyes_camera_0" not in obs_key
                            ):
                                assert False, (
                                    f"Demo {demo_idx} obs {obs_key} has unexpected rgb key (expected one of: "
                                    "observation.rgb.external_sensor0, observation.rgb.eef_link_camera_0, observation.rgb.eyes_camera_0)"
                                )
                        elif "depth_linear" in obs_key:
                            expected_shape = (obs_data.shape[0], 3, img_h, img_w)
                            assert (
                                obs_data.shape == expected_shape
                            ), f"Demo {demo_idx} obs {obs_key}: expected {expected_shape}, got {obs_data.shape}"
                            if "eef_link_camera_0" not in obs_key and "eyes_camera_0" not in obs_key:
                                assert False, (
                                    f"Demo {demo_idx} obs {obs_key} has unexpected depth_linear key (expected one of: "
                                    "observation.depth_linear.eef_link_camera_0, observation.depth_linear.eyes_camera_0)"
                                )

            if "action" in demo_grp:
                action_data = demo_grp["action"]
                if isinstance(action_data, h5py.Dataset):
                    assert action_data.shape[0] > 0, f"Demo {demo_idx} action has no data"
                    assert len(action_data.shape) == 2, f"Demo {demo_idx} action should be 2D"

            if "proprio" in demo_grp:
                proprio_data = demo_grp["proprio"]
                if isinstance(proprio_data, h5py.Dataset):
                    assert proprio_data.shape[0] > 0, f"Demo {demo_idx} proprio has no data"

    og.sim.stop()
    og.clear(
        physics_dt=1 / 120.0,
        rendering_dt=1 / 30.0,
        sim_step_dt=1 / 30.0,
    )


def test_lerobot_playback_and_dataset():
    cfg = _get_test_cfg()
    img_h, img_w = 128, 128
    _ensure_sim_stopped()

    _, collect_hdf5_path = tempfile.mkstemp("test_lerobot_playback_collect.hdf5", dir=og.tempdir)
    _, playback_hdf5_path = tempfile.mkstemp("test_lerobot_playback_output.hdf5", dir=og.tempdir)

    # --- Collect data ---
    env = og.Environment(configs=cfg)
    env = HDF5CollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
    )

    for _ in range(2):
        env.reset()
        for _ in range(3):
            env.step(env.robots[0].action_space.sample())

    env.save_data()

    # --- LeRobot Playback ---
    og.clear(
        physics_dt=0.001,
        rendering_dt=0.001,
        sim_step_dt=0.001,
    )

    robot_sensor_config, external_sensors_config = _get_playback_configs(img_h, img_w, cfg)
    obs_modalities = ["proprio", "rgb", "depth_linear"]

    lerobot_playback_kwargs = {
        "output_path": "behavior1k/test_dataset",
        "root_dir": os.path.dirname(playback_hdf5_path),
    }

    env = LeRobotPlaybackWrapper.create_from_hdf5(
        input_path=collect_hdf5_path,
        robot_obs_modalities=obs_modalities,
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=1,
        only_successes=False,
        **lerobot_playback_kwargs,
    )

    obs, info = env.reset()
    for mod in obs_modalities:
        found_obs_key = False
        for obs_key in obs.keys():
            if mod in obs_key.split("::")[-1]:
                found_obs_key = True
                break
        assert found_obs_key, f"Failed to find obs modality: {mod} in observation keys: {tuple(obs.keys())}"

    env.playback_dataset(record_data=True)
    env.save_data()

    # --- Validate LeRobot dataset ---
    dataset = LeRobotDataset(
        repo_id=lerobot_playback_kwargs["output_path"],
        root=f"{lerobot_playback_kwargs['root_dir']}/{lerobot_playback_kwargs['output_path']}",
        delta_timestamps={
            "observation.rgb.external_sensor0": [-i / 30 for i in reversed(range(5))],
        },
    )

    batch_size = 6
    data_loader = th.utils.data.DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(data_loader))

    shape = (img_h, img_w)
    for key, shape in (
        ("action", (12,)),
        ("observation.rgb.external_sensor0", (5, 3, *shape)),
        ("observation.rgb.eef_link_camera_0", (3, *shape)),
        ("observation.rgb.eyes_camera_0", (3, *shape)),
        ("observation.depth_linear.eef_link_camera_0", (3, *shape)),
        ("observation.depth_linear.eyes_camera_0", (3, *shape)),
    ):
        assert key in batch
        expected_shape = (batch_size, *shape)
        assert (
            batch[key].shape == expected_shape
        ), f"Expected key [{key}] to have shape {expected_shape}, but got {batch[key].shape}"

    # Validate metadata
    assert dataset.meta.info["omnigibson_git_hash"] is not None
    assert "cam_intrinsics" in dataset.meta.info

    og.sim.stop()
    og.clear(
        physics_dt=1 / 120.0,
        rendering_dt=1 / 30.0,
        sim_step_dt=1 / 30.0,
    )
