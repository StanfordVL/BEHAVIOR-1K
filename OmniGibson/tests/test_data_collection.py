import tempfile

import torch as th
import os

import omnigibson as og
from omnigibson.envs import HDF5CollectionWrapper, HDF5PlaybackWrapper, LeRobotPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.learning.utils.lerobot_utils import OmniGibsonLeRobotV2Dataset, OmniGibsonLeRobotV3Dataset

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION as LEROBOT_CODEBASE_VERSION


def test_data_collect_and_playback():
    activity = "laying_wood_floors"
    img_h, img_w = 720, 720
    cfg = {
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
                "type": "Fetch",
                "name": "robot0",
                "obs_modalities": [],
                "fixed_base": False,
            }
        ],
        # Task kwargs
        "task": {
            "type": "BehaviorTask",
            # BehaviorTask-specific
            "activity_name": activity,
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        },
    }

    if og.sim is None:
        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth)
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Create temp file to save data
    _, collect_hdf5_path = tempfile.mkstemp("test_data_collection.hdf5", dir=og.tempdir)
    _, playback_hdf5_path = tempfile.mkstemp("test_data_playback.hdf5", dir=og.tempdir)

    env = og.Environment(configs=cfg)

    env = HDF5CollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
        obj_attr_keys=["scale", "visible"],
    )

    # Record 3 episodes
    for i in range(3):
        env.reset()
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())
        # Manually add a random object, e.g.: a banana, and place on the floor
        obj = DatasetObject(name="banana", category="banana")
        env.scene.add_object(obj)
        obj.set_position(th.ones(3, dtype=th.float32) * 10.0)

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        # Manually remove the added object
        env.scene.remove_object(obj)

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        # Checkpoint state here for our first episode
        if i == 0:
            env.update_checkpoint()
            robot_eef_state = {arm: env.robots[0].get_eef_position(arm=arm) for arm in env.robots[0].arm_names}

            # Take one step to avoid creating the system immediately after the checkpoint is updated, which
            # will cause downstream errors during playback
            env.step(env.robots[0].action_space.sample())

        # Add water particles
        water = env.scene.get_system("water")
        pos = th.rand(10, 3, dtype=th.float32) * 10.0
        water.generate_particles(positions=pos)

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        if i == 0:
            # Rollback state here for our first episode
            env.rollback_to_checkpoint()

            # Make sure water doesn't exist
            assert "water" not in env.scene.active_systems

            # Make sure robot state is roughly the same
            for arm, pos in robot_eef_state.items():
                assert th.allclose(pos, env.robots[0].get_eef_position(arm=arm), atol=1e-5)

        elif i == 1:
            # Checkpoint state here for our second episode
            env.update_checkpoint()
            robot_eef_state = {arm: env.robots[0].get_eef_position(arm=arm) for arm in env.robots[0].arm_names}

            # Take one step to avoid clearing the system immediately after the checkpoint is updated, which
            # will cause downstream errors during playback
            env.step(env.robots[0].action_space.sample())

        # Clear the system
        env.scene.clear_system("water")

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        if i == 1:
            # Rollback state here for our first episode
            env.rollback_to_checkpoint()

            # Make sure water exists
            assert "water" in env.scene.active_systems

            # Make sure robot state is roughly the same
            for arm, pos in robot_eef_state.items():
                assert th.allclose(pos, env.robots[0].get_eef_position(arm=arm), atol=1e-5)

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        if i == 1:
            # Clear the water system since it was re-added
            env.scene.clear_system("water")

    # Save this data
    env.save_data()

    hdf5_playback_kwargs = {
        "output_path": playback_hdf5_path,
    }

    lerobot_playback_kwargs = {
        "output_path": "behavior1k/test_dataset",
        "root_dir": os.path.dirname(playback_hdf5_path),
        "lerobot_version": LEROBOT_CODEBASE_VERSION,
    }

    for playback_cls, playback_kwargs in zip(
        (LeRobotPlaybackWrapper, HDF5PlaybackWrapper),
        (lerobot_playback_kwargs, hdf5_playback_kwargs),
    ):
        # Clear the sim
        og.clear(
            physics_dt=0.001,
            rendering_dt=0.001,
            sim_step_dt=0.001,
        )

        # Define robot sensor config and external sensors to use during playback
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb"],
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
                "relative_prim_path": f"/controllable__{cfg['robots'][0]['type'].lower()}__{cfg['robots'][0]['name']}/root_link/external_sensor0",
                "modalities": ["rgb", "depth_linear"],
                "sensor_kwargs": {
                    "image_height": img_h,
                    "image_width": img_w,
                },
                "position": th.tensor([-0.26549, -0.30288, 1.0 + 0.861], dtype=th.float32),
                "orientation": th.tensor([0.36165891, -0.24745751, -0.50752921, 0.74187715], dtype=th.float32),
            },
        ]

        # Create a playback env and playback the data, collecting obs along the way
        obs_modalities = ["proprio", "rgb", "depth_linear"]
        env = playback_cls.create_from_hdf5(
            input_path=collect_hdf5_path,
            robot_obs_modalities=obs_modalities,
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
            **playback_kwargs,
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

    # Test loading env + reading data chunks from LeRobot dataset
    dataset_cls = OmniGibsonLeRobotV2Dataset if LEROBOT_CODEBASE_VERSION == "v2.1" else OmniGibsonLeRobotV3Dataset
    dataset = dataset_cls(
        repo_id=lerobot_playback_kwargs["output_path"],
        root=f"{lerobot_playback_kwargs['root_dir']}/{lerobot_playback_kwargs['output_path']}",
    )

    batch_size = 8
    data_loader = th.utils.data.DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(data_loader))

    rgb_shape = (3, img_h, img_w)
    depth_shape = (1, img_h, img_w)
    for key, shape in (
        ("action", (12,)),  # (2DOF base + 1DOF trunk + 2DOF head + 6DOF EEF + 1DOF gripper)
        ("observation.rgb.external_sensor0", rgb_shape),
        ("observation.rgb.eef_link_camera_0", rgb_shape),
        ("observation.rgb.eyes_camera_0", rgb_shape),
        ("observation.depth_linear.eef_link_camera_0", depth_shape),
        ("observation.depth_linear.eyes_camera_0", depth_shape),
    ):
        assert key in batch
        expected_shape = (batch_size, *shape)
        assert (
            batch[key].shape == expected_shape
        ), f"Expected key [{key}] to have shape {expected_shape}, but got {batch[key].shape}"
