import tempfile

import pytest
import torch as th

import omnigibson as og
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


def test_data_collect_and_playback():
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
                "obs_modalities": [],
            }
        ],
        # Task kwargs
        "task": {
            "type": "BehaviorTask",
            # BehaviorTask-specific
            "activity_name": "laying_wood_floors",
            "online_object_sampling": True,
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

    # Create the environment (wrapped as a DataCollection env)
    env = og.Environment(configs=cfg)
    log.warning("Created env")
    env = DataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
    )
    log.warning("Wrapped env")

    # Record 2 episodes
    for i in range(2):
        env.reset()
        log.warning(f"Reset env for episode {i}")
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())
        log.warning(f"Stepped env for episode {i}")
        # Manually add a random object, e.g.: a banana, and place on the floor
        obj = DatasetObject(name="banana", category="banana")
        env.scene.add_object(obj)
        log.warning(f"Added object to env for episode {i}")
        obj.set_position(th.ones(3, dtype=th.float32) * 10.0)

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        # Manually remove the added object
        env.scene.remove_object(obj)
        log.warning(f"Removed object from env for episode {i}")

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        # Add water particles
        water = env.scene.get_system("water")
        pos = th.rand(10, 3, dtype=th.float32) * 10.0
        water.generate_particles(positions=pos)
        log.warning(f"Added water particles to env for episode {i}")

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        # Clear the system
        env.scene.clear_system("water")
        log.warning(f"Cleared water particles from env for episode {i}")

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

    # Save this data
    env.save_data()
    log.warning("Saved data")

    # Clear the sim
    og.clear(
        physics_dt=0.001,
        rendering_dt=0.001,
        sim_step_dt=0.001,
    )
    log.warning("Cleared sim")

    # Define robot sensor config and external sensors to use during playback
    robot_sensor_config = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_height": 16,
                "image_width": 16,
            },
        },
    }
    external_sensors_config = [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": f"/robot0/root_link/external_sensor0",
            "modalities": ["rgb", "seg_semantic"],
            "sensor_kwargs": {
                "image_height": 16,
                "image_width": 16,
                "focal_length": 12.0,
            },
            "position": th.tensor([-0.26549, -0.30288, 1.0 + 0.861], dtype=th.float32),
            "orientation": th.tensor([0.36165891, -0.24745751, -0.50752921, 0.74187715], dtype=th.float32),
        },
    ]

    # Create a playback env and playback the data, collecting obs along the way
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=collect_hdf5_path,
        output_path=playback_hdf5_path,
        robot_obs_modalities=["proprio", "rgb", "depth_linear"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=1,
        only_successes=False,
    )
    log.warning("Created playback env")
    env.playback_dataset(record=True)
    log.warning("Played back data")
    env.save_data()
    log.warning("Saved data")
