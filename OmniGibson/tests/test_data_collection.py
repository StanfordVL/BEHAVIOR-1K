import tempfile

import torch as th

import omnigibson as og
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject


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

    # Create the environment (wrapped as a DataCollection env)
    env = og.Environment(configs=cfg)
    env = DataCollectionWrapper(
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
                assert th.all(th.isclose(pos, env.robots[0].get_eef_position(arm=arm))).item()

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
                assert th.all(th.isclose(pos, env.robots[0].get_eef_position(arm=arm))).item()

        # Take a few more steps
        for _ in range(2):
            env.step(env.robots[0].action_space.sample())

        if i == 1:
            # Clear the water system since it was re-added
            env.scene.clear_system("water")

    # Save this data
    env.save_data()

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
                "image_height": 16,
                "image_width": 16,
            },
        },
    }
    external_sensors_config = [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": "/robot0/root_link/external_sensor0",
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": 16,
                "image_width": 16,
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
    env.playback_dataset(record_data=True)
    env.save_data()
