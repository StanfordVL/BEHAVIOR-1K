# :material-controller: **Collecting Demonstrations**


## Devices
I/O Devices can be used to read user input and teleoperate simulated robots in real-time. OmniGibson leverages [TeleMoMa](https://robin-lab.cs.utexas.edu/telemoma-web/), a modular and versatile library for manipulating mobile robots in the scene. This is achieved by using devies such as keyboards, SpaceMouse, cameras, VR devices, mobile phones, or any combination thereof. More generally, we support any interface that implements the `telemoma.human_interface.teleop_core.BaseTeleopInterface` class. In order to support your own custom device, simply subclass this base class and implement the required methods. For more information on this, checkout the [TeleMoMa codebase](https://github.com/UT-Austin-RobIn/telemoma).

## Teleoperation

The following section will go through `robot_teleoperation_example.py`, which lets users to choose a robot to complete a simple pick and place task. Users are also encouraged to take a look at `vr_simple_demo.py`, which show how to actually render to VR headset and teleoperate `BehaviorRobot` with VR controllers (HTC VIVE).

We assume that we already have the scene and task setup. To initialize a teleoperation system, we first need to specify the configuration for it.  

After the config simply instantiate teh teleoperation system.

```{.python .annotate}
teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=True)
```

`TeleopSystem` takes in the config dictionary, which we just created. It also takes in the robot instance we want to teleoperate, as well as `show_control_marker`, which if set to `True`, will also create a green visual marker indicates the desired pose of the robot end effector that the user wants to robot to go.

After the `TeleopSystem` is created, start by calling
```{.python .annotate}
teleop_sys.start()
```

Then, within the simulation loop, simply call

```{.python .annotate}
action = teleop_sys.get_action(teleop_sys.get_obs())
```

to get the action based on the user teleoperation input, and pass the action to the `env.step` function.

## Data Collection and Playback

OmniGibson provides tools for collecting demonstration data and playing it back for further analysis, training, or evaluation. This is implemented via two environment wrapper classes: `DataCollectionWrapper` and `DataPlaybackWrapper`.

### DataCollectionWrapper

The `DataCollectionWrapper` is used to collect data during environment interactions. It wraps around an existing OmniGibson environment and records relevant information at each step.

Key features:

 - Records actions, states, rewards, and termination conditions
 - Optimizes the simulator for data collection
 - Tracks object and system transitions within the environment

Example usage:

```python
import omnigibson as og
from omnigibson.envs import DataCollectionWrapper

# Create your OmniGibson environment
env = og.Environment(configs=your_config)

# Wrap it with DataCollectionWrapper
wrapped_env = DataCollectionWrapper(
    env=env,
    output_path="path/to/save/data.hdf5",
    only_successes=False,  # Set to True to only save successful episodes
)

# Use the wrapped environment as you would normally
obs, info = wrapped_env.reset()
for _ in range(num_steps):
    action = your_policy(obs)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)

# Save the collected data
wrapped_env.save_data()
```

### DataPlaybackWrapper

The `DataPlaybackWrapper` is used to replay collected data and optionally record additional observations. This is particularly useful for gathering visual data or other sensor information that wasn't collected during the initial demonstration.

Key features:
 - Replays episodes from collected data
 - Can record additional observation modalities during playback
 - Supports custom robot sensor configurations and external sensors

Example usage:

```python
from omnigibson.envs import DataPlaybackWrapper

# Create a playback environment
playback_env = DataPlaybackWrapper.create_from_hdf5(
    input_path="path/to/collected/data.hdf5",
    output_path="path/to/save/playback/data.hdf5",
    robot_obs_modalities=["proprio", "rgb", "depth_linear"],
    robot_sensor_config=your_robot_sensor_config,
    external_sensors_config=your_external_sensors_config,
    n_render_iterations=5,
    only_successes=False,
)

# Playback the entire dataset and record observations
playback_env.playback_dataset(record_data=True)

# Save the recorded playback data
playback_env.save_data()
```