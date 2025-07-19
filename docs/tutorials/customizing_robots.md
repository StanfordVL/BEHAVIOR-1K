# :material-wrench-outline: **Customizing Robots**

[Robots](../omnigibson/robots.md) can have both their action spaces (types of control commands) and observation spaces (types of sensor modalities) customized to suit specific use-cases. This can be done both prior to import time (via a config) or dynamically during runtime. Below, we describe a recommended workflow for modifying both sets of these properties.

## Customizing Action Spaces

A robot is equipped with multiple controllers, each of which control a subset of the robot's low-level joint motors. Together, these controllers' inputs form the robot's corresponding action space. For example, a [Fetch](../reference/robots/fetch.md) robot consists of (a) a base controller controlling its two wheels, (b) a head controller controlling its two head joints, (c) an arm controller controlling its seven arm joints, and (d) a gripper controller controlling its two gripper joints (resulting in 13 DOF being controlled). An example set of controllers would be using a [DifferentialDriveController](../reference/controllers/dd_controller.md) for the base, [JointController](../reference/controllers/joint_controller.md)s for the head and arm, and binary  [MultiFingerGripperController](../reference/controllers/multi_finger_gripper_controller.md) for the gripper. In this case, the action space size would be 2 + 2 + 7 + 1 = 12. If we were to use an [InverseKinematicsController](../reference/controllers/ik_controller.md) commanding the 6DOF end-effector pose instead of the JointController for the arm, the action space size would be 2 + 2 + 6 + 1 = 11. Each of these controllers can be individual configured and swapped out for each robot.

### Modifying Via Config

One way to customize a robot's set of controllers is to manually set the desired controller configuration in the environment config file when creating an **OmniGibson** environment. An example is shown below:

??? code "fetch_controller_cfg.yaml"
    ``` yaml linenums="1"
    robots:
      - type: Fetch
        position: [0, 0, 0]
        orientation: [0, 0, 0, 1]
        controller_config:
          base:
            name: DifferentialDriveController
          arm_0:
            name: InverseKinematicsController
            kv: 2.0
          gripper_0:
            name: MultiFingerGripperController
            mode: binary
          camera:
            name: JointController
            use_delta_commands: False
    ```

In the above example, the types of controllers are specified for each component of the robot (`base`, `arm_0`, `gripper_0`, `camera`), and additional relevant keyword arguments to pass to the specific controller init calls can also be specified. If a controller or any keyword arguments are not specified for a given component, a default set of values will be used, which are specified in the robot class itself (`_default_controller_config` property). Please see the [Controllers](../omnigibson/controllers.md) section for additional details on controller arguments. Do note that if `action_normalize=True` is passed as a robot-level kwarg, it will automatically overwrite any `command_input_limits` passed via the controller config, since it will assume a normalization range of `[-1, 1]`.

Alternatively, if directly instantiating a robot class, the controller config can be directly passed into the constructor, e.g.:

??? code "import_fetch_controller.py"
    ``` python linenums="1"

    from omnigibson.robots import Fetch

    robot = Fetch(
        name="agent",
        controller_config={
            "base": {
                "name": "DifferentialDriveController",
            },
            "arm_0": {
                "name": "InverseKinematicsController",
                "kv": 2.0,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "binary",
            },
            "camera": {
                "name": "JointController",
                "use_delta_commands": False,
            },
        },
    )
    ```

### Modifying At Runtime

Robots' action spaces can also be modified at runtime after a robot has been imported, effectively re-loading a set of (potentially different) controllers. This is achieved by defining the new desired controller config and then calling `reload_controllers()`:

??? code "reload_fetch_controllers.py"
    ``` python linenums="1"

    import omnigibson as og
    from omnigibson.scenes import Scene
    from omnigibson.robots import Fetch

    # Launch OG
    og.launch()
    scene = Scene()
    og.sim.import_scene(scene)

    # Not specifying `controller_config` will automatically use the default set of values
    robot = Fetch(name="agent")

    # Import robot and play sim
    scene.add_object(robot)
    og.sim.play()

    # Define our custom controller config
    controller_config = {
        "base": {
            "name": "DifferentialDriveController",
        },
        "arm_0": {
            "name": "InverseKinematicsController",
            "kv": 2.0,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "mode": "binary",
        },
        "camera": {
            "name": "JointController",
            "use_delta_commands": False,
        },
    }

    # Reload the controllers
    robot.reload_controllers(controller_config=controller_config)
    ```


## Customizing Observation Spaces

A robot is equipped with multiple onboard sensors, each of which can be configured to return a unique set of observations. Together, these observation modalities form the robot's observation space. For example, a [Turtlebot](../reference/robots/turtlebot.md) robot consists of (a) a LIDAR ([ScanSensor](../reference/sensors/scan_sensor.md)) at its base, (b) an RGB-D camera ([VisionSensor](../reference/sensors/vision_sensor.md)) at its head, and (c) onboard proprioception. An example set of observations would be using modalities `["rgb", "normal", "proprio", "scan"]`, which would return RGB and surface normal maps, proprioception, and 2D radial LIDAR distances. Each of these modalities can be swapped out, depending on robot's set of equipped onboard sensors. Each of these controllers can be individual configured and swapped out for each robot. Please see the individual sensor classes for specific supported modalities.

### Modifying Via Config

One way to customize a robot's set of observations is to manually set the desired sensor configuration in the environment config file when creating an **OmniGibson** environment. An example is shown below:

??? code "turtlebot_obs_cfg.yaml"
    ``` yaml linenums="1"
    robots:
      - type: Turtlebot
        obs_modalities: [scan, rgb, depth, proprio]
        proprio_obs: [robot_lin_vel, robot_ang_vel]
        sensor_config:
          VisionSensor:
            sensor_kwargs:
              image_height: 128
              image_width: 128
          ScanSensor:
              sensor_kwargs:
                min_range: 0.05
                max_range: 10.0
    ```

In the above example, the observation modalities are specified via the `obs_modalities` kwarg. Each type of sensor can be configured as well via the `sensor_config` dictionary argument -- attributes such as image size and LIDAR range limits can be specified here. Specific proprioception values can be requested by setting the `proprio_obs` kwarg, which by default will return all available proprioception values (and can be viewed via `robot.default_proprio_obs`). Note that proprioception will only be used if `proprio` is specified in `obs_modalities`.

Alternatively, if directly instantiating a robot class, the observation modalities and sensor config can be directly passed into the constructor, e.g.:

??? code "import_turtlebot_sensor.py"
    ``` python linenums="1"

    from omnigibson.robots import Turtlebot

    robot = Turtlebot(
        name="agent",
        obs_modalities=["scan", "rgb", "depth", "proprio"],
        proprio_obs=["robot_lin_vel", "robot_ang_vel"],
        sensor_config={
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": 128,
                    "image_width": 128,
                },
            },
            "ScanSensor": {
                "sensor_kwargs": {
                    "min_range": 0.05,
                    "max_range": 10.0,
                },
            },
        },
    )
    ```

### Modifying At Runtime

In general, dynamically configuring a robot's set of observations at runtime is not supported. However, if a robot has either a `ScanSensor` or `VisionSensor` onboard, these sensors can have their set of active modalities be dynamically updated. This is achieved by directly calling `add_modality()` or `remove_modality()` on a specific sensor. An example is shown below:

??? code "modify_fetch_sensor.py"
    ``` python linenums="1"

    import omnigibson as og
    from omnigibson.scenes import Scene
    from omnigibson.robots import Turtlebot

    # Launch OG
    og.launch()
    scene = Scene()
    og.sim.import_scene(scene)

    # Import robot and play sim
    robot = Turtlebot(
        name="agent",
        obs_modalities=["scan", "rgb", "depth", "proprio"],
        proprio_obs=["robot_lin_vel", "robot_ang_vel"],
        sensor_config={
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": 128,
                    "image_width": 128,
                },
            },
            "ScanSensor": {
                "sensor_kwargs": {
                    "min_range": 0.05,
                    "max_range": 10.0,
                },
            },
        },
    )
    scene.add_object(robot)
    og.sim.play()

    # Add the occupancy grid modality to the robot's scan sensor
    robot.sensors["agent:scan_link:Lidar:0"].add_modality("occupancy_grid")

    # Remove the depth modality and add semantic segmentation ot the robot's camera sensor
    robot.sensors["agent:eyes:Camera:0"].remove_modality("depth")
    robot.sensors["agent:eyes:Camera:0"].add_modality("seg_semantic")
    ```

