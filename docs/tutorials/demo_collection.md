---
icon: material/controller
---

# 🎮 **Collecting Demonstrations**


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
