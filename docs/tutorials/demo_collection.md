---
icon: octicons/rocket-16
---

# 🕹️ **Collecting Demonstrations**


## Devices
I/O Devices can be used to read user input and teleoperate simulated robots in real-time. OmniGibson leverages [TeleMoMa](https://robin-lab.cs.utexas.edu/telemoma/), a modular and versatile library for manipulating mobile robots in the scene. This is achieved by using devies such as keyboards, SpaceMouse, cameras, VR devices, mobile phones, or any combination thereof. More generally, we support any interface that implements the `telemoma.human_interface.teleop_core.BaseTeleopInterface` class. In order to support your own custom device, simply subclass this base class and implement the required methods. For more information on this, checkout the [TeleMoMa codebase](https://github.com/UT-Austin-RobIn/telemoma).

## Teleoperation

The following section will go through `robot_teleoperation_example.py`, which lets users to choose a robot to complete a simple pick and place task. Users are also encouraged to take a look at `vr_simple_demo.py` and `hand_tracking_demo.py`, which show how to actually render to VR headset and teleoperate `BehaviorRobot` with either VR controllers (HTC VIVE) or hand tracking (Oculus Quest).

We assume that we already have the scene and task setup. To initialize a ....... TODO

## (Optional) Saving and Loading Simulation State
You can save the current state of the simulator to a json file by calling `save`:

```
og.sim.save(JSON_PATH)
```

To restore any saved state, simply call `restore`

```
og.sim.restore(JSON_PATH)
```

Alternatively, if you just want to save all the scene and objects info at the current tiemframe, you can also call `self.scene.dump_state(serialized=True)`, which will give you a numpy array containing all the relavant information. You can then stack the array together to get the full trajectory of states.
