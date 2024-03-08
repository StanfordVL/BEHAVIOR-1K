---
icon: octicons/rocket-16
---

# 🕹️ **Device**

I/O Devices are used to read user input and teleoperate simulated robots in real-time. OmniGibson leverages the [Telemoma](https://robin-lab.cs.utexas.edu/telemoma/), a modular and versatile library for manipulating mobile robots in the scene. This is achieved by using devies such as keyboards, SpaceMouse, cameras, VR devices, mobile phones, or any combination thereof. More generally, we support any interface that implements the Device abstract base class. In order to support your own custom device, simply subclass this base class and implement the required methods.

| Control | Command |
| --- | ----------- |
| Left button | rotate alone robot parts to control |
| Right button | toggle between gripper open and close |
| Move mouse laterally | move robot arm / base horizontally in xy plane|
| Move mouse vertically | move robot arm / torso vertically |
| Twist mouse about an axis | rotate robot arm / base around an axis |


