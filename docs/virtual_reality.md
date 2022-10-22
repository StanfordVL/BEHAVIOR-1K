### Virtual Reality Overview

Virtual reality is currently supported on Windows 10/11 on the HTC Vive and Oculus Rift/Quest (with link), and on Linux with the HTC Vive (there is no linux driver for Oculus).

The HTC Vive Pro Eye tracking driver is not available for Linux. You must have the latest Nvidia driver 470.XX and SteamVR 1.19.7 as asynchronous reprojection (a form of motion smoothing necessary for a usable VR experience) was only added in mid 2021.

### Setup
1. Set up the HTC Vive VR hardware according to the [setup guide](https://support.steampowered.com/steamvr/HTC_Vive/)

2. (optional) if you plan to use eye tracking on Windows, create a [vive developer account](https://hub.vive.com/sso/login) then download and install the [SRAnipal runtime](https://developer.vive.com/resources/vive-sense/sdk/vive-eye-and-facial-tracking-sdk/). Note you should [calibrate](https://developer.vive.com/us/support/sdk/category_howto/how-to-calibrate-eye-tracking.html) the Vive eye tracker before each recording session.

3. Ensure you have installed OmniGibson according to the installation [instructions](http://svl.stanford.edu/omnigibson/docs/installation.html#installation-method). 
    * Note: On Windows VR support is enabled by default. On Linux, you must install with an additional environmental variable `USE_VR=TRUE pip install -e .`. You must also have addition development headers installed: on Ubuntu `sudo apt install xorg-dev` and on Centos/Fedora: `sudo dnf install libXinerama-devel libXi-devel libXrandr-devel libXcursor-devel`.

### VR examples

We have several examples showing how to use our VR interface:

* vr demo files: `omnigibson/examples/vr`

* multi-user VR (experimental): `omnigibson/examples/vr/muvr` 

* benchmark: `omnigibson/examples/vr/in_development`
  - `vr_hand_dex_benchmark.py` -- Demonstrates various challenging manipulation tasks
  - `vr_hand_speed_benchmark.py` -- Demonstrates various tasks to assess user speed at working in VR


* data_save_replay: `omnigibson/examples/vr/data_save_replay` 
  - This folder demonstrates how to save VR recordings and "replay" them to reproduce the recorded trajectory

* robot_embodiment: `omnigibson/examples/vr/robot_embodiment` 
  - This folder demonstrates the VR interface to the Fetch robot (WIP)

Benchmarks:

We have two benchmarks - a hand and speed benchmark, both of which can be found in the top level of the vr examples folder. In these demos, you can time yourself performing various challenges,
such as putting objects away into specific containers/cabinets. Please see the comments in these demo files for more information.

### VR config and button mapping:

You can find the global vr settings in the `vr_config.yaml` in the omnigibson folder. We highly recommend reading through the default config as it is heavily documented. The three most crucial settings are **current_device**, **use_tracked_body**, and **torso_tracker_serial**.

* `current_device`: determines which of the `device_settings` keys will be used, and is used to set options specific to Oculus or the HTC Vive. The currently available keys, as seen in `device_settings` are `HTC_VIVE_PRO_EYE` and `OCULUS`
* `use_tracked_body`: determines if we will use [HTC Vive Trackers](https://developer.vive.com/us/support/sdk/category_howto/how-to-calibrate-eye-tracking.html) to track the body instead of inferring body position from the headset position.
* `torso_tracker_serial`: is the serial number of the tracker used if `use_tracked_body` is `True`.

Some additional options you may be interested in changing:
* `use_vr` (default True): whether to render to the HMD and use VR system or just render to screen (used for debugging)
* `eye_tracking` (default True): whether to use eye tracking. Turn this off if your headset does not support eye tracking (only the HTC Vive Pro Eye does)
* `touchpad_movement` (default True): whether to enable use of touchpad to move - this will help you get around large scenes if your play area is small
* `movement_controller` (default 'right'): device to controler movement - can be right or left (representing the corresponding controllers). Default is for right-handed people - please change to left if that is more comfortable.
* `relative_movement_device` (default 'hmd'): which device to use to control touchpad movement direction (can be any VR device). You should not need to change this.
* `movement_speed` (default 0.01): how fast you move when using the touchpad. This number has been calibrated carefully, however feel free to change it if you want to speed up/slow down.

We recommend looking at `omnigibson/render/mesh_renderer/mesh_renderer_vr.py` to see the VrSettings class which reads from `vr_config.yaml`. A single VrSettings object is created and passed in to the `Simulator` constructor.

Note(optional): If you are using a device not already mapped, please run `omnigibson/examples/vr/in_development/vr_button_mapping.py` to figure out which physical controller buttons correspond to which indices in OpenVR.

### Mirroring the VR view on the monitor

OmniGibson VR utilizes Steam VR's built-in companion window to visualize what the user sees in their headset. To open this window: 
* launch Steam VR
* click on the hamburger menu in the top-left corner of the VR status menu (the dark blue window with icons for the VR devices)
* then click "Display VR View" button. 

From this view, you can change which eye you are looking at (or look at both), and can make the window fullscreen.
Note that this window will be black until an application is running, and the headset is detected by the lighthouse sensors. We also support a custom-build companion window that can run in OmniGibson - this can be enabled in the vr_config file, described below (although it is off by default).

Note: Press ESCAPE to force the fullscreen rendering window to close during program execution (although fullscreen is disabled by default)

### Contributing 
* Most VR functions can be found in `omnigibson/simulator.py`
* The BehaviorRobot is located in `omnigibson/robots/behavior_robot.py`
* VR utility functions are found in `omnigibson/utils/vr_utils.py`
* The VR renderer can be found in `omnigibson/render/mesh_renderer.py`
* The underlying VR C++ code (querying controller states from openvr, renderer for VR) can be found in `omnigibson/render/cpp/vr_mesh_renderer{.cpp,.h}

