# :material-robot-industrial: **JoyLo**

JoyLo teleoperation system is first introduced in [BEHAVIOR Robot Suite](https://behavior-robot-suite.github.io/) as an effective teleop device for R1 robot, providing a cost-effective whole-body teleoperation interface that enables seamless control of mobile manipulators with high degrees of freedom. Originally designed for real-world robots, JoyLo has been extended to support simulation environments and is now fully compatible with OmniGibson.

![JoyLo System](../assets/teleop/joylo_system_overview.png)

Additionally, in OmniGibson, we support more robots beyond the original R1, including mobile manipulators R1 (6dof arm) and R1-pro (7dof arm), as well as single arm robots like A1 (6dof) and Franka (7dof). 

![JoyLo](../assets/teleop/joylo-6-7-dof.png)

## Hardware Setup

Detailed hardware setup guides are available in the [BEHAVIOR Robot Suite](https://behavior-robot-suite.github.io/) documentation for both robot variants:

### 6 DoF R1
For the 6 DoF version, refer to the [R1 setup guide](https://behavior-robot-suite.github.io/docs/sections/joylo/overview.html).

### 7 DoF R1-Pro
For the 7 DoF version, refer to the [R1-Pro setup guide](https://behavior-robot-suite.github.io/docs/sections/joylo/joylo_7dof.html).

## Software Setup

### Prerequisites

- Linux operating system
- NVIDIA RTX-enabled GPU
- External Bluetooth dongle (recommended: [[link](https://www.amazon.com/dp/B08DFBNG7F/ref=pe_386300_442618370_TE_dp_i1?th=1)])

### Step 1: BEHAVIOR Environment Setup

Refer to the [installation guide](../getting_started/installation.md) and setup the BEHAVIOR environment with the `--joylo` flag.

### Step 2: Nintendo JoyCon Configuration

1. Configure udev rules:
```bash
sudo nano /etc/udev/rules.d/50-nintendo-switch.rules
```

3. Add the following content to the file:
```
# Switch Joy-con (L) (Bluetooth only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2006.*", MODE="0666"

# Switch Joy-con (R) (Bluetooth only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2007.*", MODE="0666"

# Switch Pro controller (USB and Bluetooth)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009", MODE="0666"
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", KERNELS=="0005:057E:2009.*", MODE="0666"

# Switch Joy-con charging grip (USB only)
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="200e", MODE="0666"

KERNEL=="js0", SUBSYSTEM=="input", MODE="0666"
```

4. Refresh udev rules:
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

5. Install Bluetooth manager:
```bash
sudo add-apt-repository universe
sudo apt-get install blueman
```

### Step 3: Connect JoyCons

#### Method 1: Using Bluetooth Manager (Recommended)
1. Ensure your external Bluetooth dongle is connected
2. Open Bluetooth Manager
3. Search for JoyCon devices and connect when they appear

#### Method 2: Using Command Line (If Method 1 fails)
1. Run the following commands:
```bash
bluetoothctl
scan on
# Wait for Joy-Con (L) and (R) to appear with their MAC addresses
# For each controller:
pair <MAC_ADDRESS>
trust <MAC_ADDRESS>
connect <MAC_ADDRESS>
```
2. Verify connection: JoyCon lights should be static (not flashing)

### Step 4: Running the System

1. Ensure JoyLo is powered on (with motors NOT connected to Dynamixel software)
2. Ensure JoyCons are connected

3. In one terminal, start the recording environment with a specified task:
```bash
python joylo/experiments/launch_nodes.py --recording_path /path/to/recording_file_name.hdf5 --task_name picking_up_trash
```

4. In another terminal, run the JoyLo node:
```bash
python joylo/experiments/run_joylo.py --gello_model r1pro --joint_config_file joint_config_{your specific gello set name}.yaml
```

### Joycon Button Mapping
![Joycon instruction](../assets/teleop/joycon_button_mapping.png)

### Usage Notes

- Press the home button on the right JoyCon to save an episode and reset the scene
- To save all episodes and exit, focus your mouse on the OmniGibson window and press Escape
- Recording file will be saved to the path specified in the launch_nodes.py command
- Fast base motion mode: Activate by pressing down on the left joystick while moving it
- Object visibility toggle: Press A button on the right JoyCon to toggle between hiding non-relevant objects and showing all objects
- JoyCon connection stability: We have noticed that sometimes the JoyCon could disconnect randomly during data collection. A team member has reported that putting the Bluetooth dongle onto USB 2.0 is more stable than USB 3.0. We will look further into this issue.

### Task Information

- To learn more about available tasks, refer to the [BEHAVIOR tasks page](../challenge/tasks.md)
- For task definitions and expected goal conditions, check the corresponding `problem0.bddl` file in the BDDL repository for the task name. This file contains information about relevant objects, initial conditions, and goal conditions.

## Troubleshooting

### JoyCon connection issues
- If JoyCons won't connect, try the command line method (Method 2 above)
- Ensure you're using an external Bluetooth dongle, as built-in Bluetooth may not be compatible
- Verify that udev rules are properly configured if devices aren't recognized
- If JoyCons disconnect randomly during data collection, try connecting the Bluetooth dongle to a USB 2.0 port instead of USB 3.0
- If the JoyCon is being used as a mouse, double check [this setting](https://askubuntu.com/a/891624) (or alternatively remove `50-joystick.conf` directly)
- If the JoyCons are connected to Ubuntu in bluetooth but are still unable to be detected from Python, try `pip uninstall hidapi`, and then `pip install hid pyglm`, and then try again

### HID issues
- If you see something like `ImportError: Unable to load any of the following libraries:libhidapi-hidraw.so libhidapi-hidraw.so.0 libhidapi-libusb.so libhidapi-libusb.so.0 libhidapi-iohidmanager.so libhidapi-iohidmanager.so.0 libhidapi.dylib hidapi.dll libhidapi-0.dll`, try `sudo apt install libhidapi-hidraw0`.

## Joycon Calibration
Connect joycons and run:
```
python joylo/scripts/calibrate_joycons.py
```

## JoyLo Joint Calibration
- JoyLo sets can be assembled in slightly different ways, resulting in different orientations of the motors and offsets between the physical motor positions and the joint positions in simulation. 
- A script for automatically determining these joint signs and offsets is available in `scripts/calibrate_joints.py`

### Example Usage
```
python joylo/scripts/calibrate_joints.py --robot R1Pro --gello_name {YOUR_GELLO_NAME_HERE}
```
(Make sure to specify the same `gello_name` argument when using your set to run the simulation).

### Reference Positions
The calibration script requires each arm to be placed in two fixed reference positions, called the "zero" and "calibration" positions. These are provided below for both the R1 (6-DOF) and R1-Pro (7-DOF) JoyLo variants.

|                      | R1 (6-DOF) | R1-Pro (7-DOF) |
|----------------------|----------- |----------------| 
| Zero Position        | ![](../assets/teleop/R1_zero_L.jpg)![](../assets/teleop/R1_zero_R.jpg)  | ![](../assets/teleop/R1pro_zero_L.jpg)![](../assets/teleop/R1pro_zero_R.jpg)                |
| Calibration Position | ![](../assets/teleop/R1_calibration_L.jpg)![](../assets/teleop/R1_calibration_R.jpg)  **NOTE: Take from the front - note the forwards orientation of the notch on the wrist joint**| ![](../assets/teleop/R1pro_calibration_L.jpg)![](../assets/teleop/R1pro_calibration_R.jpg) |
