# Data Collection with JoyLo for OmniGibson

## Hardware Setup

### 1. JoyLo Assembly

- **7 DoF R1-Pro**: [Assembly Guide](ASSEMBLY.md)

[Assembly Video](https://github.com/user-attachments/assets/d6d3ee59-dfac-4ece-92f4-ea44619a2d05)

> **[Deprecated]** For the 6-DoF R1 version, please reference this [guide](https://behavior-robot-suite.github.io/docs/sections/joylo/overview.html) from the [BEHAVIOR Robot Suite](https://behavior-robot-suite.github.io/).

---

### 2. Nintendo JoyCon Configuration

#### Step 1: Configure udev rules

```bash
sudo nano /etc/udev/rules.d/50-nintendo-switch.rules
```

#### Step 2: Add the following content

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

#### Step 3: Refresh udev rules

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

#### Step 4: (Optional) Install Bluetooth manager

```bash
sudo add-apt-repository universe
sudo apt-get install blueman
```

---

### 3. Connecting JoyCons

#### Method 1: Using System Settings or Bluetooth Manager (Recommended)

1. Ensure your external Bluetooth dongle is connected
2. Open system Bluetooth settings or Bluetooth Manager
3. Search for JoyCon devices and connect when they appear

#### Method 2: Using Command Line (If Method 1 fails)

```bash
bluetoothctl
scan on
# Wait for Joy-Con (L) and (R) to appear with their MAC addresses
# For each controller:
pair <MAC_ADDRESS>
trust <MAC_ADDRESS>
connect <MAC_ADDRESS>
```

> **Note:** JoyCon lights should be static (not flashing) when connected successfully.

---

### 4. JoyLo Calibration

> **Important:** Install BEHAVIOR-1K (see [Software Setup](#software-setup)) before running calibration scripts.

JoyLo sets can be assembled in slightly different ways, resulting in different orientations of the motors and offsets between the physical motor positions and the joint positions in simulation.

Two calibration scripts are available:

| Script | Purpose |
|--------|---------|
| `scripts/calibrate_joycons.py` | Calibrates the joysticks on the JoyCons |
| `scripts/calibrate_joints.py` | Determines joint signs and offsets |

You need to run both scripts once before the first time you perform any data collection on the PC.

#### Running the calibrations

```bash
# Calibrate JoyCons
python joylo/scripts/calibrate_joycons.py
```

This will create two `joycon_calibration_xxx.yaml` files under `joylo/configs`.

```bash
# Calibrate joints
python joylo/scripts/calibrate_joints.py
```

This will create a `joint_config_default.yaml` under `joylo/configs`.

#### Reference Positions

The calibration script requires each arm to be placed in two fixed reference positions, called the **"zero"** and **"calibration"** positions. These are provided below for both the R1 (6-DoF) and R1-Pro (7-DoF) JoyLo variants.

|                    | R1 (6-DoF)                                                                      | R1-Pro (7-DoF)                                                                       |
|--------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Zero Position**  | ![](imgs/R1_zero_L.jpg) ![](imgs/R1_zero_R.jpg)                                | ![](imgs/R1pro_zero_L.jpg) ![](imgs/R1pro_zero_R.jpg)                              |
| **Calibration**   | ![](imgs/R1_calibration_L.jpg) ![](imgs/R1_calibration_R.jpg)                | ![](imgs/R1pro_calibration_L.jpg) ![](imgs/R1pro_calibration_R.jpg)                |
| **Note**          | Take from the front - note the forwards orientation of the wrist joint notch |                                                                                      |

---

## Software Setup

### 0. Prerequisites

- Ubuntu 22.04+
- NVIDIA RTX-enabled GPU
- External Bluetooth dongle (recommended: [Amazon Link](https://www.amazon.com/dp/B08DFBNG7F/ref=pe_386300_442618370_TE_dp_i1?th=1))

### 1. BEHAVIOR-1K Installation

All software dependencies (OmniGibson, BDDL, JoyLo, datasets) are installed via the `setup.sh` script in the BEHAVIOR-1K repository root.

```bash
cd /path/to/BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
```

This will create a new conda environment `behavior`.

Then, run datasets setup again to make sure everything is up-to-date:

```bash
conda activate behavior
./setup.sh --dataset
```

---

### 2. Running the System

> **Important:** Make sure you have run calibration scripts (see [Hardware Setup section 4](#4-joylo-calibration)) before running the following scripts.

The system runs two scripts in separate terminals. The scripts are located under `joylo/scripts`:

| Script | Purpose | Key Args |
|--------|---------|----------|
| `launch_og.py` | Starts the OmniGibson simulation server | `--robot`, `--task_name`, `--recording_path` |
| `run_joylo.py` | Starts the JoyLo teleoperation client | `--gello_model`, `--joint_config_file` |

#### Steps to Run

1. Ensure JoyLo is powered on (with motors NOT connected to Dynamixel software)
2. Ensure JoyCons are connected
3. In one terminal, start the recording environment with a specified task:

```bash
python joylo/scripts/launch_og.py --task_name turning_on_radio --recording_path /path/to/recording_file_name.hdf5
```

4. In another terminal, run the JoyLo node:

```bash
python joylo/scripts/run_joylo.py
```

---

## Usage Notes

- **Save Episode:** Press the home button on the right JoyCon to save an episode and reset the scene
- **Exit:** Focus your mouse on the OmniGibson window and press `Escape` to save all episodes and exit
- **Recording:** File will be saved to the path specified in the `launch_nodes.py` command
- **Fast Base Motion Mode:** Activate by pressing down on the left joystick while moving it
- **Object Visibility Toggle:** Press `A` button on the right JoyCon to toggle between hiding non-relevant objects and showing all objects
- **JoyCon Connection Stability:** We have noticed that sometimes the JoyCon could disconnect randomly during data collection. A team member has reported that putting the Bluetooth dongle onto USB 2.0 is more stable than USB 3.0. We will look further into this issue.
- **Available Tasks:** Listed in `datasets/2026-challenge-task-instances/metadata/available_tasks.yaml`

---

## Troubleshooting

### JoyCon Connection Issues

- If JoyCons won't connect, try the command line method (Method 2 above)
- Ensure you're using an external Bluetooth dongle, as built-in Bluetooth may not be compatible
- Verify that udev rules are properly configured if devices aren't recognized
- If JoyCons disconnect randomly during data collection, try connecting the Bluetooth dongle to a USB 2.0 port instead of USB 3.0
- If the JoyCon is being used as a mouse, double check [this setting](https://askubuntu.com/a/891624) (or alternatively remove `50-joystick.conf` directly)
- If the JoyCons are connected to Ubuntu in Bluetooth but are still unable to be detected from Python, try:
  ```bash
  pip uninstall hidapi
  pip install hid pyglm
  ```

### HID Issues

If you see something like:

```
ImportError: Unable to load any of the following libraries:
libhidapi-hidraw.so libhidapi-hidraw.so.0 libhidapi-libusb.so 
libhidapi-libusb.so.0 libhidapi-iohidmanager.so libhidapi-iohidmanager.so.0 
libhidapi.dylib hidapi.dll libhidapi-0.dll
```

Try:

```bash
sudo apt install libhidapi-hidraw0
```

---

## JoyCon Button Mapping

![Joycon instruction](https://github.com/user-attachments/assets/2e7d57d7-66be-490b-aa76-4d6f9b2ede52)
