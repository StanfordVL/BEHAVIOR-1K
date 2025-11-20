"""
Author:    Ji Yingwei
Created:   2025-11-20
Description:
    PinkIK-based VR teleoperation pipeline using network IPC.
"""

# VR PinkIK Teleoperation Setup

This guide describes how to set up a clean Conda environment and run the PinkIK teleoperation demo using the OmniGibson example `pink_ik` server.

---

## 📦 1. Create Conda Environment

```bash
cd OmniGibson/omnigibson/examples/teleoperation/pink_ik

# Create a new environment with Python 3.12
conda create -n vr_pinkik python=3.12
```

Activate the environment:

```bash
conda activate vr_pinkik
```

---

## 📚 2. Install Dependencies

Install PinkIK + visualization tools:

```bash
pip install pin-pink
pip install meshcat_shapes
```



---

## 🖥️ 3. Run the PinkIK Server

The IK server receives teleoperation inputs (including network IPC if enabled) and computes robot joint commands.

```bash
python r1_pinkIK_server.py
```



---

## 🎮 4. Run the VR Teleoperation Demo

In another terminal (hehaivor environment):

```bash
python -m omnigibson.examples.teleoperation.pink_ik.vr_robot_control_demo
```



