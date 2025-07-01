<h1 align="center">BEHAVIOR-1K</h1>

<span style="color:red">We are in the process of merging BEHAVIOR-1K repos into a monorepo here. You can find the usual OmniGibson source code in OmniGibson/. Thanks for your patience as we sort this move out!</span>

# 🛠️ Installation

BEHAVIOR-1K provides an installation script that handles all dependencies and components. The script supports modular installation, allowing you to install only the components you need.

## System Requirements

- **OS**: Linux (Ubuntu 20.04+), Windows 10+
- **RAM**: 32GB+ recommended
- **VRAM**: 8GB+
- **GPU**: NVIDIA RTX 2080+

## Quick Start

For most users, we recommend the full installation with a new conda environment:

### Linux
```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
. ./setup.sh --new-env --omnigibson --bddl --teleop --dataset
```

### Windows
```powershell
git clone https://github.com/StanfordVL/BEHAVIOR-1K
cd BEHAVIOR-1K
.\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
```

> **Note**: Run PowerShell as Administrator and set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Installation Options

### Available Components

| Component | Flag | Description |
|-----------|------|-------------|
| **OmniGibson** | `--omnigibson` | Core physics simulator and robotics environment |
| **BDDL** | `--bddl` | Behavior Domain Definition Language for task specification |
| **Teleoperation** | `--teleop` | JoyLo interface for robot teleoperation |

### Additional Options

| Option | Flag | Description |
|--------|------|-------------|
| **New Environment** | `--new-env` | Create a new conda environment named `behavior` |
| **Datasets** | `--dataset` | Download BEHAVIOR datasets (requires `--omnigibson`) |
| **Primitives** | `--primitives` | Install OmniGibson with action primitives support |
| **Development** | `--dev` | Install development dependencies |
| **CUDA Version** | `--cuda-version X.X` | Specify CUDA version (default: 12.1) |

To see all available options:
```bash
. ./setup.sh --help
```
