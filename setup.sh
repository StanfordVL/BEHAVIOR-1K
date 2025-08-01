#!/bin/bash
set -e

# Parse arguments
HELP=false
NEW_ENV=false
OMNIGIBSON=false
BDDL=false
TELEOP=false
DATASET=false
PRIMITIVES=false
DEV=false
CUDA_VERSION="12.4"

[ "$#" -eq 0 ] && HELP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) HELP=true; shift ;;
        --new-env) NEW_ENV=true; shift ;;
        --omnigibson) OMNIGIBSON=true; shift ;;
        --bddl) BDDL=true; shift ;;
        --teleop) TELEOP=true; shift ;;
        --dataset) DATASET=true; shift ;;
        --primitives) PRIMITIVES=true; shift ;;
        --dev) DEV=true; shift ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$HELP" = true ]; then
    cat << EOF
BEHAVIOR-1K Installation Script (Linux)
Usage: ./setup.sh [OPTIONS]

Options:
  -h, --help              Display this help message
  --new-env               Create a new conda environment 'behavior'
  --omnigibson            Install OmniGibson (core physics simulator)
  --bddl                  Install BDDL (Behavior Domain Definition Language)
  --teleop                Install JoyLo (teleoperation interface)
  --dataset               Download BEHAVIOR datasets (requires --omnigibson)
  --primitives            Install OmniGibson with primitives support
  --dev                   Install development dependencies
  --cuda-version VERSION  Specify CUDA version (default: 12.4)

Example: ./setup.sh --new-env --omnigibson --bddl --teleop --dataset
EOF
    exit 0
fi

# Validate dependencies
[ "$OMNIGIBSON" = true ] && [ "$BDDL" = false ] && { echo "ERROR: --omnigibson requires --bddl"; exit 1; }
[ "$DATASET" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --dataset requires --omnigibson"; exit 1; }
[ "$PRIMITIVES" = true ] && [ "$OMNIGIBSON" = false ] && { echo "ERROR: --primitives requires --omnigibson"; exit 1; }

WORKDIR=$(pwd)

# Create conda environment
if [ "$NEW_ENV" = true ]; then
    echo "Creating conda environment 'behavior'..."
    command -v conda >/dev/null || { echo "ERROR: Conda not found"; exit 1; }
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda env list | grep -q "^behavior " && conda env remove -n behavior -y
    conda create -n behavior python=3.10 pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION "numpy<2" -c pytorch -c nvidia -y
    conda activate behavior
    
    [[ "$CONDA_DEFAULT_ENV" != "behavior" ]] && { echo "ERROR: Failed to activate environment"; exit 1; }
fi

# Install BDDL
if [ "$BDDL" = true ]; then
    echo "Installing BDDL..."
    [ ! -d "bddl" ] && { echo "ERROR: bddl directory not found"; exit 1; }
    pip install -e "$WORKDIR/bddl"
fi

# Install OmniGibson with Isaac Sim
if [ "$OMNIGIBSON" = true ]; then
    echo "Installing OmniGibson..."
    [ ! -d "OmniGibson" ] && { echo "ERROR: OmniGibson directory not found"; exit 1; }
    
    # Pre-installation checks
    [ -z "$CONDA_PREFIX" ] && { echo "ERROR: Must run in conda environment"; exit 1; }
    
    # Check Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    [ "$PYTHON_VERSION" != "3.10" ] && { echo "ERROR: Python 3.10 required, found $PYTHON_VERSION"; exit 1; }
    
    # Check for conflicting environment variables
    if [[ -n "$EXP_PATH" || -n "$CARB_APP_PATH" || -n "$ISAAC_PATH" ]]; then
        echo "ERROR: Found existing Isaac Sim environment variables."
        echo "Please unset EXP_PATH, CARB_APP_PATH, and ISAAC_PATH and restart."
        exit 1
    fi
    
    # Build extras
    EXTRAS=""
    [ "$DEV" = true ] && [ "$PRIMITIVES" = true ] && EXTRAS="[dev,primitives]"
    [ "$DEV" = true ] && [ "$PRIMITIVES" = false ] && EXTRAS="[dev]"
    [ "$DEV" = false ] && [ "$PRIMITIVES" = true ] && EXTRAS="[primitives]"
    
    pip install -e "$WORKDIR/OmniGibson$EXTRAS"

    # Install pre-commit for dev setup
    if [ "$DEV" = true ]; then
        echo "Setting up pre-commit..."
        conda install -c conda-forge pre-commit -y
        cd "$WORKDIR/OmniGibson"
        pre-commit install
        cd "$WORKDIR"
    fi
    
    # Isaac Sim installation via pip
    export OMNI_KIT_ACCEPT_EULA=YES
    
    # Check if already installed
    if python -c "import isaacsim" 2>/dev/null; then
        echo "Isaac Sim already installed, skipping..."
    else
        echo "Installing Isaac Sim via pip..."
        
        # Helper functions
        check_glibc_old() {
            ldd --version 2>&1 | grep -qE "2\.(31|32|33)"
        }
        
        install_isaac_packages() {
            local temp_dir=$(mktemp -d)
            local packages=(
                "omniverse_kit-106.5.0.162521" "isaacsim_kernel-4.5.0.0" "isaacsim_app-4.5.0.0"
                "isaacsim_core-4.5.0.0" "isaacsim_gui-4.5.0.0" "isaacsim_utils-4.5.0.0"
                "isaacsim_storage-4.5.0.0" "isaacsim_asset-4.5.0.0" "isaacsim_sensor-4.5.0.0"
                "isaacsim_robot_motion-4.5.0.0" "isaacsim_robot-4.5.0.0" "isaacsim_benchmark-4.5.0.0"
                "isaacsim_code_editor-4.5.0.0" "isaacsim_ros1-4.5.0.0" "isaacsim_cortex-4.5.0.0"
                "isaacsim_example-4.5.0.0" "isaacsim_replicator-4.5.0.0" "isaacsim_rl-4.5.0.0"
                "isaacsim_robot_setup-4.5.0.0" "isaacsim_ros2-4.5.0.0" "isaacsim_template-4.5.0.0"
                "isaacsim_test-4.5.0.0" "isaacsim-4.5.0.0" "isaacsim_extscache_physics-4.5.0.0"
                "isaacsim_extscache_kit-4.5.0.0" "isaacsim_extscache_kit_sdk-4.5.0.0"
            )
            
            local wheel_files=()
            for pkg in "${packages[@]}"; do
                local pkg_name=${pkg%-*}
                local filename="${pkg}-cp310-none-manylinux_2_34_x86_64.whl"
                local url="https://pypi.nvidia.com/${pkg_name//_/-}/$filename"
                local filepath="$temp_dir/$filename"
                
                echo "Downloading $pkg..."
                if ! curl -sL "$url" -o "$filepath"; then
                    echo "ERROR: Failed to download $pkg"
                    rm -rf "$temp_dir"
                    return 1
                fi
                
                # Rename for older GLIBC
                if check_glibc_old; then
                    local new_filepath="${filepath/manylinux_2_34/manylinux_2_31}"
                    mv "$filepath" "$new_filepath"
                    filepath="$new_filepath"
                fi
                
                wheel_files+=("$filepath")
            done
            
            echo "Installing Isaac Sim packages..."
            pip install "${wheel_files[@]}"
            rm -rf "$temp_dir"
            
            # Verify installation
            if ! python -c "import isaacsim" 2>/dev/null; then
                echo "ERROR: Isaac Sim installation verification failed"
                return 1
            fi
        }
        
        install_isaac_packages || { echo "ERROR: Isaac Sim installation failed"; exit 1; }
        
        # Fix cryptography conflict - remove conflicting version
        if [ -n "$ISAAC_PATH" ] && [ -d "$ISAAC_PATH/exts/omni.pip.cloud/pip_prebundle/cryptography" ]; then
            echo "Fixing cryptography conflict..."
            rm -rf "$ISAAC_PATH/exts/omni.pip.cloud/pip_prebundle/cryptography"
        fi
    fi
    
    # Install datasets
    if [ "$DATASET" = true ]; then
        echo "Installing datasets..."
        python -c "
import os
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
try:
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import download_assets, download_og_dataset
    
    dataset_exists = os.path.exists(gm.DATASET_PATH)
    assets_exist = os.path.exists(gm.ASSET_PATH)
    
    if not (dataset_exists and assets_exist):
        print(f'Installing data to:')
        print(f'  Dataset (~25GB): {gm.DATASET_PATH}')
        print(f'  Assets (~2.5GB): {gm.ASSET_PATH}')
        
        if not dataset_exists:
            print('Downloading dataset...')
            download_og_dataset()
        
        if not assets_exist:
            print('Downloading assets...')
            download_assets()
    else:
        print('Datasets already exist, skipping download.')
except Exception as e:
    print(f'ERROR: Dataset installation failed: {e}')
    exit(1)
"
    fi
    
    echo "OmniGibson installation completed successfully!"
fi

# Install JoyLo
if [ "$TELEOP" = true ]; then
    echo "Installing JoyLo..."
    [ ! -d "joylo" ] && { echo "ERROR: joylo directory not found"; exit 1; }
    pip install -e "$WORKDIR/joylo"
fi

echo ""
echo "=== Installation Complete! ==="
[ "$NEW_ENV" = true ] && echo "✓ Created conda environment 'behavior'"
[ "$OMNIGIBSON" = true ] && echo "✓ Installed OmniGibson + Isaac Sim"
[ "$BDDL" = true ] && echo "✓ Installed BDDL"
[ "$TELEOP" = true ] && echo "✓ Installed JoyLo"
[ "$DATASET" = true ] && echo "✓ Downloaded datasets"
echo ""
[ "$NEW_ENV" = true ] && echo "To activate: conda activate behavior"