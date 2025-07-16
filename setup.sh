#!/bin/bash

# BEHAVIOR-1K Installation Script
# Usage: . ./setup.sh [OPTIONS]

# Enable strict error handling - exit on any command failure
set -e

# Read Arguments
TEMP=`getopt -o h --long help,new-env,omnigibson,bddl,teleop,dataset,primitives,dev,cuda-version: -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

# Initialize flags
HELP=false
NEW_ENV=false
OMNIGIBSON=false
BDDL=false
TELEOP=false
DATASET=false
PRIMITIVES=false
DEV=false
CUDA_VERSION="12.1"
ERROR=false

# Show help if no arguments provided
if [ "$#" -eq 1 ] ; then
    HELP=true
fi

# Parse arguments
while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --omnigibson) OMNIGIBSON=true ; shift ;;
        --bddl) BDDL=true ; shift ;;
        --teleop) TELEOP=true ; shift ;;
        --dataset) DATASET=true ; shift ;;
        --primitives) PRIMITIVES=true ; shift ;;
        --dev) DEV=true ; shift ;;
        --cuda-version) CUDA_VERSION="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

# Error handling
if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

# Help message
if [ "$HELP" = true ] ; then
    echo "BEHAVIOR-1K Installation Script"
    echo "Usage: setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment 'behavior'"
    echo "  --omnigibson            Install OmniGibson (core physics simulator)"
    echo "  --bddl                  Install BDDL (Behavior Domain Definition Language)"
    echo "  --teleop                Install JoyLo (teleoperation interface)"
    echo "  --dataset               Download BEHAVIOR datasets (requires --omnigibson)"
    echo "  --primitives            Install OmniGibson with primitives support"
    echo "  --dev                   Install development dependencies"
    echo "  --cuda-version VERSION  Specify CUDA version (default: 12.1)"
    echo ""
    echo "Common usage patterns:"
    echo "  # Full installation with new environment"
    echo "  ./setup.sh --new-env --omnigibson --bddl --teleop --dataset"
    echo ""
    echo "  # Development setup"
    echo "  ./setup.sh --new-env --omnigibson --bddl --teleop --dev --primitives"
    echo ""
    echo "  # Minimal OmniGibson only"
    echo "  ./setup.sh --omnigibson"
    echo ""
    echo "  # Teleoperation setup"
    echo "  ./setup.sh --new-env --omnigibson --teleop --dataset"
    return 0
fi

# Check if we're in the right directory
if [ ! -d "OmniGibson" ] && [ ! -d "bddl" ] && [ ! -d "joylo" ] ; then
    echo "[ERROR] Cannot find OmniGibson, bddl, or joylo directories"
    echo "[ERROR] Please run this script from the BEHAVIOR-1K root directory"
    return 1
fi

# Get system information
WORKDIR=$(pwd)
echo "[SYSTEM] Working directory: $WORKDIR"

# Validate dependencies
if [ "$DATASET" = true ] && [ "$OMNIGIBSON" = false ] ; then
    echo "[ERROR] --dataset requires --omnigibson"
    return 1
fi

if [ "$PRIMITIVES" = true ] && [ "$OMNIGIBSON" = false ] ; then
    echo "[ERROR] --primitives requires --omnigibson"
    return 1
fi

# Create conda environment if requested
if [ "$NEW_ENV" = true ] ; then
    echo "[ENV] Creating conda environment 'behavior'..."
    
    # Initialize conda for this shell session
    if ! command -v conda &> /dev/null; then
        echo "[ERROR] Conda is not installed or not in PATH"
        return 1
    fi
    
    # Source conda setup to enable conda activate
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^behavior "; then
        echo "[ENV] Removing existing 'behavior' environment..."
        conda env remove -n behavior -y
    fi
    
    conda create -n behavior python=3.10 pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION "numpy<2" -c pytorch -c nvidia -y
    
    echo "[ENV] Activating conda environment 'behavior'..."
    conda activate behavior
    
    # Verify environment activation
    if [[ "$CONDA_DEFAULT_ENV" != "behavior" ]]; then
        echo "[ERROR] Failed to activate conda environment 'behavior'"
        return 1
    fi
    
    # Verify environment
    PYTHON_VERSION=$(python --version)
    echo "[ENV] Python version: $PYTHON_VERSION"
    
    # Check PyTorch installation
    if python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        echo "[ENV] PyTorch installation verified"
    else
        echo "[ERROR] PyTorch not properly installed in conda environment"
        return 1
    fi
fi

# Install BDDL
if [ "$BDDL" = true ] ; then
    echo "[BDDL] Installing BDDL..."
    
    if [ ! -d "bddl" ] ; then
        echo "[ERROR] bddl directory not found"
        return 1
    fi
    
    pushd bddl > /dev/null
    pip install -e .
    popd > /dev/null
    
    echo "[BDDL] Installation completed successfully"
fi

# Install OmniGibson
if [ "$OMNIGIBSON" = true ] ; then
    echo "[OMNIGIBSON] Installing OmniGibson..."
    
    if [ ! -d "OmniGibson" ] ; then
        echo "[ERROR] OmniGibson directory not found"
        return 1
    fi
    
    pushd OmniGibson > /dev/null
    
    # Build extra requirements string
    EXTRAS=""
    if [ "$DEV" = true ] && [ "$PRIMITIVES" = true ] ; then
        EXTRAS="[dev,primitives]"
    elif [ "$DEV" = true ] ; then
        EXTRAS="[dev]"
    elif [ "$PRIMITIVES" = true ] ; then
        EXTRAS="[primitives]"
    fi
    
    echo "[OMNIGIBSON] Installing with extras: $EXTRAS"
    pip install -e .$EXTRAS
    
    popd > /dev/null
    
    # Run OmniGibson post-installation
    echo "[OMNIGIBSON] Running post-installation (Isaac Sim + assets)..."
    if [ "$DATASET" = true ] ; then
        python -m omnigibson.install
    else
        python -m omnigibson.install --no-install-datasets
    fi
    
    echo "[OMNIGIBSON] Installation completed successfully"
fi

# Install JoyLo (teleoperation)
if [ "$TELEOP" = true ] ; then
    echo "[TELEOP] Installing JoyLo teleoperation interface..."
    
    if [ ! -d "joylo" ] ; then
        echo "[ERROR] joylo directory not found"
        return 1
    fi
    
    pushd joylo > /dev/null
    pip install -e .
    popd > /dev/null
    
    echo "[TELEOP] Installation completed successfully"
fi

# Final summary
echo ""
echo "=== Installation Summary ==="
if [ "$NEW_ENV" = true ] ; then
    echo "✓ Created conda environment 'behavior'"
fi
if [ "$OMNIGIBSON" = true ] ; then
    echo "✓ Installed OmniGibson"
    if [ "$PRIMITIVES" = true ] ; then
        echo "  - With primitives support"
    fi
    if [ "$DEV" = true ] ; then
        echo "  - With development dependencies"
    fi
    if [ "$DATASET" = true ] ; then
        echo "  - With BEHAVIOR datasets"
    fi
fi
if [ "$BDDL" = true ] ; then
    echo "✓ Installed BDDL"
fi
if [ "$TELEOP" = true ] ; then
    echo "✓ Installed JoyLo (teleoperation)"
fi

echo ""
echo "Installation completed! 🎉"
if [ "$NEW_ENV" = true ] ; then
    echo "To activate the environment: conda activate behavior"
fi