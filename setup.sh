#!/bin/bash

# BEHAVIOR-1K Installation Script
# Usage: ./setup.sh [OPTIONS]

# Read Arguments
TEMP=`getopt -o h --long help,new-env,omnigibson,bddl,teleop,dataset,primitives,asset-pipeline,dev,cuda-version: -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

# Initialize flags
HELP=false
NEW_ENV=false
OMNIGIBSON=false
BDDL=false
TELEOP=false
DATASET=false
PRIMITIVES=false
ASSET_PIPELINE=false
DEV=false
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
        --asset-pipeline) ASSET_PIPELINE=true ; shift ;;
        --dev) DEV=true ; shift ;;
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
    echo "  --asset-pipeline        Install the 3D scene and object asset pipeline"
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
if [ ! -d "omnigibson" ] && [ ! -d "bddl" ] && [ ! -d "joylo" ] ; then
    echo "[ERROR] Cannot find omnigibson, bddl, or joylo directories"
    echo "[ERROR] Please run this script from the BEHAVIOR-1K root directory"
    return 1
fi

# Get system information
WORKDIR=$(pwd)
echo "[SYSTEM] Working directory: $WORKDIR"

# Validate dependencies
if [ $OMNIGIBSON = true ] && [ "$BDDL" = false ] ; then
    echo "[ERROR] --omnigibson requires --bddl"
    return 1
fi

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
    conda create -n behavior python=3.10 av "setuptools<=79" -c conda-forge -y
    
    if [ $? -ne 0 ] ; then
        echo "[ERROR] Failed to create conda environment"
        echo "[HELP] Try running: conda clean --all"
        echo "[HELP] Or manually create environment: conda create -n behavior python=3.10"
        return 1
    fi
    
    echo "[ENV] Activating conda environment 'behavior'..."
    conda activate behavior
    
    # Verify environment
    PYTHON_VERSION=$(python --version)
    echo "[ENV] Python version: $PYTHON_VERSION"
    
    # Check PyTorch installation
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ $? -eq 0 ] ; then
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        echo "[ENV] PyTorch version: $PYTORCH_VERSION, CUDA available: $CUDA_AVAILABLE"
    else
        echo "[WARNING] PyTorch not properly installed in conda environment"
    fi
fi

# Install BDDL
if [ "$BDDL" = true ] ; then
    echo "[BDDL] Installing BDDL..."
    
    if [ ! -d "bddl" ] ; then
        echo "[ERROR] bddl directory not found"
        return 1
    fi
    
    cd bddl
    pip install -e .
    
    if [ $? -ne 0 ] ; then
        echo "[ERROR] Failed to install BDDL"
        return 1
    fi
    
    cd $WORKDIR
    echo "[BDDL] Installation completed successfully"
fi

# Install OmniGibson
if [ "$OMNIGIBSON" = true ] ; then
    echo "[OMNIGIBSON] Installing OmniGibson..."
    
    if [ ! -d "OmniGibson" ] ; then
        echo "[ERROR] omnigibson directory not found"
        return 1
    fi
    
    cd OmniGibson
    
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
    
    if [ $? -ne 0 ] ; then
        echo "[ERROR] Failed to install OmniGibson"
        return 1
    fi
    
    cd $WORKDIR
    
    # Run OmniGibson post-installation
    echo "[OMNIGIBSON] Running post-installation (Isaac Sim + assets)..."
    if [ "$DATASET" = true ] ; then
        python -m omnigibson.install
    else
        python -m omnigibson.install --no-install-datasets
    fi
    
    if [ $? -ne 0 ] ; then
        echo "[WARNING] OmniGibson post-installation failed"
        echo "[HELP] You may need to run 'python -m omnigibson.install' manually later"
    else
        echo "[OMNIGIBSON] Installation completed successfully"
    fi
fi

# Install JoyLo (teleoperation)
if [ "$TELEOP" = true ] ; then
    echo "[TELEOP] Installing JoyLo teleoperation interface..."
    
    if [ ! -d "joylo" ] ; then
        echo "[ERROR] joylo directory not found"
        return 1
    fi
    
    cd joylo
    pip install -e .
    
    if [ $? -ne 0 ] ; then
        echo "[ERROR] Failed to install JoyLo"
        return 1
    fi
    
    cd $WORKDIR
    echo "[TELEOP] Installation completed successfully"
fi

# Install asset pipeline
if [ "$ASSET_PIPELINE" = true ] ; then
    echo "[ASSET PIPELINE] Installing asset pipeline..."
    
    if [ ! -d "asset_pipeline" ] ; then
        echo "[ERROR] asset_pipeline directory not found"
        return 1
    fi
    
    cd asset_pipeline
    pip install -r requirements.txt
    
    if [ $? -ne 0 ] ; then
        echo "[ERROR] Failed to install asset pipeline"
        return 1
    fi
    
    cd $WORKDIR
    echo "[ASSET PIPELINE] Installation completed successfully"
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
if [ "$ASSET_PIPELINE" = true ] ; then
    echo "✓ Installed asset pipeline"
fi

echo ""
echo "Installation completed! 🎉"
if [ "$NEW_ENV" = true ] ; then
    echo "To activate the environment: conda activate behavior"
fi