# setup.ps1 - BEHAVIOR-1K Installation Script (Windows)
param(
    [switch]$Help,
    [switch]$NewEnv,
    [switch]$OmniGibson,
    [switch]$BDDL,
    [switch]$Teleop,
    [switch]$Dataset,
    [switch]$Primitives,
    [switch]$Dev,
    [string]$CudaVersion = "12.4"
)

# Set error action preference to stop on errors
$ErrorActionPreference = "Stop"

# Show help if no parameters or help requested
if ($Help -or $PSBoundParameters.Count -eq 0) {
    Write-Host @"
BEHAVIOR-1K Installation Script (Windows)
Usage: .\setup.ps1 [OPTIONS]

Options:
  -Help                   Display this help message
  -NewEnv                 Create a new conda environment 'behavior'
  -OmniGibson             Install OmniGibson (core physics simulator)
  -BDDL                   Install BDDL (Behavior Domain Definition Language)
  -Teleop                 Install JoyLo (teleoperation interface)
  -Dataset                Download BEHAVIOR datasets (requires -OmniGibson)
  -Primitives             Install OmniGibson with primitives support
  -Dev                    Install development dependencies
  -CudaVersion VERSION    Specify CUDA version (default: 12.4)

Example: .\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
"@
    exit 0
}

# Validate dependencies
if ($OmniGibson -and -not $BDDL) {
    Write-Error "ERROR: -OmniGibson requires -BDDL"
    exit 1
}

if ($Dataset -and -not $OmniGibson) {
    Write-Error "ERROR: -Dataset requires -OmniGibson"
    exit 1
}

if ($Primitives -and -not $OmniGibson) {
    Write-Error "ERROR: -Primitives requires -OmniGibson"
    exit 1
}

$WorkDir = Get-Location

# Helper function to check if conda environment exists
function Test-CondaEnvironment {
    param([string]$EnvName)
    $envList = conda env list 2>$null | Out-String
    return $envList -match "^$EnvName\s"
}

# Helper function to activate conda environment
function Invoke-CondaActivate {
    param([string]$EnvName)
    
    # Get conda base path
    $condaInfo = conda info --base 2>$null
    if (-not $condaInfo) {
        Write-Error "ERROR: Failed to get conda info"
        exit 1
    }
    
    # Source conda initialization
    $condaHook = "$condaInfo\shell\condabin\conda-hook.ps1"
    if (Test-Path $condaHook) {
        & $condaHook
    }
    
    conda activate $EnvName
    
    if ($env:CONDA_DEFAULT_ENV -ne $EnvName) {
        Write-Error "ERROR: Failed to activate environment '$EnvName'"
        exit 1
    }
}

# Create conda environment
if ($NewEnv) {
    Write-Host "Creating conda environment 'behavior'..."
    
    # Check if conda is available
    try {
        conda --version | Out-Null
    }
    catch {
        Write-Error "ERROR: Conda not found"
        exit 1
    }
    
    # Remove existing environment if it exists
    if (Test-CondaEnvironment "behavior") {
        conda env remove -n behavior -y
    }
    
    # Create new environment
    conda create -n behavior python=3.10 pytorch torchvision torchaudio pytorch-cuda=$CudaVersion "numpy<2" -c pytorch -c nvidia -y
    
    # Activate environment
    Invoke-CondaActivate "behavior"
}

# Setup Isaac Sim conda environment helper functions
function Set-IsaacSimCondaEnv {
    param(
        [string]$IsaacSimPath,
        [string]$CondaPrefix
    )
    
    # Create directories
    $activateDir = Join-Path $CondaPrefix "etc\conda\activate.d"
    $deactivateDir = Join-Path $CondaPrefix "etc\conda\deactivate.d"
    
    New-Item -ItemType Directory -Path $activateDir -Force | Out-Null
    New-Item -ItemType Directory -Path $deactivateDir -Force | Out-Null
    
    # Create empty files
    foreach ($dir in @($activateDir, $deactivateDir)) {
        foreach ($file in @("env_vars.bat", "env_vars.ps1")) {
            $filePath = Join-Path $dir $file
            if (-not (Test-Path $filePath)) {
                New-Item -ItemType File -Path $filePath -Force | Out-Null
            }
        }
    }
    
    # Setup CMD activation script
    $cmdActFile = Join-Path $activateDir "env_vars.bat"
    @"
@echo off
set PYTHONPATH_OLD=%PYTHONPATH%
set PYTHONPATH=%PYTHONPATH%;$IsaacSimPath\site
set CARB_APP_PATH=$IsaacSimPath\kit
set EXP_PATH=$IsaacSimPath\apps
set ISAAC_PATH=$IsaacSimPath
"@ | Out-File -FilePath $cmdActFile -Encoding ascii
    
    # Setup CMD deactivation script
    $cmdDeactFile = Join-Path $deactivateDir "env_vars.bat"
    @"
@echo off
set PYTHONPATH=%PYTHONPATH_OLD%
set PYTHONPATH_OLD=""
"@ | Out-File -FilePath $cmdDeactFile -Encoding ascii
    
    # Setup PowerShell activation script
    $psActFile = Join-Path $activateDir "env_vars.ps1"
    @"
`$env:PYTHONPATH_OLD="`$env:PYTHONPATH"
`$env:PYTHONPATH="`$env:PYTHONPATH;$IsaacSimPath\site"
`$env:CARB_APP_PATH="$IsaacSimPath\kit"
`$env:EXP_PATH="$IsaacSimPath\apps"
`$env:ISAAC_PATH="$IsaacSimPath"
"@ | Out-File -FilePath $psActFile -Encoding utf8
    
    # Setup PowerShell deactivation script
    $psDeactFile = Join-Path $deactivateDir "env_vars.ps1"
    @"
`$env:PYTHONPATH="`$env:PYTHONPATH_OLD"
`$env:PYTHONPATH_OLD=`$null
"@ | Out-File -FilePath $psDeactFile -Encoding utf8
}

# Find Isaac Sim installation
function Find-IsaacSimPath {
    $basePath = Join-Path $env:USERPROFILE "AppData\Local\ov\pkg"
    
    if (-not (Test-Path $basePath)) {
        return $null
    }
    
    $isaacDirs = Get-ChildItem -Path $basePath -Directory -Name "isaac*" | Sort-Object
    if ($isaacDirs.Count -eq 0) {
        return $null
    }
    
    return Join-Path $basePath $isaacDirs[-1]
}

# Install BDDL
if ($BDDL) {
    Write-Host "Installing BDDL..."
    
    if (-not (Test-Path "bddl")) {
        Write-Error "ERROR: bddl directory not found"
        exit 1
    }
    
    pip install -e "$WorkDir\bddl"
}

# Install OmniGibson with Isaac Sim
if ($OmniGibson) {
    Write-Host "Installing OmniGibson..."
    
    if (-not (Test-Path "OmniGibson")) {
        Write-Error "ERROR: OmniGibson directory not found"
        exit 1
    }
    
    # Pre-installation checks
    if (-not $env:CONDA_PREFIX) {
        Write-Error "ERROR: Must run in conda environment"
        exit 1
    }
    
    # Check Python version
    $pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pythonVersion -ne "3.10") {
        Write-Error "ERROR: Python 3.10 required, found $pythonVersion"
        exit 1
    }
    
    # Check for conflicting environment variables
    if ($env:EXP_PATH -or $env:CARB_APP_PATH -or $env:ISAAC_PATH) {
        Write-Error @"
ERROR: Found existing Isaac Sim environment variables.
Please unset EXP_PATH, CARB_APP_PATH, and ISAAC_PATH and restart.
"@
        exit 1
    }
    
    # Build extras string
    $extras = ""
    if ($Dev -and $Primitives) { $extras = "[dev,primitives]" }
    elseif ($Dev) { $extras = "[dev]" }
    elseif ($Primitives) { $extras = "[primitives]" }
    
    pip install -e "$WorkDir\OmniGibson$extras"
    
    # Install pre-commit for dev setup
    if ($Dev) {
        Write-Host "Setting up pre-commit..."
        conda install -c conda-forge pre-commit -y
        Push-Location "$WorkDir\OmniGibson"
        pre-commit install
        Pop-Location
    }
    
    # Isaac Sim installation via pip
    $env:OMNI_KIT_ACCEPT_EULA = "YES"
    
    # Check if already installed
    $isaacInstalled = $false
    try {
        python -c "import isaacsim" 2>$null
        $isaacInstalled = $true
        Write-Host "Isaac Sim already installed, skipping..."
    }
    catch {
        Write-Host "Installing Isaac Sim via pip..."
    }
    
    if (-not $isaacInstalled) {
        # Isaac Sim packages to install
        $packages = @(
            "omniverse_kit-106.5.0.162521", "isaacsim_kernel-4.5.0.0", "isaacsim_app-4.5.0.0",
            "isaacsim_core-4.5.0.0", "isaacsim_gui-4.5.0.0", "isaacsim_utils-4.5.0.0",
            "isaacsim_storage-4.5.0.0", "isaacsim_asset-4.5.0.0", "isaacsim_sensor-4.5.0.0",
            "isaacsim_robot_motion-4.5.0.0", "isaacsim_robot-4.5.0.0", "isaacsim_benchmark-4.5.0.0",
            "isaacsim_code_editor-4.5.0.0", "isaacsim_ros1-4.5.0.0", "isaacsim_cortex-4.5.0.0",
            "isaacsim_example-4.5.0.0", "isaacsim_replicator-4.5.0.0", "isaacsim_rl-4.5.0.0",
            "isaacsim_robot_setup-4.5.0.0", "isaacsim_ros2-4.5.0.0", "isaacsim_template-4.5.0.0",
            "isaacsim_test-4.5.0.0", "isaacsim-4.5.0.0", "isaacsim_extscache_physics-4.5.0.0",
            "isaacsim_extscache_kit-4.5.0.0", "isaacsim_extscache_kit_sdk-4.5.0.0"
        )
        
        $tempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }
        $wheelFiles = @()
        
        try {
            foreach ($pkg in $packages) {
                $pkgParts = $pkg -split "-"
                $pkgName = ($pkgParts[0..$($pkgParts.Length-2)] -join "-").Replace("_", "-")
                $filename = "$pkg-cp310-none-win_amd64.whl"
                $url = "https://pypi.nvidia.com/$pkgName/$filename"
                $filepath = Join-Path $tempDir $filename
                
                Write-Host "Downloading $pkg..."
                try {
                    Invoke-WebRequest -Uri $url -OutFile $filepath -UseBasicParsing
                    $wheelFiles += $filepath
                }
                catch {
                    Write-Error "ERROR: Failed to download $pkg"
                    exit 1
                }
            }
            
            Write-Host "Installing Isaac Sim packages..."
            pip install $wheelFiles
            
            # Verify installation
            try {
                python -c "import isaacsim" 2>$null
            }
            catch {
                Write-Error "ERROR: Isaac Sim installation verification failed"
                exit 1
            }
        }
        finally {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        # Fix cryptography conflict
        if ($env:ISAAC_PATH) {
            $cryptographyPath = Join-Path $env:ISAAC_PATH "exts\omni.pip.cloud\pip_prebundle\cryptography"
            if (Test-Path $cryptographyPath) {
                Write-Host "Fixing cryptography conflict..."
                Remove-Item -Path $cryptographyPath -Recurse -Force
            }
        }
    }
    
    # Install datasets
    if ($Dataset) {
        Write-Host "Installing datasets..."
        
        $pythonScript = @"
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
"@
        
        python -c $pythonScript
    }
    
    Write-Host "OmniGibson installation completed successfully!"
}

# Install JoyLo
if ($Teleop) {
    Write-Host "Installing JoyLo..."
    
    if (-not (Test-Path "joylo")) {
        Write-Error "ERROR: joylo directory not found"
        exit 1
    }
    
    pip install -e "$WorkDir\joylo"
}

# Installation summary
Write-Host ""
Write-Host "=== Installation Complete! ==="
if ($NewEnv) { Write-Host "✓ Created conda environment 'behavior'" }
if ($OmniGibson) { Write-Host "✓ Installed OmniGibson + Isaac Sim" }
if ($BDDL) { Write-Host "✓ Installed BDDL" }
if ($Teleop) { Write-Host "✓ Installed JoyLo" }
if ($Dataset) { Write-Host "✓ Downloaded datasets" }
Write-Host ""
if ($NewEnv) { Write-Host "To activate: conda activate behavior" }