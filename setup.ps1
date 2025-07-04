#Requires -Version 5.1
<#
.SYNOPSIS
    BEHAVIOR-1K Installation Script for Windows

.DESCRIPTION
    PowerShell installation script for BEHAVIOR-1K components including OmniGibson, BDDL, and JoyLo teleoperation.

.PARAMETER Help
    Display help message

.PARAMETER NewEnv
    Create a new conda environment 'behavior'

.PARAMETER OmniGibson
    Install OmniGibson (core physics simulator)

.PARAMETER BDDL
    Install BDDL (Behavior Domain Definition Language)

.PARAMETER Teleop
    Install JoyLo (teleoperation interface)

.PARAMETER Dataset
    Download BEHAVIOR datasets (requires -OmniGibson)

.PARAMETER Primitives
    Install OmniGibson with primitives support

.PARAMETER Dev
    Install development dependencies

.PARAMETER CudaVersion
    Specify CUDA version (default: 12.1)

.EXAMPLE
    .\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset
    Full installation with new environment

.EXAMPLE
    .\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dev -Primitives  
    Development setup

.EXAMPLE
    .\setup.ps1 -OmniGibson
    Minimal OmniGibson only

.EXAMPLE
    .\setup.ps1 -NewEnv -OmniGibson -Teleop -Dataset
    Teleoperation setup

.NOTES
    Author: BEHAVIOR-1K Team
    Requires: PowerShell 5.1+, Conda, Python 3.10
#>

[CmdletBinding()]
param(
    [Parameter(HelpMessage="Display help message")]
    [switch]$Help,
    
    [Parameter(HelpMessage="Create a new conda environment 'behavior'")]
    [switch]$NewEnv,
    
    [Parameter(HelpMessage="Install OmniGibson (core physics simulator)")]
    [switch]$OmniGibson,
    
    [Parameter(HelpMessage="Install BDDL (Behavior Domain Definition Language)")]
    [switch]$BDDL,
    
    [Parameter(HelpMessage="Install JoyLo (teleoperation interface)")]
    [switch]$Teleop,
    
    [Parameter(HelpMessage="Download BEHAVIOR datasets (requires -OmniGibson)")]
    [switch]$Dataset,
    
    [Parameter(HelpMessage="Install OmniGibson with primitives support")]
    [switch]$Primitives,
    
    [Parameter(HelpMessage="Install development dependencies")]
    [switch]$Dev,
    
    [Parameter(HelpMessage="Specify CUDA version (default: 12.1)")]
    [string]$CudaVersion = "12.1"
)

# Function to display help
function Show-Help {
    Write-Host "BEHAVIOR-1K Installation Script for Windows" -ForegroundColor Cyan
    Write-Host "Usage: .\setup.ps1 [OPTIONS]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Help                   Display this help message" -ForegroundColor White
    Write-Host "  -NewEnv                 Create a new conda environment 'behavior'" -ForegroundColor White
    Write-Host "  -OmniGibson             Install OmniGibson (core physics simulator)" -ForegroundColor White
    Write-Host "  -BDDL                   Install BDDL (Behavior Domain Definition Language)" -ForegroundColor White
    Write-Host "  -Teleop                 Install JoyLo (teleoperation interface)" -ForegroundColor White
    Write-Host "  -Dataset                Download BEHAVIOR datasets (requires -OmniGibson)" -ForegroundColor White
    Write-Host "  -Primitives             Install OmniGibson with primitives support" -ForegroundColor White
    Write-Host "  -Dev                    Install development dependencies" -ForegroundColor White
    Write-Host "  -CudaVersion VERSION    Specify CUDA version (default: 12.1)" -ForegroundColor White
    Write-Host ""
    Write-Host "Common usage patterns:" -ForegroundColor Yellow
    Write-Host "  # Full installation with new environment" -ForegroundColor Green
    Write-Host "  .\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dataset" -ForegroundColor White
    Write-Host ""
    Write-Host "  # Development setup" -ForegroundColor Green
    Write-Host "  .\setup.ps1 -NewEnv -OmniGibson -BDDL -Teleop -Dev -Primitives" -ForegroundColor White
    Write-Host ""
    Write-Host "  # Minimal OmniGibson only" -ForegroundColor Green
    Write-Host "  .\setup.ps1 -OmniGibson" -ForegroundColor White
    Write-Host ""
    Write-Host "  # Teleoperation setup" -ForegroundColor Green
    Write-Host "  .\setup.ps1 -NewEnv -OmniGibson -Teleop -Dataset" -ForegroundColor White
    Write-Host ""
    exit 0
}

# Function to write colored output
function Write-Step {
    param([string]$Component, [string]$Message, [string]$Color = "Cyan")
    Write-Host "[$Component] $Message" -ForegroundColor $Color
}

function Write-Error-Step {
    param([string]$Component, [string]$Message)
    Write-Host "[$Component] $Message" -ForegroundColor Red
}

function Write-Warning-Step {
    param([string]$Component, [string]$Message)
    Write-Host "[$Component] $Message" -ForegroundColor Yellow
}

function Write-Success-Step {
    param([string]$Component, [string]$Message)
    Write-Host "[$Component] $Message" -ForegroundColor Green
}

# Show help if no parameters provided or Help switch is used
if (-not $PSBoundParameters.Count -or $Help) {
    Show-Help
}

# Check if we're in the right directory
$requiredDirs = @("omnigibson", "bddl", "joylo")
$foundDirs = @()
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir -PathType Container) {
        $foundDirs += $dir
    }
}

if ($foundDirs.Count -eq 0) {
    Write-Error-Step "ERROR" "Cannot find omnigibson, bddl, or joylo directories"
    Write-Error-Step "ERROR" "Please run this script from the BEHAVIOR-1K root directory"
    exit 1
}

# Get system information
$WorkDir = Get-Location
Write-Step "SYSTEM" "Working directory: $WorkDir"

# Validate dependencies
if ($Dataset -and -not $OmniGibson) {
    Write-Error-Step "ERROR" "-Dataset requires -OmniGibson"
    exit 1
}

if ($Primitives -and -not $OmniGibson) {
    Write-Error-Step "ERROR" "-Primitives requires -OmniGibson"
    exit 1
}

# Check if conda is available
try {
    $condaVersion = conda --version 2>$null
    if (-not $condaVersion) {
        throw "Conda not found"
    }
    Write-Step "SYSTEM" "Found conda: $condaVersion"
} catch {
    Write-Error-Step "ERROR" "Conda is not installed or not in PATH"
    Write-Error-Step "HELP" "Please install Miniconda or Anaconda first"
    exit 1
}

# Create conda environment if requested
if ($NewEnv) {
    Write-Step "ENV" "Creating conda environment 'behavior'..."
    
    $condaCmd = "conda create -n behavior python=3.10 pytorch torchvision torchaudio pytorch-cuda=$CudaVersion `"numpy<2`" av `"setuptools<=79"` -c pytorch -c nvidia -c conda-forge -y"
    
    try {
        Invoke-Expression $condaCmd
        if ($LASTEXITCODE -ne 0) {
            throw "Conda create failed"
        }
    } catch {
        Write-Error-Step "ERROR" "Failed to create conda environment"
        Write-Error-Step "HELP" "Try running: conda clean --all"
        Write-Error-Step "HELP" "Or manually create environment: conda create -n behavior python=3.10"
        exit 1
    }
    
    Write-Step "ENV" "Activating conda environment 'behavior'..."
    
    # Activate conda environment for the rest of the script
    $env:CONDA_DEFAULT_ENV = "behavior"
    
    # Verify environment by checking Python version
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion) {
            Write-Step "ENV" "Python version: $pythonVersion"
        }
        
        # Check PyTorch installation
        $pytorchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>$null
        if ($pytorchCheck) {
            Write-Step "ENV" $pytorchCheck
        } else {
            Write-Warning-Step "WARNING" "PyTorch not properly installed in conda environment"
        }
    } catch {
        Write-Warning-Step "WARNING" "Could not verify Python installation"
    }
}

# Install BDDL
if ($BDDL) {
    Write-Step "BDDL" "Installing BDDL..."
    
    if (-not (Test-Path "bddl" -PathType Container)) {
        Write-Error-Step "ERROR" "bddl directory not found"
        exit 1
    }
    
    Push-Location "bddl"
    try {
        pip install -e .
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed"
        }
        Write-Success-Step "BDDL" "Installation completed successfully"
    } catch {
        Write-Error-Step "ERROR" "Failed to install BDDL"
        Pop-Location
        exit 1
    }
    Pop-Location
}

# Install OmniGibson
if ($OmniGibson) {
    Write-Step "OMNIGIBSON" "Installing OmniGibson..."
    
    if (-not (Test-Path "omnigibson" -PathType Container)) {
        Write-Error-Step "ERROR" "omnigibson directory not found"
        exit 1
    }
    
    Push-Location "omnigibson"
    try {
        # Build extra requirements string
        $extras = @()
        if ($Dev) { $extras += "dev" }
        if ($Primitives) { $extras += "primitives" }
        
        $extrasString = ""
        if ($extras.Count -gt 0) {
            $extrasString = "[" + ($extras -join ",") + "]"
        }
        
        Write-Step "OMNIGIBSON" "Installing with extras: $extrasString"
        
        $pipCmd = "pip install -e .$extrasString"
        Invoke-Expression $pipCmd
        
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed"
        }
        
        Pop-Location
        
        # Run OmniGibson post-installation
        Write-Step "OMNIGIBSON" "Running post-installation (Isaac Sim + assets)..."
        
        if ($Dataset) {
            python -m omnigibson.install
        } else {
            python -m omnigibson.install --no-install-datasets
        }
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning-Step "WARNING" "OmniGibson post-installation failed"
            Write-Warning-Step "HELP" "You may need to run 'python -m omnigibson.install' manually later"
        } else {
            Write-Success-Step "OMNIGIBSON" "Installation completed successfully"
        }
    } catch {
        Write-Error-Step "ERROR" "Failed to install OmniGibson"
        Pop-Location
        exit 1
    }
}

# Install JoyLo (teleoperation)
if ($Teleop) {
    Write-Step "TELEOP" "Installing JoyLo teleoperation interface..."
    
    if (-not (Test-Path "joylo" -PathType Container)) {
        Write-Error-Step "ERROR" "joylo directory not found"
        exit 1
    }
    
    Push-Location "joylo"
    try {
        pip install -e .
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed"
        }
        Write-Success-Step "TELEOP" "Installation completed successfully"
    } catch {
        Write-Error-Step "ERROR" "Failed to install JoyLo"
        Pop-Location
        exit 1
    }
    Pop-Location
}

# Final summary
Write-Host ""
Write-Host "=== Installation Summary ===" -ForegroundColor Cyan
if ($NewEnv) {
    Write-Host "✓ Created conda environment 'behavior'" -ForegroundColor Green
}
if ($OmniGibson) {
    Write-Host "✓ Installed OmniGibson" -ForegroundColor Green
    if ($Primitives) {
        Write-Host "  - With primitives support" -ForegroundColor White
    }
    if ($Dev) {
        Write-Host "  - With development dependencies" -ForegroundColor White
    }
    if ($Dataset) {
        Write-Host "  - With BEHAVIOR datasets" -ForegroundColor White
    }
}
if ($BDDL) {
    Write-Host "✓ Installed BDDL" -ForegroundColor Green
}
if ($Teleop) {
    Write-Host "✓ Installed JoyLo (teleoperation)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Installation completed! 🎉" -ForegroundColor Green
if ($NewEnv) {
    Write-Host "To activate the environment: conda activate behavior" -ForegroundColor Yellow
}
Write-Host "For issues or questions, please check the documentation or create an issue on GitHub." -ForegroundColor Cyan