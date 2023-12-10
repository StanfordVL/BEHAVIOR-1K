@echo off
:: Make sure that the ISAAC_SIM_PATH variable is set correctly
dir /b /o:-n %userprofile%\AppData\Local\ov\pkg\isaac_sim* > NUL
if errorlevel 0 (
    for /f "tokens=* usebackq" %%f in (`dir /b /o:n %userprofile%\AppData\Local\ov\pkg\isaac_sim*`) do set ISAAC_SIM_PATH=%userprofile%\AppData\Local\ov\pkg\%%f
    setlocal enabledelayedexpansion
    echo We found Isaac Sim installed at [4m!ISAAC_SIM_PATH![0m. OmniGibson will use it by default.
    endlocal
    set /p ISAAC_SIM_PATH=If you want to use a different one, please type in the path containing isaac-sim.bat here ^(press enter to skip^) ^>^>^> 
) else (
    echo We did not find Isaac Sim under %userprofile%\AppData\Local\ov\pkg.
    echo If you haven't installed Isaac Sim yet, please do so before running this setup script.
    set /p ISAAC_SIM_PATH=If you have already installed it in a custom location, please type in the path containing isaac-sim.bat here ^>^>^> 
)
:check_isaac_sim_path
if not exist %ISAAC_SIM_PATH%\isaac-sim.bat (
    set /p ISAAC_SIM_PATH=isaac-sim.bat not found in [4m%ISAAC_SIM_PATH%[0m! Make sure you have entered the correct path ^>^>^>
    goto :check_isaac_sim_path
)
echo:
echo Using Isaac Sim at [4m%ISAAC_SIM_PATH%[0m
echo:

:: Choose venv name
set conda_name=omnigibson
echo The new conda environment will be named [4momnigibson[0m by default.
set /p conda_name=If you want to use a different name, please type in here ^(press enter to skip^) ^>^>^> 
echo:
echo Using [4m%conda_name%[0m as the conda environment name
echo:

:: Get Python version from Isaac Sim
FOR /F "tokens=*" %%g IN ('%ISAAC_SIM_PATH%\python.bat -c "import platform; print(platform.python_version())"') do (SET ISAAC_PYTHON_VERSION=%%g)
echo Using Python version [4m%ISAAC_PYTHON_VERSION%[0m matching your current Isaac Sim version

:: Create a conda environment with the appropriate python version
call conda create -y -n %conda_name% python=%ISAAC_PYTHON_VERSION% || goto :error
call conda activate %conda_name% || goto :error

:: We add some preprocessing information so that the Isaac Sim paths are linked to this environment upon startup
:: See https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux for reference
mkdir %CONDA_PREFIX%\etc\conda\activate.d
mkdir %CONDA_PREFIX%\etc\conda\deactivate.d
type NUL>%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
type NUL>%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.bat
type NUL>%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1
type NUL>%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.ps1

:: Add support for cmd
set CONDA_ACT_FILE_CMD=%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
echo @echo off>>%CONDA_ACT_FILE_CMD%
echo set PYTHONPATH_OLD=%%PYTHONPATH%%>>%CONDA_ACT_FILE_CMD%
echo set PYTHONPATH=%%PYTHONPATH%%;%ISAAC_SIM_PATH%\site>>%CONDA_ACT_FILE_CMD%
echo set CARB_APP_PATH=%ISAAC_SIM_PATH%\kit>>%CONDA_ACT_FILE_CMD%
echo set EXP_PATH=%ISAAC_SIM_PATH%\apps>>%CONDA_ACT_FILE_CMD%
echo set ISAAC_PATH=%ISAAC_SIM_PATH%>>%CONDA_ACT_FILE_CMD%

set CONDA_DEACT_FILE_CMD=%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.bat
echo @echo off>>%CONDA_DEACT_FILE_CMD%
echo set PYTHONPATH=%%PYTHONPATH_OLD%%>>%CONDA_DEACT_FILE_CMD%
echo set PYTHONPATH_OLD="">>%CONDA_DEACT_FILE_CMD%

:: Add support for powershell
set CONDA_ACT_FILE_PWSH=%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1
echo $env:PYTHONPATH_OLD="$env:PYTHONPATH">>%CONDA_ACT_FILE_PWSH%
echo $env:PYTHONPATH="$env:PYTHONPATH;%ISAAC_SIM_PATH%\site">>%CONDA_ACT_FILE_PWSH%
echo $env:CARB_APP_PATH="%ISAAC_SIM_PATH%\kit">>%CONDA_ACT_FILE_PWSH%
echo $env:EXP_PATH="%ISAAC_SIM_PATH%\apps">>%CONDA_ACT_FILE_PWSH%
echo $env:ISAAC_PATH="%ISAAC_SIM_PATH%">>%CONDA_ACT_FILE_PWSH%

set CONDA_DEACT_FILE_PWSH=%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.ps1
echo $env:PYTHONPATH="$env:PYTHONPATH_OLD">>%CONDA_DEACT_FILE_PWSH%
echo $env:PYTHONPATH_OLD="$null">>%CONDA_DEACT_FILE_PWSH%


:: Install omnigibson!
call pip install -e . || goto :error

:: Cycle conda environment so that all dependencies are propagated
call conda deactivate || goto :error
goto :end 

:error
echo:
echo An error occurred during installation. Please check the error message above.
echo:
exit /b

:end
echo:
echo OmniGibson successfully installed! Please run [4mconda activate %conda_name%[0m to activate the environment.
echo:
