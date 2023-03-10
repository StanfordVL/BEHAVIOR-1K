@echo off

:: Make sure that the ISAAC_SIM_PATH variable is set
if  "%ISAAC_SIM_PATH%" == "" (
    echo "Please set ISAAC_SIM_PATH!" 
    exit /b
)

:: Sanity check whether env variable is set correctly 
if not exist %ISAAC_SIM_PATH%\setup_python_env.bat (
    echo setup_python_env.bat not found in %ISAAC_SIM_PATH%! Make sure you have set the correct ISAAC_SIM_PATH
    exit /b
)

:: Create a conda environment with python 3.7
call conda create -y -n omnigibson python=3.7
call conda activate omnigibson

mkdir %CONDA_PREFIX%\etc\conda\activate.d
mkdir %CONDA_PREFIX%\etc\conda\deactivate.d
type NUL>%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
type NUL>%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.bat

:: We add some preprocessing information so that the Isaac Sim paths are linked to this environment upon startup
:: See https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux for reference
set CONDA_ACT_FILE=%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
echo @echo off>>%CONDA_ACT_FILE%
echo set PYTHONPATH_OLD=%%PYTHONPATH%%>>%CONDA_ACT_FILE%
echo call %ISAAC_SIM_PATH%\setup_python_env.bat>>%CONDA_ACT_FILE%

set CONDA_DEACT_FILE=%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.bat
echo @echo off>>%CONDA_DEACT_FILE%
echo set PYTHONPATH=%%PYTHONPATH_OLD%%>>%CONDA_DEACT_FILE%
echo set PYTHONPATH_OLD="">>%CONDA_DEACT_FILE%

:: set env variables in setup_python_env.bat
echo set PYTHONPATH=%%PYTHONPATH%%;%%~dp0site>%ISAAC_SIM_PATH%\setup_python_env.bat
echo set CARB_APP_PATH=%%~dp0kit>>%ISAAC_SIM_PATH%\setup_python_env.bat
echo set EXP_PATH=%%~dp0apps>>%ISAAC_SIM_PATH%\setup_python_env.bat
echo set ISAAC_PATH=%%~dp0>>%ISAAC_SIM_PATH%\setup_python_env.bat

:: Install omnigibson!
call pip install -e .

:: Cycle conda environment so that all dependencies are propagated
call conda deactivate
call conda activate omnigibson

echo OmniGibson successfully installed!
