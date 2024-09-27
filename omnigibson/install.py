import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import click
import pip

from omnigibson.macros import gm
from omnigibson.utils.asset_utils import download_assets, download_og_dataset

# List of NVIDIA PyPI packages needed for OmniGibson
ISAAC_SIM_PACKAGES = [
    "omniverse_kit-106.1.0.140981",
    "isaacsim_kernel-4.2.0.2",
    "isaacsim_app-4.2.0.2",
    "isaacsim_core-4.2.0.2",
    "isaacsim_gui-4.2.0.2",
    "isaacsim_utils-4.2.0.2",
    "isaacsim_storage-4.2.0.2",
    "isaacsim_asset-4.2.0.2",
    "isaacsim_sensor-4.2.0.2",
    "isaacsim_robot_motion-4.2.0.2",
    "isaacsim_robot-4.2.0.2",
    "isaacsim_benchmark-4.2.0.2",
    "isaacsim_code_editor-4.2.0.2",
    "isaacsim_ros1-4.2.0.2",
    "isaacsim_cortex-4.2.0.2",
    "isaacsim_example-4.2.0.2",
    "isaacsim_replicator-4.2.0.2",
    "isaacsim_rl-4.2.0.2",
    "isaacsim_robot_setup-4.2.0.2",
    "isaacsim_ros2-4.2.0.2",
    "isaacsim_template-4.2.0.2",
    "isaacsim_test-4.2.0.2",
    "isaacsim-4.2.0.2",
    "isaacsim_extscache_physics-4.2.0.2",
    "isaacsim_extscache_kit-4.2.0.2",
    "isaacsim_extscache_kit_sdk-4.2.0.2",
]
BASE_URL = "https://pypi.nvidia.com"


def _find_isaac_sim_path():
    """Try to find the path of a launcher-based Isaac Sim installation."""
    if platform.system() == "Windows":
        base_path = Path.home() / "AppData" / "Local" / "ov" / "pkg"
    else:
        base_path = Path.home() / ".local" / "share" / "ov" / "pkg"

    # If the pkg dir is missing, we definitely can't find an Isaac Sim installation
    if not base_path.exists():
        return None

    isaac_dirs = list(base_path.glob("isaac*"))
    if not isaac_dirs:
        return None

    return isaac_dirs[-1]


def _get_filename(package: str, temp_dir: Path):
    if platform.system() == "Windows":
        return temp_dir / f"{package}-cp310-win_amd64.whl"
    return temp_dir / f"{package}-cp310-none-manylinux_2_34_x86_64.whl"


def _download_package(url: str, filename: Path):
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        raise ValueError(f"Failed to download {url}") from e


def _rename_if_necessary(filename: Path):
    """
    Rename the file if the system's GLIBC version is older than the one used in the NVIDIA PyPI packages.

    This is permissible because the manylinux wheels are compatible with older GLIBC versions even though
    the filename suggests not - so we apply this hacky workaround. This allows pip to try to install them.
    """
    if platform.system() == "Linux" and _is_glibc_older():
        return filename.with_name(filename.name.replace("manylinux_2_34", "manylinux_2_31"))
    return filename


def _is_glibc_older():
    """Check if the system's GLIBC version is older than the one used in the NVIDIA PyPI packages."""
    try:
        dist_info = subprocess.check_output(["ldd", "--version"]).decode("utf-8")
        if any(version in dist_info for version in ["2.31", "2.32", "2.33"]):
            return True
        elif any(version in dist_info for version in ["2.34", "2.35", "2.36", "2.37", "2.38", "2.39"]):
            return False
        else:
            raise ValueError("Incompatible GLIBC version")
    except subprocess.CalledProcessError:
        raise ValueError("Failed to check GLIBC version. `ldd` was not accessible. Try running it yourself to see why.")


def _pip_install(filename: Path):
    """Install a package using pip."""
    try:
        pip.main(["install", str(filename)])
    except Exception as e:
        raise ValueError(f"Failed to install {filename}") from e


def _install_isaac_sim_package(package: str, temp_dir: Path):
    package_name = package.split("-")[0].replace("_", "-")
    filename = _get_filename(package, temp_dir)
    url = f"{BASE_URL}/{package_name}/{filename}"

    try:
        _download_package(url, filename)
        _pip_install(_rename_if_necessary(filename))
    except Exception as e:
        click.echo(f"Failed to install {package}: {str(e)}")
        raise


def _setup_windows_conda_env(isaac_sim_path: Path, conda_prefix: Path):
    # Create directories
    for dir_name in ["activate.d", "deactivate.d"]:
        os.makedirs(conda_prefix / "etc" / "conda" / dir_name, exist_ok=True)

    # Create empty files
    for file_name in ["env_vars.bat", "env_vars.ps1"]:
        for dir_name in ["activate.d", "deactivate.d"]:
            (conda_prefix / "etc" / "conda" / dir_name / file_name).touch()

    # Setup CMD activation script
    cmd_act_file = conda_prefix / "etc" / "conda" / "activate.d" / "env_vars.bat"
    with cmd_act_file.open("w") as f:
        f.write("@echo off\n")
        f.write("set PYTHONPATH_OLD=%PYTHONPATH%\n")
        f.write(f"set PYTHONPATH=%PYTHONPATH%;{isaac_sim_path}\\site\n")
        f.write(f"set CARB_APP_PATH={isaac_sim_path}\\kit\n")
        f.write(f"set EXP_PATH={isaac_sim_path}\\apps\n")
        f.write(f"set ISAAC_PATH={isaac_sim_path}\n")

    # Setup CMD deactivation script
    cmd_deact_file = conda_prefix / "etc" / "conda" / "deactivate.d" / "env_vars.bat"
    with cmd_deact_file.open("w") as f:
        f.write("@echo off\n")
        f.write("set PYTHONPATH=%PYTHONPATH_OLD%\n")
        f.write('set PYTHONPATH_OLD=""\n')

    # Setup PowerShell activation script
    ps_act_file = conda_prefix / "etc" / "conda" / "activate.d" / "env_vars.ps1"
    with ps_act_file.open("w") as f:
        f.write('$env:PYTHONPATH_OLD="$env:PYTHONPATH"\n')
        f.write(f'$env:PYTHONPATH="$env:PYTHONPATH;{isaac_sim_path}\\site"\n')
        f.write(f'$env:CARB_APP_PATH="{isaac_sim_path}\\kit"\n')
        f.write(f'$env:EXP_PATH="{isaac_sim_path}\\apps"\n')
        f.write(f'$env:ISAAC_PATH="{isaac_sim_path}"\n')

    # Setup PowerShell deactivation script
    ps_deact_file = conda_prefix / "etc" / "conda" / "deactivate.d" / "env_vars.ps1"
    with ps_deact_file.open("w") as f:
        f.write('$env:PYTHONPATH="$env:PYTHONPATH_OLD"\n')
        f.write('$env:PYTHONPATH_OLD="$null"\n')


def _setup_unix_conda_env(isaac_sim_path: Path, conda_prefix: Path):
    # Set up conda environment files
    for dir_name in ["activate.d", "deactivate.d"]:
        os.makedirs(conda_prefix / "etc" / "conda" / dir_name, exist_ok=True)

    # Create activation script
    with open(conda_prefix / "etc" / "conda" / "activate.d" / "env_vars.sh", "w") as f:
        f.write("#!/bin/sh\n")
        f.write("export LD_LIBRARY_PATH_OLD=$LD_LIBRARY_PATH\n")
        f.write("export PYTHONPATH_OLD=$PYTHONPATH\n")
        f.write(f"source {isaac_sim_path}/setup_conda_env.sh\n")

    # Create deactivation script
    with open(conda_prefix / "etc" / "conda" / "deactivate.d" / "env_vars.sh", "w") as f:
        f.write("#!/bin/sh\n")
        f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_OLD\n")
        f.write("export PYTHONPATH=$PYTHONPATH_OLD\n")
        f.write("unset ISAAC_PATH\n")
        f.write("unset CARB_APP_PATH\n")
        f.write("unset LD_LIBRARY_PATH_OLD\n")
        f.write("unset PYTHONPATH_OLD\n")


def _launcher_based_install(isaac_sim_path: Optional[Path]):
    # If we are using a launcher-based installation, we need to find the path to the Isaac Sim installation
    if isaac_sim_path is None:
        isaac_sim_path = _find_isaac_sim_path()

    # If it's not at the provided or default path, remove it
    if isaac_sim_path is None or not list(Path(isaac_sim_path).glob("isaac*.*")):
        return False

    # Update conda environment files to point to the specified Isaac Sim installation
    conda_prefix = Path(os.environ["CONDA_PREFIX"])
    if platform.system() == "Windows":
        _setup_windows_conda_env(isaac_sim_path, conda_prefix)
    else:
        _setup_unix_conda_env(isaac_sim_path, conda_prefix)

    return True


def _pip_based_install():
    try:
        # Create a temporary directory to download the packages
        with tempfile.TemporaryDirectory() as temp_dir:
            # Install all required packages
            for package in ISAAC_SIM_PACKAGES:
                _install_isaac_sim_package(package, temp_dir)

        # Check that it can now be imported
        import isaacsim
    except ImportError:
        return False

    return True


def attempt_launcher_install(isaac_sim_path: Optional[Path]):
    click.echo("Checking for an existing launcher-based Isaac Sim installation...")
    success = _launcher_based_install(isaac_sim_path)
    if success:
        click.echo("Successfully found and attached to launcher-based Isaac Sim installation.")
    else:
        click.echo("We did not find Isaac Sim installed via the launcher.")
    return success


def attempt_pip_install():
    click.echo("We will now try to install Isaac Sim via pip.")
    success = _pip_based_install()
    if success:
        click.echo("Successfully installed Isaac Sim via pip.")
    else:
        click.echo("Failed to import isaacsim. Something went wrong during the pip installation.")
    return success


@click.command()
@click.option(
    "--install-datasets", default=True, help="Install the OmniGibson dataset and assets after installing Isaac Sim"
)
@click.option(
    "--force-pip-install",
    default=False,
    help="Install Isaac Sim via pip even if a launcher-based installation is found",
)
@click.option(
    "--force-launcher-install",
    default=False,
    help="Do not install Isaac Sim via pip even if a launcher-based installation is not found",
)
@click.option(
    "--isaac-sim-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, writeable=True, readable=True, path_type=Path),
    default=None,
    help="Path to the existing launcher-based Isaac Sim installation directory, to force the setup script to use it",
)
def setup_omnigibson(
    install_datasets: bool, force_pip_install: bool, force_launcher_install: bool, isaac_sim_path: Optional[Path]
):
    if force_pip_install and force_launcher_install:
        click.echo("You can't force both pip and launcher-based installations at the same time.")
        return

    # Check that we are in a conda environment
    if "CONDA_PREFIX" not in os.environ:
        click.echo("Please run this script from within a conda environment.")
        click.echo("You can create one by running `conda create -n omnigibson python=3.10`.")
        return

    # Check that the current interpreter is Python 3.10
    if sys.version_info[:2] != (3, 10):
        click.echo("Please run this script with Python 3.10.")
        return

    # First, try to install Isaac Sim via the launcher
    installation_successful = False
    if not force_pip_install:
        installation_successful = attempt_launcher_install(isaac_sim_path)

    # If that failed, try to install it via pip
    if not installation_successful:
        if force_launcher_install:
            # If the user forced a launcher-based installation, we should not try to install via pip
            click.echo("You forced the script to use a launcher-based installation, but we couldn't find it.")
            click.echo("Please make sure you have installed Isaac Sim correctly before running this setup script.")
        else:
            installation_successful = attempt_pip_install()

    # If neither installation was successful, we should exit
    if not installation_successful:
        click.echo("Failed to install Isaac Sim. Please check the installation requirements and try again.")
        return

    # Now prompt the user to install the dataset and assets
    click.echo("Isaac Sim has been successfully installed.")

    if install_datasets:
        click.echo("We will now install the datasets.")

        # Only execute if the dataset path or asset path does not exist
        dataset_exists, assets_exist = os.path.exists(gm.DATASET_PATH), os.path.exists(gm.ASSET_PATH)
        if not (dataset_exists and assets_exist):
            # Ask user which dataset to install
            click.echo(f"OmniGibson will now install data under the following locations:")
            click.echo(f"    dataset (~25GB): {gm.DATASET_PATH}")
            click.echo(f"    assets (~2.5GB): {gm.ASSET_PATH}")
            click.echo(
                f"If you want to install data under a different path, please change the DATA_PATH variable in omnigibson/macros.py and "
                f"rerun omnigibson/download_datasets.py."
            )
            if click.confirm("Do you want to continue?", default=True):
                # Only download if the dataset path doesn't exist
                if not dataset_exists:
                    click.echo("Downloading dataset...")
                    download_og_dataset()

                # Only download if the asset path doesn't exist
                if not assets_exist:
                    click.echo("Downloading assets...")
                    download_assets()
    else:
        click.echo(
            "You chose not to install dataset for now. You can install it later by running python omnigibson/download_datasets.py."
        )

    click.echo(
        "\nOmniGibson setup completed! You can now run your experiments. "
        "Visit https://behavior.stanford.edu/omnigibson/getting_started/examples.html for some examples, "
        "and https://behavior.stanford.edu/omnigibson-develop/getting_started/quickstart.html for a quickstart "
        "guide for working with OmniGibson APIs."
    )


if __name__ == "__main__":
    setup_omnigibson()
