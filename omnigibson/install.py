import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional

import click
import tqdm

# List of NVIDIA PyPI packages needed for OmniGibson
ISAAC_SIM_PACKAGES = [
    "omniverse_kit-106.0.1.126909",
    "isaacsim_kernel-4.1.0.0",
    "isaacsim_app-4.1.0.0",
    "isaacsim_core-4.1.0.0",
    "isaacsim_gui-4.1.0.0",
    "isaacsim_utils-4.1.0.0",
    "isaacsim_storage-4.1.0.0",
    "isaacsim_asset-4.1.0.0",
    "isaacsim_sensor-4.1.0.0",
    "isaacsim_robot_motion-4.1.0.0",
    "isaacsim_robot-4.1.0.0",
    "isaacsim_benchmark-4.1.0.0",
    "isaacsim_code_editor-4.1.0.0",
    "isaacsim_ros1-4.1.0.0",
    "isaacsim_cortex-4.1.0.0",
    "isaacsim_example-4.1.0.0",
    "isaacsim_replicator-4.1.0.0",
    "isaacsim_rl-4.1.0.0",
    "isaacsim_robot_setup-4.1.0.0",
    "isaacsim_ros2-4.1.0.0",
    "isaacsim_template-4.1.0.0",
    "isaacsim_test-4.1.0.0",
    "isaacsim-4.1.0.0",
    "isaacsim_extscache_physics-4.1.0.0",
    "isaacsim_extscache_kit-4.1.0.0",
    "isaacsim_extscache_kit_sdk-4.1.0.0",
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
        return temp_dir / f"{package}-cp310-none-win_amd64.whl"
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
    # Rename the file if the system's GLIBC version is older than the one used in the NVIDIA PyPI packages
    if platform.system() == "Linux" and _is_glibc_older():
        new_filename = filename.with_name(filename.name.replace("manylinux_2_34", "manylinux_2_31"))
        shutil.move(filename, new_filename)
        return new_filename

    # If the file is not renamed, return the original filename
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


def _pip_install(filenames: List[Path]):
    """Install a package using pip."""
    try:
        subprocess.run(["pip", "install"] + [str(x) for x in filenames], check=True)
    except subprocess.CalledProcessError:
        return False

    return True


def _download_isaac_sim_package(package: str, temp_dir: Path):
    package_name = package.split("-")[0].replace("_", "-")
    filename = _get_filename(package, temp_dir)
    url = f"{BASE_URL}/{package_name}/{filename.name}"

    try:
        _download_package(url, filename)
    except Exception as e:
        click.echo(f"Failed to download {package}: {str(e)}")
        raise

    return _rename_if_necessary(filename)


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

    # Check the version string
    version_file_path = isaac_sim_path / "VERSION"
    if not version_file_path.exists():
        click.echo(f"Isaac Sim version file not found at {version_file_path}")
        return False

    with open(version_file_path, "r") as file:
        version_content = file.read().strip()
        isaac_version_str = version_content.split("-")[0]
        isaac_version_tuple = tuple(map(int, isaac_version_str.split(".")[:3]))

    if isaac_version_tuple not in ((4, 0, 0), (4, 1, 0)):
        click.echo(f"Isaac Sim version {isaac_version_str} is not supported by OmniGibson.")
        return False

    # Update conda environment files to point to the specified Isaac Sim installation
    conda_prefix = Path(os.environ["CONDA_PREFIX"])
    if platform.system() == "Windows":
        _setup_windows_conda_env(isaac_sim_path, conda_prefix)
    else:
        _setup_unix_conda_env(isaac_sim_path, conda_prefix)

    # Temporarily apply the env vars to this script, too
    sys.path.append(str(isaac_sim_path / "site"))
    os.environ["CARB_APP_PATH"] = str(isaac_sim_path / "kit")
    os.environ["EXP_PATH"] = str(isaac_sim_path / "apps")
    os.environ["ISAAC_PATH"] = str(isaac_sim_path)

    return True


def _pip_based_install():
    try:
        # Create a temporary directory to download the packages
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Download all required packages
            package_filenames = []
            for package in tqdm.tqdm(ISAAC_SIM_PACKAGES, desc="Downloading Isaac Sim packages"):
                package_filenames.append(_download_isaac_sim_package(package, temp_dir_path))

            # Install the packages
            click.echo("Installing Isaac Sim packages...")
            if not _pip_install(package_filenames):
                return False

        # Check that it can now be imported
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
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
        click.echo("We did not find a compatible Isaac Sim installed via the launcher.")
    return success


def attempt_pip_install():
    click.echo("Installing Isaac Sim via pip.")
    success = _pip_based_install()
    if success:
        click.echo("Successfully installed Isaac Sim via pip.")
    else:
        click.echo("Something went wrong during the pip installation.")
    return success


@click.command()
@click.option(
    "--install-datasets/--no-install-datasets",
    default=True,
    help="Install the OmniGibson dataset and assets after installing Isaac Sim",
)
@click.option(
    "--launcher-install/--pip-install",
    default=False,
    help="Hook up to a launcher-installed Isaac Sim installation instead of installing via pip",
)
@click.option(
    "--isaac-sim-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True, readable=True, path_type=Path),
    default=None,
    help="Path to the existing launcher-based Isaac Sim installation directory, to force the setup script to use it",
)
def setup_omnigibson(install_datasets: bool, launcher_install: bool, isaac_sim_path: Optional[Path]):
    # Check that we are in a conda environment
    if "CONDA_PREFIX" not in os.environ:
        click.echo("Please run this script from within a conda environment.")
        click.echo("You can create one by running `conda create -n omnigibson python=3.10`.")
        return

    # Check that the current interpreter is Python 3.10
    if sys.version_info[:2] != (3, 10):
        click.echo("Please run this script with Python 3.10.")
        return

    # Check that we do not have an EXP_PATH, CARB_APP_PATH or ISAAC_PATH set
    if any(env_var in os.environ for env_var in ["EXP_PATH", "CARB_APP_PATH", "ISAAC_PATH"]):
        click.echo(
            "Please unset the EXP_PATH, CARB_APP_PATH and ISAAC_PATH environment variables before running this script."
        )
        click.echo("These can stem from a dirty environment from an existing Isaac Sim installation.")
        click.echo("We recommend starting a new conda environment and running this script there.")
        click.echo("You can do this by running `conda create -n omnigibson python=3.10`.")
        return

    # Check if the isaacsim package is already installed
    try:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        import isaacsim

        click.echo("Isaac Sim is already installed via pip in your current env.")
        click.echo("If you need to download the datasets, please run omnigibson/download_datasets.py.")
        return
    except ImportError:
        pass

    # Do the actual work
    if launcher_install:
        if not attempt_launcher_install(isaac_sim_path):
            click.echo("Failed to hook environment up to launcher-based Isaac Sim installation.")
            click.echo("Please make sure you have installed Isaac Sim correctly before running this setup script.")
            return
    else:
        if not attempt_pip_install():
            click.echo("Failed to install Isaac Sim. Please check the installation requirements and try again.")
            return

    # Try to resolve the bug that occurs when `cryptography` is imported from Isaac Sim.
    # This is a known issue on Windows and the workaround is removing the copy of cryptography
    # shipped with Isaac Sim, usually included in the omni.kit.cloud package.
    cryptography_path = Path(os.environ["ISAAC_PATH"]) / "exts/omni.pip.cloud/pip_prebundle/cryptography"
    if cryptography_path.exists():
        shutil.rmtree(str(cryptography_path))

    click.echo("Isaac Sim has been successfully installed.")

    # Now prompt the user to install the dataset and assets
    if install_datasets:
        click.echo("We will now install the datasets.")

        # We import these now to avoid OmniGibson imports before torch is installed etc.
        from omnigibson.macros import gm
        from omnigibson.utils.asset_utils import download_assets, download_og_dataset

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

    # If this is a launcher install, we need to tell the user to deactivate and reactivate
    if launcher_install:
        click.echo(
            "IMPORTANT: Please deactivate and reactivate your conda environment to ensure the Isaac Sim environment variables are set correctly."
        )


if __name__ == "__main__":
    setup_omnigibson()
