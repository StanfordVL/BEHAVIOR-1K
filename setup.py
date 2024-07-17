# read the contents of your README file
import os
import urllib.request
import subprocess

from setuptools import find_packages, setup

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

isaac_sim_packages = [
    "omniverse_kit-106.0.0",
    "isaacsim_kernel-4.0.0.0", "isaacsim_app-4.0.0.0", "isaacsim_core-4.0.0.0",
    "isaacsim_gui-4.0.0.0", "isaacsim_utils-4.0.0.0", "isaacsim_storage-4.0.0.0",
    "isaacsim_asset-4.0.0.0", "isaacsim_sensor-4.0.0.0", "isaacsim_robot_motion-4.0.0.0",
    "isaacsim_robot-4.0.0.0", "isaacsim_benchmark-4.0.0.0", "isaacsim_code_editor-4.0.0.0",
    "isaacsim_ros1-4.0.0.0", "isaacsim_cortex-4.0.0.0", "isaacsim_example-4.0.0.0",
    "isaacsim_replicator-4.0.0.0", "isaacsim_rl-4.0.0.0", "isaacsim_robot_setup-4.0.0.0",
    "isaacsim_ros2-4.0.0.0", "isaacsim_template-4.0.0.0", "isaacsim_test-4.0.0.0",
    "isaacsim-4.0.0.0", "isaacsim_extscache_physics-4.0.0.0", "isaacsim_extscache_kit-4.0.0.0",
    "isaacsim_extscache_kit_sdk-4.0.0.0"
]

base_url = "https://pypi.nvidia.com"
for package in isaac_sim_packages:
    package_name = package.split('-')[0]
    old_filename = f"{package}-cp310-none-manylinux_2_34_x86_64.whl"
    new_filename = f"{package}-cp310-none-manylinux_2_31_x86_64.whl"
    url = f"{base_url}/{package_name}/{old_filename}"

    print(f"Downloading {old_filename}...")
    urllib.request.urlretrieve(url, old_filename)

    print(f"Renaming {old_filename} to {new_filename}...")
    os.rename(old_filename, new_filename)

    print(f"Installing {new_filename}...")
    subprocess.run(["pip", "install", new_filename], check=True)

    print(f"Removing {new_filename}...")
    os.remove(new_filename)

setup(
    name="omnigibson",
    version="1.0.0",
    author="Stanford University",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/StanfordVL/OmniGibson",
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy~=1.23.5",
        "scipy~=1.10.1",
        "GitPython~=3.1.40",
        "transforms3d~=0.4.1",
        "networkx~=3.2.1",
        "PyYAML~=6.0.1",
        "addict~=2.4.0",
        "ipython~=8.20.0",
        "future~=0.18.3",
        "trimesh~=4.0.8",
        "h5py~=3.10.0",
        "cryptography~=41.0.7",
        "bddl~=3.5.0",
        "opencv-python~=4.8.1",
        "nest_asyncio~=1.5.8",
        "imageio~=2.33.1",
        "imageio-ffmpeg~=0.4.9",
        "termcolor~=2.4.0",
        "progressbar~=2.5",
        "pymeshlab~=2022.2",
        "click~=8.1.3",
        "aenum~=3.1.15",
        "rtree~=1.2.0",
        "numba~=0.59.1",
        "telemoma~=0.1.2",
    ],
    tests_require=[],
    python_requires=">=3",
    include_package_data=True,
)  # yapf: disable
