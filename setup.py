# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="omnigibson",
    version="0.2.1",
    author="Stanford University",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/StanfordVL/OmniGibson",
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "gym~=0.26",
        "numpy~=1.23",
        "GitPython~=3.1",
        "transforms3d~=0.4",
        "networkx~=3.1",
        "PyYAML~=6.0",
        "addict~=2.4",
        "ipython~=8.18",
        "future~=0.18",
        "trimesh~=4.0",
        "h5py~=3.10",
        "cryptography~=41.0",
        "bddl~=3.1.0",
        "opencv-python~=4.8",
        "nest_asyncio~=1.5",
        "imageio~=2.22",
        "imageio-ffmpeg~=0.4",
        "termcolor~=2.4",
        "progressbar~=2.5",
        "pymeshlab~=2023.12",
        "click~=8.1",
        "aenum~=3.1",
    ],
    tests_require=[],
    python_requires=">=3",
    package_data={"": ["omnigibson/global_config.yaml"]},
    include_package_data=True,
)  # yapf: disable
