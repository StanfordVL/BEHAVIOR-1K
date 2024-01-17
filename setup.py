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
        "gymnasium>=0.28.1",
        "numpy~=1.23.5",
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
        "bddl~=3.2.0b1",
        "opencv-python~=4.8.1",
        "nest_asyncio~=1.5.8",
        "imageio~=2.33.1",
        "imageio-ffmpeg~=0.4.9",
        "termcolor~=2.4.0",
        "progressbar~=2.5",
        "pymeshlab~=2022.2",
        "click~=8.1.7",
        "aenum~=3.1.15",
    ],
    tests_require=[],
    python_requires=">=3",
    package_data={"": ["omnigibson/global_config.yaml"]},
    include_package_data=True,
)  # yapf: disable
