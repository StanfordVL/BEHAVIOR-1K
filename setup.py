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
        "gym>=0.26",
        "numpy>=1.20.0",
        "GitPython",
        "transforms3d>=0.3.1",
        "networkx>=2.0",
        "PyYAML",
        "addict",
        "ipython",
        "future",
        "trimesh",
        "h5py",
        "cryptography",
        "bddl~=3.1.0",
        "opencv-python",
        "nest_asyncio",
        "imageio",
        "imageio-ffmpeg",
        "termcolor",
        "progressbar",
        "pymeshlab",
        "click"
    ],
    tests_require=[],
    python_requires=">=3",
    package_data={"": ["omnigibson/global_config.yaml"]},
    include_package_data=True,
)  # yapf: disable
