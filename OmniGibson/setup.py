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
    version="3.7.2",
    author="Stanford University",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/StanfordVL/BEHAVIOR-1K",
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "huggingface-hub[cli]>=0.34.4",
        "gymnasium>=0.28.1",
        "numpy<2.0.0,>=1.23.5",
        "scipy>=1.10.1",
        "GitPython>=3.1.40",
        "transforms3d>=0.4.1",
        "networkx>=3.2.1",
        "PyYAML>=6.0.1",
        "addict>=2.4.0",
        "ipython>=8.20.0",
        "future>=0.18.3",
        "trimesh>=4.0.8",
        "h5py>=3.10.0",
        "cryptography>=41.0.7",
        "bddl~=3.7.0",
        "opencv-python>=4.8.1",
        "nest_asyncio>=1.5.6",
        "imageio>=2.33.1",
        "imageio-ffmpeg>=0.4.9",
        "termcolor>=2.4.0",
        "progressbar>=2.5",
        "pymeshlab~=2022.2",
        "click>=8.1.3",
        "aenum>=3.1.15",
        "rtree>=1.2.0",
        "graphviz>=0.20",
        "matplotlib>=3.0.0",
        "lxml>=5.2.2",
        "numba>=0.59.1",
        "cffi==1.17.1",
        "pillow~=11.0.0",
        "websockets>=15.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.3",
            "pytest-cov>=3.0.0",
            "pytest_rerunfailures",
            "mkdocs",
            "mkdocs-autorefs",
            "mkdocs-gen-files",
            "mkdocs-material",
            "mkdocs-material-extensions",
            "mkdocstrings[python]",
            "mkdocs-section-index",
            "mkdocs-literate-nav",
            "mkdocs-redirects",
            "mkdocs-include-markdown-plugin",
            "telemoma~=0.3.0",
            "gspread>=6.2.1",
        ],
        "primitives": [
            "ninja~=1.13.0",
            "nvidia-curobo @ git+https://github.com/StanfordVL/curobo@cbaf7d32436160956dad190a9465360fad6aba73",
            "ompl @ https://storage.googleapis.com/gibson_scenes/ompl-1.6.0-cp310-cp310-manylinux_2_28_x86_64.whl",
        ],
        "eval": [
            "dm_tree>=0.1.9",
            "hydra-core>=1.3.2",
            "msgpack>=1.1.0",
            "lerobot @ git+https://github.com/huggingface/lerobot@577cd10974b84bea1f06b6472eb9e5e74e07f77a",
            "gspread>=6.2.1",
            "open3d>=0.19.0",
        ]
    },
    tests_require=[],
    python_requires=">=3",
    include_package_data=True,
)  # yapf: disable
