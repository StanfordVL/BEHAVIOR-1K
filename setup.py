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
    name="igibson",
    version="3.0.0",
    author="Stanford University",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/StanfordVL/iGibson",        # TODO: Update
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "gym>=0.23",
        "numpy>=1.16.0",
        "GitPython",
        # "scipy>=1.4",
        # "pybullet-svl>=3.1.6",
        # "transforms3d>=0.3.1",
        # "opencv-python>=3.4.8",
        # "Pillow>=5.4.0",
        # "networkx>=2.0",
        # "PyYAML",
        # "tqdm",
        # "freetype-py",
        # "matplotlib",
        # "cloudpickle",
        # "aenum",
        # "GPUtil",
        # "ipython",
        # "future",
        # "trimesh",
        # "h5py",
        # "gitpython",
        # "py360convert",
        # "six",
        # "pandas",
        # "packaging",
        # "bddl>=0.0.3",
    ],
    tests_require=[],
    python_requires=">=3",
    package_data={"": ["igibson/global_config.yaml"]},
    include_package_data=True,
)  # yapf: disable
