import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="joylo",
    version="3.7.0",
    author="Stanford University",
    description="JoyLo for OmniGibson",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StanfordVL/BEHAVIOR-1K",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=[
        "dm_control",
        "dynamixel-sdk",
        "hid",
        "inputs",
        "joycon-python",
        "numpy",
        "numpy-quaternion",
        "Pillow",
        "pybullet",
        "pygame",
        "pyglm",
        "PyQt6==6.9.1",
        "pyquaternion",
        "pure-python-adb",
        "quaternion",
        "termcolor",
        "tyro",
        "zmq",
        "telemoma~=0.3.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "flake8-docstrings",
            "ipdb",
            "isort",
            "jupyterlab",
            "mypy",
            "neovim",
            "pyright",
            "pytest",
            "python-lsp-server[all]",
        ],
    },
)
