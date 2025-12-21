from setuptools import setup, find_packages

# TODO package data

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bddl",
    version="3.7.0",
    author="Stanford University",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/StanfordVL/BEHAVIOR-1K",
    zip_safe=False,
    packages=find_packages(),
    install_requires=["pytest", "numpy", "networkx", "jupytext", "future", "nltk~=3.7"],
    package_data={},
    include_package_data=True,
)
