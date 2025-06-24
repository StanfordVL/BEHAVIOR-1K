import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gello",
    version="0.0.1",
    author="Philipp Wu",
    author_email="philippwu@berkeley.edu",
    description="software for GELLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wuphilipp/gello_software",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=[
        "numpy",
    ],
)
