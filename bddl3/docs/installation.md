# Installation 
Installing BDDL is very simple. 

## System Requirements

BDDL requires python 3. It has some required packages which will be installed automatically. It has been tested on: 
- Python: >= 3.6
- Linux: Ubuntu >= 16.04 
- Windows: Windows 10 
- Mac: macOS >= 10.15

Given minimal requirements, we expect BDDL to work with most systems that have Python 3. Note that if you are using the [iGibson 2.0](https://github.com/StanfordVL/iGibson) simulator with BDDL, you will have a longer list of requirements to check. 

## Installing the library

There are two ways to install BDDL. 

### 1. pip

iGibson's simulator can be installed as a python package using pip:

```bash
pip install bddl
```

### 2. From source

```bash
git clone https://github.com/StanfordVL/bddl
cd bddl
python setup.py install
```

## Uninstalling

Uninstalling BDDL is just as simple: `pip uninstall bddl`

