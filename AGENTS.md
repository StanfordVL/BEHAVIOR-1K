# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## General Guidelines

- **Use `behavior` conda environment by default** when running Python commands, unless explicitly told otherwise. You can do this by using the `conda run` command. The conda binary is usually available as ~/miniconda3/condabin/conda.
- **Make minimal changes** - prefer small, targeted edits over large rewrites. Avoid unnecessary stylistic changes (e.g., reformatting code that isn't relevant to the task).

## Project Overview

BEHAVIOR-1K is a monorepo for a simulation benchmark testing embodied AI agents on 1,000+ household activities. It contains three main packages plus supporting tooling:

- **OmniGibson** (`OmniGibson/`) - Physics simulation engine built on NVIDIA Omniverse/Isaac Sim. Provides environments, robots, objects, sensors, controllers, and tasks as a Gymnasium-compatible interface.
- **BDDL3** (`bddl3/`) - Behavior Domain Definition Language. Defines a symbolic knowledge base with 1,000+ activity definitions, object taxonomy/ontology, and condition evaluation logic.
- **JoyLo** (`joylo/`) - Teleoperation framework for robot control using physical hardware (GELLO devices).

Supporting components: `asset_pipeline/` (3D asset conversion), `knowledgebase/` (Flask web app for browsing BDDL data), `docs/` (MkDocs site), `eval-jobqueue/` (evaluation infrastructure).

## Commands

### Installation
You should not have to install any part of this project, the user should have a pre-installed conda env for you. If that's not the case, refuse running things and ask them to install first. But for general reference, the below are the installation commands.

```bash
# Modular install via setup script (conda env creation + component selection)
bash setup.sh --new-env behavior --omnigibson --bddl
# Individual packages (editable installs)
cd bddl3 && pip install -e .
cd OmniGibson && pip install -e .[dev]           # dev dependencies (pytest, mkdocs)
cd OmniGibson && pip install -e .[dev,primitives] # + motion planning (curobo)
cd OmniGibson && pip install -e .[dev,eval]       # + evaluation dependencies
```

### Testing (OmniGibson)
Tests require an NVIDIA RTX GPU (2080Ti+) and Isaac Sim runtime. Whenever running any tests, you should set the OMNIGIBSON_HEADLESS=1 environment flag so that a DISPLAY is not needed. Run from the `OmniGibson/` directory:
```bash
pytest tests/                           # all tests
pytest tests/test_object_states.py      # single test file
pytest tests/test_envs.py -k "test_name" # single test by name
```

### Linting
Pre-commit hooks run Ruff on `OmniGibson/` only (not joylo, bddl3):
```bash
ruff check OmniGibson/                  # lint
ruff format OmniGibson/                 # format
pre-commit run --all-files              # run pre-commit hooks
```

### Documentation
```bash
mkdocs serve                            # local docs server (from repo root, uses mkdocs.yml)
```

## Architecture

### OmniGibson Core (`OmniGibson/omnigibson/`)
The simulation engine follows a registry pattern — robots, objects, scenes, tasks, controllers, and sensors each have a `REGISTERED_*` dict populated via class decorators.

Key module relationships:
- **`simulator.py`** — Singleton wrapper around Isaac Sim. All physics stepping goes through here.
- **`envs/`** — Gymnasium environments. `env_base.py` is the core env; `vec_env_base.py` for vectorized; wrappers in `env_wrapper.py` and `data_wrapper.py`.
- **`robots/`** — Robot definitions in `definitions/` (YAML-configured). `robot.py` is the base class.
- **`objects/`** — Simulated objects with state tracking (temperature, wetness, etc.).
- **`object_states/`** — State logic (e.g., `Cooked`, `Sliced`, `OnTop`). These implement BDDL predicates in simulation.
- **`scenes/`** — Scene loading and management.
- **`tasks/`** — Task definitions that pair with BDDL activity specs.
- **`controllers/`** — Low-level robot control (joint, IK, operational space).
- **`sensors/`** — Vision, scan, and other sensor modalities.
- **`transition_rules.py`** — Rules for state transitions (e.g., cooking, mixing).
- **`systems/`** — Particle systems (water, dust, etc.) and material systems.
- **`macros.py`** — Global configuration via `gm` (global macros) object.
- **`action_primitives/`** — High-level action abstractions (pick, place, navigate).
- **`learning/`** — RL training utilities.

OmniGibson is based on Isaac Sim, which will be installed in the `behavior` conda env. You can expect to find Isaac Sim source files in the conda env's site packages directory,
under the `isaacsim` package. Everything from isaacsim needs to be imported using the `omnigibson.lazy` module since these imports are only available after the app has launched
(e.g. through simulator.py's launch_app). You can follow most of these imports to the source code by finding the appropriate extension inside the isaacsim directory. Especially
relevant extensions' names start with isaacsim.core.

OmniGibson currently uses [Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/) and [Omniverse Kit 107.3.1](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.3.1/). This [documentation for USDRT](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/index.html#usdrt-scenegraph-module) may also be especially useful for understanding Fabric and USD syncing. When following the links, make sure not to add an extra "docs/" to the href.

### BDDL3 (`bddl3/bddl/`)
- **`activity_definitions/`** — One file per activity with symbolic pre/post conditions.
- **`object_taxonomy.py`** — Hierarchical object ontology.
- **`condition_evaluation.py`** — Evaluates symbolic conditions against simulation state.
- **`backend_abc.py`** — Abstract interface that OmniGibson implements to connect simulation to BDDL logic.
- **`knowledge_base/`** — Structured data about objects, scenes, and their properties.

### Config-Driven Design
OmniGibson uses YAML configs extensively (`OmniGibson/omnigibson/configs/`). Environment creation typically loads a config specifying scene, task, robot, and controller parameters.

## Code Style

- **Python 3.10+**, line length 120, indent 4 spaces
- **Ruff** for linting and formatting (rules: E4, E7, E9, F; ignores: E731, E722, E741)
- Ruff config is split: root `ruff.toml` (excludes joylo) and `OmniGibson/pyproject.toml` (detailed rules)
- Pre-commit hooks only apply to `OmniGibson/` directory
- Type checking via Pyright (configured in `OmniGibson/pyproject.toml`)

## CI/CD

GitHub Actions workflows (`.github/workflows/`):
- **tests.yml** — Runs pytest matrix across test files on self-hosted GPU runners. Also checks example list is up to date.
- **build-push-containers.yml** — Docker image builds
- **build-website.yml** — MkDocs documentation deployment
- **publish-pypi.yml** — PyPI releases
- **profiling.yml** — Performance benchmarking
