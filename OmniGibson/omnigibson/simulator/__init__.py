"""OmniGibson simulator package.

The active runtime is Newton. ``physx.py`` is retained as a reference backend
file during migration but is not selected by environment variables.
"""

from omnigibson.simulator.simulator import AbstractSimulator, SimulatorBase, UnsupportedSimulatorFeature
from omnigibson.simulator.newton import *  # noqa: F403
from omnigibson.simulator.newton import Simulator, _launch_simulator, launch_app, logo_small


BACKEND = "newton"


__all__ = [
    "AbstractSimulator",
    "BACKEND",
    "Simulator",
    "SimulatorBase",
    "UnsupportedSimulatorFeature",
    "_launch_simulator",
    "launch_app",
    "logo_small",
]
