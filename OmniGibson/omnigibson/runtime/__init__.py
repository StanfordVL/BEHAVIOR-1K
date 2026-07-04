"""Backend-neutral runtime interfaces for OmniGibson."""

from omnigibson.runtime.entity import SimBody, SimEntity, SimJoint, SimShape, SimulatorBase
from omnigibson.runtime.registry import EntityRegistry

__all__ = [
    "EntityRegistry",
    "SimBody",
    "SimEntity",
    "SimJoint",
    "SimShape",
    "SimulatorBase",
]
