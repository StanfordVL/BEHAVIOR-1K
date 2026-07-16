"""Newton backend entry points independent from the Isaac Sim runtime."""

import os

# Collider-dense BEHAVIOR USD imports have produced native failures during
# parallel OpenUSD physics traversal. Keep PXR single-threaded until the load
# stress tests in docs/other/newton_migration.md pass with the default limit.
# This must be set before any pxr module initializes.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

from omnigibson.newton.assets import (
    DatasetObjectSpec,
    RobotSpec,
    prepared_dataset_object_usd,
    resolve_data_path,
    resolve_dataset_object_usd,
    resolve_robot_asset,
)
from omnigibson.newton.config import (
    load_newton_config,
    scene_from_config,
    simulation_config_from_config,
    simulator_from_config,
    specs_from_config,
)
from omnigibson.newton.entities import NewtonBody, NewtonEntity, NewtonJoint, NewtonShape
from omnigibson.scenes.scene_base import (
    NewtonDatasetObjectSpec,
    NewtonLightSpec,
    NewtonObjectSpec,
    NewtonRobotSpec,
    NewtonSceneSpec,
)


_SIMULATOR_EXPORTS = {
    "NewtonObjectRobotSimulator",
    "NewtonSceneSimulator",
    "NewtonSimulationConfig",
}


def __getattr__(name):
    if name in _SIMULATOR_EXPORTS:
        from omnigibson import simulator as simulator_module

        return getattr(simulator_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetObjectSpec",
    "NewtonBody",
    "NewtonDatasetObjectSpec",
    "NewtonEntity",
    "NewtonJoint",
    "NewtonLightSpec",
    "NewtonObjectRobotSimulator",
    "NewtonObjectSpec",
    "NewtonRobotSpec",
    "NewtonSceneSimulator",
    "NewtonSceneSpec",
    "NewtonShape",
    "NewtonSimulationConfig",
    "RobotSpec",
    "load_newton_config",
    "prepared_dataset_object_usd",
    "resolve_data_path",
    "resolve_dataset_object_usd",
    "resolve_robot_asset",
    "scene_from_config",
    "simulation_config_from_config",
    "simulator_from_config",
    "specs_from_config",
]
