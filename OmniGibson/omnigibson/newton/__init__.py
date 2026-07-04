"""Newton backend entry points.

This package is the first native Newton integration point for OmniGibson. It is
kept independent from Isaac Sim so it can run in a Newton-only environment.
"""

import os

# Newton 1.2.0 uses OpenUSD's UsdPhysics.LoadUsdPhysicsFromRange during USD
# import. That OpenUSD helper has a known thread-safety crash for collider-dense
# assets; Newton's own test runner applies the same workaround. This must be set
# before any pxr module initializes, so keep it at the top of the Newton package.
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
