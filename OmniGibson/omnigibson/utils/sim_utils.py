import omnigibson.lazy as lazy
from omnigibson.utils import python_utils


def meets_minimum_isaac_version(minimum_version, current_version=None):
    def _transform_isaac_version(str):
        # In order to avoid issues with the version scheme change from 202X.X.X to X.X.X,
        # transform Isaac Sim versions to all not be 202x-based e.g. 2021.2.3 -> 1.2.3
        return str[3:] if str.startswith("202") else str

    # If the user has not provided the current Isaac version, get it from the system.
    if current_version is None:
        current_version = lazy.isaacsim.core.version.get_version()[0]

    # Transform and compare.
    return python_utils.meets_minimum_version(
        _transform_isaac_version(current_version), _transform_isaac_version(minimum_version)
    )
