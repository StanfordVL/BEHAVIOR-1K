from pathlib import Path
from omnigibson.robots.robot import Robot
from omnigibson.utils.asset_utils import get_dataset_path

REGISTERED_ROBOTS = []
robot_config_dir = Path(get_dataset_path("omnigibson-robot-assets")) / "models"
for yaml_file in sorted(robot_config_dir.glob("*/*.yaml")):
    if yaml_file.stem == yaml_file.parent.name:
        REGISTERED_ROBOTS.append(yaml_file.stem)

__all__ = [
    "Robot",
    "REGISTERED_ROBOTS",
]
