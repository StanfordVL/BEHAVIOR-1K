from pathlib import Path
from omnigibson.robots.robot import Robot
from omnigibson.macros import gm


REGISTERED_ROBOTS = []
for yaml_file in sorted(Path(gm.DATA_PATH).glob("*/models/*/*.yaml")):
    if yaml_file.stem == yaml_file.parent.name:
        REGISTERED_ROBOTS.append(yaml_file.stem)

__all__ = [
    "Robot",
    "REGISTERED_ROBOTS",
]
