from pathlib import Path
from omnigibson.robots.robot import Robot

REGISTERED_ROBOTS = []
robot_config_dir = Path(__file__).parent / "definitions"
for yaml_file in sorted(robot_config_dir.glob("*.yaml")):
    REGISTERED_ROBOTS.append(yaml_file.stem)

__all__ = [
    "Robot",
    "REGISTERED_ROBOTS",
]
