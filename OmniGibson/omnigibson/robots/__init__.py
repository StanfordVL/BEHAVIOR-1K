import builtins
import os
from pathlib import Path

from omnigibson.macros import gm


def _data_root_candidates():
    candidates = []
    if "OMNIGIBSON_DATA_PATH" in os.environ:
        candidates.append(Path(os.environ["OMNIGIBSON_DATA_PATH"]).expanduser())
    candidates.append(Path(gm.DATA_PATH).expanduser())
    candidates.append(Path.home() / "Research" / "BEHAVIOR-1K" / "datasets")

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _registered_robot_models():
    robot_models_dir = None
    for data_root in _data_root_candidates():
        candidate = data_root / "omnigibson-robot-assets" / "models"
        if candidate.exists():
            robot_models_dir = candidate
            break
    if robot_models_dir is None:
        return []

    models = set()
    for robot_dir in robot_models_dir.iterdir():
        if not robot_dir.is_dir() or robot_dir.name.startswith("."):
            continue
        yaml_path = robot_dir / f"{robot_dir.name}.yaml"
        usd_dir = robot_dir / "usd"
        has_usd = (
            (usd_dir / f"{robot_dir.name}.usda").exists()
            or (usd_dir / f"{robot_dir.name}.usd").exists()
            or any(usd_dir.glob("*.usda"))
            or any(usd_dir.glob("*.usd"))
        )
        if yaml_path.exists() or has_usd:
            models.add(robot_dir.name)
    return sorted(models)


REGISTERED_ROBOTS = _registered_robot_models()

if getattr(builtins, "OMNIGIBSON_NEWTON_NATIVE", False):
    Robot = None
else:
    from omnigibson.robots.robot import Robot

__all__ = [
    "Robot",
    "REGISTERED_ROBOTS",
]
