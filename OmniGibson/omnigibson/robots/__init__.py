# # Base classes and non-YAML robots (keep static imports)
# from omnigibson.robots.active_camera_robot import ActiveCameraRobot
# from omnigibson.robots.locomotion_robot import LocomotionRobot
# from omnigibson.robots.manipulation_robot import ManipulationRobot
# from omnigibson.robots.robot_base import REGISTERED_ROBOTS, BaseRobot
# from omnigibson.robots.two_wheel_robot import TwoWheelRobot

# # Load YAML-defined robot classes first (triggers auto-discovery)
# import omnigibson.robots.robot_config_parser  # noqa: F401

# # Dynamically import YAML-defined robot classes from their injected modules

# from omnigibson.robots.robot_configs.vx300s import VX300S
# from omnigibson.robots.robot_configs.a1 import A1
# from omnigibson.robots.robot_configs.fetch import Fetch
# from omnigibson.robots.robot_configs.franka import FrankaPanda
# from omnigibson.robots.robot_configs.franka_mounted import FrankaMounted
# from omnigibson.robots.robot_configs.freight import Freight
# from omnigibson.robots.robot_configs.husky import Husky
# from omnigibson.robots.robot_configs.locobot import Locobot
# from omnigibson.robots.robot_configs.r1 import R1
# from omnigibson.robots.robot_configs.r1pro import R1Pro
# from omnigibson.robots.robot_configs.stretch import Stretch
# from omnigibson.robots.robot_configs.tiago import Tiago
# from omnigibson.robots.robot_configs.turtlebot import Turtlebot
from pathlib import Path
from omnigibson.robots.robot import Robot

REGISTERED_ROBOTS = []
robot_config_dir = Path(__file__).parent / "robot_configs"
for yaml_file in sorted(robot_config_dir.glob("*.yaml")):
    REGISTERED_ROBOTS.append(yaml_file.stem)

# A1 = lambda *args, robot_type_name="a1", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# Fetch = lambda *args, robot_type_name="fetch", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# FrankaMounted = lambda *args, robot_type_name="franka_mounted", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# Tiago = lambda *args, robot_type_name="tiago", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# FrankaPanda = lambda *args, robot_type_name="franka", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# Freight = lambda *args, robot_type_name="freight", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# Husky = lambda *args, robot_type_name="husky", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# Locobot = lambda *args, robot_type_name="locobot", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# R1 = lambda *args, robot_type_name="r1", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# R1Pro = lambda *args, robot_type_name="r1pro", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# Stretch = lambda *args, robot_type_name="stretch", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# Turtlebot = lambda *args, robot_type_name="turtlebot", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)
# VX300S = lambda *args, robot_type_name="vx300s", **kwargs: Robot(*args, robot_type_name=robot_type_name, **kwargs)

__all__ = [
    "Robot",
    "REGISTERED_ROBOTS",
]
