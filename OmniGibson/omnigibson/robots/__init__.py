# Base classes and non-YAML robots (keep static imports)
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.robots.robot_base import REGISTERED_ROBOTS, BaseRobot
from omnigibson.robots.two_wheel_robot import TwoWheelRobot

# Load YAML-defined robot classes first (triggers auto-discovery)
import omnigibson.robots.robot_config_parser  # noqa: F401

# Dynamically import YAML-defined robot classes from their injected modules

from omnigibson.robots.robot_configs.vx300s import VX300S
from omnigibson.robots.robot_configs.a1 import A1
from omnigibson.robots.robot_configs.fetch import Fetch
from omnigibson.robots.robot_configs.franka import FrankaPanda
from omnigibson.robots.robot_configs.franka_mounted import FrankaMounted
from omnigibson.robots.robot_configs.freight import Freight
from omnigibson.robots.robot_configs.husky import Husky
from omnigibson.robots.robot_configs.locobot import Locobot
from omnigibson.robots.robot_configs.r1 import R1
from omnigibson.robots.robot_configs.r1pro import R1Pro
from omnigibson.robots.robot_configs.stretch import Stretch
from omnigibson.robots.robot_configs.tiago import Tiago
from omnigibson.robots.robot_configs.turtlebot import Turtlebot
__all__ = [
    "A1",
    "ActiveCameraRobot",
    "BaseRobot",
    "Fetch",
    "FrankaMounted",
    "FrankaPanda",
    "Freight",
    "Husky",
    "Locobot",
    "LocomotionRobot",
    "ManipulationRobot",
    "R1",
    "R1Pro",
    "REGISTERED_ROBOTS",
    "Stretch",
    "Tiago",
    "Turtlebot",
    "TwoWheelRobot",
    "VX300S",
]
