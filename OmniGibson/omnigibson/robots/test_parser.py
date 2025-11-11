#!/usr/bin/env python3
"""
Test script to verify that the dynamic YAML parser works correctly for Turtlebot.
This script tests:
1. Loading the robot class from YAML
2. Verifying all properties match the original Python class
3. Creating an instance and testing property access
"""

import sys
from pathlib import Path
import torch as th
from omnigibson.robots import VX300S
# from omnigibson.robots.turtlebot import Turtlebot as OriginalTurtlebot

import omnigibson as og
from omnigibson.scenes import Scene
# from omnigibson.robots import Fetch

# Launch OG
og.launch()
scene = Scene()
og.sim.import_scene(scene)

# Not specifying `controller_config` will automatically use the default set of values
robot = VX300S(name="1")

# Import robot and play sim
scene.add_object(robot)
og.sim.play()

