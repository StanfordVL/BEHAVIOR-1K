import os

import numpy as np


from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets

download_assets()


def test_fetch():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    fetch = REGISTERED_ROBOTS["Fetch"]()
    s.import_robot(fetch)
    for i in range(100):
        fetch.calc_state()
        s.step()
    s.disconnect()


def test_turtlebot():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_robot(turtlebot)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_jr2():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    jr2 = REGISTERED_ROBOTS["JR2"]()
    s.import_robot(jr2)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_ant():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    ant = REGISTERED_ROBOTS["Ant"]()
    s.import_robot(ant)
    ant2 = REGISTERED_ROBOTS["Ant"]()
    s.import_robot(ant2)
    ant2.set_position([0, 2, 2])
    nbody = p.getNumBodies()
    for i in range(100):
        s.step()
    s.disconnect()
    assert nbody == 6


def test_husky():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    husky = REGISTERED_ROBOTS["Husky"]()
    s.import_robot(husky)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_turtlebot_position():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_robot(turtlebot)

    turtlebot.set_position([0, 0, 5])

    nbody = p.getNumBodies()
    pos = turtlebot.get_position()
    s.disconnect()
    assert nbody == 5
    assert np.allclose(pos, np.array([0, 0, 5]))


def test_multiagent():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot1 = REGISTERED_ROBOTS["Turtlebot"]()
    turtlebot2 = REGISTERED_ROBOTS["Turtlebot"]()
    turtlebot3 = REGISTERED_ROBOTS["Turtlebot"]()

    s.import_robot(turtlebot1)
    s.import_robot(turtlebot2)
    s.import_robot(turtlebot3)

    turtlebot1.set_position([1, 0, 0.5])
    turtlebot2.set_position([0, 0, 0.5])
    turtlebot3.set_position([-1, 0, 0.5])

    nbody = p.getNumBodies()
    for i in range(100):
        s.step()

    s.disconnect()
    assert nbody == 7


def show_action_sensor_space():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)

    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_robot(turtlebot)
    turtlebot.set_position([0, 1, 0.5])

    ant = REGISTERED_ROBOTS["Ant"]()
    s.import_robot(ant)
    ant.set_position([0, 2, 0.5])

    jr = REGISTERED_ROBOTS["JR2"]()
    s.import_robot(jr)
    jr.set_position([0, 4, 0.5])

    jr2 = REGISTERED_ROBOTS["JR2"]()
    s.import_robot(jr2)
    jr2.set_position([0, 5, 0.5])

    husky = REGISTERED_ROBOTS["Husky"]()
    s.import_robot(husky)
    husky.set_position([0, 6, 0.5])

    for robot in scene.robots:
        print(type(robot), len(robot.joints), robot.calc_state().shape)

    for i in range(100):
        s.step()

    s.disconnect()
