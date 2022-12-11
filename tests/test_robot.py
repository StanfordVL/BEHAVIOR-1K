import numpy as np


from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.scenes.stadium_scene import StadiumScene
from omnigibson.simulator import Simulator
from omnigibson.utils.asset_utils import download_assets

download_assets()


def test_fetch():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    fetch = REGISTERED_ROBOTS["Fetch"]()
    s.import_object(fetch)
    for i in range(100):
        fetch.calc_state()
        s.step()
    s.disconnect()


def test_turtlebot():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_object(turtlebot)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_husky():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    husky = REGISTERED_ROBOTS["Husky"]()
    s.import_object(husky)
    nbody = p.getNumBodies()
    s.disconnect()
    assert nbody == 5


def test_turtlebot_position():
    s = Simulator(mode="headless")
    scene = StadiumScene()
    s.import_scene(scene)
    turtlebot = REGISTERED_ROBOTS["Turtlebot"]()
    s.import_object(turtlebot)

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

    s.import_object(turtlebot1)
    s.import_object(turtlebot2)
    s.import_object(turtlebot3)

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
    s.import_object(turtlebot)
    turtlebot.set_position([0, 1, 0.5])

    husky = REGISTERED_ROBOTS["Husky"]()
    s.import_object(husky)
    husky.set_position([0, 6, 0.5])

    for robot in scene.robots:
        print(type(robot), len(robot.joints), robot.calc_state().shape)

    for i in range(100):
        s.step()

    s.disconnect()
