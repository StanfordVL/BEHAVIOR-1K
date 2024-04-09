import numpy as np

import omnigibson as og
from omnigibson.sensors import VisionSensor
from omnigibson.object_states import ObjectsInFOVOfRobot
from omnigibson.utils.constants import semantic_class_name_to_id
from test_robot_states_flatcache import setup_environment, camera_pose_test


def test_camera_pose_flatcache_off():
    camera_pose_test(False)


def test_camera_semantic_segmentation():
    env = setup_environment(False)
    robot = env.robots[0]
    env.reset()
    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]
    env.reset()
    all_observation, all_info = vision_sensor.get_obs()
    seg_semantic = all_observation["seg_semantic"]
    seg_semantic_info = all_info["seg_semantic"]
    agent_label = semantic_class_name_to_id()["agent"]
    background_label = semantic_class_name_to_id()["background"]
    assert np.all(np.isin(seg_semantic, [agent_label, background_label]))
    assert set(seg_semantic_info.keys()) == {agent_label, background_label}
    og.sim.clear()


def test_object_in_FOV_of_robot():
    env = setup_environment(False)
    robot = env.robots[0]
    env.reset()
    assert robot.states[ObjectsInFOVOfRobot].get_value() == [robot]
    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]
    vision_sensor.set_position_orientation(position=[100, 150, 100])
    og.sim.step()
    og.sim.step()
    assert robot.states[ObjectsInFOVOfRobot].get_value() == []
    og.sim.clear()
