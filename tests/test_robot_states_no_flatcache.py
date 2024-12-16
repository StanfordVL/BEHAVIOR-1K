import torch as th
from test_robot_states_flatcache import camera_pose_test, setup_environment

import omnigibson as og
from omnigibson.object_states import ObjectsInFOVOfRobot
from omnigibson.sensors import VisionSensor
from omnigibson.utils.constants import semantic_class_name_to_id


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
    assert th.all(th.isin(seg_semantic, th.tensor([agent_label, background_label], device=seg_semantic.device)))
    assert set(seg_semantic_info.keys()) == {agent_label, background_label}
    og.clear()


def test_object_in_FOV_of_robot():
    env = setup_environment(False)
    robot = env.robots[0]
    env.reset()
    objs_in_fov = robot.states[ObjectsInFOVOfRobot].get_value()
    assert len(objs_in_fov) == 1 and next(iter(objs_in_fov)) == robot
    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    for vision_sensor in sensors:
        vision_sensor.set_position_orientation(position=[100, 150, 100])
    og.sim.step()
    for _ in range(5):
        og.sim.render()
    # Since the sensor is moved away from the robot, the robot should not see itself
    assert len(robot.states[ObjectsInFOVOfRobot].get_value()) == 0
    og.clear()
