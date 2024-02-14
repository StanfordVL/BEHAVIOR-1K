import numpy as np

import omnigibson as og
from omnigibson.macros import gm

from omnigibson.sensors import VisionSensor


def test_vision_sensor_pose():
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": [],
                "position": [150, 150, 100],
                "orientation": [0, 0, 0, 1],
                "controller_config": {
                    "arm_0": {
                        "name": "NullJointController",
                        "motor_type": "position",
                    },
                },
            }
        ]
    }
    
    # Make sure sim is stopped
    if og.sim is not None:
        og.sim.stop()
    
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = True
    
    env = og.Environment(configs=config)
    robot = env.robots[0]
    env.reset()
    
    # robot = og.sim.scene.object_registry("name", "robot0")
    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]

    # Get vision sensor world pose via directly calling get_position_orientation
    sensor_world_pos, sensor_world_ori = vision_sensor.get_position_orientation()

    sensor_world_pos_gt = np.array([150.16513062, 150.0, 101.36912537])
    sensor_world_ori_gt = np.array([-0.29444987, 0.29444981, 0.64288363, -0.64288352])
    
    assert np.allclose(sensor_world_pos, sensor_world_pos_gt)
    assert np.allclose(sensor_world_ori, sensor_world_ori_gt)
    
    # Now, we want to move the robot and check if the sensor pose is updated
    old_camera_local_pose = vision_sensor.get_local_pose()
    old_camera_world_pose = vision_sensor.get_position_orientation()
    robot.set_position_orientation(position=[100, 100, 100])
    new_camera_local_pose = vision_sensor.get_local_pose()
    new_camera_world_pose = vision_sensor.get_position_orientation()
    assert np.allclose(old_camera_local_pose[0], new_camera_local_pose[0])
    assert not np.allclose(old_camera_world_pose[0], new_camera_world_pose[0])
    
    # Then, we want to move the local pose of the camera and check 
    # 1) if the world pose is updated 2) if the robot stays in the same position
    old_camera_world_pose = vision_sensor.get_position_orientation()
    old_camera_local_pose = vision_sensor.get_local_pose()
    vision_sensor.set_local_pose(old_camera_local_pose[0]+10, [0, 0, 0, 1])
    new_camera_world_pose = vision_sensor.get_position_orientation()
    assert not np.allclose(old_camera_world_pose[0], new_camera_world_pose[0])
    assert not np.allclose(old_camera_world_pose[1], new_camera_world_pose[1])
    assert np.allclose(robot.get_position(), [100, 100, 100])
    
    # Finally, we want to move the world pose of the camera and check
    # 1) if the local pose is updated 2) if the robot stays in the same position
    old_camera_local_pose = vision_sensor.get_local_pose()
    robot.set_position_orientation(position=[150, 150, 100])
    old_camera_local_pose = vision_sensor.get_local_pose()
    vision_sensor.set_position_orientation([150, 150, 101.36912537], [-0.29444987, 0.29444981, 0.64288363, -0.64288352])
    new_camera_local_pose = vision_sensor.get_local_pose()
    assert not np.allclose(old_camera_local_pose[0], new_camera_local_pose[0])
    assert np.allclose(robot.get_position(), [150, 150, 100])
    
    og.sim.clear()

