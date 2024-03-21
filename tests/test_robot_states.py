import numpy as np

import omnigibson as og
from omnigibson.macros import gm

import omnigibson.lazy as lazy
from omnigibson.sensors import VisionSensor
from omnigibson.object_states import ObjectsInFOVOfRobot
from omnigibson.utils.transform_utils import pose2mat, mat2pose, relative_pose_transform
from omnigibson.utils.usd_utils import PoseAPI
from omnigibson.utils.constants import semantic_class_name_to_id

def setup_environment(flatcache=True):
    """
    Sets up the environment with or without flatcache based on the flatcache parameter.
    """
    # Ensure any existing simulation is stopped
    if og.sim is not None:
        og.sim.stop()
    
    # Set global flags
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = flatcache  # Set based on function parameter
    
    # Define the environment configuration
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": 'all',
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

    env = og.Environment(configs=config)
    return env

def camera_pose_test(flatcache):
    env = setup_environment(flatcache)
    robot = env.robots[0]
    env.reset()

    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]

    # Get vision sensor world pose via directly calling get_position_orientation
    robot_world_pos, robot_world_ori = robot.get_position_orientation()
    sensor_world_pos, sensor_world_ori = vision_sensor.get_position_orientation()
    
    robot_to_sensor_mat = pose2mat(relative_pose_transform(sensor_world_pos, sensor_world_ori, robot_world_pos, robot_world_ori))

    sensor_world_pos_gt = np.array([150.16513062, 150., 101.38952637])
    sensor_world_ori_gt = np.array([-0.29444987, 0.29444981, 0.64288363, -0.64288352])
    
    assert np.allclose(sensor_world_pos, sensor_world_pos_gt, atol=1e-3)
    assert np.allclose(sensor_world_ori, sensor_world_ori_gt, atol=1e-3)

    # Now, we want to move the robot and check if the sensor pose has been updated
    old_camera_local_pose = vision_sensor.get_local_pose()

    robot.set_position_orientation(position=[100, 100, 100])
    new_camera_local_pose = vision_sensor.get_local_pose()
    new_camera_world_pose = vision_sensor.get_position_orientation()
    robot_pose_mat = pose2mat(robot.get_position_orientation())
    expected_camera_world_pos, expected_camera_world_ori = mat2pose(robot_pose_mat @ robot_to_sensor_mat)
    assert np.allclose(old_camera_local_pose[0], new_camera_local_pose[0], atol=1e-3)
    assert np.allclose(new_camera_world_pose[0], expected_camera_world_pos, atol=1e-3)
    assert np.allclose(new_camera_world_pose[1], expected_camera_world_ori, atol=1e-3)
    
    # Then, we want to move the local pose of the camera and check 
    # 1) if the world pose is updated 2) if the robot stays in the same position
    old_camera_local_pose = vision_sensor.get_local_pose()
    vision_sensor.set_local_pose(position=[10, 10, 10], orientation=[0, 0, 0, 1])
    new_camera_world_pose = vision_sensor.get_position_orientation()
    camera_parent_prim = lazy.omni.isaac.core.utils.prims.get_prim_parent(vision_sensor.prim)
    camera_parent_path = str(camera_parent_prim.GetPath())
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    expected_new_camera_world_pos, expected_new_camera_world_ori = mat2pose(camera_parent_world_transform @ pose2mat([[10, 10, 10], [0, 0, 0, 1]]))
    assert np.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)
    assert np.allclose(new_camera_world_pose[1], expected_new_camera_world_ori, atol=1e-3)
    assert np.allclose(robot.get_position(), [100, 100, 100], atol=1e-3)

    
    # Finally, we want to move the world pose of the camera and check
    # 1) if the local pose is updated 2) if the robot stays in the same position
    robot.set_position_orientation(position=[150, 150, 100])
    old_camera_local_pose = vision_sensor.get_local_pose()
    vision_sensor.set_position_orientation([150, 150, 101.36912537], [-0.29444987, 0.29444981, 0.64288363, -0.64288352])
    new_camera_local_pose = vision_sensor.get_local_pose()
    assert not np.allclose(old_camera_local_pose[0], new_camera_local_pose[0], atol=1e-3)
    assert not np.allclose(old_camera_local_pose[1], new_camera_local_pose[1], atol=1e-3)
    assert np.allclose(robot.get_position(), [150, 150, 100], atol=1e-3)
    
    # Another test we want to try is setting the camera's parent scale and check if the world pose is updated
    camera_parent_prim.GetAttribute('xformOp:scale').Set(lazy.pxr.Gf.Vec3d([2.0, 2.0, 2.0]))
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    camera_local_pose = vision_sensor.get_local_pose()
    expected_new_camera_world_pos, _ = mat2pose(camera_parent_world_transform @ pose2mat(camera_local_pose))
    new_camera_world_pose = vision_sensor.get_position_orientation()
    assert np.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)
    
    og.sim.clear()

def test_camera_pose_flatcache_on():
    camera_pose_test(True)

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
    seg_semantic = all_observation['seg_semantic']
    seg_semantic_info = all_info['seg_semantic']
    agent_label = semantic_class_name_to_id()['agent']
    background_label = semantic_class_name_to_id()['background']
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
