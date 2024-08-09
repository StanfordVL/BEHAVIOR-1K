import numpy as np

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T


def setup_multi_environment(num_of_envs, robot="Tiago", additional_objects_cfg=[]):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls"],
        },
        "robots": [
            {
                "type": "Tiago",
                "obs_modalities": [],
            }
        ],
    }

    cfg["objects"] = additional_objects_cfg
    cfg["robots"][0]["type"] = robot

    if og.sim is None:
        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth)
        gm.RENDER_VIEWER_CAMERA = False
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    vec_env = og.VectorEnvironment(num_of_envs, cfg)
    return vec_env

def test_tiago_getter():

    vec_env = setup_multi_environment(2)
    robot1 = vec_env.envs[0].scene.robots[0]

    robot1_world_position, robot1_world_orientation = robot1.get_position_orientation()
    robot1_scene_position, robot1_scene_orientation = robot1.get_position_orientation(frame="scene")
    robot1_parent_position, robot1_parent_orientation = robot1.get_position_orientation(frame="parent")
    
    # Test the get_position_orientation method for 3 different frames
    # since the robot is at the origin, the position and orientation should be the same
    assert np.allclose(robot1_world_position, robot1_parent_position, atol=1e-3)
    assert np.allclose(robot1_world_position, robot1_scene_position, atol=1e-3)
    assert np.allclose(robot1_world_orientation, robot1_parent_orientation, atol=1e-3)
    assert np.allclose(robot1_world_orientation, robot1_scene_orientation, atol=1e-3)

    # test if the scene position is non-zero, the getter with parent and world frame should return different values
    robot2 = vec_env.envs[1].scene.robots[0]
    scene_position, scene_orientation = vec_env.envs[1].scene.prim.get_position_orientation()
    
    robot2_world_position, robot2_world_orientation = robot2.get_position_orientation()
    robot2_scene_position, robot2_scene_orientation = robot2.get_position_orientation(frame="scene")
    robot2_parent_position, robot2_parent_orientation = robot2.get_position_orientation(frame="parent")

    assert np.allclose(robot2_parent_position, robot2_scene_position, atol=1e-3)
    assert np.allclose(robot2_parent_orientation, robot2_scene_orientation, atol=1e-3)

    combined_position, combined_orientation = T.pose_transform(scene_position, scene_orientation, robot2_parent_position, robot2_parent_orientation)
    assert np.allclose(robot2_world_position, combined_position, atol=1e-3)
    assert np.allclose(robot2_world_orientation, combined_orientation, atol=1e-3)

    # Clean up
    og.clear()

def test_tiago_setter():
    vec_env = setup_multi_environment(2)

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]
    
    # Test setting position and orientation in world frame
    new_world_pos = np.array([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat([0, 0, np.pi/2])
    robot.set_position_orientation(new_world_pos, new_world_ori)
    
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert np.allclose(got_world_ori, new_world_ori, atol=1e-3)
    
    # Test setting position and orientation in scene frame
    new_scene_pos = np.array([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat([0, np.pi/4, 0])
    robot.set_position_orientation(new_scene_pos, new_scene_ori, frame="scene")
    
    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert np.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert np.allclose(got_scene_ori, new_scene_ori, atol=1e-3)
    
    # Test setting position and orientation in parent frame
    new_parent_pos = np.array([-1.0, -2.0, 0.1])
    new_parent_ori = T.euler2quat([np.pi/6, 0, 0])
    robot.set_position_orientation(new_parent_pos, new_parent_ori, frame="parent")
    
    got_parent_pos, got_parent_ori = robot.get_position_orientation(frame="parent")
    assert np.allclose(got_parent_pos, new_parent_pos, atol=1e-3)
    assert np.allclose(got_parent_ori, new_parent_ori, atol=1e-3)
    
    # Verify that world frame position/orientation has changed after setting in parent frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not np.allclose(got_world_ori, new_world_ori, atol=1e-3)
    
    # Clean up
    og.clear()

    # assert that when the simulator is stopped, the behavior for getter/setter is not affected
    vec_env = setup_multi_environment(2)
    og.sim.stop()

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]
    
    # Test setting position and orientation in world frame
    new_world_pos = np.array([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat([0, 0, np.pi/2])
    robot.set_position_orientation(new_world_pos, new_world_ori)
    
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert np.allclose(got_world_ori, new_world_ori, atol=1e-3)
    
    # Test setting position and orientation in scene frame
    new_scene_pos = np.array([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat([0, np.pi/4, 0])
    robot.set_position_orientation(new_scene_pos, new_scene_ori, frame="scene")
    
    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert np.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert np.allclose(got_scene_ori, new_scene_ori, atol=1e-3)
    
    # Test setting position and orientation in parent frame
    new_parent_pos = np.array([-1.0, -2.0, 0.1])
    new_parent_ori = T.euler2quat([np.pi/6, 0, 0])
    robot.set_position_orientation(new_parent_pos, new_parent_ori, frame="parent")
    
    got_parent_pos, got_parent_ori = robot.get_position_orientation(frame="parent")
    assert np.allclose(got_parent_pos, new_parent_pos, atol=1e-3)
    assert np.allclose(got_parent_ori, new_parent_ori, atol=1e-3)
    
    # Verify that world frame position/orientation has changed after setting in parent frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not np.allclose(got_world_ori, new_world_ori, atol=1e-3)

    og.clear()

def test_behavior_getter():

    vec_env = setup_multi_environment(2, robot="BehaviorRobot")
    robot1 = vec_env.envs[0].scene.robots[0]

    robot1_world_position, robot1_world_orientation = robot1.get_position_orientation()
    robot1_scene_position, robot1_scene_orientation = robot1.get_position_orientation(frame="scene")
    robot1_parent_position, robot1_parent_orientation = robot1.get_position_orientation(frame="parent")
    
    # Test the get_position_orientation method for 3 different frames
    # since the robot is at the origin, the position and orientation should be the same
    assert np.allclose(robot1_world_position, robot1_parent_position, atol=1e-3)
    assert np.allclose(robot1_world_position, robot1_scene_position, atol=1e-3)
    assert np.allclose(robot1_world_orientation, robot1_parent_orientation, atol=1e-3)
    assert np.allclose(robot1_world_orientation, robot1_scene_orientation, atol=1e-3)

    # test if the scene position is non-zero, the getter with parent and world frame should return different values
    robot2 = vec_env.envs[1].scene.robots[0]
    scene_position, scene_orientation = vec_env.envs[1].scene.prim.get_position_orientation()
    
    robot2_world_position, robot2_world_orientation = robot2.get_position_orientation()
    robot2_scene_position, robot2_scene_orientation = robot2.get_position_orientation(frame="scene")
    robot2_parent_position, robot2_parent_orientation = robot2.get_position_orientation(frame="parent")

    assert np.allclose(robot2_parent_position, robot2_scene_position, atol=1e-3)
    assert np.allclose(robot2_parent_orientation, robot2_scene_orientation, atol=1e-3)

    combined_position, combined_orientation = T.pose_transform(scene_position, scene_orientation, robot2_parent_position, robot2_parent_orientation)
    assert np.allclose(robot2_world_position, combined_position, atol=1e-3)
    assert np.allclose(robot2_world_orientation, combined_orientation, atol=1e-3)

    # Clean up
    og.clear()

def test_behavior_setter():
    vec_env = setup_multi_environment(2, robot="BehaviorRobot")

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]
    
    # Test setting position and orientation in world frame
    new_world_pos = np.array([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat([0, 0, np.pi/2])

    robot.set_position_orientation(new_world_pos, new_world_ori)
    
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert np.allclose(got_world_ori, new_world_ori, atol=1e-3)
    
    # Test setting position and orientation in scene frame
    new_scene_pos = np.array([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat([0, np.pi/4, 0])
    robot.set_position_orientation(new_scene_pos, new_scene_ori, frame="scene")
    
    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert np.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert np.allclose(got_scene_ori, new_scene_ori, atol=1e-3)
    
    # Test setting position and orientation in parent frame
    new_parent_pos = np.array([-1.0, -2.0, 0.1])
    new_parent_ori = T.euler2quat([np.pi/6, 0, 0])
    robot.set_position_orientation(new_parent_pos, new_parent_ori, frame="parent")
    
    got_parent_pos, got_parent_ori = robot.get_position_orientation(frame="parent")
    assert np.allclose(got_parent_pos, new_parent_pos, atol=1e-3)
    assert np.allclose(got_parent_ori, new_parent_ori, atol=1e-3)
    
    # Verify that world frame position/orientation has changed after setting in parent frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not np.allclose(got_world_ori, new_world_ori, atol=1e-3)
    
    # Clean up
    og.clear()

    # assert that when the simulator is stopped, the behavior for getter/setter is not affected
    vec_env = setup_multi_environment(2)
    og.sim.stop()

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]
    
    # Test setting position and orientation in world frame
    new_world_pos = np.array([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat([0, 0, np.pi/2])
    robot.set_position_orientation(new_world_pos, new_world_ori)
    
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert np.allclose(got_world_ori, new_world_ori, atol=1e-3)
    
    # Test setting position and orientation in scene frame
    new_scene_pos = np.array([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat([0, np.pi/4, 0])
    robot.set_position_orientation(new_scene_pos, new_scene_ori, frame="scene")
    
    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert np.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert np.allclose(got_scene_ori, new_scene_ori, atol=1e-3)
    
    # Test setting position and orientation in parent frame
    new_parent_pos = np.array([-1.0, -2.0, 0.1])
    new_parent_ori = T.euler2quat([np.pi/6, 0, 0])
    robot.set_position_orientation(new_parent_pos, new_parent_ori, frame="parent")
    
    got_parent_pos, got_parent_ori = robot.get_position_orientation(frame="parent")
    assert np.allclose(got_parent_pos, new_parent_pos, atol=1e-3)
    assert np.allclose(got_parent_ori, new_parent_ori, atol=1e-3)
    
    # Verify that world frame position/orientation has changed after setting in parent frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not np.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not np.allclose(got_world_ori, new_world_ori, atol=1e-3)