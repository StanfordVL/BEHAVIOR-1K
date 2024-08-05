import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.macros import gm
from omnigibson.robots import *
from omnigibson.sensors import VisionSensor
from omnigibson.utils.transform_utils import mat2pose, pose2mat, relative_pose_transform
from omnigibson.utils.usd_utils import PoseAPI


def setup_environment(flatcache):
    """
    Sets up the environment with or without flatcache based on the flatcache parameter.
    """
    if og.sim is None:
        # Set global flags
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = flatcache  # Set based on function parameter
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Define the environment configuration
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb", "seg_semantic", "seg_instance"],
                "position": [150, 150, 100],
                "orientation": [0, 0, 0, 1],
            }
        ],
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

    robot_to_sensor_mat = pose2mat(
        relative_pose_transform(sensor_world_pos, sensor_world_ori, robot_world_pos, robot_world_ori)
    )

    sensor_world_pos_gt = np.array([150.16513062, 150.0, 101.39360809])
    sensor_world_ori_gt = np.array([-0.29444984, 0.29444979, 0.64288365, -0.64288352])

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
    expected_new_camera_world_pos, expected_new_camera_world_ori = mat2pose(
        camera_parent_world_transform @ pose2mat([[10, 10, 10], [0, 0, 0, 1]])
    )
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
    camera_parent_prim.GetAttribute("xformOp:scale").Set(lazy.pxr.Gf.Vec3d([2.0, 2.0, 2.0]))
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    camera_local_pose = vision_sensor.get_local_pose()
    expected_new_camera_world_pos, _ = mat2pose(camera_parent_world_transform @ pose2mat(camera_local_pose))
    new_camera_world_pose = vision_sensor.get_position_orientation()
    assert np.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)

    og.clear()


def test_camera_pose_flatcache_on():
    camera_pose_test(True)


def test_robot_load_drive():
    if og.sim is None:
        # Set global flags
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    config = {
        "scene": {
            "type": "Scene",
        },
    }

    env = og.Environment(configs=config)

    # Iterate over all robots and test their motion
    for robot_name, robot_cls in REGISTERED_ROBOTS.items():
        robot = robot_cls(
            name=robot_name,
            obs_modalities=[],
        )
        og.sim.stop()
        env.scene.add_object(robot)
        og.sim.play()

        # At least one step is always needed while sim is playing for any imported object to be fully initialized
        og.sim.play()
        og.sim.step()

        # Reset robot and make sure it's not moving
        robot.reset()
        robot.keep_still()

        # Set viewer in front facing robot
        og.sim.viewer_camera.set_position_orientation(
            position=np.array([2.69918369, -3.63686664, 4.57894564]),
            orientation=np.array([0.39592411, 0.1348514, 0.29286304, 0.85982]),
        )

        if not robot_name in ["Husky", "BehaviorRobot"]:
            # Husky base motion is a little messed up because of the 4-wheel drive; skipping for now
            # BehaviorRobot does not work with the primitive actions at the moment

            # If this is a manipulation robot, we want to test moving the arm
            if isinstance(robot, ManipulationRobot):
                # load IK controller
                controller_config = {
                    f"arm_{robot.default_arm}": {"name": "InverseKinematicsController", "mode": "pose_absolute_ori"}
                }
                robot.reload_controllers(controller_config=controller_config)
                env.scene.update_initial_state()

                action_primitives = StarterSemanticActionPrimitives(env)

                eef_pos = env.robots[0].get_eef_position()
                eef_orn = env.robots[0].get_eef_orientation()
                if isinstance(robot, Stretch):  # Stretch arm faces the y-axis
                    target_eef_pos = (eef_pos[0], eef_pos[1] - 0.1, eef_pos[2])
                else:
                    target_eef_pos = (eef_pos[0] + 0.1, eef_pos[1], eef_pos[2])
                target_eef_orn = eef_orn
                for action in action_primitives._move_hand_direct_ik((target_eef_pos, target_eef_orn)):
                    env.step(action)
                assert np.linalg.norm(robot.get_eef_position() - target_eef_pos) < 0.05

            # If this is a locomotion robot, we want to test driving
            if isinstance(robot, LocomotionRobot):
                # load diff drive controller
                controller_config = {"base": {"name": "DifferentialDriveController"}}
                action_primitives = StarterSemanticActionPrimitives(env)
                goal_location = (0, 1, 0)
                for action in action_primitives._navigate_to_pose_direct(goal_location):
                    env.step(action)
                assert np.linalg.norm(robot.get_position()[:2] - goal_location[:2]) < 0.1

        # Stop the simulator and remove the robot
        og.sim.stop()
        og.sim.remove_object(obj=robot)

    env.close()
