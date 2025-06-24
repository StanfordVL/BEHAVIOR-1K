import time

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.macros import gm
from omnigibson.macros import macros as m
from omnigibson.robots import REGISTERED_ROBOTS, Fetch, LocomotionRobot, ManipulationRobot, Stretch
from omnigibson.sensors import VisionSensor
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.transform_utils import mat2pose, pose2mat, quaternions_close, relative_pose_transform
from omnigibson.utils.usd_utils import PoseAPI
from omnigibson.utils.sim_utils import prim_paths_to_rigid_prims


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
    og.sim.step()

    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]

    # Get vision sensor world pose via directly calling get_position_orientation
    robot_world_pos, robot_world_ori = robot.get_position_orientation()
    sensor_world_pos, sensor_world_ori = vision_sensor.get_position_orientation()

    robot_to_sensor_mat = pose2mat(
        relative_pose_transform(sensor_world_pos, sensor_world_ori, robot_world_pos, robot_world_ori)
    )

    sensor_world_pos_gt = th.tensor([150.5134, 149.8278, 101.0816])
    sensor_world_ori_gt = th.tensor([0.0176, -0.1205, 0.9910, -0.0549])

    assert th.allclose(sensor_world_pos, sensor_world_pos_gt, atol=1e-3)
    assert quaternions_close(sensor_world_ori, sensor_world_ori_gt, atol=1e-3)

    # Now, we want to move the robot and check if the sensor pose has been updated
    old_camera_local_pose = vision_sensor.get_position_orientation(frame="parent")

    robot.set_position_orientation(position=[100, 100, 100])
    new_camera_local_pose = vision_sensor.get_position_orientation(frame="parent")
    new_camera_world_pose = vision_sensor.get_position_orientation()
    robot_pose_mat = pose2mat(robot.get_position_orientation())
    expected_camera_world_pos, expected_camera_world_ori = mat2pose(robot_pose_mat @ robot_to_sensor_mat)
    assert th.allclose(old_camera_local_pose[0], new_camera_local_pose[0], atol=1e-3)
    assert th.allclose(new_camera_world_pose[0], expected_camera_world_pos, atol=1e-3)
    assert quaternions_close(new_camera_world_pose[1], expected_camera_world_ori, atol=1e-3)

    # Then, we want to move the local pose of the camera and check
    # 1) if the world pose is updated 2) if the robot stays in the same position
    old_camera_local_pose = vision_sensor.get_position_orientation(frame="parent")
    vision_sensor.set_position_orientation(position=[10, 10, 10], orientation=[0, 0, 0, 1], frame="parent")
    new_camera_world_pose = vision_sensor.get_position_orientation()
    camera_parent_prim = lazy.isaacsim.core.utils.prims.get_prim_parent(vision_sensor.prim)
    camera_parent_path = str(camera_parent_prim.GetPath())
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    expected_new_camera_world_pos, expected_new_camera_world_ori = mat2pose(
        camera_parent_world_transform
        @ pose2mat((th.tensor([10, 10, 10], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32)))
    )
    assert th.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)
    assert quaternions_close(new_camera_world_pose[1], expected_new_camera_world_ori, atol=1e-3)
    assert th.allclose(robot.get_position_orientation()[0], th.tensor([100, 100, 100], dtype=th.float32), atol=1e-3)

    # Finally, we want to move the world pose of the camera and check
    # 1) if the local pose is updated 2) if the robot stays in the same position
    robot.set_position_orientation(position=[150, 150, 100])
    old_camera_local_pose = vision_sensor.get_position_orientation(frame="parent")
    vision_sensor.set_position_orientation(
        position=[150, 150, 101.36912537], orientation=[-0.29444987, 0.29444981, 0.64288363, -0.64288352]
    )
    new_camera_local_pose = vision_sensor.get_position_orientation(frame="parent")
    assert not th.allclose(old_camera_local_pose[0], new_camera_local_pose[0], atol=1e-3)
    assert not quaternions_close(old_camera_local_pose[1], new_camera_local_pose[1], atol=1e-3)
    assert th.allclose(robot.get_position_orientation()[0], th.tensor([150, 150, 100], dtype=th.float32), atol=1e-3)

    # Another test we want to try is setting the camera's parent scale and check if the world pose is updated
    camera_parent_prim.GetAttribute("xformOp:scale").Set(lazy.pxr.Gf.Vec3d([2.0, 2.0, 2.0]))
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    camera_local_pose = vision_sensor.get_position_orientation(frame="parent")
    expected_mat = camera_parent_world_transform @ pose2mat(camera_local_pose)
    expected_mat[:3, :3] = expected_mat[:3, :3] / th.norm(expected_mat[:3, :3], dim=0, keepdim=True)
    expected_new_camera_world_pos, _ = mat2pose(expected_mat)
    new_camera_world_pose = vision_sensor.get_position_orientation()
    assert th.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)

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
    og.sim.stop()

    # Iterate over all robots and test their motion
    for robot_name, robot_cls in REGISTERED_ROBOTS.items():
        if robot_name in ["FrankaMounted", "Stretch"]:
            # TODO: skipping FrankaMounted and Stretch for now because CI doesn't have the required assets
            continue

        if robot_name in ["Husky", "BehaviorRobot"]:
            # Husky base motion is a little messed up because of the 4-wheel drive; skipping for now
            # BehaviorRobot does not work with the primitive actions at the moment
            continue

        robot = robot_cls(
            name=robot_name,
            obs_modalities=[],
        )
        env.scene.add_object(robot)

        # At least one step is always needed while sim is playing for any imported object to be fully initialized
        og.sim.play()

        # Reset robot and make sure it's not moving
        robot.reset()
        robot.keep_still()

        og.sim.step()

        # Set viewer in front facing robot
        og.sim.viewer_camera.set_position_orientation(
            position=[2.69918369, -3.63686664, 4.57894564],
            orientation=[0.39592411, 0.1348514, 0.29286304, 0.85982],
        )

        # If this is a manipulation robot, we want to test moving the arm
        if isinstance(robot, ManipulationRobot):
            # load IK controller
            controller_config = {
                f"arm_{robot.default_arm}": {"name": "InverseKinematicsController", "mode": "pose_absolute_ori"}
            }
            robot.reload_controllers(controller_config=controller_config)
            env.scene.update_initial_file()

            action_primitives = StarterSemanticActionPrimitives(env, robot, skip_curobo_initilization=True)

            eef_pos = env.robots[0].get_eef_position()
            eef_orn = env.robots[0].get_eef_orientation()
            if isinstance(robot, Stretch):  # Stretch arm faces the y-axis
                target_eef_pos = th.tensor([eef_pos[0], eef_pos[1] - 0.1, eef_pos[2]], dtype=th.float32)
            else:
                target_eef_pos = th.tensor([eef_pos[0] + 0.1, eef_pos[1], eef_pos[2]], dtype=th.float32)
            target_eef_orn = eef_orn
            for action in action_primitives._move_hand_direct_ik((target_eef_pos, target_eef_orn)):
                env.step(action)
            assert th.norm(robot.get_eef_position() - target_eef_pos) < 0.05

        # If this is a locomotion robot, we want to test driving
        if isinstance(robot, LocomotionRobot):
            action_primitives = StarterSemanticActionPrimitives(env, robot, skip_curobo_initilization=True)
            goal_location = th.tensor([0, 1, 0], dtype=th.float32)
            for action in action_primitives._navigate_to_pose_direct(goal_location):
                env.step(action)
            assert th.norm(robot.get_position()[:2] - goal_location[:2]) < 0.1
            assert robot.get_rpy()[2] - goal_location[2] < 0.1

        # Stop the simulator and remove the robot
        og.sim.stop()
        env.scene.remove_object(obj=robot)

    og.clear()


def test_grasping_mode():
    if og.sim is None:
        # Set global flags
        gm.ENABLE_FLATCACHE = True
    else:
        # Make sure sim is stopped
        og.sim.stop()

    scene_cfg = dict(type="Scene")
    objects_cfg = []
    objects_cfg.append(
        dict(
            type="DatasetObject",
            name="table",
            category="breakfast_table",
            model="lcsizg",
            bounding_box=[0.5, 0.5, 0.8],
            fixed_base=True,
            position=[0.7, -0.1, 0.6],
        )
    )
    objects_cfg.append(
        dict(
            type="PrimitiveObject",
            name="box",
            primitive_type="Cube",
            rgba=[1.0, 0, 0, 1.0],
            size=0.05,
            position=[0.53, 0.0, 0.87],
        )
    )
    cfg = dict(scene=scene_cfg, objects=objects_cfg)

    env = og.Environment(configs=cfg)
    og.sim.viewer_camera.set_position_orientation(
        position=[1.0170, 0.5663, 1.0554],
        orientation=[0.1734, 0.5006, 0.8015, 0.2776],
    )
    og.sim.stop()

    grasping_modes = dict(
        sticky="Sticky Mitten - Objects are magnetized when they touch the fingers and a CLOSE command is given",
        assisted="Assisted Grasping - Objects are magnetized when they touch the fingers, are within the hand, and a CLOSE command is given",
        physical="Physical Grasping - No additional grasping assistance applied",
    )

    def object_is_in_hand(robot, obj, grasping_mode):
        if grasping_mode in ["sticky", "assisted"]:
            return robot._ag_obj_in_hand[robot.default_arm] == obj
        elif grasping_mode == "physical":
            prim_paths = robot._find_gripper_raycast_collisions()
            return obj in {obj for (obj, _) in prim_paths_to_rigid_prims(prim_paths, obj.scene)}
        else:
            raise ValueError(f"Unknown grasping mode: {grasping_mode}")

    for grasping_mode in grasping_modes:
        robot = Fetch(
            name="Fetch",
            obs_modalities=[],
            controller_config={"arm_0": {"name": "InverseKinematicsController", "mode": "pose_absolute_ori"}},
            grasping_mode=grasping_mode,
        )
        env.scene.add_object(robot)

        # At least one step is always needed while sim is playing for any imported object to be fully initialized
        og.sim.play()

        env.scene.reset(hard=False)

        # Reset robot and make sure it's not moving
        robot.reset()
        robot.keep_still()

        # Let the box settle
        for _ in range(10):
            og.sim.step()

        action_primitives = StarterSemanticActionPrimitives(env=env, robot=robot, skip_curobo_initilization=True)

        box_object = env.scene.object_registry("name", "box")
        target_eef_pos = box_object.get_position_orientation()[0]
        target_eef_orn = robot.get_eef_orientation()

        # Move eef to the box
        for action in action_primitives._move_hand_direct_ik((target_eef_pos, target_eef_orn), pos_thresh=0.01):
            env.step(action)

        gripper_controller = robot.controllers["gripper_0"]

        # Grasp the box
        gripper_controller.update_goal(cb.array([-1]), robot.get_control_dict())
        for _ in range(10):
            og.sim.step()

        curr_time = time.time()
        time_required = m.robots.manipulation_robot.GRASP_WINDOW
        while time.time() - curr_time < time_required:
            og.sim.step()

        assert object_is_in_hand(
            robot, box_object, grasping_mode
        ), f"Grasping mode {grasping_mode} failed to grasp the object"

        # Move eef
        eef_offset = th.tensor([0.0, 0.2, 0.2])
        for action in action_primitives._move_hand_direct_ik((target_eef_pos + eef_offset, target_eef_orn)):
            env.step(action)

        assert object_is_in_hand(
            robot, box_object, grasping_mode
        ), f"Grasping mode {grasping_mode} failed to keep the object in hand"

        # Release the box
        gripper_controller.update_goal(cb.array([1]), robot.get_control_dict())
        for _ in range(20):
            og.sim.step()

        assert not object_is_in_hand(
            robot, box_object, grasping_mode
        ), f"Grasping mode {grasping_mode} failed to release the object"

        # Stop the simulator and remove the robot
        og.sim.stop()
        env.scene.remove_object(obj=robot)

    og.clear()
