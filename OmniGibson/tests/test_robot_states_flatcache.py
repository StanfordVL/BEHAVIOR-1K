import torch as th
import pytest
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.controllers import ControllerView
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS, Robot
from omnigibson.sensors import VisionSensor
from omnigibson.utils.transform_utils import mat2pose, pose2mat, quaternions_close, relative_pose_transform
from omnigibson.utils.usd_utils import PoseAPI
from omnigibson.object_states.robot_related_states import ObjectsInFOVOfRobot
from omnigibson.utils.constants import semantic_class_name_to_id


def setup_environment():
    """
    Sets up the environment.
    """
    if og.sim is None:
        # Set global flags
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Clear the simulator
        og.clear()

    # Define the environment configuration
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "model": "fetch",
                "obs_modalities": ["rgb", "seg_semantic", "seg_instance"],
                "position": [150, 150, 100],
                "orientation": [0, 0, 0, 1],
            }
        ],
    }

    env = og.Environment(configs=config)
    return env


def test_camera_pose():
    env = setup_environment()
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
    camera_parent_prim.scale = th.tensor([2.0, 2.0, 2.0])
    camera_parent_world_transform = PoseAPI.get_world_pose_with_scale(camera_parent_path)
    camera_local_pose = vision_sensor.get_position_orientation(frame="parent")
    expected_mat = camera_parent_world_transform @ pose2mat(camera_local_pose)
    expected_mat[:3, :3] = expected_mat[:3, :3] / th.norm(expected_mat[:3, :3], dim=0, keepdim=True)
    expected_new_camera_world_pos, _ = mat2pose(expected_mat)
    new_camera_world_pose = vision_sensor.get_position_orientation()
    assert th.allclose(new_camera_world_pose[0], expected_new_camera_world_pos, atol=1e-3)

    og.clear()


@pytest.mark.parametrize("robot_name", REGISTERED_ROBOTS)
def test_robot_load_drive(robot_name):
    if robot_name == "stretch" or robot_name == "locobot":
        pytest.skip(
            f"Skipping {robot_name} for now due to issues with turning"
        )  # TODO: https://github.com/StanfordVL/BEHAVIOR-1K/issues/2018

    if robot_name == "husky":
        pytest.skip("Husky base motion is a little messed up because of the 4-wheel drive; skipping for now")

    try:
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

        robot = Robot(
            name=robot_name,
            model=robot_name.lower(),
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
        if robot.is_manipulation:
            # load IK controller
            controller_config = {
                f"arm_{robot.default_arm}": {"name": "InverseKinematicsController", "mode": "pose_absolute_ori"}
            }
            robot.reload_controllers(controller_config=controller_config)
            env.scene.update_initial_file()

            action_primitives = StarterSemanticActionPrimitives(env, robot, skip_curobo_initilization=True)

            eef_pos = env.robots[0].get_eef_position()
            eef_orn = env.robots[0].get_eef_orientation()
            if robot.model == "stretch":  # Stretch arm faces the y-axis
                target_eef_pos = th.tensor([eef_pos[0], eef_pos[1] - 0.1, eef_pos[2]], dtype=th.float32)
            else:
                target_eef_pos = th.tensor([eef_pos[0] + 0.1, eef_pos[1], eef_pos[2]], dtype=th.float32)
            target_eef_orn = eef_orn
            for action in action_primitives._move_hand_direct_ik((target_eef_pos, target_eef_orn)):
                env.step(action)
            assert th.norm(robot.get_eef_position() - target_eef_pos) < 0.05

        # If this is a locomotion robot, we want to test driving
        if robot.is_locomotion:
            action_primitives = StarterSemanticActionPrimitives(env, robot, skip_curobo_initilization=True)
            goal_location = th.tensor([0, 1, 0], dtype=th.float32)
            for action in action_primitives._navigate_to_pose_direct(goal_location):
                env.step(action)
            assert th.norm(robot.get_position()[:2] - goal_location[:2]) < 0.1
            yaw_diff = robot.get_rpy()[2] - goal_location[2]
            wrapped_yaw_diff = th.atan2(th.sin(yaw_diff), th.cos(yaw_diff))
            assert th.abs(wrapped_yaw_diff) < 0.1

        # Stop the simulator and remove the robot
        og.sim.stop()
        env.scene.remove_object(obj=robot)
    finally:
        if og.sim is not None:
            og.clear()


def test_grasping_mode():
    if og.sim is not None:
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
            return len(prim_paths.intersection(obj.link_prim_paths)) > 0
        else:
            raise ValueError(f"Unknown grasping mode: {grasping_mode}")

    for grasping_mode in grasping_modes:
        robot = Robot(
            name="Fetch",
            model="fetch",
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

        group_key, controller_idx = robot.controllers["gripper_0"]

        # Grasp the box
        ControllerView.update_goal(group_key, controller_idx, th.tensor([-1.0]))
        for _ in range(30):
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
        ControllerView.update_goal(group_key, controller_idx, th.tensor([1.0]))
        for _ in range(20):
            og.sim.step()

        assert not object_is_in_hand(
            robot, box_object, grasping_mode
        ), f"Grasping mode {grasping_mode} failed to release the object"

        # Stop the simulator and remove the robot
        og.sim.stop()
        env.scene.remove_object(obj=robot)

    og.clear()


def test_camera_semantic_segmentation():
    env = setup_environment()
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
    env = setup_environment()
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


def test_holonomic_robot_tuck_untuck_base_joint_invariance():
    """
    Test that calling tuck() and untuck() on a holonomic base robot
    should not move the robot's body.
    """
    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = False
    else:
        og.sim.stop()

    # Use R1 which has holonomic base and mobile_manipulation (tucked/untucked)
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "model": "r1",
                "obs_modalities": [],
                "position": [10.0, 20.0, 0.5],
                "orientation": [0, 0, 0.3827, 0.9239],  # 45 degree rotation around z
            }
        ],
    }

    env = og.Environment(configs=config)
    robot = env.robots[0]
    env.reset()
    og.sim.step()

    assert robot.is_holonomic_base, "R1 should have holonomic base"
    assert robot.is_mobile_manipulation, "R1 should have mobile manipulation capability"

    # Record initial base joint positions (the 6 DoF controlling robot pose)
    initial_base_joint_pos = robot.get_joint_positions()[robot.base_idx].clone()
    initial_pos, initial_ori = robot.get_position_orientation()

    # Test tuck() - should preserve base joint positions and pose
    robot.tuck()
    base_joint_pos_after_tuck = robot.get_joint_positions()[robot.base_idx]
    pos_after_tuck, ori_after_tuck = robot.get_position_orientation()
    assert th.allclose(
        initial_base_joint_pos, base_joint_pos_after_tuck, atol=1e-6
    ), f"tuck() changed base joint positions! Initial: {initial_base_joint_pos}, After tuck: {base_joint_pos_after_tuck}"
    assert th.allclose(initial_pos, pos_after_tuck, atol=1e-6), "tuck() changed robot position"
    assert th.allclose(initial_ori, ori_after_tuck, atol=1e-6), "tuck() changed robot orientation"

    # Test untuck() - should preserve base joint positions and pose
    robot.untuck()
    base_joint_pos_after_untuck = robot.get_joint_positions()[robot.base_idx]
    pos_after_untuck, ori_after_untuck = robot.get_position_orientation()
    assert th.allclose(
        initial_base_joint_pos, base_joint_pos_after_untuck, atol=1e-6
    ), f"untuck() changed base joint positions! Initial: {initial_base_joint_pos}, After untuck: {base_joint_pos_after_untuck}"
    assert th.allclose(initial_pos, pos_after_untuck, atol=1e-6), "untuck() changed robot position"
    assert th.allclose(initial_ori, ori_after_untuck, atol=1e-6), "untuck() changed robot orientation"

    og.clear()
