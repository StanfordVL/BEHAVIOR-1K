import torch as th
from test_robot_states_flatcache import camera_pose_test, setup_environment

import omnigibson as og
from omnigibson.macros import gm
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


def test_holonomic_robot_tuck_untuck_base_joint_invariance():
    """
    Test that calling tuck() and untuck() on a holonomic base robot
    preserves the base joint positions (the 6 DoF that control robot pose).
    """
    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
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
                "type": "R1",
                "model": "R1",
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
    
    assert robot.holonomic_base, "R1 should have holonomic base"
    assert robot.mobile_manipulation, "R1 should have mobile_manipulation capability"

    # Record initial base joint positions (the 6 DoF controlling robot pose)
    initial_base_joint_pos = robot.get_joint_positions()[robot.base_idx].clone()

    # Test tuck() - should preserve base joint positions
    robot.tuck()
    base_joint_pos_after_tuck = robot.get_joint_positions()[robot.base_idx]
    assert th.allclose(initial_base_joint_pos, base_joint_pos_after_tuck, atol=1e-6), (
        f"tuck() changed base joint positions! Initial: {initial_base_joint_pos}, After tuck: {base_joint_pos_after_tuck}"
    )

    # Test untuck() - should preserve base joint positions
    robot.untuck()
    base_joint_pos_after_untuck = robot.get_joint_positions()[robot.base_idx]
    assert th.allclose(initial_base_joint_pos, base_joint_pos_after_untuck, atol=1e-6), (
        f"untuck() changed base joint positions! Initial: {initial_base_joint_pos}, After untuck: {base_joint_pos_after_untuck}"
    )

    og.clear()

