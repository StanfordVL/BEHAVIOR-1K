import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition
from omnigibson.utils.transform_utils import quat_multiply


def setup_multi_environment(num_of_envs, robot="Fetch", additional_objects_cfg=[]):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls"],
        },
        "robots": [
            {
                "type": robot,
                "obs_modalities": [],
            }
        ],
    }

    cfg["objects"] = additional_objects_cfg

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


def test_multi_scene_dump_load_states():
    vec_env = setup_multi_environment(3)
    robot_0 = vec_env.envs[0].scene.robots[0]
    robot_1 = vec_env.envs[1].scene.robots[0]
    robot_2 = vec_env.envs[2].scene.robots[0]

    robot_0_pos = robot_0.get_position_orientation()[0]
    robot_1_pos = robot_1.get_position_orientation()[0]
    robot_2_pos = robot_2.get_position_orientation()[0]

    dist_0_1 = robot_1_pos - robot_0_pos
    dist_1_2 = robot_2_pos - robot_1_pos

    assert th.allclose(dist_0_1, dist_1_2, atol=1e-3)

    # Set different poses for the cube in each environment
    pose_1 = (th.tensor([1, 1, 1], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32))
    pose_2 = (th.tensor([0, 2, 1], dtype=th.float32), th.tensor([0, 0, 0.7071, 0.7071], dtype=th.float32))
    pose_3 = (th.tensor([-1, -1, 0.5], dtype=th.float32), th.tensor([0.5, 0.5, 0.5, 0.5], dtype=th.float32))

    robot_0.set_position_orientation(*pose_1, frame="scene")
    robot_1.set_position_orientation(*pose_2, frame="scene")
    robot_2.set_position_orientation(*pose_3, frame="scene")

    # Run simulation for a bit
    for _ in range(10):
        og.sim.step()

    initial_robot_pos_scene_1 = robot_1.get_position_orientation(frame="scene")
    initial_robot_pos_scene_2 = robot_2.get_position_orientation(frame="scene")
    initial_robot_pos_scene_0 = robot_0.get_position_orientation(frame="scene")

    # Save states
    robot_0_state = vec_env.envs[0].scene._dump_state()
    robot_1_state = vec_env.envs[1].scene._dump_state()
    robot_2_state = vec_env.envs[2].scene._dump_state()

    # reset the object_positions
    vec_env.reset()

    # Load the states in a different order
    vec_env.envs[1].scene._load_state(robot_1_state)
    vec_env.envs[2].scene._load_state(robot_2_state)
    vec_env.envs[0].scene._load_state(robot_0_state)

    post_robot_pos_scene_1 = vec_env.envs[1].scene.robots[0].get_position_orientation(frame="scene")
    post_robot_pos_scene_2 = vec_env.envs[2].scene.robots[0].get_position_orientation(frame="scene")
    post_robot_pos_scene_0 = vec_env.envs[0].scene.robots[0].get_position_orientation(frame="scene")

    # Check that the poses are the same
    assert th.allclose(initial_robot_pos_scene_0[0], post_robot_pos_scene_0[0], atol=1e-3)
    assert th.allclose(initial_robot_pos_scene_1[0], post_robot_pos_scene_1[0], atol=1e-3)
    assert th.allclose(initial_robot_pos_scene_2[0], post_robot_pos_scene_2[0], atol=1e-3)

    assert th.allclose(initial_robot_pos_scene_0[1], post_robot_pos_scene_0[1], atol=1e-3)
    assert th.allclose(initial_robot_pos_scene_1[1], post_robot_pos_scene_1[1], atol=1e-3)
    assert th.allclose(initial_robot_pos_scene_2[1], post_robot_pos_scene_2[1], atol=1e-3)

    og.clear()


def test_multi_scene_get_local_position():
    vec_env = setup_multi_environment(3)

    robot_1_pos_local = vec_env.envs[1].scene.robots[0].get_position_orientation(frame="scene")[0]
    robot_1_pos_global = vec_env.envs[1].scene.robots[0].get_position_orientation()[0]

    pos_scene = vec_env.envs[1].scene.get_position_orientation()[0]

    assert th.allclose(robot_1_pos_global, pos_scene + robot_1_pos_local, atol=1e-3)
    og.clear()


def test_multi_scene_set_local_position():
    vec_env = setup_multi_environment(3)

    # Get the robot from the second environment
    robot = vec_env.envs[1].scene.robots[0]

    # Get the initial global position of the robot
    initial_global_pos = robot.get_position_orientation()[0]

    # Define a new global position
    new_global_pos = initial_global_pos + th.tensor([1.0, 0.5, 0.0], dtype=th.float32)

    # Set the new global position
    robot.set_position_orientation(position=new_global_pos)

    # Get the updated global position
    updated_global_pos = robot.get_position_orientation()[0]

    # Get the scene's global position
    scene_pos = vec_env.envs[1].scene.get_position_orientation()[0]

    # Get the updated local position
    updated_local_pos = robot.get_position_orientation(frame="scene")[0]

    # Calculate expected local position
    expected_local_pos = new_global_pos - scene_pos

    # Assert that the global position has been updated correctly
    assert th.allclose(
        updated_global_pos, new_global_pos, atol=1e-3
    ), f"Updated global position {updated_global_pos} does not match expected {new_global_pos}"

    # Assert that the local position has been updated correctly
    assert th.allclose(
        updated_local_pos, expected_local_pos, atol=1e-3
    ), f"Updated local position {updated_local_pos} does not match expected {expected_local_pos}"

    # Assert that the change in global position is correct
    global_pos_change = updated_global_pos - initial_global_pos
    expected_change = th.tensor([1.0, 0.5, 0.0], dtype=th.float32)
    assert th.allclose(
        global_pos_change, expected_change, atol=1e-3
    ), f"Global position change {global_pos_change} does not match expected change {expected_change}"

    og.clear()


def test_multi_scene_scene_prim():
    vec_env = setup_multi_environment(1)
    original_robot_pos = vec_env.envs[0].scene.robots[0].get_position_orientation()[0]
    scene_prim_displacement = th.tensor([10.0, 0.0, 0.0], dtype=th.float32)
    original_scene_prim_pos = vec_env.envs[0].scene._scene_prim.get_position_orientation()[0]
    vec_env.envs[0].scene.set_position_orientation(position=original_scene_prim_pos + scene_prim_displacement)
    new_scene_prim_pos = vec_env.envs[0].scene._scene_prim.get_position_orientation()[0]
    new_robot_pos = vec_env.envs[0].scene.robots[0].get_position_orientation()[0]
    assert th.allclose(new_scene_prim_pos - original_scene_prim_pos, scene_prim_displacement, atol=1e-3)
    assert th.allclose(new_robot_pos - original_robot_pos, scene_prim_displacement, atol=1e-3)

    og.clear()


def test_multi_scene_particle_source():
    sink_cfg = dict(
        type="DatasetObject",
        name="sink",
        category="furniture_sink",
        model="czyfhq",
        abilities={
            "toggleable": {},
            "particleSource": {
                "conditions": {
                    "water": [
                        (ParticleModifyCondition.TOGGLEDON, True)
                    ],  # Must be toggled on for water source to be active
                },
                "initial_speed": 0.0,  # Water merely falls out of the spout
            },
            "particleSink": {
                "conditions": {
                    "water": [],  # No conditions, always sinking nearby particles
                },
            },
        },
        position=[0.0, -1.5, 0.0],
    )

    vec_env = setup_multi_environment(3, additional_objects_cfg=[sink_cfg])

    for env in vec_env.envs:
        sink = env.scene.object_registry("name", "sink")
        assert sink.states[object_states.ToggledOn].set_value(True)

    for _ in range(50):
        og.sim.step()

    og.clear()


def test_multi_scene_position_orientation_relative_to_scene():
    vec_env = setup_multi_environment(3)

    # Get the robot from the second environment
    robot = vec_env.envs[1].scene.robots[0]

    # Define a new position and orientation relative to the scene
    new_relative_pos = th.tensor([1.0, 2.0, 0.5])
    new_relative_ori = th.tensor([0, 0, 0.7071, 0.7071])  # 90 degrees rotation around z-axis

    # Set the new position and orientation relative to the scene
    robot.set_position_orientation(position=new_relative_pos, orientation=new_relative_ori, frame="scene")

    # Get the updated position and orientation relative to the scene
    updated_relative_pos, updated_relative_ori = robot.get_position_orientation(frame="scene")

    # Assert that the relative position has been updated correctly
    assert th.allclose(
        updated_relative_pos, new_relative_pos, atol=1e-3
    ), f"Updated relative position {updated_relative_pos} does not match expected {new_relative_pos}"

    # Assert that the relative orientation has been updated correctly
    assert th.allclose(
        updated_relative_ori, new_relative_ori, atol=1e-3
    ), f"Updated relative orientation {updated_relative_ori} does not match expected {new_relative_ori}"

    # Get the scene's global position and orientation
    scene_pos, scene_ori = vec_env.envs[1].scene.get_position_orientation()

    # Get the robot's global position and orientation
    global_pos, global_ori = robot.get_position_orientation()

    # Calculate expected global position
    expected_global_pos = scene_pos + updated_relative_pos

    # Assert that the global position is correct
    assert th.allclose(
        global_pos, expected_global_pos, atol=1e-3
    ), f"Global position {global_pos} does not match expected {expected_global_pos}"

    # Calculate expected global orientation
    expected_global_ori = quat_multiply(scene_ori, new_relative_ori)

    # Assert that the global orientation is correct
    assert th.allclose(
        global_ori, expected_global_ori, atol=1e-3
    ), f"Global orientation {global_ori} does not match expected {expected_global_ori}"

    og.clear()


def test_tiago_getter():
    vec_env = setup_multi_environment(2, robot="Tiago")
    robot1 = vec_env.envs[0].scene.robots[0]

    robot1_world_position, robot1_world_orientation = robot1.get_position_orientation()
    robot1_scene_position, robot1_scene_orientation = robot1.get_position_orientation(frame="scene")

    # Test the get_position_orientation method for 3 different frames
    # since the robot is at the origin, the position and orientation should be the same
    assert th.allclose(robot1_world_position, robot1_scene_position, atol=1e-3)
    assert th.allclose(robot1_world_orientation, robot1_scene_orientation, atol=1e-3)

    # test if the scene position is non-zero, the getter with scene and world frame should return different values
    robot2 = vec_env.envs[1].scene.robots[0]
    scene_position, scene_orientation = vec_env.envs[1].scene.get_position_orientation()

    robot2_world_position, robot2_world_orientation = robot2.get_position_orientation()
    robot2_scene_position, robot2_scene_orientation = robot2.get_position_orientation(frame="scene")

    combined_position, combined_orientation = T.pose_transform(
        scene_position, scene_orientation, robot2_scene_position, robot2_scene_orientation
    )
    assert th.allclose(robot2_world_position, combined_position, atol=1e-3)
    assert th.allclose(robot2_world_orientation, combined_orientation, atol=1e-3)

    # Clean up
    og.clear()


def test_tiago_setter():
    vec_env = setup_multi_environment(2, robot="Tiago")

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]

    # Test setting position and orientation in world frame
    new_world_pos = th.tensor([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
    robot.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([-1.0, -2.0, 0.1])
    new_scene_ori = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    # Verify that world frame position/orientation has changed after setting in scene frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not th.allclose(got_world_ori, new_world_ori, atol=1e-3)

    # Clean up
    og.clear()

    # assert that when the simulator is stopped, the behavior for getter/setter is not affected
    vec_env = setup_multi_environment(2)
    og.sim.stop()

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]

    # Test setting position and orientation in world frame
    new_world_pos = th.tensor([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
    robot.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([-1.0, -2.0, 0.1])
    new_scene_ori = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

    # Verify that world frame position/orientation has changed after setting in scene frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not th.allclose(got_world_ori, new_world_ori, atol=1e-3)

    og.clear()


@pytest.mark.skip("Behavior getter is currently broken")
def test_behavior_getter():
    vec_env = setup_multi_environment(2, robot="BehaviorRobot")
    robot1 = vec_env.envs[0].scene.robots[0]

    robot1_world_position, robot1_world_orientation = robot1.get_position_orientation()
    robot1_scene_position, robot1_scene_orientation = robot1.get_position_orientation(frame="scene")

    # Test the get_position_orientation method for 3 different frames
    # since the robot is at the origin, the position and orientation should be the same
    assert th.allclose(robot1_world_position, robot1_scene_position, atol=1e-3)
    assert th.allclose(robot1_world_position, robot1_scene_position, atol=1e-3)
    assert th.allclose(robot1_world_orientation, robot1_scene_orientation, atol=1e-3)
    assert th.allclose(robot1_world_orientation, robot1_scene_orientation, atol=1e-3)

    # test if the scene position is non-zero, the getter with scene and world frame should return different values
    robot2 = vec_env.envs[1].scene.robots[0]
    scene_position, scene_orientation = vec_env.envs[1].scene.get_position_orientation()
    robot2_world_position, robot2_world_orientation = robot2.get_position_orientation()
    robot2_scene_position, robot2_scene_orientation = robot2.get_position_orientation(frame="scene")

    combined_position, combined_orientation = T.pose_transform(
        scene_position, scene_orientation, robot2_scene_position, robot2_scene_orientation
    )
    assert th.allclose(robot2_world_position, combined_position, atol=1e-3)
    assert th.allclose(robot2_world_orientation, combined_orientation, atol=1e-3)

    # Clean up
    og.clear()


@pytest.mark.skip("Behavior setter is currently broken")
def test_behavior_setter():
    vec_env = setup_multi_environment(2, robot="BehaviorRobot")

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]

    # Test setting position and orientation in world frame
    new_world_pos = th.tensor([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))

    robot.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([-1.0, -2.0, 0.1])
    new_scene_ori = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

    # Verify that world frame position/orientation has changed after setting in scene frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not th.allclose(got_world_ori, new_world_ori, atol=1e-3)

    # Clean up
    og.clear()

    # assert that when the simulator is stopped, the behavior for getter/setter is not affected
    vec_env = setup_multi_environment(2)
    og.sim.stop()

    # use a robot with non-zero scene position
    robot = vec_env.envs[1].scene.robots[0]

    # Test setting position and orientation in world frame
    new_world_pos = th.tensor([1.0, 2.0, 0.5])
    new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
    robot.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([0.5, 1.0, 0.25])
    new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

    # Test setting position and orientation in scene frame
    new_scene_pos = th.tensor([-1.0, -2.0, 0.1])
    new_scene_ori = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
    robot.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

    got_scene_pos, got_scene_ori = robot.get_position_orientation(frame="scene")
    assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
    assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

    # Verify that world frame position/orientation has changed after setting in scene frame
    got_world_pos, got_world_ori = robot.get_position_orientation()
    assert not th.allclose(got_world_pos, new_world_pos, atol=1e-3)
    assert not th.allclose(got_world_ori, new_world_ori, atol=1e-3)
