import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition
from omnigibson.utils.transform_utils import quat_multiply


def setup_multi_environment(num_of_envs, additional_objects_cfg=[]):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls"],
        },
        "robots": [
            {
                "type": "Fetch",
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


def test_multi_scene_dump_and_load():
    vec_env = setup_multi_environment(3)
    robot_displacement = [1.0, 0.0, 0.0]
    scene_three_robot = vec_env.envs[2].scene.robots[0]
    robot_new_pos = scene_three_robot.get_position_orientation()[0] + robot_displacement
    scene_three_robot.set_position_orientation(position=robot_new_pos)
    scene_three_state = vec_env.envs[2].scene._dump_state()
    og.clear()

    vec_env = setup_multi_environment(3)
    initial_robot_pos_scene_one = vec_env.envs[0].scene.robots[0].get_position_orientation()[0]
    vec_env.envs[0].scene._load_state(scene_three_state)
    new_robot_pos_scene_one = vec_env.envs[0].scene.robots[0].get_position_orientation()[0]
    assert np.allclose(new_robot_pos_scene_one - initial_robot_pos_scene_one, robot_displacement, atol=1e-3)

    og.clear()


def test_multi_scene_displacement():
    vec_env = setup_multi_environment(3)
    robot_0_pos = vec_env.envs[0].scene.robots[0].get_position_orientation()[0]
    robot_1_pos = vec_env.envs[1].scene.robots[0].get_position_orientation()[0]
    robot_2_pos = vec_env.envs[2].scene.robots[0].get_position_orientation()[0]

    dist_0_1 = robot_1_pos - robot_0_pos
    dist_1_2 = robot_2_pos - robot_1_pos
    assert np.allclose(dist_0_1, dist_1_2, atol=1e-3)
    og.clear()


def test_multi_scene_get_local_position():
    vec_env = setup_multi_environment(3)

    robot_1_pos_local = vec_env.envs[1].scene.robots[0].get_position_orientation(frame="parent")[0]
    robot_1_pos_global = vec_env.envs[1].scene.robots[0].get_position_orientation(frame="world")[0]

    scene_prim = vec_env.envs[1].scene.prim
    pos_scene = scene_prim.get_position_orientation(frame="world")[0]

    assert np.allclose(robot_1_pos_global, pos_scene + robot_1_pos_local, atol=1e-3)
    og.clear()


def test_multi_scene_set_local_position():

    vec_env = setup_multi_environment(3)

    # Get the robot from the second environment
    robot = vec_env.envs[1].scene.robots[0]

    # Get the initial global position of the robot
    initial_global_pos = robot.get_position_orientation(frame="world")[0]

    # Define a new global position
    new_global_pos = initial_global_pos + np.array([1.0, 0.5, 0.0])

    # Set the new global position
    robot.set_position_orientation(position=new_global_pos)

    # Get the updated global position
    updated_global_pos = robot.get_position_orientation(frame="world")[0]

    # Get the scene's global position
    scene_pos = vec_env.envs[1].scene.prim.get_position_orientation(frame="world")[0]

    # Get the updated local position
    updated_local_pos = robot.get_position_orientation(frame="parent")[0]

    # Calculate expected local position
    expected_local_pos = new_global_pos - scene_pos

    # Assert that the global position has been updated correctly
    assert np.allclose(
        updated_global_pos, new_global_pos, atol=1e-3
    ), f"Updated global position {updated_global_pos} does not match expected {new_global_pos}"

    # Assert that the local position has been updated correctly
    assert np.allclose(
        updated_local_pos, expected_local_pos, atol=1e-3
    ), f"Updated local position {updated_local_pos} does not match expected {expected_local_pos}"

    # Assert that the change in global position is correct
    global_pos_change = updated_global_pos - initial_global_pos
    expected_change = np.array([1.0, 0.5, 0.0])
    assert np.allclose(
        global_pos_change, expected_change, atol=1e-3
    ), f"Global position change {global_pos_change} does not match expected change {expected_change}"

    og.clear()


def test_multi_scene_scene_prim():
    vec_env = setup_multi_environment(1)
    original_robot_pos = vec_env.envs[0].scene.robots[0].get_position_orientation()[0]
    scene_state = vec_env.envs[0].scene._dump_state()
    scene_prim_displacement = [10.0, 0.0, 0.0]
    original_scene_prim_pos = vec_env.envs[0].scene._scene_prim.get_position_orientation()[0]
    vec_env.envs[0].scene._scene_prim.set_position_orientation(
        position=original_scene_prim_pos + scene_prim_displacement
    )
    vec_env.envs[0].scene._load_state(scene_state)
    new_scene_prim_pos = vec_env.envs[0].scene._scene_prim.get_position_orientation()[0]
    new_robot_pos = vec_env.envs[0].scene.robots[0].get_position_orientation()[0]
    assert np.allclose(new_scene_prim_pos - original_scene_prim_pos, scene_prim_displacement, atol=1e-3)
    assert np.allclose(new_robot_pos - original_robot_pos, scene_prim_displacement, atol=1e-3)


def test_multi_scene_particle_source():
    sink_cfg = dict(
        type="DatasetObject",
        name="sink",
        category="sink",
        model="egwapq",
        bounding_box=[2.427, 0.625, 1.2],
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
        position=[0.0, -1.5, 0.42],
    )

    vec_env = setup_multi_environment(3, additional_objects_cfg=[sink_cfg])

    for env in vec_env.envs:
        sink = env.scene.object_registry("name", "sink")
        assert sink.states[object_states.ToggledOn].set_value(True)

    for _ in range(50):
        og.sim.step()

def test_multi_scene_position_orientation_relative_to_scene():
    vec_env = setup_multi_environment(3)

    # Get the robot from the second environment
    robot = vec_env.envs[1].scene.robots[0]

    # Get the scene prim
    scene_prim = vec_env.envs[1].scene.prim

    # Define a new position and orientation relative to the scene
    new_relative_pos = np.array([1.0, 2.0, 0.5])
    new_relative_ori = np.array([0, 0, 0.7071, 0.7071])  # 90 degrees rotation around z-axis

    # Set the new position and orientation relative to the scene
    robot.set_position_orientation(position=new_relative_pos, orientation=new_relative_ori, frame="scene")

    # Get the updated position and orientation relative to the scene
    updated_relative_pos, updated_relative_ori = robot.get_position_orientation(frame="scene")

    # Assert that the relative position has been updated correctly
    assert np.allclose(updated_relative_pos, new_relative_pos, atol=1e-3), \
        f"Updated relative position {updated_relative_pos} does not match expected {new_relative_pos}"

    # Assert that the relative orientation has been updated correctly
    assert np.allclose(updated_relative_ori, new_relative_ori, atol=1e-3), \
        f"Updated relative orientation {updated_relative_ori} does not match expected {new_relative_ori}"
    
    # Get the scene's global position and orientation
    scene_pos, scene_ori = scene_prim.get_position_orientation(frame="world")

    # Get the robot's global position and orientation
    global_pos, global_ori = robot.get_position_orientation(frame="world")

    # Calculate expected global position
    expected_global_pos = scene_pos + updated_relative_pos

    # Assert that the global position is correct
    assert np.allclose(global_pos, expected_global_pos, atol=1e-3), \
        f"Global position {global_pos} does not match expected {expected_global_pos}"

    # Calculate expected global orientation
    expected_global_ori = quat_multiply(scene_ori, new_relative_ori)

    # Assert that the global orientation is correct
    assert np.allclose(global_ori, expected_global_ori, atol=1e-3), \
        f"Global orientation {global_ori} does not match expected {expected_global_ori}"

    og.clear() 