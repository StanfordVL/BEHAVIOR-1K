import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.utils.constants import ParticleModifyCondition
from omnigibson.utils.transform_utils import quat_multiply

from utils import (
    MULTI_ENV_ROBOTS,
    multi_env_progress,
    multi_env_passed,
    setup_multi_environment,
)

# Test counter for progress tracking
_test_counter = {"current": 0, "total": 18}


def _progress(test_name):
    multi_env_progress(test_name, _test_counter)


def _passed(test_name):
    multi_env_passed(test_name)


# ===================================================================
#  Section 4 – Scene coordinate system & state tests
# ===================================================================


class TestSceneCoordinates:
    """Multi-scene position/orientation and state dump/load tests."""

    def test_multi_scene_dump_load_states(self):
        _progress("TestSceneCoordinates::test_multi_scene_dump_load_states")
        env = setup_multi_environment(3)
        robot_0 = env.scenes[0].robots[0]
        robot_1 = env.scenes[1].robots[0]
        robot_2 = env.scenes[2].robots[0]

        robot_0_pos = robot_0.get_position_orientation()[0]
        robot_1_pos = robot_1.get_position_orientation()[0]
        robot_2_pos = robot_2.get_position_orientation()[0]

        dist_0_1 = robot_1_pos - robot_0_pos
        dist_1_2 = robot_2_pos - robot_1_pos

        print(f"  dist_0_1={dist_0_1}, dist_1_2={dist_1_2}")
        # Check x/y spacing is even (z can drift slightly due to physics settling)
        assert th.allclose(dist_0_1[:2], dist_1_2[:2], atol=1e-3)

        # Set different poses for the robot in each environment
        pose_1 = (th.tensor([1, 1, 1], dtype=th.float32), th.tensor([0, 0, 0, 1], dtype=th.float32))
        pose_2 = (th.tensor([0, 2, 1], dtype=th.float32), th.tensor([0, 0, 0.7071, 0.7071], dtype=th.float32))
        pose_3 = (th.tensor([-1, -1, 0.5], dtype=th.float32), th.tensor([0.5, 0.5, 0.5, 0.5], dtype=th.float32))

        robot_0.set_position_orientation(*pose_1, frame="scene")
        robot_1.set_position_orientation(*pose_2, frame="scene")
        robot_2.set_position_orientation(*pose_3, frame="scene")

        print("  Running 10 sim steps...")
        for _ in range(10):
            og.sim.step()

        initial_robot_pos_scene_1 = robot_1.get_position_orientation(frame="scene")
        initial_robot_pos_scene_2 = robot_2.get_position_orientation(frame="scene")
        initial_robot_pos_scene_0 = robot_0.get_position_orientation(frame="scene")

        # Save states
        print("  Saving states...")
        robot_0_state = env.scenes[0]._dump_state()
        robot_1_state = env.scenes[1]._dump_state()
        robot_2_state = env.scenes[2]._dump_state()

        print("  Resetting env...")
        env.reset()

        # Load the states in a different order
        print("  Loading states in different order...")
        env.scenes[1]._load_state(robot_1_state)
        env.scenes[2]._load_state(robot_2_state)
        env.scenes[0]._load_state(robot_0_state)

        post_robot_pos_scene_1 = env.scenes[1].robots[0].get_position_orientation(frame="scene")
        post_robot_pos_scene_2 = env.scenes[2].robots[0].get_position_orientation(frame="scene")
        post_robot_pos_scene_0 = env.scenes[0].robots[0].get_position_orientation(frame="scene")

        print(f"  scene 0: initial={initial_robot_pos_scene_0[0]} -> post={post_robot_pos_scene_0[0]}")
        print(f"  scene 1: initial={initial_robot_pos_scene_1[0]} -> post={post_robot_pos_scene_1[0]}")
        print(f"  scene 2: initial={initial_robot_pos_scene_2[0]} -> post={post_robot_pos_scene_2[0]}")

        assert th.allclose(initial_robot_pos_scene_0[0], post_robot_pos_scene_0[0], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_1[0], post_robot_pos_scene_1[0], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_2[0], post_robot_pos_scene_2[0], atol=1e-3)

        assert th.allclose(initial_robot_pos_scene_0[1], post_robot_pos_scene_0[1], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_1[1], post_robot_pos_scene_1[1], atol=1e-3)
        assert th.allclose(initial_robot_pos_scene_2[1], post_robot_pos_scene_2[1], atol=1e-3)

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_dump_load_states")

    def test_multi_scene_get_local_position(self):
        _progress("TestSceneCoordinates::test_multi_scene_get_local_position")
        env = setup_multi_environment(3)

        robot_1_pos_local = env.scenes[1].robots[0].get_position_orientation(frame="scene")[0]
        robot_1_pos_global = env.scenes[1].robots[0].get_position_orientation()[0]

        pos_scene = env.scenes[1].get_position_orientation()[0]

        print(f"  local={robot_1_pos_local}, global={robot_1_pos_global}, scene_origin={pos_scene}")
        assert th.allclose(robot_1_pos_global, pos_scene + robot_1_pos_local, atol=1e-3)
        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_get_local_position")

    def test_multi_scene_set_local_position(self):
        _progress("TestSceneCoordinates::test_multi_scene_set_local_position")
        env = setup_multi_environment(3)

        robot = env.scenes[1].robots[0]
        initial_global_pos = robot.get_position_orientation()[0]
        new_global_pos = initial_global_pos + th.tensor([1.0, 0.5, 0.0], dtype=th.float32)

        robot.set_position_orientation(position=new_global_pos)

        updated_global_pos = robot.get_position_orientation()[0]
        scene_pos = env.scenes[1].get_position_orientation()[0]
        updated_local_pos = robot.get_position_orientation(frame="scene")[0]
        expected_local_pos = new_global_pos - scene_pos

        print(f"  updated_global={updated_global_pos}, expected={new_global_pos}")
        print(f"  updated_local={updated_local_pos}, expected_local={expected_local_pos}")

        assert th.allclose(
            updated_global_pos, new_global_pos, atol=1e-3
        ), f"Updated global position {updated_global_pos} does not match expected {new_global_pos}"
        assert th.allclose(
            updated_local_pos, expected_local_pos, atol=1e-3
        ), f"Updated local position {updated_local_pos} does not match expected {expected_local_pos}"

        global_pos_change = updated_global_pos - initial_global_pos
        expected_change = th.tensor([1.0, 0.5, 0.0], dtype=th.float32)
        assert th.allclose(
            global_pos_change, expected_change, atol=1e-3
        ), f"Global position change {global_pos_change} does not match expected change {expected_change}"

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_set_local_position")

    def test_multi_scene_scene_prim(self):
        _progress("TestSceneCoordinates::test_multi_scene_scene_prim")
        env = setup_multi_environment(1)
        original_robot_pos = env.scenes[0].robots[0].get_position_orientation()[0]
        scene_prim_displacement = th.tensor([10.0, 0.0, 0.0], dtype=th.float32)
        original_scene_prim_pos = env.scenes[0]._scene_prim.get_position_orientation()[0]
        env.scenes[0].set_position_orientation(position=original_scene_prim_pos + scene_prim_displacement)
        new_scene_prim_pos = env.scenes[0]._scene_prim.get_position_orientation()[0]
        new_robot_pos = env.scenes[0].robots[0].get_position_orientation()[0]
        print(f"  scene_prim moved: {original_scene_prim_pos} -> {new_scene_prim_pos}")
        print(f"  robot moved: {original_robot_pos} -> {new_robot_pos}")
        assert th.allclose(new_scene_prim_pos - original_scene_prim_pos, scene_prim_displacement, atol=1e-3)
        assert th.allclose(new_robot_pos - original_robot_pos, scene_prim_displacement, atol=1e-2)

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_scene_prim")

    def test_multi_scene_position_orientation_relative_to_scene(self):
        _progress("TestSceneCoordinates::test_multi_scene_position_orientation_relative_to_scene")
        env = setup_multi_environment(3)

        robot = env.scenes[1].robots[0]
        new_relative_pos = th.tensor([1.0, 2.0, 0.5])
        new_relative_ori = th.tensor([0, 0, 0.7071, 0.7071])

        robot.set_position_orientation(position=new_relative_pos, orientation=new_relative_ori, frame="scene")
        updated_relative_pos, updated_relative_ori = robot.get_position_orientation(frame="scene")

        print(f"  set relative pos={new_relative_pos}, got={updated_relative_pos}")
        assert th.allclose(
            updated_relative_pos, new_relative_pos, atol=1e-3
        ), f"Updated relative position {updated_relative_pos} does not match expected {new_relative_pos}"
        assert th.allclose(
            updated_relative_ori, new_relative_ori, atol=1e-3
        ), f"Updated relative orientation {updated_relative_ori} does not match expected {new_relative_ori}"

        scene_pos, scene_ori = env.scenes[1].get_position_orientation()
        global_pos, global_ori = robot.get_position_orientation()

        expected_global_pos = scene_pos + updated_relative_pos
        print(f"  global_pos={global_pos}, expected={expected_global_pos}")
        assert th.allclose(
            global_pos, expected_global_pos, atol=1e-3
        ), f"Global position {global_pos} does not match expected {expected_global_pos}"

        expected_global_ori = quat_multiply(scene_ori, new_relative_ori)
        assert th.allclose(
            global_ori, expected_global_ori, atol=1e-3
        ), f"Global orientation {global_ori} does not match expected {expected_global_ori}"

        og.clear()
        _passed("TestSceneCoordinates::test_multi_scene_position_orientation_relative_to_scene")


# ===================================================================
#  Section 5 – Robot-specific getter/setter tests (parametrized)
# ===================================================================


@pytest.mark.parametrize("robot", MULTI_ENV_ROBOTS)
class TestRobotGetterSetter:
    """Position/orientation getter and setter correctness across robots."""

    def test_getter(self, robot):
        test_id = f"TestRobotGetterSetter::test_getter[{robot}]"
        _progress(test_id)
        env = setup_multi_environment(2, robot=robot)
        robot1 = env.scenes[0].robots[0]

        robot1_world_position, robot1_world_orientation = robot1.get_position_orientation()
        robot1_scene_position, robot1_scene_orientation = robot1.get_position_orientation(frame="scene")

        print(f"  scene 0 robot: world_pos={robot1_world_position}, scene_pos={robot1_scene_position}")
        # Robot in scene 0 is at origin, so world == scene
        assert th.allclose(robot1_world_position, robot1_scene_position, atol=1e-3)
        assert th.allclose(robot1_world_orientation, robot1_scene_orientation, atol=1e-3)

        # For scene 1 (non-zero offset), verify coordinate transform
        robot2 = env.scenes[1].robots[0]
        scene_position, scene_orientation = env.scenes[1].get_position_orientation()

        robot2_world_position, robot2_world_orientation = robot2.get_position_orientation()
        robot2_scene_position, robot2_scene_orientation = robot2.get_position_orientation(frame="scene")

        print(f"  scene 1 robot: world_pos={robot2_world_position}, scene_pos={robot2_scene_position}")
        combined_position, combined_orientation = T.pose_transform(
            scene_position, scene_orientation, robot2_scene_position, robot2_scene_orientation
        )
        assert th.allclose(robot2_world_position, combined_position, atol=1e-3)
        assert th.allclose(robot2_world_orientation, combined_orientation, atol=1e-3)

        og.clear()
        _passed(test_id)

    def test_setter(self, robot):
        test_id = f"TestRobotGetterSetter::test_setter[{robot}]"
        _progress(test_id)
        env = setup_multi_environment(2, robot=robot)

        robot_obj = env.scenes[1].robots[0]

        # Test setting in world frame
        new_world_pos = th.tensor([1.0, 2.0, 0.5])
        new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
        robot_obj.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

        got_world_pos, got_world_ori = robot_obj.get_position_orientation()
        print(f"  set world pos={new_world_pos}, got={got_world_pos}")
        assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
        assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

        # Test setting in scene frame
        new_scene_pos = th.tensor([0.5, 1.0, 0.25])
        new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

        got_scene_pos, got_scene_ori = robot_obj.get_position_orientation(frame="scene")
        print(f"  set scene pos={new_scene_pos}, got={got_scene_pos}")
        assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
        assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

        # Setting a different scene-frame pose should change world-frame result
        new_scene_pos2 = th.tensor([-1.0, -2.0, 0.1])
        new_scene_ori2 = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos2, orientation=new_scene_ori2, frame="scene")

        got_world_pos2, got_world_ori2 = robot_obj.get_position_orientation()
        assert not th.allclose(got_world_pos2, new_world_pos, atol=1e-3)
        assert not th.allclose(got_world_ori2, new_world_ori, atol=1e-3)

        og.clear()
        _passed(test_id)

    def test_setter_sim_stopped(self, robot):
        """Getter/setter should work even when the simulator is stopped."""
        test_id = f"TestRobotGetterSetter::test_setter_sim_stopped[{robot}]"
        _progress(test_id)
        env = setup_multi_environment(2, robot=robot)
        og.sim.stop()
        print("  Sim stopped")

        robot_obj = env.scenes[1].robots[0]

        new_world_pos = th.tensor([1.0, 2.0, 0.5])
        new_world_ori = T.euler2quat(th.tensor([0, 0, th.pi / 2]))
        robot_obj.set_position_orientation(position=new_world_pos, orientation=new_world_ori)

        got_world_pos, got_world_ori = robot_obj.get_position_orientation()
        print(f"  world frame: set={new_world_pos}, got={got_world_pos}")
        assert th.allclose(got_world_pos, new_world_pos, atol=1e-3)
        assert th.allclose(got_world_ori, new_world_ori, atol=1e-3)

        new_scene_pos = th.tensor([0.5, 1.0, 0.25])
        new_scene_ori = T.euler2quat(th.tensor([0, th.pi / 4, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos, orientation=new_scene_ori, frame="scene")

        got_scene_pos, got_scene_ori = robot_obj.get_position_orientation(frame="scene")
        print(f"  scene frame: set={new_scene_pos}, got={got_scene_pos}")
        assert th.allclose(got_scene_pos, new_scene_pos, atol=1e-3)
        assert th.allclose(got_scene_ori, new_scene_ori, atol=1e-3)

        new_scene_pos2 = th.tensor([-1.0, -2.0, 0.1])
        new_scene_ori2 = T.euler2quat(th.tensor([th.pi / 6, 0, 0]))
        robot_obj.set_position_orientation(position=new_scene_pos2, orientation=new_scene_ori2, frame="scene")

        got_scene_pos2, got_scene_ori2 = robot_obj.get_position_orientation(frame="scene")
        assert th.allclose(got_scene_pos2, new_scene_pos2, atol=1e-3)
        assert th.allclose(got_scene_ori2, new_scene_ori2, atol=1e-3)

        got_world_pos2, got_world_ori2 = robot_obj.get_position_orientation()
        assert not th.allclose(got_world_pos2, new_world_pos, atol=1e-3)
        assert not th.allclose(got_world_ori2, new_world_ori, atol=1e-3)

        og.clear()
        _passed(test_id)


# ===================================================================
#  Section 6 – Particle system test
# ===================================================================


class TestParticles:
    def test_multi_scene_particle_source(self):
        _progress("TestParticles::test_multi_scene_particle_source")
        sink_cfg = dict(
            type="DatasetObject",
            name="sink",
            category="furniture_sink",
            model="czyfhq",
            abilities={
                "toggleable": {},
                "particleSource": {
                    "conditions": {
                        "water": [(ParticleModifyCondition.TOGGLEDON, True)],
                    },
                    "initial_speed": 0.0,
                },
                "particleSink": {
                    "conditions": {
                        "water": [],
                    },
                },
            },
            position=[0.0, -1.5, 0.0],
        )

        env = setup_multi_environment(3, additional_objects_cfg=[sink_cfg])

        for i, scene in enumerate(env.scenes):
            sink = scene.object_registry("name", "sink")
            assert sink.states[object_states.ToggledOn].set_value(True)
            print(f"  scene {i}: sink toggled on")

        print("  Running 50 sim steps...")
        for _ in range(50):
            og.sim.step()

        og.clear()
        _passed("TestParticles::test_multi_scene_particle_source")
