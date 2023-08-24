import os
import pytest
import yaml

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.symbolic_semantic_action_primitives import SymbolicSemanticActionPrimitiveSet, SymbolicSemanticActionPrimitives

@pytest.fixture
def env():
  config = {
    "env": {
      "initial_pos_z_offset": 0.1
    },
    "render": {
      "viewer_width": 1280,
      "viewer_height": 720
    },
    "scene": {
      "type": "InteractiveTraversableScene",
      "scene_model": "Wainscott_0_int",
      "trav_map_resolution": 0.1,
      "trav_map_erosion": 2,
      "trav_map_with_objects": True,
      "build_graph": True,
      "num_waypoints": 1,
      "waypoint_resolution": 0.2,
      "load_object_categories": ["floors", "walls", "fridge", "sink"],
      "not_load_object_categories": None,
      "load_room_types": None,
      "load_room_instances": None,
      "load_task_relevant_only": False,
      "seg_map_resolution": 0.1,
      "scene_source": "OG",
      "include_robots": True
    },
    "robots": [
      {
        "type": "Fetch",
        "obs_modalities": [
          "scan",
          "rgb",
          "depth"
        ],
        "scale": 1,
        "self_collisions": True,
        "action_normalize": False,
        "action_type": "continuous",
        "grasping_mode": "sticky",
        "rigid_trunk": False,
        "default_trunk_offset": 0.365,
        "default_arm_pose": "diagonal30",
        "reset_joint_pos": "tuck",
        "controller_config": {
          "base": {
            "name": "DifferentialDriveController"
          },
          "arm_0": {
            "name": "JointController",
            "motor_type": "position",
            "command_input_limits": None,
            "command_output_limits": None,
            "use_delta_commands": False
          },
          "gripper_0": {
            "name": "JointController",
            "motor_type": "position",
            "command_input_limits": [
              -1,
              1
            ],
            "command_output_limits": None,
            "use_delta_commands": True,
            "use_single_command": True
          },
          "camera": {
            "name": "JointController",
            "use_delta_commands": False
          }
        }
      }
    ],
    "objects": []
  }

  env = og.Environment(configs=config)

  # TODO: Add some new objects
  return env

@pytest.fixture
def prim_gen(env):
  scene = env.scene
  robot = env.robots[0]
  return SymbolicSemanticActionPrimitives(None, scene, robot)

@pytest.fixture
def fridge(env):
  return next(iter(env.scene.object_registry("category", "fridge")))

@pytest.fixture
def sink(env):
  return next(iter(env.scene.object_registry("category", "sink")))

# def test_navigate():
#    pass

def test_open(env, prim_gen, fridge):
  assert not fridge.states[object_states.Open].get_value()
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, fridge):
    env.step(action)
  assert fridge.states[object_states.Open].get_value()

def test_close():
  fridge.states[object_states.Open].set_value(True)
  assert fridge.states[object_states.Open].get_value()
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.CLOSE, fridge):
    env.step(action)
  assert not fridge.states[object_states.Open].get_value()

# def test_place_inside():
#    pass

# def test_place_ontop():
#    pass

# def test_toggle_on():
#    pass

# def test_soak_under():
#    pass

# def test_soak_inside():
#    pass

# def test_wipe():
#    pass

# def test_cut():
#    pass

# def test_place_near_heating_element():
#    pass

# def test_wait_for_cooked():
#    pass