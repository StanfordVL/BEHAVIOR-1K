import os
import pytest
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson import object_states
from omnigibson.action_primitives.symbolic_semantic_action_primitives import SymbolicSemanticActionPrimitiveSet, SymbolicSemanticActionPrimitives

gm.USE_GPU_DYNAMICS = True
gm.USE_FLATCACHE = True

def start_env():
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
      "load_object_categories": ["floors", "walls", "countertop", "fridge", "sink", "stove"],
      "scene_source": "OG",
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
        "disable_grasp_handling": True,
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
    "objects": [
      {
          "type": "DatasetObject",
          "name": "pan",
          "category": "frying_pan",
          "model": "mhndon",
          "position": [5.31, 10.75, 1.],
      },
      {
          "type": "DatasetObject",
          "name": "knife",
          "category": "carving_knife",
          "model": "awvoox",
          "position": [5.31, 10.75, 1.2],
      },
      {
          "type": "DatasetObject",
          "name": "apple",
          "category": "apple",
          "model": "agveuv",
          "position": [4.75, 10.75, 1.],
          "bounding_box": [0.098, 0.098, 0.115]
      },
      # {
      #     "type": "DatasetObject",
      #     "name": "sponge",
      #     "category": "sponge",
      #     "model": "qewotb",
      #     "position": [4.75, 10.75, 1.],
      # },
    ]
  }

  env = og.Environment(configs=config)
  
  return env

@pytest.fixture(scope="module")
def shared_env():
  """Load the environment just once using module scope."""
  return start_env()

@pytest.fixture(scope="function")
def env(shared_env):
  """Reset the environment before each test function."""
  og.sim.scene.reset()
  return shared_env

@pytest.fixture
def prim_gen(env):
  scene = env.scene
  robot = env.robots[0]
  return SymbolicSemanticActionPrimitives(None, scene, robot)

@pytest.fixture
def countertop(env):
  return next(iter(env.scene.object_registry("category", "countertop")))

@pytest.fixture
def fridge(env):
  return next(iter(env.scene.object_registry("category", "fridge")))

@pytest.fixture
def sink(env):
  return next(iter(env.scene.object_registry("category", "sink")))

@pytest.fixture
def pan(env):
  return next(iter(env.scene.object_registry("category", "frying_pan")))

@pytest.fixture
def steak(env):
  # TODO: Why apple? Use steak.
  return next(iter(env.scene.object_registry("category", "apple")))

@pytest.fixture
def sponge(env):
  return next(iter(env.scene.object_registry("category", "sponge")))

@pytest.fixture
def knife(env):
  return next(iter(env.scene.object_registry("category", "carving_knife")))

# def test_navigate():
#    pass

def test_open(env, prim_gen, fridge):
  assert not fridge.states[object_states.Open].get_value()
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, fridge):
    env.step(action)
  assert fridge.states[object_states.Open].get_value()

def test_close(env, prim_gen, fridge):
  fridge.states[object_states.Open].set_value(True)
  assert fridge.states[object_states.Open].get_value()
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.CLOSE, fridge):
    env.step(action)
  assert not fridge.states[object_states.Open].get_value()

def test_place_inside(env, prim_gen, steak, fridge):
  assert not steak.states[object_states.Inside].get_value(fridge)
  assert not fridge.states[object_states.Open].get_value()
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, fridge):
    env.step(action)
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, steak):
    env.step(action)
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE, fridge):
    env.step(action)
  assert steak.states[object_states.Inside].get_value(fridge)

def test_place_ontop(env, prim_gen, steak, pan):
  assert not steak.states[object_states.OnTop].get_value(pan)
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, steak):
    env.step(action)
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, pan):
    env.step(action)
  assert steak.states[object_states.OnTop].get_value(pan)

def test_toggle_on(env, prim_gen, stove):
    assert not stove.states[object_states.ToggledOn].get_value()
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_ON, stove):
      env.step(action)
    assert stove.states[object_states.ToggledOn].get_value()


# def test_soak_under():
#    pass

# def test_soak_inside():
#    pass

# def test_wipe():
#    pass

@pytest.mark.skip(reason="A bug with object addition causes the robot to crash after slicing.")
def test_cut(env, prim_gen, steak, knife, countertop):
  # assert not steak.states[object_states.Cut].get_value(knife)
  print("Grasping knife")
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, knife):
    env.step(action)
  for _ in range(60): og.sim.step()
  print("Cutting steak")
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.CUT, steak):
    env.step(action)
  for _ in range(60): og.sim.step()
  print("Putting knife back on countertop")
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, countertop):
    env.step(action)

# def test_place_near_heating_element():
#    pass

# def test_wait_for_cooked():
#    pass

def main():
  env = start_env()
  scene = env.scene
  robot = env.robots[0]
  prim_gen = SymbolicSemanticActionPrimitives(None, scene, robot)
  steak = next(iter(env.scene.object_registry("category", "apple")))
  knife = next(iter(env.scene.object_registry("category", "carving_knife")))
  countertop = next(iter(env.scene.object_registry("category", "countertop")))

  print("Will start in 3 seconds")
  for _ in range(120): og.sim.step()

  try:
    test_cut(env, prim_gen, steak, knife, countertop)
  except:
    raise
  while True:
    og.sim.step()

if __name__ == "__main__":
  main()