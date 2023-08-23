import os
import pytest
import yaml

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.symbolic_semantic_action_primitives import SymbolicSemanticActionPrimitiveSet, SymbolicSemanticActionPrimitives

@pytest.fixture
def env():
  config_filename = os.path.join(og.example_config_path, "homeboy.yaml")
  config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
  config["scene"]["scene_model"] = "Pomaria_1_int"
  config["scene"]["load_object_categories"] = ["floors", "walls", "fridge", "sink"]
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
  return env.scene.object_registry("category", "fridge")[0]

@pytest.fixture
def sink(env):
  return env.scene.object_registry("category", "sink")[0]

# def test_navigate():
#    pass

def test_open(env, prim_gen, fridge):
  assert not fridge.states[object_states.Open].get_value()
  for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, fridge):
    env.step(action)
  assert fridge.states[object_states.Open].get_value()

# def test_close():
#    pass

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