import os
import pytest
import yaml

from omnigibson.macros import gm
gm.USE_GPU_DYNAMICS = True
gm.USE_FLATCACHE = True

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.symbolic_semantic_action_primitives import SymbolicSemanticActionPrimitiveSet, SymbolicSemanticActionPrimitives
from omnigibson.systems import get_system


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
        "default_reset_mode": "tuck",
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
      {
          "type": "DatasetObject",
          "name": "sponge",
          "category": "sponge",
          "model": "qewotb",
          "position": [4.75, 10.75, 1.],
      },
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
def robot(env):
  return env.robots[0]

@pytest.fixture
def prim_gen(env):
  return SymbolicSemanticActionPrimitives(env)

@pytest.fixture
def countertop(env):
  return next(iter(env.scene.object_registry("category", "countertop")))

@pytest.fixture
def fridge(env):
  return next(iter(env.scene.object_registry("category", "fridge")))

@pytest.fixture
def stove(env):
  return next(iter(env.scene.object_registry("category", "stove")))

@pytest.fixture
def sink(env):
  return next(iter(env.scene.object_registry("category", "sink")))

@pytest.fixture
def pan(env):
  return next(iter(env.scene.object_registry("category", "frying_pan")))

@pytest.fixture
def apple(env):
  return next(iter(env.scene.object_registry("category", "apple")))

@pytest.fixture
def sponge(env):
  return next(iter(env.scene.object_registry("category", "sponge")))

@pytest.fixture
def knife(env):
  return next(iter(env.scene.object_registry("category", "carving_knife")))

class TestSymbolicPrimitives:
  @pytest.mark.skip(reason="primitives are broken")
  def test_in_hand_state(self, env, robot, prim_gen, apple):
    assert not robot.states[object_states.IsGrasping].get_value(apple)
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, apple):
      env.step(action)
    assert robot.states[object_states.IsGrasping].get_value(apple)

  # def test_navigate():
  #    pass

  @pytest.mark.skip(reason="primitives are broken")
  def test_open(self, env, prim_gen, fridge):
    assert not fridge.states[object_states.Open].get_value()
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, fridge):
      env.step(action)
    assert fridge.states[object_states.Open].get_value()

  @pytest.mark.skip(reason="primitives are broken")
  def test_close(self, env, prim_gen, fridge):
    fridge.states[object_states.Open].set_value(True)
    assert fridge.states[object_states.Open].get_value()
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.CLOSE, fridge):
      env.step(action)
    assert not fridge.states[object_states.Open].get_value()

  @pytest.mark.skip(reason="primitives are broken")
  def test_place_inside(self, env, prim_gen, apple, fridge):
    assert not apple.states[object_states.Inside].get_value(fridge)
    assert not fridge.states[object_states.Open].get_value()
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, fridge):
      env.step(action)
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, apple):
      env.step(action)
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE, fridge):
      env.step(action)
    assert apple.states[object_states.Inside].get_value(fridge)

  @pytest.mark.skip(reason="primitives are broken")
  def test_place_ontop(self, env, prim_gen, apple, pan):
    assert not apple.states[object_states.OnTop].get_value(pan)
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, apple):
      env.step(action)
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, pan):
      env.step(action)
    assert apple.states[object_states.OnTop].get_value(pan)

  @pytest.mark.skip(reason="primitives are broken")
  def test_toggle_on(self, env, prim_gen, stove):
    assert not stove.states[object_states.ToggledOn].get_value()
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_ON, stove):
      env.step(action)
    assert stove.states[object_states.ToggledOn].get_value()

  @pytest.mark.skip(reason="primitives are broken")
  def test_soak_under(self, env, prim_gen, robot, sponge, sink):
    water_system = get_system("water", force_active=True)
    assert not sponge.states[object_states.Saturated].get_value(water_system)
    assert not sink.states[object_states.ToggledOn].get_value()

    # First toggle on the sink
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_ON, sink):
      env.step(action)
    assert sink.states[object_states.ToggledOn].get_value()

    # Then grasp the sponge
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, sponge):
      env.step(action)
    assert robot.states[object_states.IsGrasping].get_value(sponge)

    # Then soak the sponge under the water
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.SOAK_UNDER, sink):
      env.step(action)
    assert sponge.states[object_states.Saturated].get_value(water_system)

  # def test_soak_inside():
  #    pass

  @pytest.mark.skip(reason="primitives are broken")
  def test_wipe(self, env, prim_gen, sponge, sink, countertop):
    # Some pre-assertions
    water_system = get_system("water", force_active=True)
    assert not sponge.states[object_states.Saturated].get_value(water_system)
    assert not sink.states[object_states.ToggledOn].get_value()

    # Dirty the countertop as the setup
    mud_system = get_system("mud", force_active=True)
    countertop.states[object_states.Covered].set_value(mud_system, True)
    assert countertop.states[object_states.Covered].get_value(mud_system)

    # First toggle on the sink
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_ON, sink):
      env.step(action)
    assert sink.states[object_states.ToggledOn].get_value()

    # Then grasp the sponge
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, sponge):
      env.step(action)
    assert robot.states[object_states.IsGrasping].get_value(sponge)

    # Then soak the sponge under the water
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.SOAK_UNDER, sink):
      env.step(action)
    assert sponge.states[object_states.Saturated].get_value(water_system)

    # Wipe the countertop with the sponge
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.WIPE, countertop):
      env.step(action)
    assert not countertop.states[object_states.Covered].get_value(mud_system)

  @pytest.mark.skip(reason="primitives are broken")
  def test_cut(self, env, prim_gen, apple, knife, countertop):
    # assert not apple.states[object_states.Cut].get_value(knife)
    print("Grasping knife")
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, knife):
      env.step(action)
    for _ in range(60): env.step(prim_gen._empty_action())
    print("Cutting apple")
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.CUT, apple):
      env.step(action)
    for _ in range(60): env.step(prim_gen._empty_action())
    print("Putting knife back on countertop")
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, countertop):
      env.step(action)

  def test_persistent_sticky_grasping(self, env, robot, prim_gen, apple):
    assert not robot.states[object_states.IsGrasping].get_value(apple)
    for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, apple):
      env.step(action)
    assert robot.states[object_states.IsGrasping].get_value(apple)
    state = og.sim.dump_state()
    og.sim.stop()
    og.sim.play()
    og.sim.load_state(state)
    assert robot.states[object_states.IsGrasping].get_value(apple)

    for _ in range(10):
      env.step(prim_gen._empty_action())

    assert robot.states[object_states.IsGrasping].get_value(apple)

  # def test_place_near_heating_element():
  #    pass

  # def test_wait_for_cooked():
  #    pass

  def teardown_class(cls):
    og.sim.clear()

def main():
  env = start_env()
  prim_gen = SymbolicSemanticActionPrimitives(env)
  apple = next(iter(env.scene.object_registry("category", "apple")))
  knife = next(iter(env.scene.object_registry("category", "carving_knife")))
  countertop = next(iter(env.scene.object_registry("category", "countertop")))

  print("Will start in 3 seconds")
  for _ in range(180): env.step(prim_gen._empty_action())

  try:
    test_cut(env, prim_gen, apple, knife, countertop)
  except:
    raise
  while True:
    og.sim.step()

if __name__ == "__main__":
  main()