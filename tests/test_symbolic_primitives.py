import os

import pytest
import yaml

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm

def load_robot_config(robot_name):
    config_filename = os.path.join(og.example_config_path, f"{robot_name.lower()}_symbolic_primitives.yaml")
    with open(config_filename, "r") as file:
        return yaml.safe_load(file)

def start_env(enable_transition_rule=False, use_gpu_dynamics=True, robot_type="Fetch"):
    if og.sim:
        og.sim.stop()

    if robot_type not in ["Fetch", "Tiago"]:
        raise ValueError("Invalid robot configuration")
    robots = load_robot_config(robot_type)
    config = {
        "env": {"initial_pos_z_offset": 0.1},
        "render": {"viewer_width": 1280, "viewer_height": 720},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Wainscott_0_int",
            "load_object_categories": ["floors", "walls", "countertop", "fridge", "sink", "stove"],
            "scene_source": "OG",
        },
        "robots": [robots],
        "objects": [
            {
                "type": "DatasetObject",
                "name": "pan",
                "category": "frying_pan",
                "model": "mhndon",
                "position": [5.31, 10.75, 1.0],
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
                "position": [4.75, 10.75, 1.0],
                "bounding_box": [0.098, 0.098, 0.115],
            },
            {
                "type": "DatasetObject",
                "name": "sponge",
                "category": "sponge",
                "model": "qewotb",
                "position": [4.75, 10.75, 1.0],
            },
        ],
    }

    gm.USE_GPU_DYNAMICS = use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = enable_transition_rule
    env = og.Environment(configs=config)

    return env

@pytest.fixture(scope="module")
def shared_env():
    """Load the environment just once using module scope."""
    return start_env()

@pytest.fixture(scope="function")
def env(shared_env):
    """Reset the environment before each test function."""
    shared_env.scene.reset()
    return shared_env

@pytest.fixture(scope="function")
def env_with_transition_rules():
    """Create an environment with transition rules enabled for each test function."""
    env = start_env(enable_transition_rule=True, use_gpu_dynamics=False)
    yield env
    env.scene.reset()

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
    def test_in_hand_state(self, env, robot, prim_gen, apple):
        assert not robot.states[object_states.IsGrasping].get_value(apple)
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, apple):
            env.step(action)
        assert robot.states[object_states.IsGrasping].get_value(apple)

    # def test_navigate():
    #    pass

    def test_open(self, env, prim_gen, fridge):
        assert not fridge.states[object_states.Open].get_value()
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.OPEN, fridge):
            env.step(action)
        assert fridge.states[object_states.Open].get_value()

    def test_close(self, env, prim_gen, fridge):
        fridge.states[object_states.Open].set_value(True)
        assert fridge.states[object_states.Open].get_value()
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.CLOSE, fridge):
            env.step(action)
        assert not fridge.states[object_states.Open].get_value()

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

    def test_place_ontop(self, env, prim_gen, apple, pan):
        assert not apple.states[object_states.OnTop].get_value(pan)
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, apple):
            env.step(action)
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, pan):
            env.step(action)
        assert apple.states[object_states.OnTop].get_value(pan)

    def test_toggle_on(self, env, prim_gen, stove, sink):
        assert not stove.states[object_states.ToggledOn].get_value()
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_ON, stove):
            env.step(action)
        assert stove.states[object_states.ToggledOn].get_value()

        # toggle off the stove after the test is done
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_OFF, stove):
            env.step(action)
        assert not stove.states[object_states.ToggledOn].get_value()

        assert not sink.states[object_states.ToggledOn].get_value()
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_ON, sink):
            env.step(action)
        assert sink.states[object_states.ToggledOn].get_value()

        # toggle off the stove after the test is done
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.TOGGLE_OFF, sink):
            env.step(action)
        assert not sink.states[object_states.ToggledOn].get_value()

    def test_soak_under(self, env, prim_gen, robot, sponge, sink):
        water_system = env.scene.get_system("water")
        assert not sponge.states[object_states.Saturated].get_value(water_system)
        assert not sink.states[object_states.ToggledOn].get_value()

        # First grasp the sponge
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, sponge):
            env.step(action)
        assert robot.states[object_states.IsGrasping].get_value(sponge)

        # Then toggle on the sink
        sink.states[object_states.ToggledOn].set_value(True)
        assert sink.states[object_states.ToggledOn].get_value()

        # Then soak the sponge under the water
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.SOAK_UNDER, sink):
            env.step(action)
        assert sponge.states[object_states.Saturated].get_value(water_system)

        # toggle off the sink after the test is done
        sink.states[object_states.ToggledOn].set_value(False)
        assert not sink.states[object_states.ToggledOn].get_value()

    def test_wipe(self, env, prim_gen, robot, sponge, sink, countertop):
        # Some pre-assertions
        water_system = env.scene.get_system("water")
        assert not sponge.states[object_states.Saturated].get_value(water_system)
        assert not sink.states[object_states.ToggledOn].get_value()

        # Dirty the countertop as the setup
        mud_system = env.scene.get_system("mud")
        countertop.states[object_states.Covered].set_value(mud_system, True)

        # First grasp the sponge
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, sponge):
            env.step(action)
        assert robot.states[object_states.IsGrasping].get_value(sponge)

        # Then toggle on the sink
        sink.states[object_states.ToggledOn].set_value(True)
        assert sink.states[object_states.ToggledOn].get_value()

        # Then soak the sponge under the water
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.SOAK_UNDER, sink):
            env.step(action)
        assert sponge.states[object_states.Saturated].get_value(water_system)

        # Then toggle on the sink
        sink.states[object_states.ToggledOn].set_value(False)
        assert not sink.states[object_states.ToggledOn].get_value()

        # Wipe the countertop with the sponge
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.WIPE, countertop):
            env.step(action)
        assert not countertop.states[object_states.Covered].get_value(mud_system)

    def test_cut(self, env_with_transition_rules, prim_gen, apple, knife, countertop):
        # assert not apple.states[object_states.Cut].get_value(knife)
        # start a new environment to enable transition rules
        print("Grasping knife")
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.GRASP, knife):
            env_with_transition_rules.step(action)
        print("Cutting apple")
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.CUT, apple):
            env_with_transition_rules.step(action)
        print("Putting knife back on countertop")
        for action in prim_gen.apply_ref(SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP, countertop):
            env_with_transition_rules.step(action)

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
        og.clear()


def main():
    test = TestSymbolicPrimitives()
    env = start_env(enable_transition_rule=True)
    prim_gen = SymbolicSemanticActionPrimitives(env)
    apple = next(iter(env.scene.object_registry("category", "apple")))
    sponge = next(iter(env.scene.object_registry("category", "sponge")))
    knife = next(iter(env.scene.object_registry("category", "carving_knife")))
    sink = next(iter(env.scene.object_registry("category", "sink")))
    fridge = next(iter(env.scene.object_registry("category", "fridge")))
    robot = env.robots[0]
    countertop = next(iter(env.scene.object_registry("category", "countertop")))

    try:
        test.test_open(env, prim_gen, fridge)
        test.test_close(env, prim_gen, fridge)
    except:
        raise

if __name__ == "__main__":
    main()
