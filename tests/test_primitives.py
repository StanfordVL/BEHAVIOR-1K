import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject

pytestmark = pytest.mark.skip("Skip all primitive tests for multiple-envs PR; will fix in a follow-up")

# Make sure that Omniverse is launched before setting up the tests.
og.launch()


def setup_environment(load_object_categories):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": load_object_categories,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["scan", "rgb", "depth"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "controller_config": {
                    "base": {
                        "name": "DifferentialDriveController",
                    },
                    "arm_0": {
                        "name": "InverseKinematicsController",
                        "command_input_limits": "default",
                        "command_output_limits": [
                            th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5], dtype=th.float32),
                            th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5], dtype=th.float32),
                        ],
                        "mode": "pose_absolute_ori",
                        "kp": 300.0,
                    },
                    "gripper_0": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                    },
                    "camera": {"name": "JointController", "use_delta_commands": False},
                },
            }
        ],
    }

    if og.sim is None:
        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    return env


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def primitive_tester(env, objects, primitives, primitives_args):
    for obj in objects:
        env.scene.add_object(obj["object"])
        obj["object"].set_position_orientation(position=obj["position"], orientation=obj["orientation"])
        og.sim.step()

    controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    try:

        for primitive, args in zip(primitives, primitives_args):
            try:
                execute_controller(controller.apply_ref(primitive, *args), env)
            except Exception as e:
                return False
    finally:
        # Clear the sim
        og.clear()

    return True


def test_navigate():
    categories = ["floors", "ceilings", "walls"]
    env = setup_environment(categories)

    objects = []
    obj_1 = {
        "object": DatasetObject(name="cologne", category="bottle_of_cologne", model="lyipur"),
        "position": [-0.3, -0.8, 0.5],
        "orientation": [0, 0, 0, 1],
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.NAVIGATE_TO]
    primitives_args = [(obj_1["object"],)]

    assert primitive_tester(env, objects, primitives, primitives_args)


def test_grasp():
    categories = ["floors", "ceilings", "walls", "coffee_table"]
    env = setup_environment(categories)

    objects = []
    obj_1 = {
        "object": DatasetObject(name="cologne", category="bottle_of_cologne", model="lyipur"),
        "position": [-0.3, -0.8, 0.5],
        "orientation": [0, 0, 0, 1],
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.GRASP]
    primitives_args = [(obj_1["object"],)]

    assert primitive_tester(env, objects, primitives, primitives_args)


def test_place():
    categories = ["floors", "ceilings", "walls", "coffee_table"]
    env = setup_environment(categories)

    objects = []
    obj_1 = {
        "object": DatasetObject(name="table", category="breakfast_table", model="rjgmmy", scale=[0.3, 0.3, 0.3]),
        "position": [-0.7, 0.5, 0.2],
        "orientation": [0, 0, 0, 1],
    }
    obj_2 = {
        "object": DatasetObject(name="cologne", category="bottle_of_cologne", model="lyipur"),
        "position": [-0.3, -0.8, 0.5],
        "orientation": [0, 0, 0, 1],
    }
    objects.append(obj_1)
    objects.append(obj_2)

    primitives = [StarterSemanticActionPrimitiveSet.GRASP, StarterSemanticActionPrimitiveSet.PLACE_ON_TOP]
    primitives_args = [(obj_2["object"],), (obj_1["object"],)]

    assert primitive_tester(env, objects, primitives, primitives_args)


@pytest.mark.skip(reason="primitives are broken")
def test_open_prismatic():
    categories = ["floors"]
    env = setup_environment(categories)

    objects = []
    obj_1 = {
        "object": DatasetObject(
            name="bottom_cabinet", category="bottom_cabinet", model="bamfsz", scale=[0.7, 0.7, 0.7]
        ),
        "position": [-1.2, -0.4, 0.5],
        "orientation": [0, 0, 0, 1],
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.OPEN]
    primitives_args = [(obj_1["object"],)]

    assert primitive_tester(env, objects, primitives, primitives_args)


@pytest.mark.skip(reason="primitives are broken")
def test_open_revolute():
    categories = ["floors"]
    env = setup_environment(categories)

    objects = []
    obj_1 = {
        "object": DatasetObject(name="fridge", category="fridge", model="dszchb", scale=[0.7, 0.7, 0.7]),
        "position": [-1.2, -0.4, 0.5],
        "orientation": [0, 0, 0, 1],
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.OPEN]
    primitives_args = [(obj_1["object"],)]

    assert primitive_tester(env, objects, primitives, primitives_args)
