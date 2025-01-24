import os
import random

import numpy as np
import torch as th
import yaml
from pytest_rerunfailures import pytest

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject

# Make sure that Omniverse is launched before setting up the tests.
og.launch()


def load_robot_config(robot_name):
    config_filename = os.path.join(og.example_config_path, f"{robot_name.lower()}_primitives.yaml")
    with open(config_filename, "r") as file:
        full_config = yaml.safe_load(file)
        return full_config.get("robots", {})[0]


def setup_environment(load_object_categories, robot="R1"):
    if robot not in ["R1", "Tiago"]:
        raise ValueError("Invalid robot configuration")

    robots = load_robot_config(robot)

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": load_object_categories,
        },
        "robots": [robots],
    }

    seed = 40
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

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

    # Let the objects settle
    for _ in range(30):
        og.sim.step()

    controller = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=False, curobo_batch_size=1)
    try:
        for primitive, args in zip(primitives, primitives_args):
            execute_controller(controller.apply_ref(primitive, *args, attempts=1), env)
    finally:
        # Clear the sim
        og.clear()


@pytest.mark.parametrize("robot", ["Tiago", "R1"])
class TestPrimitives:
    def test_navigate(self, robot):
        categories = ["floors", "ceilings", "walls"]
        env = setup_environment(categories, robot=robot)

        objects = []
        obj_1 = {
            "object": DatasetObject(name="cologne", category="bottle_of_cologne", model="lyipur"),
            "position": [-0.3, -0.8, 0.5],
            "orientation": [0, 0, 0, 1],
        }
        objects.append(obj_1)

        primitives = [StarterSemanticActionPrimitiveSet.NAVIGATE_TO]
        primitives_args = [(obj_1["object"],)]

        primitive_tester(env, objects, primitives, primitives_args)

    def test_grasp(self, robot):
        categories = ["floors", "ceilings", "walls", "coffee_table"]
        env = setup_environment(categories, robot=robot)

        objects = []
        obj_1 = {
            "object": DatasetObject(name="cologne", category="bottle_of_cologne", model="lyipur"),
            "position": [-0.3, -0.8, 0.5],
            "orientation": [0, 0, 0, 1],
        }
        objects.append(obj_1)

        primitives = [StarterSemanticActionPrimitiveSet.GRASP]
        primitives_args = [(obj_1["object"],)]

        primitive_tester(env, objects, primitives, primitives_args)

    def test_place(self, robot):
        categories = ["floors", "ceilings", "walls", "coffee_table"]
        env = setup_environment(categories, robot=robot)

        objects = []
        obj_1 = {
            "object": DatasetObject(name="table", category="breakfast_table", model="rjgmmy", scale=[0.3, 0.3, 0.3]),
            "position": [-0.7, 0.5, 0.09],
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

        primitive_tester(env, objects, primitives, primitives_args)

    @pytest.mark.skip(reason="primitives are broken")
    def test_open_prismatic(self, robot):
        categories = ["floors"]
        env = setup_environment(categories, robot=robot)

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

        primitive_tester(env, objects, primitives, primitives_args)

    @pytest.mark.skip(reason="primitives are broken")
    def test_open_revolute(self, robot):
        categories = ["floors"]
        env = setup_environment(categories, robot=robot)

        objects = []
        obj_1 = {
            "object": DatasetObject(name="fridge", category="fridge", model="dszchb", scale=[0.7, 0.7, 0.7]),
            "position": [-1.2, -0.4, 0.5],
            "orientation": [0, 0, 0, 1],
        }
        objects.append(obj_1)

        primitives = [StarterSemanticActionPrimitiveSet.OPEN]
        primitives_args = [(obj_1["object"],)]

        primitive_tester(env, objects, primitives, primitives_args)
