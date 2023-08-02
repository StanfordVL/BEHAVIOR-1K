import IPython
import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

import cProfile, pstats, io
import time
import os
import argparse

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action.tolist())
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = os.path.join(og.example_config_path, "homeboy.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    robot._links['base_link'].mass = 10000

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # table = DatasetObject(
    #     name="table",
    #     category="breakfast_table",
    #     model="rjgmmy",
    # )
    # og.sim.import_object(table)
    # table.set_position([1.0, 1.0, 0.58])

    # grasp_obj = DatasetObject(
    #     name="potato",
    #     category="bottle_of_cologne",
    #     model="lyipur",
    # )

    # og.sim.import_object(grasp_obj)
    # grasp_obj.set_position([-0.3, -0.8, 0.5])
    # og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot, teleport=True)

    def test_grasp():
        grasp_obj, = scene.object_registry("category", "bottle_of_vodka")
        execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, False, grasp_obj), env)

    def test_place():
        box, = scene.object_registry("category", "storage_box")
        execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_INSIDE, False, box), env)

    # Work more reliably
    # IPython.embed()
    # test_navigate_to_obj()
    # test_grasp_no_navigation()
    # test_grasp_replay_and_place()

    og.sim.step()

    # Don't work as reliably
    test_grasp()
    test_place()

if __name__ == "__main__":
    main()