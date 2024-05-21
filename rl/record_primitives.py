import os

import numpy as np
import yaml

import omnigibson as og
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode Finished:")
            print("rew:", rew)
            print(terminated, truncated, info)

            break


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in a crowded scene.
    """
    print("Starting to collect trajectories")

    # Load the config
    config_filename = "service/omni_grpc_offline.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    # config["scene"]["scene_model"] = "Rs_int"
    # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    # config["objects"] = [
    #     {
    #         "type": "DatasetObject",
    #         "name": "cologne",
    #         "category": "bottle_of_cologne",
    #         "model": "lyipur",
    #         "position": [-0.3, -0.8, 0.5],
    #         "orientation": [0, 0, 0, 1],
    #     },
    #     {
    #         "type": "DatasetObject",
    #         "name": "table",
    #         "category": "breakfast_table",
    #         "model": "rjgmmy",
    #         "scale": [0.3, 0.3, 0.3],
    #         "position": [-0.7, 0.5, 0.2],
    #         "orientation": [0, 0, 0, 1],
    #     },
    # ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = env.task._primitive_controller
    cologne = scene.object_registry("name", "cologne")

    for episode_idx in range(10000):

        obs = env.reset()

        # Grasp the object
        try:
            print("Executing controller")
            execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, cologne), env)
            print("Finished executing grasp")

        except ActionPrimitiveError as e:
            print(e)
            continue


if __name__ == "__main__":
    main()
