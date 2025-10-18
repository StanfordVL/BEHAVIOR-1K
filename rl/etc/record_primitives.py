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
from pathlib import Path
import zmq
import json
import pickle

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

# Create a ZMQ context
context = zmq.Context()

# Create a socket to send messages (PUB)
data_queue = context.socket(zmq.PUB)
data_queue.bind("tcp://127.0.0.1:5555")

# Create a socket to receive messages (SUB)
state_queue = context.socket(zmq.SUB)
state_queue.connect("tcp://127.0.0.1:5556")

# Subscribe to all messages
state_queue.setsockopt_string(zmq.SUBSCRIBE, "")

def execute_controller(ctrl_gen, env):
    episode_data = []
    for action in ctrl_gen:
        obs, rew, terminated, truncated, info = env.step(action)
        episode_data.append((obs, action, rew))
        # breakpoint()
        if terminated or truncated:
            data_queue.send(pickle.dumps(episode_data))
            break

def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in a crowded scene.
    """
    print("Starting to collect trajectories")

    # Load the config
    config_filename = "primitives.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    reset_poses_path = reset_poses_path = Path(__file__).resolve().parent.parent / "reset_poses.json"
    config["task"]["precached_reset_pose_path"] = reset_poses_path

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # controller = env.task._primitive_controller
    controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    cologne = scene.object_registry("name", "cologne")

    for episode_idx in range(10000):
        state = pickle.loads(state_queue.recv())
        og.sim.load_state(state)
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
