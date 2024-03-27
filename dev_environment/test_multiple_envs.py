import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision, set_base_and_detect_collision

import cProfile, pstats, io
import time
import os
import argparse

def pause(time):
    for _ in range(int(time*100)):
        og.sim.render()

def pause_step(time):
    for _ in range(int(time*100)):
        og.sim.step()

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action)
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def main():
    # Load the config
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    config_filename = "rl.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    vec_env = og.VectorEnvironment(20, config)
    # while True:
    #     og.sim.step()
    import time
    while True:
        actions = []
        start_time = time.time()
        for e in vec_env.envs:
            actions.append(e.action_space.sample())
        vec_env.step(actions)
        print("fps", 1/(time.time() - start_time))

    # scene = env.scene
    # robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # grasp_obj = DatasetObject(
    #     name="cologne",
    #     category="bottle_of_cologne",
    #     model="lyipur"
    # )
    # og.sim.import_object(grasp_obj)
    # grasp_obj.set_position([-0.3, -0.8, 0.5])
    # og.sim.step()
    
    pause_step(100)

if __name__ == "__main__":
    main()