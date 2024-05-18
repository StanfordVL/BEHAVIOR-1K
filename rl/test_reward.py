import argparse
import math
import os
import uuid
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

import omnigibson as og
from omnigibson.envs.sb3_vec_env import SB3VectorEnvironment
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.sensors.scan_sensor import ScanSensor
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.utils.ui_utils import KeyboardRobotController


def step_sim(time):
    for _ in range(int(time * 100)):
        og.sim.step()


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action[0])


def main(iterations):
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.HEADLESS = False

    config = yaml.load(open("rl_reward.yaml", "r"), Loader=yaml.FullLoader)
    reset_poses_path = os.path.dirname(__file__) + "/reset_poses.json"
    config["task"]["precached_reset_pose_path"] = reset_poses_path

    vec_env = SB3VectorEnvironment(1, config, render_on_step=True)
    env = vec_env.envs[0]
    robot = env.robots[0]
    # controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

    # Testing primitives with env
    #############################
    # obj = env.scene.object_registry("name", "cologne")
    # for i in tqdm(range(int(iterations))):
    #     try:
    #         obs = vec_env.reset()
    #         timestep = 0
    #         for action in controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj):
    #             obs, reward, done, info = vec_env.step(np.array([action]))
    #             print(reward)
    #             timestep += 1
    #             if done[0] or timestep >= 400:
    #                 for action in controller._execute_release():
    #                     vec_env.step(np.array([action]))
    #                 break
    #     except Exception as e:
    #         print("Error in iteration: ", i)
    #         print(e)
    #         print('--------------------')

    # Testing with keyboard controller
    ##################################
    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()
    # pose = controller._get_robot_pose_from_2d_pose([-0.433881, -0.210183, -2.0118])
    pose = ([-0.433881, -0.210183,  0.01], [ 0.,  0., -0.8446441, 0.53532825])
    robot.set_position_orientation(*pose)
    # obj = env.scene.object_registry("name", "cologne")
    while True:
        action = action_generator.get_teleop_action()
        obs, reward, done, info = vec_env.step(np.array([action]))
        # print(reward)

    # Testing random actions with env
    #############################
    # import traceback

    # for i in tqdm(range(int(iterations))):
    #     try:
    #         done = False
    #         env.reset()
    #         while not done:
    #             action = env.action_space.sample()
    #             obs, reward, terminated, truncated, info = env.step(action)
    #             done = terminated or truncated
    #             break
    #     except Exception as e:
    #         print("Error in iteration: ", i)
    #         print(e)
    #         traceback.print_exc()
    #         print("--------------------")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run worker")
    # parser.add_argument("iterations")

    # args = parser.parse_args()
    main(5)

    # seg semantic - 224 x 224
    # seg instance - 224 x 224
    # depth - 224 x 224
    # rgb - 224 x 224 x 4
