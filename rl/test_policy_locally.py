import argparse
from datetime import datetime
import math
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import yaml
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
from omnigibson.sensors.scan_sensor import ScanSensor
from omnigibson.sensors.vision_sensor import VisionSensor
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
import h5py

from PIL import Image
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


def step_sim(time):
    for _ in range(int(time*100)):
        og.sim.step()

def main():
    gm.USE_GPU_DYNAMICS = True
    cfg = yaml.load(open("./service/omni_grpc.yaml", "r"), Loader=yaml.FullLoader)
    env = og.Environment(configs=cfg)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=5)
    # breakpoint()
    model = PPO.load("_480000_steps.zip")

    evaluate_policy(model, env, n_eval_episodes=10)
    
    # while True:
    #     done = False
    #     obs = env.reset()
    #     print("reset")
    #     print("------------------------------------")
    #     while not done:
    #         action = model.predict(obs, deterministic=True)[0]
    #         # print(action)
    #         # breakpoint()
    #         obs, reward, done, info = env.step(action)
    #         print(reward)
    #         done = done[0]


if __name__ == "__main__":
    main()