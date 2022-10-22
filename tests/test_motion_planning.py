import os

import matplotlib.pyplot as plt
import numpy as np

import omnigibson
from omnigibson.envs.omnigibson_env import OmniGibsonEnv
from omnigibson.utils.asset_utils import download_assets, download_demo_data
from omnigibson.utils.motion_planning_wrapper import MotionPlanningWrapper


def test_occupancy_grid():
    print("Test env")
    download_assets()
    download_demo_data()
    config_filename = os.path.join(os.path.dirname(__file__), "test_house_occupancy_grid.yaml")

    nav_env = OmniGibsonEnv(config_file=config_filename, mode="headless")
    nav_env.reset()
    nav_env.robots[0].set_position_orientation([0, 0, 0], [0, 0, 0, 1])
    nav_env.simulator.step()

    action = nav_env.action_space.sample()
    ts = nav_env.step(action)
    assert np.sum(ts[0]["occupancy_grid"] == 0) > 0
    assert np.sum(ts[0]["occupancy_grid"] == 1) > 0
    plt.imshow(ts[0]["occupancy_grid"][:, :, 0])
    plt.colorbar()
    plt.savefig(os.path.join(os.path.dirname(__file__), "occupancy_grid.png"))
    nav_env.clean()


def test_base_planning():
    print("Test env")
    download_assets()
    download_demo_data()
    config_filename = os.path.join(os.path.dirname(__file__), "test_house_occupancy_grid.yaml")

    nav_env = OmniGibsonEnv(config_file=config_filename, mode="headless")
    motion_planner = MotionPlanningWrapper(nav_env)
    nav_env.reset()
    nav_env.robots[0].set_position_orientation([0, 0, 0], [0, 0, 0, 1])
    nav_env.simulator.step()
    plan = None
    itr = 0
    while plan is None and itr < 10:
        plan = motion_planner.plan_base_motion([0.5, 0, 0])
        print(plan)
        itr += 1
    motion_planner.dry_run_base_plan(plan)

    assert len(plan) > 0
    nav_env.clean()
