#!/usr/bin/env python

import os
import time
import matplotlib.pyplot as plt
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots.turtlebot import Turtlebot
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.simulator import launch_simulator
from omnigibson.utils.asset_utils import get_og_assets_version


# Params to be set as needed.
SCENES = ["Rs_int"]
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
NUM_STEPS = 2000

gm.HEADLESS = False
gm.GUI_VIEWPORT_ONLY = True
gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False

# Launch the simulator
launch_simulator(physics_dt=1/60., rendering_dt=1/60.)


def benchmark_scene(scene_name, optimized=False, import_robot=True):
    assets_version = get_og_assets_version()
    print("assets_version", assets_version)
    scene = InteractiveTraversableScene(scene_name)
    start = time.time()
    og.sim.import_scene(scene)
    print(time.time() - start)

    if import_robot:
        turtlebot = Turtlebot(prim_path="/World/robot", name="agent", obs_modalities=['rgb'])
        og.sim.import_object(turtlebot)
        og.sim.step()
        if scene_name == "restaurant_hotel":
            turtlebot.set_position_orientation([-12.2627573, -10.60511875, -3.4790891], [0, 0, 0, 1.0])

    og.sim.play()
    og.sim.step()
    fps = []
    physics_fps = []
    render_fps = []
    obj_awake = []
    for i in range(NUM_STEPS):
        start = time.time()
        if import_robot:
            # Apply random actions.
            turtlebot.apply_action(turtlebot.action_space.sample())
        og.sim.step(render=False)
        physics_end = time.time()

        og.sim.render()
        end = time.time()

        if i % 100 == 0:
            print("Elapsed time: ", end - start)
            print("Render Frequency: ", 1 / (end - physics_end))
            print("Physics Frequency: ", 1 / (physics_end - start))
            print("Step Frequency: ", 1 / (end - start))
        fps.append(1 / (end - start))
        physics_fps.append(1 / (physics_end - start))
        render_fps.append(1 / (end - physics_end))
        # obj_awake.append(sim.body_links_awake)

    plt.figure(figsize=(7, 25))
    ax = plt.subplot(6, 1, 1)
    plt.hist(render_fps)
    ax.set_xlabel("Render fps")
    ax.set_title(
        "Scene {} version {}\noptimized {} num_obj {}\n import_robot {}".format(
            scene_name, assets_version, optimized, scene.n_objects, import_robot
        )
    )
    ax = plt.subplot(6, 1, 2)
    plt.hist(physics_fps)
    ax.set_xlabel("Physics fps")
    ax = plt.subplot(6, 1, 3)
    plt.hist(fps)
    ax.set_xlabel("Step fps")
    ax = plt.subplot(6, 1, 4)
    plt.plot(render_fps)
    ax.set_xlabel("Render fps with time, converge to {}".format(np.mean(render_fps[-100:])))
    ax.set_ylabel("fps")
    ax = plt.subplot(6, 1, 5)
    plt.plot(physics_fps)
    ax.set_xlabel("Physics fps with time, converge to {}".format(np.mean(physics_fps[-100:])))
    ax.set_ylabel("fps")
    ax = plt.subplot(6, 1, 6)
    plt.plot(fps)
    ax.set_xlabel("Overall fps with time, converge to {}".format(np.mean(fps[-100:])))
    ax.set_ylabel("fps")
    # TODO! Reading objects' wake state not available yet from omniverse
    # ax = plt.subplot(6, 1, 6)
    # plt.plot(obj_awake)
    # ax.set_xlabel("Num object links awake, converge to {}".format(np.mean(obj_awake[-100:])))
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR,
        "scene_benchmark_{}_o_{}_r_{}.pdf".format(scene_name, optimized, import_robot)))


def main():
    for scene in SCENES:
        benchmark_scene(scene, optimized=True, import_robot=True)

    og.shutdown()


if __name__ == "__main__":
    main()
