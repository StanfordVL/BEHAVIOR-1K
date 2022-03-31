#!/usr/bin/env python

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import igibson
from igibson import app, ig_dataset_path
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.simulator_omni import Simulator
from igibson.utils.assets_utils import get_ig_assets_version
from igibson.utils.utils import parse_config


def benchmark_scene(sim, scene_name, optimized=False, import_robot=True):
    assets_version = get_ig_assets_version()
    print("assets_version", assets_version)
    scene_path = f"{ig_dataset_path}/scenes/{scene_name}/urdf/{scene_name}_best_template.usd"
    scene = InteractiveTraversableScene(scene_name, usd_path=scene_path, texture_randomization=False, object_randomization=False)
    start = time.time()
    sim.import_scene(scene)
    sim.play()
    print(time.time() - start)

    if import_robot:
        turtlebot = Turtlebot(prim_path="/World/robot", name="agent")
        sim.import_object(turtlebot, auto_initialize=True)
        sim.step()

    fps = []
    physics_fps = []
    render_fps = []
    obj_awake = []
    for i in range(2000):
        # if i % 100 == 0:
        #     scene.randomize_texture()
        start = time.time()
        if import_robot:
            # apply random actions
            turtlebot.apply_action(turtlebot.action_space.sample())
        sim.step(render=False)
        physics_end = time.time()
        
        # TODO: We need a way to measure rendering time. Currently omni already renders when we call s.step()
        # if import_robot:
        #     _ = s.renderer.render_robot_cameras(modes=("rgb"))
        # else:
        #     _ = s.renderer.render(modes=("rgb"))
        sim.render()
        end = time.time()

        # print("Elapsed time: ", end - start)
        # print("Render Frequency: ", 1 / (end - physics_end))
        print("Physics Frequency: ", 1 / (physics_end - start))
        print("Step Frequency: ", 1 / (end - start))
        fps.append(1 / (end - start))
        physics_fps.append(1 / (physics_end - start))
        # render_fps.append(1 / (end - physics_end))
        # obj_awake.append(sim.body_links_awake)
    plt.figure(figsize=(7, 25))

    ax = plt.subplot(6, 1, 1)
    plt.hist(render_fps)
    ax.set_xlabel("Render fps")
    ax.set_title(
        "Scene {} version {}\noptimized {} num_obj {}\n import_robot {}".format(
            scene_name, assets_version, optimized, scene.get_num_objects(), import_robot
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
    ax.set_xlabel("Render fps with time")
    ax.set_ylabel("fps")
    ax = plt.subplot(6, 1, 5)
    plt.plot(physics_fps)
    ax.set_xlabel("Physics fps with time, converge to {}".format(np.mean(physics_fps[-100:])))
    ax.set_ylabel("fps")

    # TODO! Objs awake not implemented yet in simulator_omni.py
    # ax = plt.subplot(6, 1, 6)
    # plt.plot(obj_awake)
    # ax.set_xlabel("Num object links awake, converge to {}".format(np.mean(obj_awake[-100:])))

    plt.savefig("scene_benchmark_{}_o_{}_r_{}.pdf".format(scene_name, optimized, import_robot))


def main():
    sim = Simulator(
        viewer_width=512,
        viewer_height=512,
    )
    benchmark_scene(sim, "Rs_int", optimized=True, import_robot=True)
    sim.stop()
    benchmark_scene(sim, "Rs_int", optimized=True, import_robot=False)

    # scenes = ["Beechwood_0_int",
    #           "Beechwood_1_int",
    #           "Benevolence_0_int",
    #           "Benevolence_1_int",
    #           "Benevolence_2_int",
    #           "Ihlen_0_int",
    #           "Ihlen_1_int",
    #           "Merom_0_int",
    #           "Merom_1_int",
    #           "Pomaria_0_int",
    #           "Pomaria_1_int",
    #           "Pomaria_2_int",
    #           "Rs_int",
    #           "Wainscott_0_int",
    #           "Wainscott_1_int"]

    # for scene in scenes:
    #     benchmark_scene(scene, True)

    app.close()


if __name__ == "__main__":
    main()
