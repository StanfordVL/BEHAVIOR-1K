#!/usr/bin/env python

import os
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np

import igibson
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_assets_version
from igibson.utils.constants import AVAILABLE_MODALITIES, NamedRenderingPresets
from igibson.utils.utils import parse_config


def benchmark_rendering(scene_list, rendering_presets_list, modality_list):
    config = parse_config(os.path.join(os.path.dirname(__file__), "..", "test.yaml"))
    assets_version = get_ig_assets_version()
    print("assets_version", assets_version)
    result = {}
    for scene_name in scene_list:
        for rendering_preset in rendering_presets_list:
            scene = InteractiveIndoorScene(scene_name, texture_randomization=False, object_randomization=False)
            settings = NamedRenderingPresets[rendering_preset]
            if rendering_preset == "VISUAL_RL":
                image_width = 128
                image_height = 128
            else:
                image_width = 512
                image_height = 512
            s = Simulator(
                mode="headless",
                image_width=image_width,
                image_height=image_height,
                device_idx=0,
                rendering_settings=settings,
                physics_timestep=1 / 240.0,
            )
            s.import_scene(scene)
            turtlebot = Turtlebot()
            s.import_object(turtlebot)

            for mode in modality_list:
                for _ in range(10):
                    s.step()
                    _ = s.renderer.render_robot_cameras(modes=(mode))
                start = time.time()
                for _ in range(200):
                    _ = s.renderer.render_robot_cameras(modes=(mode))
                end = time.time()
                fps = 200 / (end - start)
                result[(scene_name, rendering_preset, mode)] = fps
            s.disconnect()
    return result


def main():
    scenes = [
        "Beechwood_0_int",
        "Beechwood_1_int",
        "Benevolence_0_int",
        "Benevolence_1_int",
        "Benevolence_2_int",
        "Ihlen_0_int",
        "Ihlen_1_int",
        "Merom_0_int",
        "Merom_1_int",
        "Pomaria_0_int",
        "Pomaria_1_int",
        "Pomaria_2_int",
        "Rs_int",
        "Wainscott_0_int",
        "Wainscott_1_int",
    ]
    rendering_settings = ["VISUAL_RL", "PERCEPTION"]
    modalities = list(AVAILABLE_MODALITIES)

    result = benchmark_rendering(scenes, rendering_settings, modalities)

    aggregated_result = {}
    for rendering_setting in rendering_settings:
        for modality in modalities:
            all_scenes = []
            for item in result.keys():
                if item[1] == rendering_setting and item[2] == modality:
                    all_scenes.append(result[item])
            aggregated_result[("MEAN", rendering_setting, modality)] = np.mean(all_scenes)
            aggregated_result[("MAX", rendering_setting, modality)] = np.max(all_scenes)
            aggregated_result[("MIN", rendering_setting, modality)] = np.min(all_scenes)

    print(result)
    plt.figure(figsize=(5, 30))
    plt.tight_layout()
    plt.barh(["-".join(item) for item in result.keys()], result.values())
    for i, v in enumerate(result.values()):
        plt.text(v + 3, i, "{:.1f}".format(v), color="blue", fontweight="bold")
    plt.xlabel("fps")
    plt.savefig("benchmark_rendering.pdf", bbox_inches="tight")
    pkl.dump(result, open("rendering_benchmark_results.pkl", "wb"))

    plt.figure(figsize=(5, 30))
    plt.tight_layout()
    plt.barh(["-".join(item) for item in aggregated_result.keys()], aggregated_result.values())
    for i, v in enumerate(aggregated_result.values()):
        plt.text(v + 3, i, "{:.1f}".format(v), color="blue", fontweight="bold")
    plt.xlabel("fps")
    plt.savefig("benchmark_rendering_stats.pdf", bbox_inches="tight")
    pkl.dump(aggregated_result, open("rendering_benchmark_results_stats.pkl", "wb"))


if __name__ == "__main__":
    main()
