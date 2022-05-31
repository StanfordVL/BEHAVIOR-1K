#!/usr/bin/env python

import argparse
import os
import random

from igibson.scenes.interactive_traversable_scene import SCENE_SOURCE, InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_3dfront_scene_path, get_cubicasa_scene_path, get_ig_scene_path
from igibson.utils.map_utils import gen_trav_map

"""
script to generate all traversability maps:

for file in ../../igibson/ig_dataset/scenes/*
  python generate_trav_map.py $(basename $file)

"""


def generate_trav_map(scene_name, scene_source, load_full_scene=True):
    if scene_source not in SCENE_SOURCE:
        raise ValueError("Unsupported scene source: {}".format(scene_source))
    if scene_source == "IG":
        scene_dir = get_ig_scene_path(scene_name)
    elif scene_source == "CUBICASA":
        scene_dir = get_cubicasa_scene_path(scene_name)
    else:
        scene_dir = get_3dfront_scene_path(scene_name)
    random.seed(0)
    scene = InteractiveIndoorScene(
        scene_name, build_graph=False, texture_randomization=False, scene_source=scene_source
    )
    if not load_full_scene:
        scene._set_first_n_objects(3)
    s = Simulator(mode="headless", image_width=512, image_height=512, device_idx=0)
    s.import_scene(scene)

    if load_full_scene:
        scene.open_all_doors()

    for i in range(20):
        s.step()

    vertices_info, faces_info = s.renderer.dump()
    s.disconnect()

    if load_full_scene:
        trav_map_filename_format = "floor_trav_{}.png"
        obstacle_map_filename_format = "floor_{}.png"
    else:
        trav_map_filename_format = "floor_trav_no_obj_{}.png"
        obstacle_map_filename_format = "floor_no_obj_{}.png"

    gen_trav_map(
        vertices_info,
        faces_info,
        output_folder=os.path.join(scene_dir, "layout"),
        trav_map_filename_format=trav_map_filename_format,
        obstacle_map_filename_format=obstacle_map_filename_format,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Traversability Map")
    parser.add_argument("scene_names", metavar="s", type=str, nargs="+", help="The name of the scene to process")
    parser.add_argument(
        "--source", dest="source", help="Source of the scene, should be among [CUBICASA, IG, THREEDFRONT]"
    )

    args = parser.parse_args()
    for scene_name in args.scene_names:
        generate_trav_map(scene_name, args.source, False)
        generate_trav_map(scene_name, args.source, True)


if __name__ == "__main__":
    main()
