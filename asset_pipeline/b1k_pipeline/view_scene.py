import itertools
import sys


import json
import pathlib
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
import tqdm

from b1k_pipeline.utils import PIPELINE_ROOT

import igibson

igibson.ignore_visual_shape = False
igibson.ig_dataset_path = r"/scr/dataset-10-6"

from igibson.objects.articulated_object import URDFObject
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.external.pybullet_tools import utils


def get_cube(limits=None):
    """get the vertices, edges, and faces of a cuboid defined by its limits

    limits = np.array([[x_min, x_max],
                       [y_min, y_max],
                       [z_min, z_max]])
    """
    v = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    )

    if limits is not None:
        v = limits[np.arange(3)[np.newaxis, :].repeat(8, axis=0), v]

    e = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ],
        dtype=int,
    )

    f = np.array(
        [
            [0, 2, 3, 1],
            [0, 4, 5, 1],
            [0, 4, 6, 2],
            [1, 5, 7, 3],
            [2, 6, 7, 3],
            [4, 6, 7, 5],
        ],
        dtype=int,
    )

    return v, e, f


def main():
    scene_name = sys.argv[1]
    # target = "scenes/" + scene_name
    # input_dir = PIPELINE_ROOT / "artifacts" / "aggregate" / target
    # scene_filename = input_dir / f"urdf/{scene_name}_best.urdf"

    # Load the scene into iGibson 2
    s = Simulator(mode="headless", use_pb_gui=True)
    try:
        scene = InteractiveIndoorScene(
            scene_name, not_load_object_categories=["ceilings"]
        )
        s.import_scene(scene)

        # Get the points this object thinks are its bounding box
        # doors: List[URDFObject] = list(scene.objects_by_category["door"])
        # colors = list(itertools.product([1, 0], [1, 0], [1, 0]))
        # for door, color in zip(doors, colors):
        #     print(door.model_path, list(color))
        #     pos, orn = door.get_base_link_position_orientation()
        #     rotated_offset = p.multiplyTransforms([0, 0, 0], orn, door.scaled_bbxc_in_blf, [0, 0, 0, 1])[0]
        #     bbox_ctr = pos - rotated_offset
        #     bbox_min = bbox_ctr - door.bounding_box / 2
        #     bbox_max = bbox_ctr + door.bounding_box / 2
        #     limits = np.stack([bbox_min, bbox_max], axis=1)
        #     v, e, f = get_cube(limits)
        #     for v_from, v_to in e:
        #         pos_from = v[v_from]
        #         pos_to = v[v_to]
        #         p.addUserDebugLine(pos_from, pos_to, list(color), 0.1)

        print("Stepping")
        while True:
            s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
