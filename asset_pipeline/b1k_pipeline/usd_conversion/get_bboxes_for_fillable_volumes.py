import hashlib
import os
import re
import random
import sys
import glob
import pathlib
import numpy as np
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.macros import gm
from omnigibson.prims import XFormPrim
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
import trimesh
import json
import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import torch as th
import networkx as nx
from collections import defaultdict
from fs.zipfs import ZipFS
import fs.path

gm.HEADLESS = False
gm.USE_ENCRYPTED_ASSETS = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = False
gm.DATASET_PATH = r"D:\fillable-10-21"

BATCH_SIZE = 100

META_PATTERN = re.compile(
    "(particlesink|lights|particlesource|togglebutton|heatsource|attachment)_.*_.*_link"
)


def view_object(cat, mdl):
    if og.sim:
        og.clear()
    else:
        og.launch()

    if og.sim.is_playing():
        og.sim.stop()

    orn = [0, 0, 0, 1]

    cfg = {
        "scene": {
            "type": "Scene",
            "use_floor_plane": False,
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "fillable",
                "category": cat,
                "model": mdl,
                "orientation": orn,
                "kinematic_only": False,
                "fixed_base": True,
            },
        ],
    }

    env = og.Environment(configs=cfg)
    og.sim.step()

    fillable = env.scene.object_registry("name", "fillable")
    fillable.set_position([0, 0, fillable.aabb_extent[2]])
    og.sim.step()

    points_world = [
        link.visual_boundary_points_world
        for link in fillable._links.values()
        if not META_PATTERN.fullmatch(link.prim_path.split("/")[-1])
    ]
    all_points = th.cat([p for p in points_world if p is not None], dim=0)
    aabb_lo = th.min(all_points, dim=0).values
    aabb_hi = th.max(all_points, dim=0).values

    bbox_extents = (aabb_hi - aabb_lo).tolist()
    bbox_center = ((aabb_hi + aabb_lo) / 2).tolist()
    base_pos = fillable.get_position_orientation()[0].numpy().tolist()
    base_orn = fillable.get_position_orientation()[1].numpy().tolist()

    data = {
        "bbox_center": bbox_center,
        "bbox_extents": bbox_extents,
        "base_pos": base_pos,
        "base_orn": base_orn,
    }
    target_path = (
        pathlib.Path(DatasetObject.get_usd_path(cat, mdl)).parent.parent / "bbox.json"
    )
    with open(target_path, "w") as f:
        json.dump(data, f)


def main():
    print("Fillable annotator version 11.6.0")

    fillables = [
        tuple(pathlib.Path(x).parts[-4:-2])
        for x in glob.glob(f"{gm.DATASET_PATH}/objects/*/*/usd/*.usd")
        if not (pathlib.Path(x).parent.parent / "bbox.json").exists()
    ]
    random.shuffle(fillables)

    for cat, mdl in tqdm.tqdm(fillables[:BATCH_SIZE]):
        if not os.path.exists(
            DatasetObject.get_usd_path(cat, mdl).replace(".usd", ".encrypted.usd")
        ):
            print(f"Skipping {cat}/{mdl} because it does not exist")
            continue
        view_object(cat, mdl)

    if len(fillables) == 0:
        with open(os.path.join(gm.DATASET_PATH, "done.txt"), "w") as f:
            pass


if __name__ == "__main__":
    main()
