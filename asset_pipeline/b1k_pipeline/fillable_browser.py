import json
import pathlib
import sys
import glob
import os
import pybullet as p

import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm

import igibson

igibson.ig_dataset_path = "/scr/BEHAVIOR-1K/asset_pipeline/tmp/fedafr"

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.empty_scene import EmptyScene
from igibson.objects.articulated_object import URDFObject
from igibson.external.pybullet_tools import utils


def main():
    # Load the scene into iGibson 2
    s = Simulator(mode="headless", use_pb_gui=True, image_height=1080, image_width=1920)
    scene = EmptyScene()
    s.import_scene(scene)

    fillable_models = json.loads(
        pathlib.Path(
            "/scr/BEHAVIOR-1K/asset_pipeline/artifacts/pipeline/fillable_ids.json"
        ).read_text()
    )
    fillable_paths = sorted(
        [
            path
            for fm in fillable_models
            for path in glob.glob(
                os.path.join(
                    igibson.ig_dataset_path, "objects", "*", fm, "urdf", f"{fm}.urdf"
                )
            )
        ]
    )
    print("Found", len(fillable_paths), "fillable objects out of", len(fillable_models))

    START_ID = 0
    for i, path in enumerate(fillable_paths[START_ID : START_ID + 250]):
        ctr_x = i * 1.2
        metadata = json.loads(
            (pathlib.Path(path).parent / "misc/metadata.json").read_text()
        )
        native_bbox_size = np.array(metadata["bbox_size"])
        scale = 1.0 / np.max(native_bbox_size)
        obj = URDFObject(path, scale=scale * np.array([1.0, 1.0, 1.0]), fixed_base=True)
        s.import_object(obj)
        obj.set_bbox_center_position_orientation(
            np.array([ctr_x, 0, 1]), np.array([0, 0, 0, 1])
        )
        p.addUserDebugText(
            text=pathlib.Path(path).parts[-2],
            textPosition=[ctr_x - 0.5, 0, 1.75],
            textOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
            textSize=0.4,
        )
        print(i)

    # Step the simulation by 5 seconds.
    while True:
        s.step()


if __name__ == "__main__":
    main()
