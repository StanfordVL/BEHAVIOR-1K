import csv
import hashlib
import pathlib
import traceback
import shutil
import os
from tqdm import tqdm
import sys
import json
import torch as th
from scipy.spatial.transform import Rotation as R
import glob

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.scenes import Scene
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


def convert_csv_to_dict(filepath):
    """
    Converts a 2-column CSV file into a key:value dictionary.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        dict: A dictionary created from the CSV data.
    """
    with open(filepath, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        # Use a dictionary comprehension for a clean and efficient conversion
        return {row[0]: row[1] for row in reader}


AI2_FIXEDNESS_MAPPING_FN = "/home/cgokmen/projects/BEHAVIOR-1K/slurm/ai2thor-fixedness.csv"
AI2_FIXEDNESS_MAPPING = {k: str(v).lower() == "true" for k, v in convert_csv_to_dict(AI2_FIXEDNESS_MAPPING_FN).items()}


def load_habitat_scene(dataset_name, scene_input_json):
    scene_input_json = pathlib.Path(scene_input_json)

    ROTATE_EVERYTHING_BY = th.as_tensor(R.from_euler("x", 90, degrees=True).as_quat())

    DATASET_ROOT = pathlib.Path(gm.DATA_PATH) / dataset_name
    object_mapping = json.loads((DATASET_ROOT / "object_name_mapping.json").read_text())

    # Load the scene JSON
    scene_contents = json.loads(scene_input_json.read_text())
    assert scene_contents["translation_origin"] == "asset_local"

    # Load all the objects manually into a scene
    scene = Scene(use_floor_plane=True, floor_plane_visible=False)
    og.sim.import_scene(scene)

    # Load the scene template. This is not an actual path but is namespaced as stages/iThor/etc and we just want the etc.
    stage_instance = os.path.basename(scene_contents["stage_instance"]["template_name"])
    stage_category, stage_model = object_mapping[stage_instance]
    tmpl = DatasetObject(
        name="stage", category=stage_category, model=stage_model, fixed_base=True, dataset_name=dataset_name
    )
    scene.add_object(tmpl)

    stage_orn = ROTATE_EVERYTHING_BY
    if "ProcTHOR" in stage_instance:
        # Additional -90 rotation around world Z+
        stage_orn = T.quat_multiply(T.euler2quat(th.tensor([0, 0, -th.pi / 2])), stage_orn)

    tmpl.set_position_orientation(position=th.zeros(3), orientation=stage_orn)

    for i, obj_instance in enumerate(scene_contents["object_instances"]):
        try:
            # This can similarly be prefixed, so we undo that.
            template_name = os.path.basename(obj_instance["template_name"])
            category, model = object_mapping[template_name]
            pos = th.as_tensor(obj_instance["translation"])
            orn = th.as_tensor(obj_instance["rotation"])[[1, 2, 3, 0]]
            scale = th.as_tensor(obj_instance["non_uniform_scale"])

            # Is the dataset ai2thor? If so, we don't trust its fixedness flag.
            if dataset_name == "ai2thor":
                fixed_base = not AI2_FIXEDNESS_MAPPING[category]
                print("Object", category, "fixedness", fixed_base)
            else:
                fixed_base = obj_instance["motion_type"] == "STATIC"

            obj = DatasetObject(
                name=f"{category}_{i}",
                category=category,
                model=model,
                scale=scale,
                fixed_base=fixed_base,
                dataset_name=dataset_name,
            )
        except:
            print("Skipping object", obj_instance)
            continue
        scene.add_object(obj)

        rotated_pos, rotated_orn = T.pose_transform(th.zeros(3), ROTATE_EVERYTHING_BY, pos, orn)
        obj.set_position_orientation(rotated_pos, rotated_orn)

    # Play the simulator, then save
    og.sim.play()

    # Take a sim step
    for _ in range(500):
        og.sim.step()


if __name__ == "__main__":
    if og.sim:
        og.clear()
    else:
        og.launch()

    load_habitat_scene("ai2thor", "/home/cgokmen/Downloads/ProcTHOR-Train-1001.scene_instance.json")

    while True:
        og.sim.step()
