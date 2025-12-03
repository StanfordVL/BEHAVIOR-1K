import logging
import os
import json
import sys
import nltk
import numpy as np
import csv
import pybullet as p
import shutil

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import wordnet as wn

import igibson

from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import (
    get_all_object_categories,
    get_ig_avg_category_specs,
    get_ig_model_path,
    get_object_models_of_category,
)

from bddl.object_taxonomy import ObjectTaxonomy
from b1k_pipeline.utils import PIPELINE_ROOT

skip_file_path = PIPELINE_ROOT / "qa-logs/category-pass-skips.json"

# INVENTORY_PATH = PIPELINE_ROOT / "artifacts"/ "pipeline"/"object_inventory.json"
# with open(INVENTORY_PATH, "r") as f:
#     INVENTORY_DICT = json.load(f)["providers"]

CATEGORY_TO_SYNSET = {}
with open(PIPELINE_ROOT / "metadata/category_mapping.csv", "r") as f:
    r = csv.DictReader(f)
    for row in r:
        CATEGORY_TO_SYNSET[row["category"].strip()] = row["synset"].strip()


def main(dataset_path, record_path):
    """
    Minimal example to visualize all the models available in the iG dataset
    It queries the user to select an object category and a model of that category, loads it and visualizes it
    No physical simulation
    """
    igibson.ignore_visual_shape = False
    igibson.ig_dataset_path = dataset_path

    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    processed_cats = set()
    if os.path.exists(record_path):
        # Load the processed objects from the pass record file
        with open(record_path) as f:
            processed_cats = set(json.load(f))

    skipped_cats = set()
    if os.path.exists(skip_file_path):
        # Load the processed objects from the pass record file
        with open(skip_file_path) as f:
            skipped_cats = set(json.load(f))
    # Get all the objects in the dataset
    all_cats = set(get_all_object_categories())

    # Compute the remaining objects to be processed
    remaining_cats = all_cats - (processed_cats - skipped_cats)

    print(f"{len(remaining_cats)} cats remaining out of {len(all_cats)}.")

    for i, obj_category in enumerate(sorted(remaining_cats)):
        print(f"Cat {i + 1}/{len(remaining_cats)}: {obj_category}.")
        try:
            sim = Simulator(mode="headless", use_pb_gui=True)
            sim.import_scene(EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1]))
            for i, obj_model in enumerate(get_object_models_of_category(obj_category)):
                obj_name = "{}_{}".format(obj_category, obj_model)
                model_path = get_ig_model_path(obj_category, obj_model)
                filename = os.path.join(model_path, "urdf", obj_model + ".urdf")

                print(
                    "Visualizing category {}, model {}".format(obj_category, obj_model)
                )
                bbox = URDFObject(
                    filename,
                    name=obj_name,
                    category=obj_category,
                    model_path=model_path,
                    fixed_base=True,
                ).bounding_box

                # Get the right scale
                scale = np.min(1 / bbox)
                simulator_obj = URDFObject(
                    filename,
                    name=obj_name,
                    category=obj_category,
                    model_path=model_path,
                    fixed_base=True,
                    scale=np.array([scale, scale, scale]),
                )

                sim.import_object(simulator_obj)
                simulator_obj.set_bbox_center_position_orientation(
                    np.array([i + 0.5, 0.5, 0.5]), np.array([0, 0, 0, 1])
                )

            print("Synset info:", get_synset(obj_category))
            user_input = input("Hit enter to continue, 's' to skip")
            if user_input == "s":
                save_to_skip_file(obj_category, skip_file_path)

            with open(record_path, "w") as f:
                processed_cats.add(obj_category)
                json.dump(sorted(processed_cats), f)

        finally:
            sim.disconnect()
            shutil.rmtree(
                PIPELINE_ROOT / "artifacts/aggregate/scene_instances",
                ignore_errors=True,
            )


def save_to_skip_file(category, skip_file_path):
    skipped_categories = set()

    if os.path.exists(skip_file_path):
        with open(skip_file_path, "r") as f:
            skipped_categories = set(json.load(f))

    skipped_categories.add(category)

    with open(skip_file_path, "w") as f:
        json.dump(sorted(skipped_categories), f)


def get_synset(category):
    if category not in CATEGORY_TO_SYNSET:
        return "", ""

    synset_name = CATEGORY_TO_SYNSET[category]

    # Read the custom synsets from the CSV file
    custom_synsets = []
    with open(PIPELINE_ROOT / "metadata/synsets.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if synset_name == row[0].strip():
                return row[1] + " (custom synset)", row[2] + "(hypernyms): " + (
                    wn.synset(row[2])
                ).definition()
    try:
        synset = wn.synset(synset_name)
    except:
        return synset_name, "No definition found"
    return synset.name(), synset.definition()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print(
            "Usage: python -m b1k_pipeline.category_qa_viewer dataset_path record_file_path.json"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
