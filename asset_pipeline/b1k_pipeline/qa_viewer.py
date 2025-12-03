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

import igibson.external.pybullet_tools.utils as pb_utils
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


INVENTORY_PATH = r"D:\object_inventory.json"
with open(INVENTORY_PATH, "r") as f:
    INVENTORY_DICT = json.load(f)["providers"]

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

    processed_objs = set()
    if os.path.exists(record_path):
        # Load the processed objects from the pass record file
        with open(record_path) as f:
            processed_objs = {tuple(obj) for obj in json.load(f)}

    # Get all the objects in the dataset
    all_objs = {
        (cat, model)
        for cat in get_all_object_categories()
        for model in get_object_models_of_category(cat)
    }

    # Compute the remaining objects to be processed
    remaining_objs = all_objs - processed_objs

    print(f"{len(remaining_objs)} objects remaining out of {len(all_objs)}.")

    for i, (obj_category, obj_model) in enumerate(sorted(remaining_objs)):
        sim = Simulator(mode="headless", use_pb_gui=True)
        sim.import_scene(EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1]))

        obj_name = "{}_{}".format(obj_category, obj_model)
        model_path = get_ig_model_path(obj_category, obj_model)
        filename = os.path.join(model_path, "urdf", f"{obj_model}.urdf")

        print(
            "\n\n-----------------------------------------------------------------------------"
        )
        print(f"Object {i + 1}/{len(remaining_objs)}: ")
        print(f"{obj_category}-{obj_model}")

        try:
            simulator_obj = URDFObject(
                filename,
                name=obj_name,
                category=obj_category,
                model_path=model_path,
                fixed_base=True,
            )

            sim.import_object(simulator_obj)
            z_pos = (
                simulator_obj.bounding_box[2] / 2 + simulator_obj.scaled_bbxc_in_blf[2]
            )
            simulator_obj.set_position([0.0, 0.0, z_pos])

            dist = 3 * np.max(simulator_obj.bounding_box)
            p.resetDebugVisualizerCamera(
                cameraDistance=dist,
                cameraYaw=30,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
            )

            user_complained_synset(simulator_obj)
            user_complained_appearance(simulator_obj)
            user_complained_bbox(simulator_obj)
            user_complained_softbody(simulator_obj)
            # user_complained_properties(simulator_obj)
            user_complained_metas(simulator_obj)
            user_complained_articulation(simulator_obj)

            with open(record_path, "w") as f:
                processed_objs.add((obj_category, obj_model))
                json.dump(sorted(processed_objs), f, indent=4)
        finally:
            sim.disconnect()
            shutil.rmtree(
                PIPELINE_ROOT / "artifacts/aggregate/scene_instances",
                ignore_errors=True,
            )


def process_complaint(message, simulator_obj):
    print("-----------------------")
    print(message)
    response = input(
        "Do you think anything is wrong? Enter a complaint (hit enter if all's good): "
    )
    if response:
        model = os.path.basename(simulator_obj.model_path)
        object_key = simulator_obj.category + "-" + model
        complaint = {
            "object": object_key,
            "message": message,
            "complaint": response,
            "processed": False,
        }
        target_name = INVENTORY_DICT[object_key]

        complaints_file = PIPELINE_ROOT / "cad" / target_name / "complaints.json"
        complaints = []
        if os.path.exists(complaints_file):
            with open(complaints_file, "r") as f:
                complaints = json.load(f)
        complaints.append(complaint)
        with open(complaints_file, "w") as f:
            json.dump(complaints, f, indent=4)


def get_synset(category):
    if category not in CATEGORY_TO_SYNSET:
        return "", ""

    synset_name = CATEGORY_TO_SYNSET[category]

    # Read the custom synsets from the CSV file
    custom_synsets = []
    with open(PIPELINE_ROOT / "metadata/synsets.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if synset_name == row[1]:
                return row[1] + " (custom synset)", row[2] + "(hypernyms): " + (
                    wn.synset(row[2])
                ).definition()
    try:
        synset = wn.synset(synset_name)
    except:
        return "", ""
    return synset.name(), synset.definition()


def user_complained_synset(simulator_obj):
    # Get the synset assigned to the object
    synset, definition = get_synset(simulator_obj.category)

    # Print the synset name and definition
    message = (
        "Confirm object synset assignment.\n"
        f"Object assigned to synset: {synset}\n"
        f"Definition: {definition}\n"
        "Reminder: synset should match the object and not its contents.\n"
        "(e.g. orange juice bottle needs to match bottled__orange_juice.n.01\n"
        "and not orange_juice.n.01)"
    )
    process_complaint(message, simulator_obj)


def user_complained_appearance(simulator_obj):
    message = (
        "Confirm object visual appearance.\n"
        "Requirements:\n"
        "- make sure there is only one rigid body.\n"
        "- make sure the object has a valid texture or appearance.\n"
        "- make sure the object has all parts necessary."
    )
    process_complaint(message, simulator_obj)


def user_complained_softbody(simulator_obj):
    message = (
        "Check if the object looks like it absolutely MUST be a soft body.\n"
        "Requirements:\n"
        "- type 'cloth' if you think the object can reasonably be simulated by 2D cloth.\n"
        "- type 'soft' if it needs to be soft but still needs to have a volume.\n"
        "- just hit enter without typing anything if the object does NOT have to be a soft body."
    )
    process_complaint(message, simulator_obj)


def user_complained_properties(simulator_obj):
    BAD_PROPERTIES = {"breakable", "timeSetable", "perishable", "screwable"}

    taxonomy = ObjectTaxonomy()
    synset = taxonomy.get_class_name_from_igibson_category(simulator_obj.category)
    try:
        abilities = taxonomy.get_abilities(synset)
    except:
        abilities = []
        print("synset not in taxonomy")

    all_abilities = sorted(
        {a for s in taxonomy.taxonomy.nodes for a in taxonomy.get_abilities(s).keys()}
        - BAD_PROPERTIES
    )
    message = "Confirm object properties:\n"
    for ability in abilities:
        if ability not in BAD_PROPERTIES:
            message += f"- {ability}\n"
    message += f"Full list of iGibson abilities: {', '.join(all_abilities)}\n"
    message += "Check incorrect or missing properties, especially liquid property\n"
    process_complaint(message, simulator_obj)


def user_complained_metas(simulator_obj):
    meta_links = sorted(
        {
            meta_name
            for link_metas in simulator_obj.metadata["meta_links"].values()
            for meta_name in link_metas
        }
    )
    message = f"Confirm object meta links listed below:\n"
    if meta_links:
        for meta_link in meta_links:
            message += f"- {meta_link}\n"
    else:
        message += f"- N/A\n"
    message += "\nMake sure these match mechanisms you expect from this object."
    process_complaint(message, simulator_obj)


def user_complained_bbox(simulator_obj):
    bounding_box = simulator_obj.bounding_box
    message = "Confirm reasonable bounding box size:\n"
    bb_items = []
    for k in range(3):
        size = bounding_box[k]
        size_m = size
        size_cm = size * 100
        size_mm = size * 1000
        if size_m > 1:
            bb_items.append("%.2fm" % size_m)
        elif size_cm > 1:
            bb_items.append("%.2fcm" % size_cm)
        else:
            bb_items.append("%.2fmm" % size_mm)
    message += ", ".join(bb_items) + "\n"
    message += "Make sure these sizes are within the same order of magnitude you expect from this object IRL.\n"
    message += "If the scale is off, please let us know by a factor of how much (for example, 1000 if something\n"
    message += "that's supposed to be 1mm is 1m instead) in your complaint note."
    # TODO: Print avg category specs too
    process_complaint(message, simulator_obj)


def user_complained_articulation(simulator_obj):
    message = f"Confirm articulation:\n"
    message += "This object has the below movable links annotated:\n"
    joint_ids = [
        (bid, joint)
        for bid in simulator_obj.get_body_ids()
        for joint in pb_utils.get_movable_joints(bid)
    ]
    if joint_ids:
        for bid, joint in joint_ids:
            name = pb_utils.get_link_name(bid, joint)
            _, upper = pb_utils.get_joint_limits(bid, joint)
            pb_utils.set_joint_position(bid, joint, upper)
            message += f"- {name}\n"
    else:
        message += f"- N/A\n"
    message += "\nThey have now all been set to their upper (maximum) limits.\n"
    message += (
        "Verify that these are all the moving parts you expect from this object\n"
    )
    message += "and that the joint limits look reasonable."
    process_complaint(message, simulator_obj)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print(
            "Usage: python -m b1k_pipeline.qa_viewer dataset_path record_file_path.json"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
