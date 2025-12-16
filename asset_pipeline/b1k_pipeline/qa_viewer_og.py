import logging
import multiprocessing
import os
import json
import sys
import nltk
import numpy as np
import csv
import networkx as nx
import sys

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import wordnet as wn

from bddl.object_taxonomy import ObjectTaxonomy
from b1k_pipeline.utils import PIPELINE_ROOT


# Mapping from object model name (e.g. apple-xyzabc) to the source 3dsmax file.
INVENTORY_PATH = PIPELINE_ROOT / "artifacts/pipeline/object_inventory.json"
with open(INVENTORY_PATH, "r") as f:
    INVENTORY_DICT = json.load(f)["providers"]

# Mapping from object category to synset (WordNet + custom)
CATEGORY_TO_SYNSET = {}
with open(PIPELINE_ROOT / "metadata/category_mapping.csv", "r") as f:
    r = csv.DictReader(f)
    for row in r:
        CATEGORY_TO_SYNSET[row["category"].strip()] = row["synset"].strip()

SYNSET_TO_PROPERTY = {}
with open(PIPELINE_ROOT / "metadata/synset_property.csv", "r") as f:
    r = csv.DictReader(f)
    for row in r:
        SYNSET_TO_PROPERTY[row["synset"].strip()] = row


def sanity_check_category_synset():
    modifiers = [f"{state}__" for state in ["cooked", "half", "diced"]]
    known_exceptions = {"coconut.n.01"}

    for category, synset in CATEGORY_TO_SYNSET.items():
        assert category != ""
        assert synset != ""
        assert (
            synset in SYNSET_TO_PROPERTY
        ), f"synset {synset} not in synset_property.csv"

    for synset, properties in SYNSET_TO_PROPERTY.items():
        assert synset != ""
        try:
            wn.synset(synset)
            in_wn = True
        except:
            in_wn = False

        synset_splits = synset.split(".")
        assert (
            synset_splits[1] == "n" and len(synset_splits[2]) == 2
        ), f"Invalid synset {synset}."
        is_custom_bool = bool(int(properties["is_custom"]))
        if in_wn:
            assert (
                not is_custom_bool
            ), f"synset {synset} is in WordNet but marked as custom."
        else:
            assert (
                is_custom_bool
            ), f"synset {synset} is not in WordNet but not marked as custom."

        if in_wn:
            wn_synset = wn.synset(synset)
            expected_hypernyms = ",".join([h.name() for h in wn_synset.hypernyms()])
            assert (
                expected_hypernyms == properties["hypernyms"]
            ), f"synset {synset} has hypernyms {properties['hypernyms']} but expected {expected_hypernyms}."

        for hypernym in properties["hypernyms"].split(","):
            try:
                wn.synset(hypernym)
            except:
                assert (
                    False
                ), f"hypernym {hypernym} of synset {synset} is not in WordNet."

        for modifier in modifiers:
            if modifier in synset:
                synset_parents = set(properties["hypernyms"].split(","))

                # cooked__batter.n.01 -> batter.n.01
                base_synset = synset
                for modifier in modifiers:
                    base_synset = base_synset.replace(modifier, "")

                # cooked__batter.n.01's base synset should be the smallest index of all synsets that start with batter.n. (batter.n.04, batter.n.08)
                index = int(base_synset[-2:]) - 1

                synset_candidates = sorted(
                    [
                        key
                        for key in SYNSET_TO_PROPERTY
                        if key.startswith(base_synset[:-2])
                    ]
                )
                if not index < len(synset_candidates):
                    print(
                        f"base synset of synset {synset} not found in synset_property.csv"
                    )
                    break

                base_synset = synset_candidates[index]
                base_synset_parents = set(
                    SYNSET_TO_PROPERTY[base_synset]["hypernyms"].split(",")
                )

                # The cooked/diced/half version should share the same parent as the base version.
                assert (
                    base_synset_parents == synset_parents
                    or base_synset in known_exceptions
                ), f"base synset {base_synset} of synset {synset} has parent {base_synset_parents} but synset parent is {synset_parents}"

                break

    taxonomy = nx.DiGraph()
    for synset, properties in SYNSET_TO_PROPERTY.items():
        first_layer = True
        current_synsets = [synset]
        while len(current_synsets) > 0:
            next_synsets = []
            for current_synset in current_synsets:
                if first_layer:
                    parents = [
                        wn.synset(hypernym)
                        for hypernym in properties["hypernyms"].split(",")
                    ]
                else:
                    parents = current_synset.hypernyms()
                for parent in parents:
                    taxonomy.add_edge(
                        parent.name(),
                        current_synset if first_layer else current_synset.name(),
                    )
                next_synsets.extend(parents)

            current_synsets = next_synsets
            first_layer = False

    for synset in SYNSET_TO_PROPERTY:
        assert (
            taxonomy.out_degree(synset) == 0
        ), f"synset {synset} has out degree {taxonomy.out_degree(synset)} (non-leaf)"

    for synset, properties in SYNSET_TO_PROPERTY.items():
        if properties["is_custom"] == "1" and "__" in synset:
            if ("__of__" in synset and "heap__of__" not in synset) or (
                "ed__" in synset
                and "cooked__" not in synset
                and "diced__" not in synset
                and "melted" not in synset
                and "_seed__" not in synset
                and "_feed__" not in synset
                and "chopped__" not in synset
                and "sliced__" not in synset
            ):
                assert (
                    properties["hypernyms"]
                    in [
                        "grocery.n.02",
                        "durables.n.01",
                        "waste.n.01",
                    ]
                ), f"Custom synset {synset} with __of__xyz or ed__xyz should have hypernym grocery.n.02"
                assert (
                    properties["fillable"] == "0"
                ), f"Custom synset {synset} with __of__xyz or ed__xyz should not be fillable"
            else:
                tokens = synset.split("__")
                container = tokens[-1]
                fillable = None
                if container in ["bottle.n.01", "jar.n.01", "bowl.n.01"]:
                    fillable = True
                    assert (
                        properties["hypernyms"] == "vessel.n.03"
                    ), f"Custom synset {synset} with __bottle/jar/bowl.n.01 should have hypernym vessel.n.03"
                elif container in ["carton.n.01"]:
                    fillable = True
                    assert (
                        properties["hypernyms"] == "box.n.01"
                    ), f"Custom synset {synset} with __carton.n.01 should have hypernym box.n.01"
                elif container in ["sack.n.01"]:
                    fillable = True
                    assert (
                        properties["hypernyms"] == "bag.n.01"
                    ), f"Custom synset {synset} with __sack.n.01 should have hypernym bag.n.01"
                elif container in [
                    "bag.n.01",
                    "box.n.01",
                    "tin.n.01",
                    "cup.n.01",
                    "package.n.01",
                ]:
                    fillable = True
                    assert (
                        properties["hypernyms"] == "container.n.01"
                    ), f"Custom synset {synset} with __bag/box/tin/cup/package.n.01 should have hypernym container.n.01"
                elif container in ["can.n.01"]:
                    assert (
                        properties["hypernyms"]
                        in [
                            "container.n.01",
                            "dispenser.n.01",
                        ]
                    ), f"Custom synset {synset} with __can.n.01 should have hypernym container.n.01 or dispenser.n.01"
                    fillable = (
                        properties["hypernyms"] == "container.n.01"
                    )  # tin can versus spray can
                elif container in ["atomizer.n.01"]:
                    fillable = False
                    assert (
                        properties["hypernyms"] == "dispenser.n.01"
                    ), f"Custom synset {synset} with __atomizer.n.01 should have hypernym dispenser.n.01"
                elif container in ["dispenser.n.01", "shaker.n.01"]:
                    fillable = False
                    assert (
                        properties["hypernyms"] == "container.n.01"
                    ), f"Custom synset {synset} with __dispenser/shaker.n.01 should have hypernym container.n.01"
                elif tokens[0] in [
                    "cooked",
                    "diced",
                    "sliced",
                    "half",
                    "heap",
                    "melted",
                    "chopped",
                    "pottable",
                    "bagged",
                    "broken",
                ]:
                    pass
                else:
                    assert False, f"Unexpected custom synset {synset}."

                if fillable is not None:
                    if fillable:
                        assert (
                            properties["fillable"] == "1"
                        ), f"Custom synset {synset} with __xyz should be fillable"
                        assert (
                            properties["particleApplier"] == "0"
                        ), f"Custom synset {synset} with __xyz should not be particleApplier"
                    else:
                        assert (
                            properties["fillable"] == "0"
                        ), f"Custom synset {synset} with __xyz should not be fillable"
                        assert (
                            properties["particleApplier"] == "1"
                        ), f"Custom synset {synset} with __xyz should be particleApplier"

                    assert (
                        properties["particleSource"] == "0"
                    ), f"Custom synset {synset} with __xyz should not be particleSource"


def main(dataset_path, record_path):
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import (
        get_all_object_categories,
        get_all_object_category_models,
    )
    from omnigibson.objects.dataset_object import DatasetObject
    import omnigibson.utils.transform_utils as T

    gm.DATASET_PATH = dataset_path

    sanity_check_category_synset()

    processed_objs = set()
    if os.path.exists(record_path):
        # Load the processed objects from the pass record file
        with open(record_path) as f:
            processed_objs = {tuple(obj) for obj in json.load(f)}

    # Get all the objects in the dataset
    all_objs = {
        (cat, model)
        for cat in get_all_object_categories()
        for model in get_all_object_category_models(cat)
    }

    # Compute the remaining objects to be processed
    remaining_objs = all_objs - processed_objs
    print(f"{len(remaining_objs)} objects remaining out of {len(all_objs)}.")

    # Create the scene config to load -- empty scene
    cfg = {"scene": {"type": "Scene"}}

    # Create the environment
    env = og.Environment(configs=cfg)

    # Make it brighter
    dome_light = og.sim.scene.skybox
    dome_light.intensity = 0.5e4

    for i, (obj_category, obj_model) in enumerate(sorted(remaining_objs)):
        print(f"Object {i + 1}/{len(remaining_objs)}: {obj_category}-{obj_model}.")

        og.sim.stop()

        obj = DatasetObject(
            name=obj_model,
            category=obj_category,
            model=obj_model,
            position=[0, 0, 10.0],
            fixed_base=True,
        )

        og.sim.import_object(obj)
        # Make sure the object fits in a box of unit size 1
        obj.scale = (np.ones(3) / obj.aabb_extent).min()
        og.sim.play()

        og.sim.viewer_camera.set_position_orientation(
            position=np.array([-0.00913503, -1.95750906, 1.36407314]),
            orientation=np.array([0.6350064, 0.0, 0.0, 0.77250687]),
        )
        og.sim.enable_viewer_camera_teleoperation()

        # Get all the questions
        messages = get_questions(obj)

        # Launch the complaint thread
        complaint_process = multiprocessing.Process(
            target=process_object_complaints,
            args=[obj.category, obj.name, messages, record_path, sys.stdin.fileno()],
            daemon=True,
        )
        complaint_process.start()

        # Keep simulating until the thread is done
        while complaint_process.is_alive():
            target_pos = np.array([0.0, 0.0, 1.0])

            obj.set_orientation([0, 0, 0, 1])
            center_offset = obj.get_position() - obj.aabb_center

            steps = 600
            steps_per_joint = steps // 6
            for i in range(steps):
                quat = T.euler2quat([0, 0, 2 * np.pi * i / steps])
                pos = T.quat2mat(quat) @ center_offset + target_pos
                obj.set_position_orientation(pos, quat)
                if obj.n_dof > 0:
                    frac = (i % steps_per_joint) / steps_per_joint
                    j_frac = (
                        -1.0 + 2.0 * frac
                        if (i // steps_per_joint) % 2 == 0
                        else 1.0 - 2.0 * frac
                    )
                    obj.set_joint_positions(
                        positions=j_frac * np.ones(obj.n_dof),
                        normalized=True,
                        drive=False,
                    )
                og.sim.step()
                og.sim.render()

        # Join the finished thread
        complaint_process.join()
        assert complaint_process.exitcode == 0, "Complaint process exited."

        # Remove the object and move on
        og.sim.remove_object(obj)

    og.shutdown()


def get_questions(obj):
    messages = [
        user_complained_synset(obj),
        user_complained_bbox(obj),
        user_complained_appearance(obj),
        user_complained_collision(obj),
        user_complained_articulation(obj),
    ]

    _, properties = get_synset_and_properties(obj.category)
    if properties["objectType"] in ["rope", "cloth"]:
        messages.append(user_complained_cloth(obj))

    # messages.append(user_complained_metas(_obj))
    return messages


def process_object_complaints(category, model, messages, record_path, stdin_fileno):
    # Open stdin so that we can read from it
    sys.stdin = os.fdopen(stdin_fileno)

    for m in messages:
        process_complaint(m, category, model)

    processed_objs = set()
    if os.path.exists(record_path):
        with open(record_path, "r") as f:
            processed_objs = set(json.load(f))
    processed_objs.add((category, model))
    with open(record_path, "w") as f:
        json.dump(sorted(processed_objs), f)


def process_complaint(message, category, model):
    print(message)
    while True:
        response = input(
            "Do you think anything is wrong? Enter a complaint (hit enter if all's good): "
        )
        if response:
            obj_key = f"{category}-{model}"
            complaint = {
                "object": obj_key,
                "message": message,
                "complaint": response,
                "processed": False,
            }
            target_name = INVENTORY_DICT[obj_key]

            complaints_file = PIPELINE_ROOT / "cad" / target_name / "complaints.json"
            complaints = []
            if os.path.exists(complaints_file):
                with open(complaints_file, "r") as f:
                    complaints = json.load(f)
            complaints.append(complaint)
            with open(complaints_file, "w") as f:
                json.dump(complaints, f, indent=4)
        break


def get_synset_and_properties(category):
    assert category in CATEGORY_TO_SYNSET
    synset = CATEGORY_TO_SYNSET[category]
    assert synset in SYNSET_TO_PROPERTY
    return synset, SYNSET_TO_PROPERTY[synset]


def get_synset_and_definition(category):
    synset, properties = get_synset_and_properties(category)
    if bool(int(properties["is_custom"])):
        s = wn.synset(properties["hypernyms"])
        return f"{synset} (custom synset)", f"(hypernyms: {s.name()}): {s.definition()}"
    else:
        s = wn.synset(synset)
        return s.name(), s.definition()


def user_complained_synset(obj):
    # Get the synset assigned to the object
    synset, definition = get_synset_and_definition(obj.category)

    # Print the synset name and definition
    message = (
        "Confirm object synset assignment.\n"
        f"Object assigned to category: {obj.category}\n"
        f"Object assigned to synset: {synset}\n"
        f"Definition: {definition}\n"
        "Reminder: synset should match the object and not its contents.\n"
        "(e.g. orange juice bottle needs to match orange_juice__bottle.n.01\n"
        "and not orange_juice.n.01)\n"
        "If the object category is wrong, please add this object to the Object Rename tab.\n"
        "If the object synset is empty or wrong, please modify the Object Category Mapping tab."
    )
    return message


def user_complained_bbox(obj):
    original_bounding_box = obj.aabb_extent / obj.scale
    message = (
        "Confirm reasonable bounding box size (in meters):\n"
        f"{', '.join([str(item) for item in original_bounding_box])}\n"
        "Make sure these sizes are within the same order of magnitude you expect from this object in real life.\n"
        "Press Enter if the size is good. Otherwise, enter the scaling factor you want to apply to the object.\n"
        "2 means the object should be scaled 2x larger and 0.5 means the object should be shrinked to half."
    )
    return message


def user_complained_appearance(obj):
    message = (
        "Confirm object visual appearance.\n"
        "Requirements:\n"
        "- make sure there is only one rigid body (e.g. one shoe instead of a pair of shoes).\n"
        "- make sure the object has a valid texture or appearance (e.g. texture not black, transparency rendered correctly, etc).\n"
        "- make sure the object has all parts necessary."
    )
    return message


def user_complained_collision(obj):
    message = (
        "Confirm object collision meshes.\n"
        "Requirements:\n"
        "- make sure the collision meshes well approximate the original visual meshes\n"
        "- make sure the collision meshes don't lose any affordance (e.g. holes and handles are preserved)."
    )
    return message


def user_complained_articulation(obj):
    message = f"Confirm articulation:\n"
    message += "This object has the below movable links annotated:\n"
    for j_name, j in obj.joints.items():
        message += f"- {j_name}, {j.joint_type}\n"
    message += (
        "Verify that these are all the moving parts you expect from this object\n"
    )
    message += "and that the joint limits look reasonable."
    return message


def user_complained_cloth(obj):
    message = "Confirm the default state of the rope/cloth object is unfolded."
    return message


# TODO: after metalinks are added to the USD with the final format, add a check for them here.
def user_complained_metas(obj):
    meta_links = sorted(
        {
            meta_name
            for link_metas in obj.metadata["meta_links"].values()
            for meta_name in link_metas
        }
    )
    message = f"Confirm object meta links listed below:\n"
    for meta_link in meta_links:
        message += f"- {meta_link}\n"
    message += "\nMake sure these match mechanisms you expect from this object."
    return message


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print(
            "Usage: python -m b1k_pipeline.qa_viewer dataset_path record_file_path.json"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
