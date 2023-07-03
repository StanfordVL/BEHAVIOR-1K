import json
from pathlib import Path
import os
import shutil
import csv

import bddl
from bddl.object_taxonomy import ObjectTaxonomy
from bddl.activity import Conditions, get_all_activities, get_instance_count

from nltk.corpus import wordnet as wn


SUBSTANCE_CSV_FPATH = f"{os.path.dirname(bddl.__file__)}/generated_data/substance_hyperparams.csv"
DATASET_PATH = "/scr/chengshu/iGibson3/omnigibson/data/og_dataset"

object_taxonomy = ObjectTaxonomy()

def canonicalize(s):
    try:
        new_s = wn.synset(s).name()
        assert new_s == s, (new_s, s)
        return new_s
    except:
        return s

rows = []
with open(SUBSTANCE_CSV_FPATH) as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"')
    for row in reader:
        rows.append(row)

annotated_systems = set()
for system_name, metadata_str in rows:
    annotated_systems.add(system_name)

task_relevant_synsets = set()
tasks = [(act, inst) for act in sorted(get_all_activities()) for inst in range(get_instance_count(act))]
for act, inst in tasks:
    # Load task definition
    conds = Conditions(act, inst, "omnigibson")
    synsets = set(synset for synset in conds.parsed_objects if synset != "agent.n.01")
    canonicalized_synsets = set(canonicalize(synset) for synset in synsets)
    task_relevant_synsets |= canonicalized_synsets

substance_synsets = set()
for node in object_taxonomy.taxonomy.nodes:
    if object_taxonomy.has_ability(node, "substance") and object_taxonomy.is_leaf(node):
        substance_synsets.add(node)

task_relevant_substance_synsets = task_relevant_synsets & substance_synsets

task_relevant_substance_categories = set()
for synset in task_relevant_substance_synsets:
    assert len(object_taxonomy.get_categories(synset)) == 1
    task_relevant_substance_categories.add(object_taxonomy.get_categories(synset)[0])

missing_annotations = task_relevant_substance_categories - annotated_systems

for missing_category in missing_annotations:
    assert missing_category.startswith("cooked__"), f"only cooked__xyz is allowed to miss substance hyperparam annotation, got {missing_category}"
    assert missing_category.replace("cooked__", "") in annotated_systems, f"the non-cooked version of {missing_category} should be annotated"

system_root_dir = f"{DATASET_PATH}/systems"
for system_name, metadata_str in rows:
    metadata = json.loads(metadata_str)

    old_system_dir = f"{DATASET_PATH}/objects/{system_name}"
    system_dir = f"{system_root_dir}/{system_name}"

    # Move files if they exist
    if os.path.exists(old_system_dir):
        shutil.move(old_system_dir, system_root_dir)
    else:
        Path(system_dir).mkdir(parents=True, exist_ok=True)

    metadata_fpath = f"{system_dir}/metadata.json"
    with open(metadata_fpath, "w+") as f:
        json.dump(metadata, f)

    coooked_system_name = "cooked__" + system_name
    if coooked_system_name in missing_annotations:
        system_dir = f"{system_root_dir}/{coooked_system_name}"
        if os.path.exists(old_system_dir):
            shutil.move(old_system_dir, system_root_dir)
        else:
            Path(system_dir).mkdir(parents=True, exist_ok=True)

        metadata_fpath = f"{system_dir}/metadata.json"
        with open(metadata_fpath, "w+") as f:
            json.dump(metadata, f)
