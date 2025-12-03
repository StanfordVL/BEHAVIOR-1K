import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name, PIPELINE_ROOT

import json

import random
import string

import pymxs

rt = pymxs.runtime

objs_inv_path = PIPELINE_ROOT / "artifacts/pipeline/object_inventoty.json"
current_file = rt.maxFilePath.split("cad", 1)[1][1:-1]
current_file = current_file.replace("\\", "/")

# get error objects and objects in scene
with open(objs_inv_path, "r") as f:
    all_inv = json.load(f)

repeated_objs = []  # list of dictionaries, key is obj name, value is list of scenes where obj is repeated
for obj in all_inv["error_multiple_provided"]:
    repeated_objs.append(obj)

all_objs = {}  # dictionary of lists, key is obj base_name e.g. 'trophy-id', value is list of obj instances
for obj in rt.objects:
    n = parse_name(obj.name)
    if not n:  # assume objs have correct names
        continue
    name = obj.name[n.start("category") : n.end("model_id")]
    if name not in all_objs:
        all_objs[name] = []
    all_objs[name].append(obj)


# rename necessary objs
for obj_base_name in repeated_objs:
    # do not rename if obj not in scene or if file doesn't have error with this object
    if (
        obj_base_name not in all_objs
        or current_file not in all_inv["error_multiple_provided"][obj_base_name]
    ):
        continue
    first_token = obj_base_name.split("-")[0]
    obj_instances = all_objs[obj_base_name]
    if first_token == "walls" or first_token == "ceilings" or first_token == "floors":
        # rename id
        old_id = parse_name(obj_instances[0].name).group("model_id")
        new_id = "".join(random.choices(string.ascii_lowercase, k=6))
        # two instances - obj and manual cmesh
        for obj in obj_instances:
            obj.name = obj.name.replace(old_id, new_id)
    else:
        # only rename instances when not last file in repeated-list
        if current_file != all_inv["error_multiple_provided"][obj_base_name][-1]:
            for obj in obj_instances:
                assert not obj.name.startswith("B-"), obj.name
                obj.name = "B-" + obj.name
