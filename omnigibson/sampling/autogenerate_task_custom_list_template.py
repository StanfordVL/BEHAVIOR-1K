from bddl.activity import (
    Conditions,
    get_object_scope,
    get_initial_conditions,
)
from bddl.object_taxonomy import ObjectTaxonomy
import json
import argparse
from omnigibson.utils.bddl_utils import (
    OmniGibsonBDDLBackend,
)

parser = argparse.ArgumentParser()
parser.add_argument("--activity", type=str, required=True)

ot = ObjectTaxonomy()

def print_task_custom_list_template(activity_name):
    activity_conditions = Conditions(
        activity_name,
        0,
        simulator_name="omnigibson",
        predefined_problem=None,
    )
    obj_scope = get_object_scope(activity_conditions)
    backend = OmniGibsonBDDLBackend()
    init_conds = get_initial_conditions(activity_conditions, backend, obj_scope)
    synsets = set()
    for init_cond in init_conds:
        body = init_cond.body
        if len(body) == 3:
            synset = "_".join(body[1].split("_")[:-1])
            if "sceneObject" in ot.get_abilities(synset):
                continue
            if "agent" in synset:
                continue
            synsets.add(synset)
    task_custom_template = {
        activity_name: {
            "house_single_floor": {
                "whitelist": {synset: {synset.split(".")[0]: {"__TODO__MODEL__": None}} for synset in sorted(synsets)},
                "blacklist": {}
            }
        }
    }

    json_str = json.dumps(task_custom_template, indent=4)
    print("*" * 40)
    print()
    print(json_str)
    print()
    print("*" * 40)


if __name__ == "__main__":
    args = parser.parse_args()
    print_task_custom_list_template(args.activity)
