from bddl.activity import Conditions
from bddl.object_taxonomy import ObjectTaxonomy
import json
import argparse

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
    init_conds = activity_conditions.parsed_initial_conditions
    synsets = set()
    room_types = set()
    for init_cond in init_conds:
        if len(init_cond) == 3:
            if "inroom" == init_cond[0]:
                room_types.add(init_cond[2])
            synset = "_".join(init_cond[1].split("_")[:-1])
            if "sceneObject" in ot.get_abilities(synset):
                continue
            if "agent" in synset:
                continue
            synsets.add(synset)
    task_custom_template = {
        activity_name: {
            "room_types": list(room_types),
            "__TODO__SCENE__": {
                "whitelist": {synset: {synset.split(".")[0]: {"__TODO__MODEL__": None}} for synset in sorted(synsets)},
                "blacklist": {},
            },
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
