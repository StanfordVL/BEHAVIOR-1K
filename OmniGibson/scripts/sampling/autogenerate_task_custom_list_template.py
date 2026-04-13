from omnigibson.utils.bddl_utils import get_knowledge_base
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--activity", type=str, required=True)


def print_task_custom_list_template(activity_name):
    task = get_knowledge_base().get_task(f"{activity_name}-0")
    task._ensure_compiled()
    init_conds = task.conditions.parsed_initial_conditions
    synsets = set()
    room_types = set()
    for init_cond in init_conds:
        if len(init_cond) == 3:
            if "inroom" == init_cond[0]:
                room_types.add(init_cond[2])
            synset = "_".join(init_cond[1].split("_")[:-1])
            synset_obj = get_knowledge_base().get_synset(synset)
            if synset_obj is not None and "sceneObject" in synset_obj.abilities:
                continue
            if "agent" in synset:
                continue
            synsets.add(synset)
    task_custom_template = {
        activity_name: {
            "room_types": list(room_types),
            "__TODO__SCENE__": {
                synset: {cat.name: ["__TODO__MODEL__"] for cat in get_knowledge_base().get_synset(synset).categories}
                for synset in synsets
            },
        }
    }
    print(json.dumps(task_custom_template, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()
    print_task_custom_list_template(args.activity)
