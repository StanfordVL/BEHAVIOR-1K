import json
import argparse
import os
from omnigibson.utils.bddl_utils import get_knowledge_base
from omnigibson.macros import gm
from constants import DATASET_2026_PATH, TASK_CUSTOM_LIST_PATH


PREFERRED_SCENES = ["house_double_floor_lower", "house_double_floor_upper", "house_single_floor"]
SYNSET_BASE_URL = "https://behavior.stanford.edu/knowledgebase/synsets"
SCENES_PATH = os.path.join(gm.DATA_PATH, "behavior-1k-assets", "scenes")
OBJECTS_PATH = os.path.join(gm.DATA_PATH, "behavior-1k-assets", "objects")

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--activity", type=str, required=True)


def prompt_choice(prompt, options, multi=False):
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Enter index or name" + (" (comma-separated for multiple)" if multi else "") + ": ").strip()
        chosen = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and int(part) < len(options):
                chosen.append(options[int(part)])
            elif part in options:
                chosen.append(part)
            else:
                print(f"  Invalid choice: {part!r}")
                chosen = []
                break
        if chosen:
            return chosen if multi else chosen[0]


def autogenerate_task_custom_list(activity_name):
    assert os.path.exists(DATASET_2026_PATH), f"2026 dataset not found: {DATASET_2026_PATH}"
    assert os.path.exists(TASK_CUSTOM_LIST_PATH), f"task_custom_lists.json not found: {TASK_CUSTOM_LIST_PATH}"

    task = get_knowledge_base().get_task(f"{activity_name}-0")
    conditions = task.parse_base_scope()[0]
    init_conds = conditions.parsed_initial_conditions
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

    # Prompt for scene
    print(f"\nSelect scene for activity '{activity_name}':")
    for i, s in enumerate(PREFERRED_SCENES):
        print(f"  [{i}] {s}")
    while True:
        raw = input("Enter index, name, or custom string: ").strip()
        if raw.isdigit() and int(raw) < len(PREFERRED_SCENES):
            scene = PREFERRED_SCENES[int(raw)]
            break
        elif raw:
            scene = raw
            break

    # Prompt for models per synset/category
    whitelist = {}
    for synset in sorted(synsets):
        synset_obj = get_knowledge_base().get_synset(synset)
        if synset_obj is None:
            continue
        whitelist[synset] = {}
        for cat in synset_obj.categories:
            cat_name = cat.name
            cat_models_dir = os.path.join(OBJECTS_PATH, cat_name)
            available_models = sorted(os.listdir(cat_models_dir)) if os.path.exists(cat_models_dir) else []
            if not available_models:
                print(f"\n  No models found for category '{cat_name}', skipping.")
                continue
            models = prompt_choice(
                f"Select model(s) for {synset} / {cat_name} ({SYNSET_BASE_URL}/{synset}.html):",
                available_models,
                multi=True,
            )
            whitelist[synset][cat_name] = {m: None for m in models}

    task_entry = {
        activity_name: {
            "room_types": list(room_types),
            scene: {
                "whitelist": whitelist,
                "blacklist": {},
            },
        }
    }

    # Load, update, and write back
    with open(TASK_CUSTOM_LIST_PATH, "r") as f:
        existing = json.load(f)

    existing.update(task_entry)

    with open(TASK_CUSTOM_LIST_PATH, "w") as f:
        json.dump(existing, f, indent=4)

    print(f"\nWrote entry for '{activity_name}' to {TASK_CUSTOM_LIST_PATH}")


if __name__ == "__main__":
    args = parser.parse_args()
    autogenerate_task_custom_list(args.activity)
