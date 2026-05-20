import json
import argparse
import glob
import os
from omnigibson.utils.bddl_utils import get_knowledge_base, GOOD_MODELS, BAD_CLOTH_MODELS
from omnigibson.utils.asset_utils import get_all_object_category_models
from constants import DATASET_2025_PATH, DATASET_2026_PATH, TASK_CUSTOM_LIST_PATH


SYNSET_BASE_URL = "https://behavior.stanford.edu/knowledgebase/synsets"

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--activity", type=str, required=True)


def get_2025_models_for_task(activity_name):
    """Return {synset: {category: [model_id, ...]}} for a task found in the 2025 dataset."""
    pattern = os.path.join(DATASET_2025_PATH, "scenes", "*", "json", f"*_task_{activity_name}_0_0_template.json")
    results = {}
    for template_path in glob.glob(pattern):
        try:
            with open(template_path) as f:
                d = json.load(f)
            inst_to_name = d["metadata"]["task"]["inst_to_name"]
            objs_info = d["objects_info"]["init_info"]
            for bddl_inst, obj_name in inst_to_name.items():
                if "agent" in bddl_inst or obj_name not in objs_info:
                    continue
                args = objs_info[obj_name]["args"]
                synset = "_".join(bddl_inst.split("_")[:-1])
                category = args["category"]
                model = args["model"]
                results.setdefault(synset, {}).setdefault(category, set()).add(model)
        except Exception:
            pass
    return results


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

    # Prompt for scene — only offer scenes that match the task per the knowledge base
    matched_scene_names = sorted(s.name for s in task.matched_scenes)
    print(f"\nSelect scene for activity '{activity_name}' ({len(matched_scene_names)} matching):")
    for i, s in enumerate(matched_scene_names):
        print(f"  [{i}] {s}")
    while True:
        raw = input("Enter index, name, or custom string: ").strip()
        if raw.isdigit() and int(raw) < len(matched_scene_names):
            scene = matched_scene_names[int(raw)]
            break
        elif raw:
            scene = raw
            break

    # Prompt for models per synset/category
    models_2025 = get_2025_models_for_task(activity_name)
    whitelist = {}
    for synset in sorted(synsets):
        synset_obj = get_knowledge_base().get_synset(synset)
        if synset_obj is None:
            continue
        whitelist[synset] = {}
        # Non-leaf synsets have no direct categories; walk to leaf descendants
        leaf_synsets = [synset_obj] if synset_obj.is_leaf else [d for d in synset_obj.descendants if d.is_leaf]
        all_cats = [cat for s in leaf_synsets for cat in s.categories]
        for cat in all_cats:
            cat_name = cat.name
            available_models = set(get_all_object_category_models(cat_name))
            available_models = (
                available_models
                if cat_name not in GOOD_MODELS
                else available_models.intersection(GOOD_MODELS[cat_name])
            )
            available_models = sorted(available_models - BAD_CLOTH_MODELS.get(cat_name, set()))
            if not available_models:
                print(f"\n  No models found for category '{cat_name}', skipping.")
                continue
            used_in_2025 = sorted(models_2025.get(synset, {}).get(cat_name, []))
            hint = f"  (used in 2025: {', '.join(used_in_2025)})" if used_in_2025 else "  (not found in 2025 dataset)"
            models = prompt_choice(
                f"Select model(s) for {synset} / {cat_name} ({SYNSET_BASE_URL}/{synset}.html):\n{hint}",
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
