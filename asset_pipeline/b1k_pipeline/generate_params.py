"""Regenerate the params.yaml file to contain all of the object and scenes."""

import os
import re
import sys
import yaml
import hashlib
import json

OBJS_DIR = "../cad/objects"
SCENES_DIR = "../cad/scenes"
OUT_PATH = "../params.yaml"
ROOT_PATH = ".."

DVC_TMPL_PATH = "../dvc-tmpl.yaml"
DVC_OUT_PATH = "../dvc.yaml"

DEP_REGEX = re.compile(r"#\{(.*?)\}")

FINAL_SCENES = [
    "Beechwood_0_int",
    "Beechwood_1_int",
    "Benevolence_0_int",
    "Benevolence_1_int",
    "Benevolence_2_int",
    "Ihlen_0_int",
    "Ihlen_1_int",
    "Merom_0_int",
    "Merom_1_int",
    "Pomaria_0_int",
    "Pomaria_1_int",
    "Pomaria_2_int",
    "Rs_int",
    "Wainscott_0_int",
    "Wainscott_1_int",
    "Beechwood_0_garden",
    "Rs_garden",
    "Pomaria_0_garden",
    "Merom_0_garden",
    "Wainscott_0_garden",
    "house_single_floor",
    "house_double_floor_lower",
    "house_double_floor_upper",
    "grocery_store_asian",
    "grocery_store_cafe",
    "grocery_store_convenience",
    "grocery_store_half_stocked",
    "hall_arch_wood",
    "hall_train_station",
    "hall_glass_ceiling",
    "hall_conference_large",
    "hotel_suite_large",
    "hotel_suite_small",
    "hotel_gym_spa",
    "office_bike",
    "office_cubicles_left",
    "office_cubicles_right",
    "office_large",
    "office_vendor_machine",
    "restaurant_asian",
    "restaurant_cafeteria",
    "restaurant_diner",
    "restaurant_brunch",
    "restaurant_urban",
    "restaurant_hotel",
    "school_gym",
    "school_geography",
    "school_biology",
    "school_chemistry",
    "school_computer_lab_and_infirmary",
    "gates_bedroom",
]

CHALLENGE_SCENES = [
    "house_single_floor",
    "house_double_floor_lower",
    "house_double_floor_upper",
    "restaurant_diner",
    "hotel_suite_large",
    "office_cubicles_right",
    "Rs_int",
]

APPROVED_OBJS = {
    ".*",
}

REJECTED_OBJS = set()

# APPROVED_SCENES = set()

APPROVED_SCENES = {
    ".*",
}

REJECTED_SCENES = set()

VERIFIED_SCENES = {
    ".*",
}


def get_challenge_scene_deps(challenge_scenes, approved_scenes):
    challenge_scenes_and_deps = set(challenge_scenes)
    for scene_name in sorted(challenge_scenes):
        object_list_path = os.path.join(
            os.path.dirname(__file__),
            SCENES_DIR,
            scene_name,
            "artifacts",
            "object_list.json",
        )
        if not os.path.exists(object_list_path):
            print(f"Missing object list for challenge scene: {scene_name}")
            continue

        with open(object_list_path, "r") as f:
            room_object_list = json.load(f)

        challenge_scenes_and_deps.update(room_object_list.get("outgoing_portals", {}).keys())

    missing_challenge_deps = challenge_scenes_and_deps - set(approved_scenes)
    if missing_challenge_deps:
        print(f"Missing challenge scene dependencies: {missing_challenge_deps}")

    return sorted(challenge_scenes_and_deps & set(approved_scenes))


def main():
    objects_path = os.path.join(os.path.dirname(__file__), OBJS_DIR)
    all_object_list = [
        x
        for x in os.listdir(objects_path)
        if os.path.exists(os.path.join(objects_path, x, "processed.max"))
    ]
    approved_objects = sorted(
        x
        for x in all_object_list
        if any(re.fullmatch(exp, x) for exp in APPROVED_OBJS)
        and not any(re.fullmatch(exp, x) for exp in REJECTED_OBJS)
    )
    objects = sorted(
        ["objects/" + x for x in approved_objects]
    )

    scenes_path = os.path.join(os.path.dirname(__file__), SCENES_DIR)
    all_scenes_list = [
        x
        for x in os.listdir(scenes_path)
        if os.path.exists(os.path.join(scenes_path, x, "processed.max"))
    ]
    approved_scenes = sorted(
        x
        for x in all_scenes_list
        if any(re.fullmatch(exp, x) for exp in APPROVED_SCENES)
        and not any(re.fullmatch(exp, x) for exp in REJECTED_SCENES)
    )
    scenes = sorted(["scenes/" + x for x in approved_scenes])  #

    combined = objects + scenes

    found_final_scenes = set(FINAL_SCENES) & set(approved_scenes)
    missing_final_scene_paths = set(FINAL_SCENES) - found_final_scenes
    if missing_final_scene_paths:
        print(f"Missing scenes: {missing_final_scene_paths}")
    final_scenes = sorted(["scenes/" + x for x in found_final_scenes])

    found_challenge_scenes = set(CHALLENGE_SCENES) & set(approved_scenes)
    missing_challenge_scene_paths = set(CHALLENGE_SCENES) - found_challenge_scenes
    if missing_challenge_scene_paths:
        print(f"Missing challenge scenes: {missing_challenge_scene_paths}")
    challenge_scenes = sorted(["scenes/" + x for x in found_challenge_scenes])
    challenge_scenes_and_deps = sorted(
        "scenes/" + x
        for x in get_challenge_scene_deps(found_challenge_scenes, approved_scenes)
    )

    found_verified_scenes = set(
        x
        for x in all_scenes_list
        if any(re.fullmatch(exp, x) for exp in VERIFIED_SCENES)
    ) & set(approved_scenes)
    verified_scenes = sorted(["scenes/" + x for x in found_verified_scenes])

    out_path = os.path.join(os.path.dirname(__file__), OUT_PATH)
    params = {
        "objects": objects,
        "scenes": scenes,
        "final_scenes": final_scenes,
        "challenge_scenes": challenge_scenes,
        "challenge_scenes_and_deps": challenge_scenes_and_deps,
        "verified_scenes": verified_scenes,
        "combined": combined,
    }
    with open(out_path, "w") as f:
        yaml.dump(params, f)

    # Now update the # placeholders in deps
    dvc_tmpl_path = os.path.join(os.path.dirname(__file__), DVC_TMPL_PATH)
    with open(dvc_tmpl_path, "r") as f:
        dvc_conf = yaml.load(f, Loader=yaml.SafeLoader)

    for stage in dvc_conf["stages"].values():
        deps_list = None
        if "deps" in stage:
            deps_list = stage["deps"]
        elif "do" in stage and "deps" in stage["do"]:
            deps_list = stage["do"]["deps"]
        else:
            continue

        new_deps_list = []
        for dep in deps_list:
            match = DEP_REGEX.search(dep)
            if match is None:
                new_deps_list.append(dep)
                continue

            key = match.group(1)
            assert key in params, f"Unknown placeholder {key}"
            replacements = params[key]
            for replacement in replacements:
                new_deps_list.append(DEP_REGEX.sub(replacement, dep))

        if "deps" in stage:
            stage["deps"] = new_deps_list
        elif "do" in stage and "deps" in stage["do"]:
            stage["do"]["deps"] = new_deps_list

    # Write the DVC file.
    dvc_out_path = os.path.join(os.path.dirname(__file__), DVC_OUT_PATH)
    with open(dvc_out_path, "w") as f:
        yaml.dump(dvc_conf, f)

    print("Params updated successfully.")


if __name__ == "__main__":
    main()
