import xml.etree.ElementTree as ET
import os
from collections import defaultdict
import numpy as np

SCENE_NAME = "Benevolence_0_int"


SCENES = [
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
]


def get_unique_coords(p):
    tree = ET.parse(p)
    root = tree.getroot()
    link_dict = defaultdict(list)
    for child in root.findall("link"):
        if child.attrib["name"] in ("world", "walls", "floors", "ceilings"):
            continue
        category = child.attrib["category"]
        if category:
            link_dict[category].append(child.attrib["name"])

    unique_names = {k: v[0] for k, v in link_dict.items() if len(v) == 1}
    unique_coords = {}

    for cat, unique_name in unique_names.items():
        joint_name = "j_" + unique_name
        joint = root.find(f"joint[@name='{joint_name}']")
        assert joint
        coordinates_str = joint.find("origin").attrib["xyz"]
        coordinates = np.array([float(x) for x in coordinates_str.split()])
        unique_coords[cat] = coordinates

    return unique_coords


def process_scene(scene_name):
    path_new = os.path.join(
        f"D:/BEHAVIOR-1K/asset_pipeline/cad/scenes/{scene_name}/artifacts/scene/urdf/{scene_name}_best.urdf"
    )
    path_old = os.path.join(
        f"C:/Users/Cem/research/iGibson-dev/igibson/data/ig_dataset/scenes/{scene_name}/urdf/{scene_name}_best.urdf"
    )
    unique_new = get_unique_coords(path_new)
    # print("new keys", unique_new.keys())
    unique_old = get_unique_coords(path_old)
    # print("old keys", unique_old.keys())
    common_keys = set(unique_new.keys()) & set(unique_old.keys())

    diffs = []
    for key in common_keys:
        diffs.append(unique_old[key] - unique_new[key])

    print("\n", scene_name)
    print("avg %.3f %.3f %.3f" % tuple(np.mean(diffs, axis=0)))
    print("med %.3f %.3f %.3f" % tuple(np.median(diffs, axis=0)))
    print("std %.3f %.3f %.3f" % tuple(np.std(diffs, axis=0)))


def main():
    for scene_name in SCENES:
        process_scene(scene_name)


if __name__ == "__main__":
    main()
