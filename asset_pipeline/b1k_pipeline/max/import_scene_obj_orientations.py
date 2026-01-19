import pathlib
import sys
import traceback
from collections import defaultdict

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import os

import numpy as np
import pymxs
import yaml
from scipy.spatial.transform import Rotation as R

from b1k_pipeline.utils import parse_name

rt = pymxs.runtime

import xml.etree.ElementTree as ET

IN_DATASET_ROOT = r"C:\Users\Cem\research\iGibson-dev\igibson\data\ig_dataset"

TRANSLATION_PATH = os.path.join(IN_DATASET_ROOT, "metadata", "model_rename.yaml")
with open(TRANSLATION_PATH, "r") as f:
    TRANSLATION_DICT = yaml.load(f, Loader=yaml.SafeLoader)


def fix():
    # Prepare a list of objects to position.
    objs = [x for x in rt.objects if rt.classOf(x) == rt.Editable_Poly]
    candidates = defaultdict(list)
    for obj in objs:
        match = parse_name(obj.name)
        if not match:
            continue

        if match.group("link_name") and match.group("link_name") != "base_link":
            continue

        if match.group("category") in (
            "ceilings",
            "floors",
            "walls",
            "electric_switch",
        ):
            continue

        q = obj.rotation
        mag = R.from_quat([q.x, q.y, q.z, q.w]).magnitude()
        assert np.isclose(mag, 0, atol=1e-3), f"{obj.name} has nonzero rotation."
        assert np.all(obj.scale, 0), f"{obj.name} has unit scale."
        assert np.all(obj.objectoffsetscale, 0), f"{obj.name} has unit scale."

        cat = match.group("category")
        model = match.group("model_id")
        com = (
            np.asarray(obj.position) / 1000
        )  # np.mean(np.array(rt.polyop.getVerts(obj, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj)))), axis=0) / 1000
        candidates[(cat, model)].append((obj, com))

    scene_name = pathlib.Path(rt.maxFilePath).parts[-1]
    file_path = (
        pathlib.Path(IN_DATASET_ROOT)
        / "scenes"
        / scene_name
        / "urdf"
        / f"{scene_name}_best.urdf"
    )
    assert file_path.exists(), f"Could not find {str(file_path)}"
    tree = ET.parse(file_path)
    root = tree.getroot()
    found = defaultdict(list)
    for link in root.findall("link"):
        if "category" not in link.attrib:
            continue

        cat = link.attrib["category"]
        model = link.attrib["model"]
        name = link.attrib["name"]

        if cat in ("ceilings", "floors", "walls"):
            continue

        new_category_name, new_model_name = TRANSLATION_DICT[cat + "/" + model].split(
            "/"
        )

        joint_name = "j_" + name
        joint_xpath = f"joint[@name='{joint_name}']"
        # print("looking for ", joint_xpath)
        (joint,) = root.findall(joint_xpath)
        xyz = np.asarray(
            [float(x) for x in joint.find("origin").attrib["xyz"].split(" ")]
        )
        rpy = np.asarray(
            [float(x) for x in joint.find("origin").attrib["rpy"].split(" ")]
        )
        quat = R.from_euler("xyz", rpy).as_quat()

        found[(new_category_name, new_model_name)].append((name, xyz, quat))

    missing_in_file = set(found.keys()) - set(candidates.keys())
    assert not missing_in_file, f"Missing in file: {missing_in_file}"

    missing_in_urdf = set(candidates.keys()) - set(found.keys())
    assert not missing_in_urdf, f"Missing in urdf: {missing_in_urdf}"

    # Check that the length is the same
    problems = []
    for x in found.keys():
        count_in_urdf = len(found[x])
        count_in_file = len(candidates[x])
        if not count_in_file == count_in_urdf:
            problems.append(
                f"{x} has {count_in_file} in file but {count_in_urdf} in URDF."
            )

    assert not problems, "\n".join(problems)

    positions_from_file = np.asarray(
        [com for objs in candidates.values() for _, com in objs]
    )
    positions_from_urdf = np.asarray(
        [xyz for objs in found.values() for _, xyz, _ in objs]
    )

    assert (
        len(positions_from_file) == len(positions_from_urdf)
    ), f"File contains {len(positions_from_file)} things while URDF contains {len(positions_from_urdf)}"
    file_mean = np.mean(positions_from_file, axis=0)
    urdf_mean = np.mean(positions_from_urdf, axis=0)
    displacement = file_mean - urdf_mean

    # Start mapping everything to the nearest
    for x in found.keys():
        x_candidates = candidates[x]
        cand_pos = np.asarray([com for _, com in x_candidates])
        x_found = found[x]
        found_pos = np.asarray([xyz for _, xyz, _ in x_found]) + displacement

        # For each found pos, get the index of the closest
        closest_candidates = []
        for f in found_pos:
            dists = np.linalg.norm(cand_pos - f[None, :], axis=-1)
            closest_candidate = np.argmin(dists)
            closest_candidates.append(closest_candidate)

        if len(closest_candidates) != len(set(closest_candidates)):
            for urdf_idx, file_idx in enumerate(closest_candidates):
                print(
                    f"Closest object for {x_found[urdf_idx][0]} is {x_candidates[file_idx][0].name}"
                )

        # Now that everything has its closest candidate, set the orientations.
        for i, (_, _, quat) in enumerate(x_found):
            closest_idx = closest_candidates[i]
            closest_obj = x_candidates[closest_idx][0]

            current_pos = closest_obj.position
            current_rot = closest_obj.rotation
            current_oor = closest_obj.objectoffsetrot
            current_oop = closest_obj.objectoffsetpos

            target_rot = rt.Quat(*quat.tolist())
            diff_rot = rt.inverse(current_rot) * target_rot
            inv_diff_rot = rt.inverse(diff_rot)
            inv_diff_rot_R = R.from_quat(
                [inv_diff_rot.x, inv_diff_rot.y, inv_diff_rot.z, inv_diff_rot.w]
            )
            target_oop = inv_diff_rot_R.apply(current_oop)
            closest_obj.rotation = current_rot * diff_rot
            closest_obj.objectoffsetrot = current_oor * diff_rot
            closest_obj.position = current_pos
            closest_obj.objectoffsetpos = rt.Point3(*target_oop.tolist())

            print("Fixed", closest_obj.name)


if __name__ == "__main__":
    fix()
