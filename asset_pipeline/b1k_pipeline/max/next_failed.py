import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import glob
import json
import pathlib
import re
from collections import defaultdict

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils

rt = pymxs.runtime

file_name = "sanitycheck.json"
# file_name = "object_list.json"
# file_name = "export_meshes.json"


def file_eligible(objdir):
    sanitycheck = objdir / "artifacts" / file_name

    if not sanitycheck.exists():
        print(
            f"{str(objdir)} is missing object list or sanitycheck. Unsure if it failed."
        )
        return False

    with open(sanitycheck, "r") as f:
        x = json.load(f)
        if not x["success"]:
            return True

    return False


def next_failed():
    eligible_max = []
    for target in b1k_pipeline.utils.get_targets("combined"):
        objdir = b1k_pipeline.utils.PIPELINE_ROOT / "cad" / target
        if objdir.exists() and file_eligible(objdir):
            eligible_max.append(objdir / "processed.max")

    eligible_max.sort()
    print(len(eligible_max), "files remaining.")
    print("\n".join(str(x) for x in eligible_max))
    if eligible_max:
        # Find current file in the list
        current_max_dir = pathlib.Path(rt.maxFilePath).resolve()
        next_idx = 0
        try:
            next_idx = (
                next(
                    i
                    for i, max in enumerate(eligible_max)
                    if max.parent.resolve() == current_max_dir
                )
                + 1
            ) % len(eligible_max)
        except:
            pass

        print(f"File {next_idx} of {len(eligible_max)}")
        scene_file = eligible_max[next_idx]
        assert (
            not scene_file.is_symlink()
        ), f"File {scene_file} should not be a symlink."
        assert rt.loadMaxFile(
            str(scene_file), useFileUnits=False, quiet=True
        ), f"Could not load {scene_file}"

        sanitycheck_file = eligible_max[next_idx].parent / "artifacts" / file_name
        print(sanitycheck_file.read_text())


def next_failed_button():
    try:
        next_failed()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    next_failed_button()
