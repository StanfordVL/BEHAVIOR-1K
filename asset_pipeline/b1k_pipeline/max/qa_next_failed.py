import re
import sys
import bisect

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import json
import pathlib

import pymxs

import b1k_pipeline.utils

rt = pymxs.runtime

type_re = re.compile(r"([A-Z\-]+):")
ignore_messages = ["SYNSET", "CATEGORY", "ABILITIES", "SUBSTANCE", "STRUCTURE-UNCLOSED"]


def file_eligible(objdir):
    complaint_path = objdir / "complaints.json"

    if not complaint_path.exists():
        return False

    with open(complaint_path, "r") as f:
        x = json.load(f)
        for complaint in x:
            if complaint["processed"]:
                continue

            complaint_type = "PREVIOUS PASS"
            m = type_re.match(complaint["message"])
            if m:
                complaint_type = m.group(1)

            if complaint_type in ignore_messages:
                continue

            return True

    return False


def next_failed():
    eligible_max = []
    for target in b1k_pipeline.utils.get_targets("objects"):
        objdir = b1k_pipeline.utils.PIPELINE_ROOT / "cad" / target
        if objdir.exists() and file_eligible(objdir):
            eligible_max.append(objdir / "processed.max")

    eligible_max.sort()
    symlinks = [x for x in eligible_max if x.is_symlink()]
    # If there are any symlinks, print dvc unprotect command and stop
    if symlinks:
        print("There are symlinks! Run the following commands:")
        for start in range(0, len(symlinks), 50):
            batch = symlinks[start : start + 50]
            print(
                "dvc unprotect",
                " ".join(
                    str(x.relative_to(b1k_pipeline.utils.PIPELINE_ROOT)) for x in batch
                ),
            )
        return

    print(len(eligible_max), "files remaining.")
    print("\n".join(str(x) for x in eligible_max))
    if eligible_max:
        current_max = pathlib.Path(rt.maxFilePath).resolve() / "processed.max"
        next_idx = 0
        try:
            next_idx = bisect.bisect(eligible_max, current_max) % len(eligible_max)
        except Exception as e:
            print(e)

        scene_file = eligible_max[next_idx]
        assert (
            not scene_file.is_symlink()
        ), f"File {scene_file} should not be a symlink."
        assert rt.loadMaxFile(
            str(scene_file), useFileUnits=False, quiet=True
        ), f"Could not load {scene_file}"


def next_failed_button():
    try:
        next_failed()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    next_failed_button()
