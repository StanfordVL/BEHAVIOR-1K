import csv
import json
import os
import sys
import textwrap

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

rt = pymxs.runtime

import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import wordnet as wn

from b1k_pipeline.utils import parse_name, PIPELINE_ROOT

IGNORE_CATEGORIES = {"walls", "floors", "ceilings"}

PASS_NAME = "benjamin-scene-looseness-category"
RECORD_PATH = PIPELINE_ROOT / "qa-logs" / f"{PASS_NAME}.json"


def main():
    # Read completed groups record
    completed_groups = set()
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH) as f:
            completed_groups = set(json.load(f))

    # Find incomplete groups
    parsed_names = [parse_name(x.name) for x in rt.objects]
    available_groups = {
        x.group("category")
        for x in parsed_names
        if x is not None
        and int(x.group("instance_id")) == 0
        and x.group("category") not in IGNORE_CATEGORIES
    }
    remaining_groups = sorted(available_groups - completed_groups)
    if not remaining_groups:
        rt.messageBox("Scene complete.")
        return

    # Find objects corresponding to the next remaining group's instance zero
    next_group = remaining_groups[0]
    next_group_objects = []
    has_loose = False
    has_fixed = False
    has_clutter = False
    for obj in rt.objects:
        n = parse_name(obj.name)
        if n is None:
            continue
        if int(n.group("instance_id")) != 0:
            continue
        if n.group("category") != next_group:
            continue
        next_group_objects.append(obj)
        obj.isHidden = False
        if n.group("loose") == "L-":
            has_loose = True
        elif n.group("loose") == "C-":
            has_clutter = True
        else:
            has_fixed = True

    # Select that object and print
    rt.select(next_group_objects)
    rt.IsolateSelection.EnterIsolateSelectionMode()
    rt.execute("max tool zoomextents")
    print(
        f"{len(available_groups) - len(remaining_groups) + 1} / {len(available_groups)}: {next_group}"
    )

    if has_loose and not has_fixed and not has_clutter:
        print("Used as loose")
    elif has_fixed and not has_loose and not has_clutter:
        print("Used as fixed")
    elif has_clutter and not has_loose and not has_fixed:
        print("Used as clutter")
    else:
        print(
            textwrap.fill(
                f"Used as loose: {has_loose}. Used as fixed: {has_fixed}. Used as clutter: {has_clutter}"
            )
        )

    # Show a popup with the synset info
    # category_name = next_group.split("-")[0]
    # synset_name, synset_desc = get_synset(category_name)
    # print(textwrap.fill(f"Category {category_name} is currently mapped to synset {synset_name} ({synset_desc})"))

    # Record that object as completed
    with open(RECORD_PATH, "w") as f:
        json.dump(list(completed_groups | {next_group}), f)


if __name__ == "__main__":
    main()
