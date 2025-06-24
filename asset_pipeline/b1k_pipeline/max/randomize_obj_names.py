import os
import random
import re
import string
from collections import defaultdict

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs
import yaml
import b1k_pipeline.utils

rt = pymxs.runtime

TRANSLATION_PATH = b1k_pipeline.utils.PIPELINE_ROOT / "b1k_pipeline/model_rename.yml"
with open(TRANSLATION_PATH, "r") as f:
    TRANSLATION_DICT = yaml.load(f, Loader=yaml.SafeLoader)

PATTERN = re.compile(
    r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-zA-Z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RPF])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z_]+)(_(?P<meta_id>[0-9]+))?)?(?P<tag>(?:-T[a-z]+)*)$"
)


def parse_name(name):
    return PATTERN.fullmatch(name)


def randomize_names():
    objs = list(rt.selection)

    # Start by grouping objects by category / model.
    objs_by_model = defaultdict(list)
    for obj in objs:
        if obj in rt.cameras:
            continue

        result = parse_name(obj.name)
        assert result is not None, "{} does not match naming convention".format(
            obj.name
        )
        category, model_id = result.group("category"), result.group("model_id")
        if result.group("bad"):
            # For bad objects we can process directly. We simply perform necessary translations.
            translation_key = category + "/" + model_id
            if translation_key in TRANSLATION_DICT:
                new_cat, new_model = TRANSLATION_DICT[translation_key].split("/")
                old_str = "-".join([category, model_id])
                new_str = "-".join([new_cat, new_model])
                obj.name = obj.name.replace(old_str, new_str)

        objs_by_model[(category, model_id)].append(obj)

    # Then for each group, we assign a new random model ID.
    for category, model_id in objs_by_model:
        # if re.fullmatch(r"^[a-z]{6}$", model_id):
        #     continue

        new_cat = category
        new_model = "".join(random.choice(string.ascii_lowercase) for _ in range(6))

        old_str = "-".join([category, model_id])
        new_str = "-".join([new_cat, new_model])
        print(f"renaming {old_str} to {new_str}")
        for obj in objs_by_model[(category, model_id)]:
            obj.name = obj.name.replace(old_str, new_str)


def randomize_names_safe():
    try:
        randomize_names()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Names randomized!")


if __name__ == "__main__":
    randomize_names_safe()
