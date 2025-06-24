import random
import re
import string
from collections import defaultdict

import pymxs
import yaml

rt = pymxs.runtime

PATTERN = re.compile(
    r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[A-Za-z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z]+)_(?P<meta_id>[0-9]+))?$"
)


def parse_name(name):
    return PATTERN.fullmatch(name)


def translate_names():
    with open(r"D:\BEHAVIOR-1K\asset_pipeline\scripts\model_rename.yml") as f:
        mapping = yaml.load(f, Loader=yaml.FullLoader)

    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        result = parse_name(obj.name)
        assert result is not None, "{} does not match naming convention".format(
            obj.name
        )

        category, model_id, is_bad = (
            result.group("category"),
            result.group("model_id"),
            result.group("bad"),
        )
        key = f"{category}/{model_id}"
        if key not in mapping:
            continue

        assert is_bad, f"Object {category}-{model_id} found in mapping but is not bad."

        new_category, new_model_id = mapping[key].split("/")
        old_str = "-".join([category, model_id])
        new_str = "-".join([new_category, new_model_id])
        new_name = obj.name.replace(old_str, new_str)

        print(f"Replacing {obj.name} with {new_name}.")
        obj.name = new_name


def translate_names_safe():
    try:
        translate_names()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Names translated!")


if __name__ == "__main__":
    translate_names_safe()
