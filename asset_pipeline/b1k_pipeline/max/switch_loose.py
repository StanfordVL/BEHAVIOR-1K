import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import parse_name


def main():
    # Find groups of visible objects
    parsed_names = [parse_name(x.name) for x in rt.objects if not x.isHidden]
    groups = {
        x.group("category")
        for x in parsed_names
        if x is not None and int(x.group("instance_id")) == 0
    }
    assert len(groups) == 1, "Multiple object groups are visible"

    # Find objects corresponding to the next remaining group's instance zero
    (this_group,) = groups
    this_group_objects = []
    current_prefix = ""
    for obj in rt.objects:
        n = parse_name(obj.name)
        if n is None:
            continue
        if n.group("category") != this_group:
            continue
        this_group_objects.append(obj)
        current_prefix = n.group("loose")

    # Decide which way to go
    if current_prefix == "L-":  # Loose nonclutter -> Loose clutter
        prefix = "C-"
        message = "Switched to clutter"
    elif current_prefix == "C-":  # Loose clutter -> Fixed
        prefix = ""
        message = "Switched to fixed"
    else:  # Fixed -> Loose nonluuter
        prefix = "L-"
        message = "Switched to loose nonclutter"

    # Apply the prefix
    for obj in this_group_objects:
        n = parse_name(obj.name)
        bad = n.group("bad") or ""
        randomization_disabled = n.group("randomization_disabled") or ""
        before = f"{bad}{randomization_disabled}"
        after = obj.name[n.start("category") :]
        new_name = before + prefix + after
        assert parse_name(obj.name), f"Almost generated invalid name {new_name}"
        obj.name = new_name

    print(message)


if __name__ == "__main__":
    main()
