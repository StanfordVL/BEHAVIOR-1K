import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name


import pymxs

rt = pymxs.runtime


def switch_metalink():
    metalinks = list(rt.selection)

    for obj in metalinks:
        n = parse_name(obj.name)
        if n is None:
            continue
        current_metatype = n.group("meta_type")

        if current_metatype == "togglebutton":
            new_metatype = "slicer"
        elif current_metatype == "slicer":
            new_metatype = "fillable"
        elif current_metatype == "fillable":
            new_metatype = "fluidsink"
        elif current_metatype == "fluidsink":
            new_metatype = "fluidsource"
        elif current_metatype == "fluidsource":
            new_metatype = "heatsource"
        elif current_metatype == "heatsource":
            new_metatype = "particleapplier"
        elif current_metatype == "particleapplier":
            new_metatype = "togglebutton"

        before = obj.name[: n.start("meta_type")]
        after = obj.name[n.end("meta_type") :]
        new_name = before + new_metatype + after
        assert parse_name(obj.name), f"Almost generated invalid name {new_name}"
        obj.name = new_name


def switch_metalink_button():
    try:
        switch_metalink()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    switch_metalink_button()
