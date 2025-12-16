import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

from b1k_pipeline.utils import parse_name

rt = pymxs.runtime


def assign_toggle():
    target = rt.pickobject()

    assert (
        len(rt.selection) == 1
    ), "You have to assign toggle button one at a time - Select toggle button first, then click this button, then select the target object."

    button = rt.selection[0]

    # Parent the collision object to the target object
    button.parent = target

    # Rename the first object to match the selected object
    button.name = target.name + "-Mtogglebutton"

    # Validate that the object name is valid
    assert (
        parse_name(button.name) is not None
    ), f"Done, but please fix invalid name {button.name} for collision object"


def assign_toggle_button():
    try:
        assign_toggle()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    assign_toggle_button()
