import re
from collections import defaultdict

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R

rt = pymxs.runtime


def assign_light():
    physical_light = rt.pickobject()

    assert (
        len(rt.selection) == 1
    ), "You have to assign light one at a time - Select light first, then click this button, then select the physical light object."

    light = rt.selection[0]

    existing_lights = [
        obj for obj in rt.objects if physical_light.name + "-L" in obj.name
    ]
    light.name = physical_light.name + "-L{}".format(len(existing_lights))


def assign_light_button():
    try:
        assign_light()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Assign light complete - no errors found!")


if __name__ == "__main__":
    assign_light_button()
