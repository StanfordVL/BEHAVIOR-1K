import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import b1k_pipeline.utils

import pymxs

rt = pymxs.runtime


def merge_collision(collision_objs, target, metatype="collision"):
    # Convert all the objects to editable poly
    for obj in collision_objs:
        rt.addmodifier(obj, rt.Edit_Poly())
        rt.maxOps.collapseNodeTo(obj, 1, True)

    # Attach all of them to the first obj
    baseObj = collision_objs[0]
    for obj in collision_objs[1:]:
        rt.polyop.attach(baseObj, obj)

    # Triangulate the faces
    ttp = rt.Turn_To_Poly()
    ttp.limitPolySize = True
    ttp.maxPolySize = 3
    rt.addmodifier(baseObj, ttp)
    rt.maxOps.collapseNodeTo(baseObj, 1, True)

    # Parent the collision object to the target object
    baseObj.parent = target

    # Rename the first object to match the selected object
    baseObj.name = target.name + "-M" + metatype

    # Validate that the object name is valid
    assert (
        b1k_pipeline.utils.parse_name(baseObj.name) is not None
    ), f"Done, but please fix invalid name {baseObj.name} for collision object"

    return baseObj


def select_and_merge_collision():
    collision_objs = list(rt.selection)

    # Ask the user to pick a target object
    target = rt.pickobject()

    # Merge the collision objects
    merge_collision(collision_objs, target)


def merge_collision_button():
    try:
        select_and_merge_collision()
        # rt.messageBox("Success!")
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    merge_collision_button()
