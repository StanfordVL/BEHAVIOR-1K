import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import b1k_pipeline.utils
import tqdm

import pymxs

rt = pymxs.runtime


def create_collision_from_self(obj):
    # Check that the object is parentless and not already a collision object
    assert obj.parent is None, f"Object {obj.name} has a parent {obj.parent.name}, expected it to be parentless"
    assert "-Mcollision" not in obj.name, f"Object {obj.name} already has '-Mcollision' in its name, expected it to not already be a collision object"
    assert b1k_pipeline.utils.parse_name(obj.name) is not None, f"Object {obj.name} does not have a valid name according to the naming convention, please rename it before creating collision from self"
    assert rt.classOf(obj) == rt.Editable_Poly, f"Object {obj.name} is of type {rt.classOf(obj)}, expected it to be an Editable Poly"
    for child in obj.children:
        assert "-Mcollision" not in child.name, f"Child object {child.name} of {obj.name} already has '-Mcollision' in its name, expected it to not already be a collision object"

    # Clone the object
    success, baseObj = rt.maxOps.cloneNodes(
        obj,
        cloneType=rt.name("copy"),
        newNodes=pymxs.byref(None),
    )
    assert success, f"Could not clone {obj.name}"
    (baseObj,) = baseObj

    # Triangulate the faces
    ttp = rt.Turn_To_Poly()
    ttp.limitPolySize = True
    ttp.maxPolySize = 3
    rt.addmodifier(baseObj, ttp)
    rt.maxOps.collapseNodeTo(baseObj, 1, True)

    # Parent the collision object to the target object
    baseObj.parent = obj

    # Rename the first object to match the selected object
    baseObj.name = obj.name + "-Mcollision"

    # Remove the material
    baseObj.material = None

    # Validate that the object name is valid
    assert (
        b1k_pipeline.utils.parse_name(baseObj.name) is not None
    ), f"Done, but please fix invalid name {baseObj.name} for collision object"


def create_collision_from_all_selections():
    for obj in tqdm.tqdm(list(rt.selection)):
        try:
            # Merge the collision objects
            create_collision_from_self(obj)
        except AssertionError as e:
            print(f"Failed to create collision for {obj.name}, skipping.", e)

def create_collision_from_self_button():
    try:
        create_collision_from_all_selections()
        # rt.messageBox("Success!")
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    create_collision_from_self_button()
