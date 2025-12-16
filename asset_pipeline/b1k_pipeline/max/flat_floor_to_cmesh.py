import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import b1k_pipeline.utils

import pymxs
import numpy as np

rt = pymxs.runtime


def flat_floor_to_cmesh(collision_objs, target):
    # for obj in collision_objs:
    #     assert rt.classOf(obj) in [
    #         rt.Line,
    #         rt.Plane,
    #     ], f"Object {obj.name} is not a line or plane"

    # Convert all the objects to editable poly
    for obj in collision_objs:
        rt.addmodifier(obj, rt.Edit_Poly())
        rt.maxOps.collapseNodeTo(obj, 1, True)

    # Attach all of them to the first obj
    baseObj = collision_objs[0]
    for obj in collision_objs[1:]:
        rt.polyop.attach(baseObj, obj)

    # Decide which axis we're going to extrude along
    points = np.array(
        rt.polyop.getVerts(
            baseObj.baseObject, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(baseObj))
        )
    )
    bbox_min, bbox_max = np.min(points, axis=0), np.max(points, axis=0)
    bbox_extent = bbox_max - bbox_min
    print("Bbox extents:", bbox_extent)
    extrusion_axis = int(np.argmin(bbox_extent))

    # Make the points planar
    print("Planarizing along", "xyz"[extrusion_axis], "axis")
    plane_normal = np.zeros(3)
    plane_normal[extrusion_axis] = 1
    rt.polyop.moveVertsToPlane(
        baseObj.baseObject,
        list(range(1, len(points) + 1)),
        rt.Point3(*plane_normal.tolist()),
        float(np.mean(points[:, extrusion_axis])),
    )

    # Flip the faces' normals for extrusion
    nm = rt.NormalModifier()
    nm.flip = True
    rt.addmodifier(baseObj, nm)
    rt.maxOps.collapseNodeTo(baseObj, 1, True)

    # Select all the faces
    rt.polyop.setFaceSelection(
        baseObj, list(range(1, rt.polyop.getNumFaces(baseObj) + 1))
    )

    # Apply the Face Extrusion modifier
    fe = rt.Face_Extrude()
    fe.amount = 300
    rt.addmodifier(baseObj, fe)
    rt.maxOps.collapseNodeTo(baseObj, 1, True)

    # Convert to editable poly again
    rt.addmodifier(baseObj, rt.Edit_Poly())
    rt.maxOps.collapseNodeTo(baseObj, 1, True)

    # And then cap holes
    rt.addmodifier(baseObj, rt.Cap_Holes())
    rt.maxOps.collapseNodeTo(baseObj, 1, True)

    # Triangulate the faces
    ttp = rt.Turn_To_Poly()
    ttp.limitPolySize = True
    ttp.maxPolySize = 3
    rt.addmodifier(baseObj, ttp)
    rt.maxOps.collapseNodeTo(baseObj, 1, True)

    # Parent the collision object to the target object
    baseObj.parent = target

    # Rename the first object to match the selected object
    baseObj.name = target.name + "-Mcollision"

    # Validate that the object name is valid
    assert (
        b1k_pipeline.utils.parse_name(baseObj.name) is not None
    ), f"Done, but please fix invalid name {baseObj.name} for collision object"

    return baseObj


def select_and_flat_floor_to_cmesh():
    collision_objs = list(rt.selection)

    # Ask the user to pick a target object
    target = rt.pickobject()

    # Merge the collision objects
    flat_floor_to_cmesh(collision_objs, target)


def flat_floor_to_cmesh_button():
    try:
        select_and_flat_floor_to_cmesh()
        # rt.messageBox("Success!")
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return


if __name__ == "__main__":
    flat_floor_to_cmesh_button()
