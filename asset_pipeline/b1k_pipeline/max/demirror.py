import re
from collections import defaultdict

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R

rt = pymxs.runtime

X_MIRROR_MATRIX = rt.Matrix3(
    rt.Point3(-1, 0, 0), rt.Point3(0, 1, 0), rt.Point3(0, 0, 1), rt.Point3(0, 0, 0)
)


def demirror():
    objects = rt.selection

    # First, find all the qualifying objects.
    assert all(
        np.allclose(obj.scale, 1) for obj in objects
    ), "Objects should have 1, 1, 1 scale before this feature is executed."
    bases = defaultdict(set)
    for obj in objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        neg_scale = any(x < 0 for x in obj.objectoffsetscale)
        bases[obj.baseObject].add(neg_scale)

    # Bases qualify only if they are negative in all instances.
    qualifying_bases = {base for base, scales in bases.items() if scales == {True}}
    qualifying_objects_per_base = {
        base: [obj for obj in objects if obj.baseObject == base]
        for base in qualifying_bases
    }

    # Start by adding the modifier to the first one.
    for base, objs in qualifying_objects_per_base.items():
        start_positions = {
            obj: np.mean(
                np.array(
                    rt.polyop.getVerts(
                        obj, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj))
                    )
                ),
                axis=0,
            )
            for obj in objs
        }

        # First apply the geometry mirroring
        baseobj = objs[0]
        modifier = rt.mirror()
        rt.addmodifier(baseobj, modifier)

        # TODO: Base this on bounding box and not the object
        object_in_pivot_frame = baseobj.objecttransform * rt.inverse(baseobj.transform)
        pivot_in_object_frame = rt.inverse(object_in_pivot_frame)

        modifier.Mirror_Center.rotation = object_in_pivot_frame.rotation
        modifier.Mirror_Center.position = pivot_in_object_frame.position
        assert np.allclose(
            modifier.Mirror_Center.position, pivot_in_object_frame.position
        )
        assert np.allclose(
            modifier.Mirror_Center.rotation, object_in_pivot_frame.rotation
        )

        rt.maxOps.collapseNodeTo(baseobj, 1, True)

        # Then apply the transform mirroring
        for obj in objs:
            obj.transform = X_MIRROR_MATRIX * obj.transform
            end_position = np.mean(
                np.array(
                    rt.polyop.getVerts(
                        obj, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj))
                    )
                ),
                axis=0,
            )
            diff = np.linalg.norm(end_position - start_positions[obj])
            assert (
                diff < 1
            ), f"{obj.name} should not have moved, but it moved by {diff} units."
            print("Demirrored ", obj.name)


def demirror_button():
    try:
        demirror()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Demirror complete - no errors found!")


if __name__ == "__main__":
    demirror_button()
