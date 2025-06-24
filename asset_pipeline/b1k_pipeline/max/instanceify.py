import collections
import random

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R

rt = pymxs.runtime
RMSD_THRESHOLD = 1e-3

local_coordsys = pymxs.runtime.Name("local")


def are_equal(target, base):
    # Check with align_vectors' RMSD
    pick_100 = random.sample(range(1, rt.polyop.getNumVerts(base) + 1), 100)
    src_pc = np.array(rt.polyop.getVerts(base, list(pick_100)))
    src_pc -= np.mean(src_pc, axis=0)
    src_quat = base.rotation
    src_rot = R.from_quat([src_quat.x, src_quat.y, src_quat.z, src_quat.w])
    src_pc = src_rot.inv().apply(src_pc)
    src_pc /= np.array(base.scale)

    tgt_pc = np.array(rt.polyop.getVerts(target, list(pick_100)))
    tgt_pc -= np.mean(tgt_pc, axis=0)
    tgt_quat = target.rotation
    tgt_rot = R.from_quat([tgt_quat.x, tgt_quat.y, tgt_quat.z, tgt_quat.w])
    tgt_pc = tgt_rot.inv().apply(tgt_pc)
    tgt_pc /= np.array(target.scale)
    assert src_pc.shape == tgt_pc.shape, "objects have different vertex counts"

    is_src_diff_scaled = not np.allclose(base.scale, base.scale[0])
    is_tgt_diff_scaled = not np.allclose(target.scale, target.scale[0])
    is_diff_scaled = is_src_diff_scaled or is_tgt_diff_scaled

    rmsd = R.align_vectors(
        tgt_pc,
        src_pc,
    )[1]

    # print(rmsd)

    return rmsd < RMSD_THRESHOLD, is_diff_scaled


def align_pivots(tgt_obj, src_obj):
    # Center pivot to object
    src_obj.pivot = src_obj.center
    pick_1000 = random.sample(range(1, rt.polyop.getNumVerts(src_obj) + 1), 100)
    src_pc = np.array(rt.polyop.getVerts(src_obj, list(pick_1000)))
    old_src_quat = [
        src_obj.rotation.x,
        src_obj.rotation.y,
        src_obj.rotation.z,
        src_obj.rotation.w,
    ]
    old_src_R = R.from_quat(old_src_quat)

    # Center pivot to object
    tgt_obj.pivot = tgt_obj.center
    tgt_pc = np.array(rt.polyop.getVerts(tgt_obj, list(pick_1000)))
    assert (
        src_pc.shape == tgt_pc.shape
    ), "source object ({}) and target object ({}) are not instances of each other - they have different number of vertices.".format(
        src_obj.name, tgt_obj.name
    )

    r = R.align_vectors(
        tgt_pc - np.mean(tgt_pc, axis=0),
        src_pc - np.mean(src_pc, axis=0),
    )[0]

    new_rotation = r * old_src_R  # desired orientation in the world frame

    old_target_quat = [
        tgt_obj.rotation.x,
        tgt_obj.rotation.y,
        tgt_obj.rotation.z,
        tgt_obj.rotation.w,
    ]
    old_target_R = R.from_quat(old_target_quat)

    new_rotation_local = (
        old_target_R.inv() * new_rotation
    )  # desired orientation in the local frame of the target object
    delta_quat = rt.quat(*new_rotation_local.as_quat().tolist())

    coordsys = getattr(pymxs.runtime, "%coordsys_context")
    prev_coordsys = coordsys(local_coordsys, None)
    tgt_obj.rotation *= delta_quat
    tgt_obj.objectOffsetRot *= delta_quat
    tgt_obj.objectOffsetPos *= delta_quat
    coordsys(prev_coordsys, None)


def clone_and_align(target, base):
    new_objs = []
    success, new_objs = rt.maxOps.cloneNodes(
        base, cloneType=rt.name("instance"), newNodes=pymxs.byref(new_objs)
    )
    assert success, "Could not clone."
    (new_obj,) = new_objs
    new_obj.transform = target.transform
    new_obj.scale = target.scale

    return new_obj


def instanceify():
    message = ""

    # Group objects by vertex count and material
    coarse_groups = collections.defaultdict(list)
    for obj in rt.objects:
        if rt.classOf(obj) == rt.Editable_Poly:
            if len(obj.vertices) >= 1000:
                coarse_groups[(obj.material, len(obj.vertices))].append(obj)
            # else:
            #     message += f"Ignored low-vertex object {obj.name}\n"
        # else:
        #     message += f"Ignored non-EP object {obj.name}\n"
    print(message)

    # For each group, do narrower groups.
    print("Finding object groups.")
    real_groups = []
    for cg in coarse_groups.values():
        this_groups = []

        # We will sort objects by whether or not they belong to the most populous instance group
        bases = collections.Counter(obj.baseObject for obj in cg)
        cg.sort(key=lambda x: -bases[x.baseObject])

        for obj in cg:
            # Check every object against every group's leader
            found_group = False
            for group in this_groups:
                # TODO: Only check objects that are not already instances of the group's leader.
                if are_equal(obj, group[0]):
                    found_group = True
                    # Only append if this is not already an instance.
                    if obj.baseObject != group[0].baseObject:
                        group.append(obj)
                    break

            # If we did not find a group, create a new one.
            if not found_group:
                this_groups.append([obj])

        # Add these groups to the big list
        real_groups.extend(this_groups)

    print("Filtering object groups.")
    # Let's only do the 1+-member groups
    real_groups = [x for x in real_groups if len(x) > 1]
    real_groups.sort(key=lambda x: -sum(len(obj.vertices) for obj in x))

    # For now we only do the top group.
    if len(real_groups) == 0:
        rt.messageBox("No groups of uninstanced objects found.")
        return

    # Now select the objects for clone-and-align use.
    # print("Selecting.")
    # rt.select(group[1:])

    # rt.messageBox(f"Objects are selected now and can be replaced with clone-and-align. Use object {head.name} as the cloning base.")

    # Now go through the groups and make everything an instance
    replacements = 0
    for group in real_groups:
        if len(group) == 1:
            continue

        # Align the pivots if possible.
        head = group[0]
        any_diff_scaled = any(are_equal(head, obj)[1] for obj in group[1:])
        if not any_diff_scaled:
            print("Aligning pivots.")
            for other in group[1:]:
                align_pivots(other, head)

        # Get ready to clone.
        for other in group[1:]:
            new_obj = clone_and_align(other, head)
            new_obj.name = other.name
            rt.delete(other)
            replacements += 1

    # Report the results.
    rt.messageBox(f"Replaced {replacements} objects.")


if __name__ == "__main__":
    instanceify()
