import traceback
from pymxs import runtime as rt
import trimesh
import numpy as np

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name


def get_verts_for_obj(obj):
    return np.array(
        [rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))]
    )


def get_faces_for_obj(obj):
    try:
        return (
            np.array(
                rt.polyop.getFacesVerts(
                    obj.baseObject, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
                )
            )
            - 1
        )
    except:
        raise ValueError(f"Error getting faces for {obj.name}. Did you triangulate?")


def validate_collision_mesh(obj, max_elements=40, max_vertices_per_element=60):
    # Check that the name can be parsed correctly and that it contains the correct meta type
    # and no meta ID or sub ID
    parsed_name = parse_name(obj.name)
    assert parsed_name, f"{obj.name} has an invalid name."
    assert parsed_name.group("meta_type") in (
        "collision",
        "fillable",
        "openfillable",
    ), f"{obj.name} has an invalid meta type ({parsed_name.group('meta_type')})."
    assert (
        parsed_name.group("meta_id") is None
    ), f"{obj.name} has a meta ID ({parsed_name.group('meta_id')}). Remove it."
    assert (
        parsed_name.group("meta_subid") is None
    ), f"{obj.name} has a meta sub ID ({parsed_name.group('meta_subid')}). Remove it."

    # Check that it has a parent and the parent has the same mesh basename.
    assert obj.parent, f"{obj.name} has no parent."
    parent_parsed_name = parse_name(obj.parent.name)
    assert parent_parsed_name, f"{obj.parent.name} has an invalid name."
    assert (
        parsed_name.group("mesh_basename") == parent_parsed_name.group("mesh_basename")
    ), f"{obj.name} and {obj.parent.name} have different mesh basenames ({parsed_name.group('mesh_basename')} vs {parent_parsed_name.group('mesh_basename')})."

    # Check that the parent is not bad and is not an upper side
    assert not parent_parsed_name.group(
        "bad"
    ), f"Bad object {obj.parent.name} does not need a collision mesh."
    assert (
        parent_parsed_name.group("joint_side") != "upper"
    ), f"Upper side {obj.parent.name} does not need a collision mesh. Put one on the lower side if needed."

    # Expect that collision meshes do not share instances in the scene
    assert not [
        x for x in rt.objects if x.baseObject == obj.baseObject and x != obj
    ], f"{obj.name} should not have instances."

    # Check that there are no dead elements
    assert (
        rt.polyop.GetHasDeadStructs(obj) == 0
    ), f"{obj.name} has dead structs. Apply the Triangulate script."

    # Get vertices and faces into numpy arrays for conversion
    verts = get_verts_for_obj(obj)
    faces = get_faces_for_obj(obj)
    assert len(faces) > 0, f"{obj.name} has no faces."
    assert all(
        len(f) == 3 for f in faces
    ), f"{obj.name} has non-triangular faces. Apply the Triangulate script."

    assert not any(x < 0 for x in obj.scale), (
        f"{obj.name} has negative scale, probably because you mirrored it. Apply the Reset Xform button from the Utilities tab of the right-side menu. "
        "Then return to the Modify tab, right click on the Xform modifier in the modifier stack and select Collapse To."
    )

    # Split the faces into elements
    faces_not_yet_found = np.zeros(faces.shape[0], dtype=bool)
    elems = []
    while not np.all(faces_not_yet_found):
        next_not_found_face = int(np.where(~faces_not_yet_found)[0][0])
        elem = np.array(rt.polyop.GetElementsUsingFace(obj, [next_not_found_face + 1]))
        assert elem[next_not_found_face], "Searched face not found in element."
        elems.append(elem)
        faces_not_yet_found[elem] = True
    elems = np.array(elems)
    assert not np.any(
        np.sum(elems.astype(int), axis=0) > 1
    ), f"{obj.name} has same face appear in multiple elements"

    elems_by_volume = []

    # Iterate through the elements
    for i, elem in enumerate(elems):
        # Load the mesh into trimesh and assert convexity
        relevant_faces = faces[elem]
        m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
        m.remove_unreferenced_vertices()
        if max_vertices_per_element is not None:
            assert (
                len(m.vertices) <= max_vertices_per_element
            ), f"{obj.name} element {i} has too many vertices ({len(m.vertices)} > {max_vertices_per_element})"
        if not m.is_volume:
            # Get the element faces (True indices in `elem`) and select them in 3ds Max
            element_faces = (np.where(elem)[0] + 1).tolist()
            rt.polyop.setFaceSelection(obj, element_faces)
            raise ValueError(
                f"{obj.name} element {i} is not a volume. It's now selected for your viewing."
            )
        elems_by_volume.append((elem, m.volume))
        if not m.is_convex:
            print(
                f"WARNING: {obj.name} element {i} may be non-convex. The checker says so, but it's not 100% accurate, so please verify that all elements are indeed convex."
            )
        assert (
            len(m.split()) == 1
        ), f"{obj.name} element {i} has elements trimesh still finds splittable e.g. are not watertight / connected"

    if max_elements is not None:
        if len(elems) > max_elements:
            # Select the smallest element
            elems_by_volume.sort(key=lambda x: x[1])
            elem, volume = elems_by_volume[0]
            element_faces = (np.where(elem)[0] + 1).tolist()
            rt.polyop.setFaceSelection(obj, element_faces)

            raise ValueError(
                f"{obj.name} should not have more than {max_elements} elements. Has {len(elems)} elements. Selected the smallest element."
            )


if __name__ == "__main__":
    # assert len(rt.selection) == 1, "Please select a single object."
    objs = list(rt.selection)
    if not objs:
        objs = [x for x in rt.objects if "Mcollision" in x.name]
    for obj in objs:
        try:
            validate_collision_mesh(obj)
            print("Collision mesh is VALID:", obj.name)
        except Exception as e:
            print("Collision mesh is INVALID:", obj.name)
            print(traceback.format_exc())
