import trimesh
import numpy as np

import pymxs

rt = pymxs.runtime

MAX_VERTEX_COUNT = 60
REDUCTION_STEP = 5


def reduce_mesh(mesh):
    # Check its vertex count and reduce as necessary
    reduction_target_vertex_count = MAX_VERTEX_COUNT + REDUCTION_STEP
    reduced_mesh = mesh
    while len(reduced_mesh.vertices) > MAX_VERTEX_COUNT:
        # Reduction function takes a number of faces as its input. We estimate this, if it doesn't
        # work out, we keep trying with a smaller estimate.
        reduction_target_vertex_count -= REDUCTION_STEP
        if reduction_target_vertex_count < MAX_VERTEX_COUNT / 2:
            # Don't want to reduce too far
            raise ValueError("Vertex reduction failed.")

        reduction_target_face_count = int(
            reduction_target_vertex_count / len(mesh.vertices) * len(mesh.faces)
        )
        reduced_mesh = mesh.simplify_quadratic_decimation(reduction_target_face_count)

    return reduced_mesh.convex_hull


def convert_to_trimesh(obj, checks=True):
    # Get vertices and faces into numpy arrays for conversion
    verts = np.array(
        [rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))]
    )
    faces = (
        np.array(
            rt.polyop.getFacesVerts(
                obj, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
            )
        )
        - 1
    )
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"

    # Split the faces into elements
    elems = {
        tuple(rt.polyop.GetElementsUsingFace(obj, i + 1))
        for i in range(rt.polyop.GetNumFaces(obj))
    }
    # assert len(elems) <= 40, f"{obj.name} should not have more than 40 elements."
    elems = np.array(list(elems))
    assert not np.any(
        np.sum(elems, axis=0) > 1
    ), f"{obj.name} has same face appear in multiple elements"

    # Iterate through the elements
    meshes = []
    for i, elem in enumerate(elems):
        # Load the mesh into trimesh and assert convexity
        relevant_faces = faces[elem]
        m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
        m.remove_unreferenced_vertices()
        assert not checks or m.is_volume, f"{obj.name} element {i} is not a volume"
        # assert not checks or m.is_convex, f"{obj.name} element {i} is not convex"
        assert (
            not checks or len(m.split()) == 1
        ), f"{obj.name} element {i} has elements trimesh still finds splittable"
        meshes.append(m)

    return meshes


def process_convex_obj(obj, force=False):
    # Get the collision meshes into a set of trimeshes
    convex_meshes = convert_to_trimesh(obj, checks=False)

    # Check if any of the meshes have too many verts
    too_many_verts = any(len(m.vertices) > MAX_VERTEX_COUNT for m in convex_meshes)

    # Check if any of the meshes is not a volume
    not_volume = any(not m.is_volume for m in convex_meshes)

    if not force and not too_many_verts and not not_volume:
        return

    print("Reducing", obj.name)

    # Individually reduce each of the trimeshes. This has multiple effects: it will reduce the vertices
    # if the vertex count is too high, and replace the shape with its convex hull helping with the volumeness
    # issues.
    reduced_trimeshes = [reduce_mesh(m) for m in convex_meshes]
    assert all(
        m.is_volume for m in reduced_trimeshes
    ), f"{obj.name} reduced part is not a volume. Not quite sure how that can be."

    # Get a flattened list of vertices and faces
    all_vertices = []
    all_faces = []
    for split in reduced_trimeshes:
        vertices = [rt.Point3(*v.tolist()) for v in split.vertices]
        # Offsetting here by the past vertex count
        faces = [[v + len(all_vertices) + 1 for v in f.tolist()] for f in split.faces]
        all_vertices.extend(vertices)
        all_faces.extend(faces)

    # Remove the original mesh
    rt.polyop.deleteVerts(obj, rt.execute("#{1..%d}" % rt.polyop.GetNumVerts(obj)))

    # Add the vertices
    for v in all_vertices:
        rt.polyop.createVert(obj, v)

    # Add the faces
    for f in all_faces:
        rt.polyop.createPolygon(obj, f)

    # Update the mesh to reflect changes
    rt.update(obj)

    # Check that the new element count is the same as the split count
    elems = {
        tuple(rt.polyop.GetElementsUsingFace(obj, i + 1))
        for i in range(rt.polyop.GetNumFaces(obj))
    }
    assert len(elems) == len(
        convex_meshes
    ), f"{obj.name} has different number of elements in input vs output"
    elems = np.array(list(elems))
    assert not np.any(
        np.sum(elems, axis=0) > 1
    ), f"{obj.name} has same face appear in multiple elements"

    # Convert back to trimesh this time with checks to check volumeness, convexness, etc.
    # convert_to_trimesh(obj, checks=True)


def process_all_convex_meshes(objs=None):
    if objs is None:
        objs = list(rt.objects)
    for obj in objs:
        if (
            "Mcollision" not in obj.name
            and "Mfillable" not in obj.name
            and "Mopenfillable" not in obj.name
        ):
            continue

        process_convex_obj(obj)


def main():
    # for obj in rt.selection:
    #     assert (
    #         "Mcollision" in obj.name
    #         or "Mfillable" in obj.name
    #         or "Mopenfillable" in obj.name
    #     ), "Please select a collision or fillable object"
    #     process_convex_obj(obj, force=True)

    process_all_convex_meshes()


if __name__ == "__main__":
    main()
