def select_bad_element(obj):
    # Assert that collision meshes do not share instances in the scene
    assert not [
        x for x in rt.objects if x.baseObject == obj.baseObject and x != obj
    ], f"{obj.name} should not have instances."

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
    assert len(elems) <= 32, f"{obj.name} should not have more than 32 elements."
    elems = np.array(list(elems))
    assert not np.any(
        np.sum(elems, axis=0) > 1
    ), f"{obj.name} has same face appear in multiple elements"

    # Iterate through the elements
    for i, elem in enumerate(elems):
        # Load the mesh into trimesh and assert convexity
        relevant_faces = faces[elem]
        m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
        m.remove_unreferenced_vertices()
        if not m.is_volume:
            rt.polyop.SetFaceSelection(obj, (elem + 1).tolist())
            print("Selected element", i)
            return

    print("No bad elements found!")


if __name__ == "__main__":
    select_bad_element(rt.selection[0])
