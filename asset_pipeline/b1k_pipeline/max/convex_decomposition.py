import numpy as np
import coacd
import trimesh
import subprocess

import pymxs

rt = pymxs.runtime

import os
import sys
import tempfile

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name
from b1k_pipeline.max.collision_vertex_reduction import reduce_mesh

VHACD_EXECUTABLE = r"D:\BEHAVIOR-1K\asset_pipeline\b1k_pipeline\vhacd2.exe"


def run_coacd(input_mesh, hull_count):
    coacd_mesh = coacd.Mesh(input_mesh.vertices, input_mesh.faces)
    result = coacd.run_coacd(
        coacd_mesh,
        max_convex_hull=hull_count,
    )
    output_meshes = []
    for vertices, faces in result:
        output_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        output_meshes.append(output_trimesh)
    return output_meshes


def run_vhacd(input_mesh, hull_count):
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.obj")
        out_path = os.path.join(
            td, "decomp.obj"
        )  # This is the path that VHACD outputs to.
        input_mesh.export(in_path)

        vhacd_cmd = [
            str(VHACD_EXECUTABLE),
            in_path,
            "-r",
            "1000000",
            "-d",
            "20",
            "-f",
            "flood",
            "-e",
            "0.01",
            "-p",
            "true",
            "-l",
            "2",
            "-v",
            "60",
            "-h",
            str(hull_count),
        ]
        try:
            proc = subprocess.run(
                vhacd_cmd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=td,
                check=True,
            )
            if not os.path.exists(out_path):
                raise ValueError(
                    "VHACD failed to produce an output file. VHACD output:\n"
                    + proc.stdout.decode("utf-8")
                )
            out_mesh = trimesh.load(
                out_path,
                file_type="obj",
                force="mesh",
                skip_material=True,
                merge_tex=True,
                merge_norm=True,
            )
            return out_mesh.split(only_watertight=False)
        except subprocess.CalledProcessError as e:
            print(e.output.decode("utf-8"))
            raise ValueError(f"VHACD failed with exit code {e.returncode}")


HULL_COUNTS = [16]
USE_METHODS = {
    "coacd": run_coacd,
    # "vhacd": run_vhacd,
}


def _create_collision_obj_from_verts_faces(vertices, faces, parent, tag):
    # Create a new node for the collision mesh
    collision_obj = rt.Editable_Mesh()
    rt.ConvertToPoly(collision_obj)
    parsed_name = parse_name(parent.name)
    # if parsed_name:
    #     collision_obj.name = f"{parsed_name.group('mesh_basename')}-Mcollision{tag}"
    # else:
    #     collision_obj.name = parent.name + "-Mcollision"
    collision_obj.name = "collisioncandidate" + tag
    collision_obj.rotation = parent.rotation
    collision_obj.position = parent.position

    # Add the vertices
    for v in vertices:
        rt.polyop.createVert(collision_obj, rt.Point3(*v.tolist()))

    # Add the faces
    for f in faces:
        rt.polyop.createPolygon(collision_obj, (f + 1).tolist())

    # Optionally set its wire color
    collision_obj.wirecolor = rt.yellow

    # Update the mesh to reflect changes
    rt.update(collision_obj)

    # Parent the mesh
    collision_obj.parent = parent

    return collision_obj


def generate_convex_decompositions(
    obj, preferred_method=None, preferred_hull_count=None
):
    if rt.classOf(obj) != rt.Editable_Poly:
        return

    # Get the vertices and faces
    verts = np.array(
        [rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))]
    )
    face_selection = rt.polyop.GetFaceSelection(obj)
    if face_selection.isEmpty:
        face_selection = rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
    faces = np.array(rt.polyop.getFacesVerts(obj, face_selection)) - 1
    assert all(
        len(f) == 3 for f in faces
    ), f"{obj.name} has non-triangular faces. Apply the Triangulate script."
    print("\nGenerating collision meshes for", obj.name)

    # Run the convex hull option
    tm = trimesh.Trimesh(vertices=verts, faces=faces)
    tm.remove_unreferenced_vertices()

    if preferred_method is None or preferred_method == "chull":
        chull = reduce_mesh(tm.convex_hull)
        # TODO: Quadric decimation
        convex_hull_obj = _create_collision_obj_from_verts_faces(
            chull.vertices,
            chull.faces,
            obj,
            ("_chull" if preferred_method is None else ""),
        )
        print("Generated convex hull", convex_hull_obj.name)

    # Run CoACD a number of times
    hull_counts = (
        [preferred_hull_count] if preferred_hull_count is not None else HULL_COUNTS
    )
    for method_name, method in USE_METHODS.items():
        if preferred_method is not None and method_name != preferred_method:
            continue
        for hull_count in hull_counts:
            meshes = method(tm, hull_count)
            reduced_meshes = [reduce_mesh(m) for m in meshes]

            # Get a flattened list of vertices and faces
            all_vertices = []
            all_faces = []
            for cmesh in reduced_meshes:
                # Offsetting here by the past vertex count
                all_faces.extend(cmesh.faces + len(all_vertices))
                all_vertices.extend(cmesh.vertices)
            all_vertices = np.array(all_vertices)
            all_faces = np.array(all_faces)
            collision_obj = _create_collision_obj_from_verts_faces(
                all_vertices,
                all_faces,
                obj,
                (
                    f"_{method_name}{hull_count}"
                    if preferred_method is None or preferred_hull_count is None
                    else ""
                ),
            )

            # Check that the new element count is the same as the split count
            elems = {
                tuple(rt.polyop.GetElementsUsingFace(collision_obj, i + 1))
                for i in range(rt.polyop.GetNumFaces(collision_obj))
            }
            assert (
                len(elems) == len(reduced_meshes)
            ), f"{obj.name} has different number of faces in collision mesh than in splits"
            elems = np.array(list(elems))
            assert not np.any(
                np.sum(elems, axis=0) > 1
            ), f"{obj.name} has same face appear in multiple elements"

            print(
                f"Generated {method_name} w/ max hull count {hull_count}, actual hull count {len(reduced_meshes)}, name {collision_obj.name}"
            )

    if preferred_method is None or preferred_hull_count is None:
        print("Don't forget to make a selection!")


def generate_all_missing_collision_meshes():
    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        parsed_name = parse_name(obj.name)
        if not parsed_name:
            continue
        if parsed_name.group("bad"):
            continue
        if parsed_name.group("instance_id") != "0":
            continue
        if parsed_name.group("meta_type"):
            continue
        if parsed_name.group("joint_side") == "upper":
            continue

        # Does it already have a collision mesh? If so, move on.
        for child in obj.children:
            parsed_child_name = parse_name(child.name)
            if not parsed_child_name:
                continue

            # Skip parts etc.
            if parsed_child_name.group("mesh_basename") != parsed_name.group(
                "mesh_basename"
            ):
                continue

            if parsed_child_name.group("meta_type") == "collision":
                print("Collision mesh already exists for", obj.name, ", skipping.")
                break
        else:
            # Generate teh collision mesh if we didnt already find one.
            generate_convex_decompositions(
                obj,
            )


def main():
    for obj in rt.selection:
        generate_convex_decompositions(obj)


if __name__ == "__main__":
    main()
