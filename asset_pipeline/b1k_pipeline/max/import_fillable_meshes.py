from collections import defaultdict
import pathlib
import numpy as np
import json
from fs.osfs import OSFS

import pymxs

rt = pymxs.runtime

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name, load_mesh, PIPELINE_ROOT
from b1k_pipeline.max.replace_bad_object import (
    node_bounding_box_incl_children,
    rotation_only_transform,
)

FILLABLE_DIR = pathlib.Path(r"D:\fillable-10-21")
FILLABLE_ASSIGNMENTS = {
    k: v
    for k, v in json.loads(
        (FILLABLE_DIR / "fillable_assignments_2.json").read_text()
    ).items()
    if v in {"dip", "ray", "combined", "generated"}
}
BOUNDING_BOX_DATA_PATH = FILLABLE_DIR / "fillable_bboxes.json"
BOUNDING_BOX_DATA = json.loads(BOUNDING_BOX_DATA_PATH.read_text())
KEY_PATH = FILLABLE_DIR / "omnigibson.key"
LOG_PATH = FILLABLE_DIR / "import_fillable_meshes.log"
REMOVE_EXISTING = True
CHECK_MATCHING_SIZE = True


def get_bounding_box_from_usd(obj_dir):
    offset, size = BOUNDING_BOX_DATA[obj_dir.parts[-1]]
    return np.array(offset), np.array(size)


def import_fillable_volumes(model_id, object_links):
    print("Importing fillable mesh for", model_id)

    # Remove any fillable volumes that this object already has
    if REMOVE_EXISTING:
        to_remove = []
        for cand_obj in rt.objects:
            match = parse_name(cand_obj.name)
            if not match:
                continue
            if match.group("model_id") != model_id:
                continue
            if match.group("meta_type") not in ("fillable", "openfillable"):
                continue
            to_remove.append(cand_obj)

        for obj in to_remove:
            rt.delete(obj)

    # Find the directory corresponding to this object
    (model_dir,) = list(
        {x.parent for x in FILLABLE_DIR.glob(f"objects/*/{model_id}/bbox.json")}
    )

    # Find the fillable mesh files
    fillable_files = list(model_dir.glob("fillable---*---*---*.obj"))
    assert (
        len(fillable_files) >= 1
    ), f"Expected at least one fillable mesh for {model_id}"
    by_link_and_kind = defaultdict(lambda: defaultdict(list))
    for fillable_file in fillable_files:
        link, idx, kind = fillable_file.stem.split("---")[1:]
        by_link_and_kind[link][kind].append(fillable_file)

    # From the USD read the bounding box info
    usd_bbox_offset, usd_bbox_size = get_bounding_box_from_usd(model_dir)

    # Get information about the base link of the object
    base_link = object_links["base_link"]
    base_rot = base_link.rotation
    base_link_rotation_transform = rotation_only_transform(base_link.transform)

    # Compute the full object bbox by looking at all the links
    bbox_points = []
    for link_obj in object_links.values():
        bbox_points.extend(
            node_bounding_box_incl_children(
                link_obj, base_link_rotation_transform, only_canonical=True
            )
        )
    bbox_min = np.min(bbox_points, axis=0)
    bbox_max = np.max(bbox_points, axis=0)
    bbox_ctr_rotated = (bbox_min + bbox_max) / 2
    base_link_rotated = bbox_ctr_rotated - usd_bbox_offset
    base_link_world = (
        base_link_rotation_transform @ np.concatenate([base_link_rotated, [1]])
    )[:3]
    bbox_size = bbox_max - bbox_min

    # Compare the bounding box sizes
    if CHECK_MATCHING_SIZE and not np.allclose(bbox_size, usd_bbox_size, atol=10):
        msg = f"Bounding box sizes do not match for {model_id}. USD has {usd_bbox_size}, max has {bbox_size}, skipping."
        print(msg)

        # Append the message to the log file too
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

        return False

    for link_name, kind_files in by_link_and_kind.items():
        link_obj = object_links[link_name]
        parsed_link_name = parse_name(link_obj.name)
        for kind, files in kind_files.items():
            fillable_meshes = []
            for path in files:
                fs = OSFS(str(path.parent))
                filename = path.name
                fillable_mesh = load_mesh(
                    fs, filename, force="mesh", skip_materials=True
                )
                if not fillable_mesh.is_volume:
                    fillable_mesh = load_mesh(
                        fs, filename, force="mesh", process=False, skip_materials=True
                    )
                fillable_meshes.append(fillable_mesh)

            # Get a flattened list of vertices and faces
            all_vertices = []
            all_faces = []
            for split in fillable_meshes:
                vertices = [rt.Point3(*(v * 1000).tolist()) for v in split.vertices]
                # Offsetting here by the past vertex count
                faces = [
                    [v + len(all_vertices) + 1 for v in f.tolist()] for f in split.faces
                ]
                all_vertices.extend(vertices)
                all_faces.extend(faces)

            # Create a new node for the fillable mesh
            fillable_obj = rt.Editable_Mesh()
            rt.ConvertToPoly(fillable_obj)
            fillable_type = "Mfillable" if kind == "enclosed" else "Mopenfillable"
            fillable_obj.name = (
                f"{parsed_link_name.group('mesh_basename')}-{fillable_type}"
            )

            # Add the vertices
            for v in all_vertices:
                rt.polyop.createVert(fillable_obj, v)

            # Add the faces
            for f in all_faces:
                rt.polyop.createPolygon(fillable_obj, f)

            # So far, everything is in the local coordinate system, e.g. where they would go if the
            # object base link was at the origin and had no rotation. We need to move it to where we
            # think the base link is (this is unknown in 3ds Max but we read it from the USD).
            fillable_obj.rotation = base_rot
            fillable_obj.position = rt.Point3(*base_link_world.tolist())

            # Optionally set its wire color
            fillable_obj.wirecolor = rt.yellow

            # Update the mesh to reflect changes
            rt.update(fillable_obj)

            # Parent the mesh
            fillable_obj.parent = link_obj

            # Check that the new element count is the same as the split count
            elems = {
                tuple(rt.polyop.GetElementsUsingFace(fillable_obj, i + 1))
                for i in range(rt.polyop.GetNumFaces(fillable_obj))
            }
            assert (
                len(elems) == len(fillable_meshes)
            ), f"{fillable_obj.name} has different number of faces in fillable mesh than in splits"
            elems = np.array(list(elems))
            assert not np.any(
                np.sum(elems, axis=0) > 1
            ), f"{fillable_obj.name} has same face appear in multiple elements"

            # Hide the mesh
            fillable_obj.isHidden = True

    return True


def process_current_file():
    # Iterate through the objects in the file and build the link lists
    object_links = defaultdict(dict)
    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        match = parse_name(obj.name)
        if not match:
            continue

        if match.group("instance_id") != "0":
            continue

        if match.group("bad"):
            continue

        if match.group("meta_type"):
            continue

        if match.group("joint_side") == "upper":
            continue

        link_name = (
            match.group("link_name") if match.group("link_name") else "base_link"
        )
        object_links[match.group("model_id")][link_name] = obj

    # For each object, try to import the fillable volumes
    availables = set(FILLABLE_ASSIGNMENTS.keys()) & set(object_links.keys())

    return any(
        [
            import_fillable_volumes(model_id, object_links[model_id])
            for model_id in availables
        ]
    )


if __name__ == "__main__":
    process_current_file()
