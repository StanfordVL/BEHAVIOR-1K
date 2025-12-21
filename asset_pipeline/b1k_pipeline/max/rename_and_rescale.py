from collections import defaultdict
import csv
import json
import sys
import tqdm

import numpy as np

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pathlib

import pymxs
from trimesh.transformations import transform_points


import b1k_pipeline.utils
from b1k_pipeline.max.new_sanity_check import SanityCheck

rt = pymxs.runtime


RECORD_RELPATH = "rename_and_rescale.success"


def compute_bounding_box(objs):
    objs = [x for x in objs if rt.classOf(x) == rt.Editable_Poly]
    (base,) = [
        x
        for x in objs
        if b1k_pipeline.utils.parse_name(x.name).group("link_name")
        in (None, "", "base_link")
        and not b1k_pipeline.utils.parse_name(x.name).group("meta_type")
    ]
    transform = np.eye(4)
    transform[:3] = b1k_pipeline.utils.mat2arr(base.transform).T
    invt = np.linalg.inv(transform)

    all_pts = []
    for obj in objs:
        # Skip non-lower ends
        if b1k_pipeline.utils.parse_name(obj.name).group("joint_side") == "upper":
            continue

        # Get all the vertices
        X = np.array(
            rt.polyop.getVerts(obj, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj)))
        )

        # Convert them into the base frame
        pts = transform_points(X, invt)

        # Add them into the list
        all_pts.append(pts.min(axis=0))
        all_pts.append(pts.max(axis=0))

    # Now fit a bounding box in that space
    all_pts = np.array(all_pts)
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    return np.abs(bb_max - bb_min)


def processFile(pipeline_fs, target, renames, deletions, avg_dims):
    # Load file, fixing the units
    max_path = pipeline_fs.target(target).getsyspath("processed.max")
    assert rt.loadMaxFile(max_path, useFileUnits=False, quiet=True)
    assert rt.units.systemScale == 1, "System scale not set to 1mm."
    assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

    # Take care of resizing.
    if False:  # "objects/legacy_" in target:
        # First, group things by object ID
        objs_by_model = defaultdict(list)
        for obj in rt.objects:
            match = b1k_pipeline.utils.parse_name(obj.name)
            if not match:
                continue
            cat = match.group("category")
            model_id = match.group("model_id")
            instance_id = match.group("instance_id")
            objs_by_model[(cat, model_id, instance_id)].append(obj)

        # Identify the base object.
        # This is the object that has the same ID as the file name.
        base_key = next(
            k for k in objs_by_model.keys() if k[1] == target.split("-")[-1]
        )
        base_objs = objs_by_model[base_key]
        bb_ext = np.sort(compute_bounding_box(base_objs))
        avg_cat_dims = np.sort(avg_dims[base_key[0]])
        scale_factors = avg_cat_dims / bb_ext

        # Make sure that the max scale is not more than x times the min scale
        if scale_factors.max() / scale_factors.min() > 10:
            raise ValueError(
                f"Object {base_key} has scales that vary by too much: {scale_factors}"
            )
        scale_factor = float(scale_factors.min())

        # Apply the scale
        for obj in rt.objects:
            # Multiply everything
            obj.position *= scale_factor
            obj.objectoffsetpos *= scale_factor
            obj.objectoffsetscale *= scale_factor

    # Now take care of renames and deletions
    objects = list(rt.objects)
    for obj in objects:
        match = b1k_pipeline.utils.parse_name(obj.name)
        if not match:
            continue

        # Check if it's on the deletion queue
        model_id = match.group("model_id")
        assert len(model_id) == 6, f"Object ID {model_id} is not 6 digits long."
        # if match.group("model_id") in deletions:
        #     rt.delete(obj)
        #     continue

        # Otherwise check if it's in the rename list
        category = match.group("category")
        rename_key = f"{category}-{model_id}"
        if model_id in renames:
            old_name, new_name = renames[model_id]
            # assert rename_key == old_name, f"Rename key {rename_key} does not match old name {old_name}."
            obj.name = obj.name.replace(rename_key, new_name)

    # Run a final sanity check
    sc = SanityCheck().run()
    if sc["ERROR"]:
        raise ValueError(f"Sanity check failed for {target}:\n{sc['ERROR']}")

    # Save again.
    rt.saveMaxFile(max_path)

    # Touch file to record the file as processed
    pipeline_fs.target_output(target).touch(RECORD_RELPATH)


def rename_and_rescale_all_files():
    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    targets = b1k_pipeline.utils.get_targets("combined")

    # Load data for renames
    with pipeline_fs.open("metadata/object_renames.csv", "r") as f:
        reader = csv.DictReader(f)
        renames = {}
        for row in reader:
            obj_id = row["ID (auto)"]
            old_cat = row["Original category (auto)"]
            new_cat = row["New Category"]
            in_name = f"{old_cat}-{obj_id}"
            out_name = f"{new_cat}-{obj_id}"
            renames[obj_id] = (in_name, out_name)

    # Load files for deletion
    with pipeline_fs.open("metadata/deletion_queue.csv", "r") as f:
        reader = csv.DictReader(f)
        deletions = set()
        # for row in reader:
        #     obj_name = row["Object"]
        #     obj_id = obj_name.split("-")[1]
        #     assert len(obj_id) == 6, f"Deletion object ID {obj_id} is not 6 digits long."
        #     deletions.add(obj_id)

    # Load the scale needed for each category
    # with open(r"C:\Users\Cem\research\iGibson-dev\igibson\data\ig_dataset\metadata\avg_category_specs.json", "r") as f:
    #     avg_specs = json.load(f)
    # avg_dims = {cat: np.array(spec["size"]) * 1000 for cat, spec in avg_specs.items()}
    avg_dims = {}

    # Find the targets that contain these objects
    selected_targets = []
    print("Preprocessing targets...")
    for target in tqdm.tqdm(targets):
        # See if the target was already processed
        # if pipeline_fs.target_output(target).exists(RECORD_RELPATH):
        #     continue

        # Open the target's object list
        # with pipeline_fs.target_output(target).open("object_list.json", "r") as f:
        #     object_list = set(json.load(f)["needed_objects"])
        mesh_list = rt.getMAXFileObjectNames(
            pipeline_fs.target(target).getsyspath("processed.max"), quiet=True
        )
        match_list = [b1k_pipeline.utils.parse_name(mesh) for mesh in mesh_list]
        object_list = {
            match.group("category") + "-" + match.group("model_id")
            for match in match_list
            if match
        }
        ids = {x.split("-")[1] for x in object_list}

        # See if the target needs any of the operations
        has_rename = ids & set(renames.keys())
        has_deletion = ids & deletions
        has_scale = False  # "objects/legacy_" in target
        if has_rename or has_deletion or has_scale:
            selected_targets.append(target)

    print("Remaining files:", len(selected_targets))

    # Before processing, check if any of the remaining targets are symlinks, and unprotect them
    paths = [
        pathlib.Path(pipeline_fs.target(target).getsyspath("processed.max"))
        for target in selected_targets
    ]
    symlinks = [p for p in paths if p.is_symlink()]
    if symlinks:
        print("Found symlinks. Run below commands to unprotect them in DVC.")
        for batch_start in range(0, len(symlinks), 50):
            print(
                "dvc unprotect",
                " ".join(
                    str(p.relative_to(b1k_pipeline.utils.PIPELINE_ROOT))
                    for p in symlinks[batch_start : batch_start + 50]
                ),
            )
        return

    for i, f in enumerate(sorted(selected_targets)):
        print(f"Processing file {i + 1}/{len(selected_targets)}: {f}")
        try:
            processFile(pipeline_fs, f, renames, deletions, avg_dims)
        except Exception as e:
            print(f, e)


if __name__ == "__main__":
    rename_and_rescale_all_files()
