from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import os
import re
import sys
import numpy as np
import tqdm

import matplotlib.pyplot as plt
import pybullet as p
import json
from fs.zipfs import ZipFS
from fs.osfs import OSFS
import fs.copy
import trimesh
import b1k_pipeline.utils


def load_mesh(mesh_fs, option_name, mesh_fns, output_fs, offset):
    # First, load into trimesh
    hulls = [
        b1k_pipeline.utils.load_mesh(
            mesh_fs, mesh_fn, force="mesh", skip_materials=True
        )
        for mesh_fn in mesh_fns
    ]
    # Check that none of the hulls has zero volume
    # if index != 0 and any(x.volume == 0 for x in hulls):
    #     return
    m = trimesh.util.concatenate(hulls)

    # Apply the desired offset if one is provided. Otherwise, center.
    m.apply_translation(-offset)

    # Scale the object to fit in the [1, 1, 1] bounding box
    # if scale is None:
    #     scale = 1 / m.bounding_box.extents.max()
    # m.apply_scale(scale)
    # for mesh in hulls:
    #     mesh.apply_scale(scale)

    # Apply a different color to each part
    b1k_pipeline.utils.save_mesh(m, output_fs, f"{option_name}.obj")


def select_mesh(target_output_fs, mesh_name, out_fs):
    with (
        target_output_fs.open("meshes.zip", "rb") as zip_file,
        ZipFS(zip_file) as zip_fs,
        zip_fs.opendir(mesh_name) as mesh_fs,
        out_fs.makedir(mesh_name) as mesh_out_fs,
    ):
        fs.copy.copy_fs(mesh_fs, mesh_out_fs)
        fs.copy.copy_file(mesh_fs, f"{mesh_name}.obj", mesh_out_fs, "visual.obj")
        # fs.copy.copy_file(mesh_fs, f"{mesh_name}.mtl", mesh_out_fs, f"{mesh_name}.mtl")
        # fs.copy.copy_dir(mesh_fs, "material", mesh_out_fs, "material")

        offset = b1k_pipeline.utils.load_mesh(
            mesh_fs, f"{mesh_name}.obj", force="mesh", skip_materials=True
        ).centroid

        # Load in each of the meshes
        with (
            target_output_fs.open("collision_meshes.zip", "rb") as zip_file,
            ZipFS(zip_file) as zip_fs,
            zip_fs.opendir(mesh_name) as mesh_fs,
        ):
            filenames = [x.name for x in mesh_fs.filterdir("/", files=["*.obj"])]
            filename_bases = {x.rsplit("-", 1)[0] for x in filenames}
            for i, fn in enumerate(sorted(filename_bases)):
                # Load the candidate
                selection_matching_pattern = re.compile(fn + r"-(\d+).obj")
                load_mesh(
                    mesh_fs,
                    fn,
                    [x for x in filenames if selection_matching_pattern.fullmatch(x)],
                    mesh_out_fs,
                    offset,
                )


# Start iterating.
def process_target(target, target_meshes):
    with OSFS(target) as target_output_fs, OSFS("/scr/cmesh_test_dataset") as out_fs:
        # Load the meshes and do the selection.
        for mesh_name in target_meshes:
            select_mesh(target_output_fs, mesh_name, out_fs)


def preprocess_target(target):
    candidates = []
    with OSFS(target) as target_output_fs:
        if not target_output_fs.exists("collision_meshes.zip"):
            return

        with (
            target_output_fs.open("collision_meshes.zip", "rb") as zip_file,
            ZipFS(zip_file) as zip_fs,
            target_output_fs.open("meshes.zip", "rb") as meshes_zip_file,
            ZipFS(meshes_zip_file) as meshes_zip_fs,
        ):
            for mesh_name in meshes_zip_fs.listdir("/"):
                parsed_name = b1k_pipeline.utils.parse_name(mesh_name)
                if not parsed_name:
                    print("Bad name", parsed_name)
                    continue
                mesh_model = parsed_name.group("model_id")
                mesh_link = parsed_name.group("link_name")
                if not mesh_link:
                    mesh_link = "base_link"
                should_convert = (
                    int(parsed_name.group("instance_id")) == 0
                    and not parsed_name.group("bad")
                    and parsed_name.group("joint_side") != "upper"
                )
                if not should_convert:
                    continue

                # collision_found = any("Mcollision" in item for item in meshes_zip_fs.listdir(mesh_name))
                # if collision_found:
                #     continue

                if not zip_fs.exists(mesh_name):
                    print("Missing mesh", mesh_name)
                    continue

                candidates.append(mesh_name)

    return candidates


def main():
    all_targets = ["/scr/BEHAVIOR-1K/asset_pipeline/obj_out"]

    # Now get a list of all the objects that we can process.
    print("Getting list of objects to process...")
    candidates = defaultdict(list)
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {}
        for target in tqdm.tqdm(all_targets):
            futures[executor.submit(preprocess_target, target)] = target

        for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
            target = futures[future]
            candidates[target] = future.result()

    print(
        "Total objects (including completed) in your batch:",
        sum(len(x) for x in candidates.values()),
    )

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = []
        for target, mesh_names in candidates.items():
            futures.append(executor.submit(process_target, target, mesh_names))

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    main()
