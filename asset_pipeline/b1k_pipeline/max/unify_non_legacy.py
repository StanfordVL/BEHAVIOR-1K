from collections import defaultdict, Counter
import filecmp
import glob
import hashlib
import os
import pathlib
import random
import shutil
import numpy as np

import tqdm

import pymxs

rt = pymxs.runtime

MARGIN = 500  # mm, or 50 centimeters
from b1k_pipeline.max.new_sanity_check import SanityCheck

PRECHECK_COLLISION = False
PRECHECK_EMPTY = False


def bin_files():
    max_files = glob.glob(r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects\*\processed.max")
    max_files = sorted(
        [pathlib.Path(x) for x in max_files if "legacy_" not in x and "batch-" not in x]
    )

    # Deterministic shuffle
    random.seed(1337)
    random.shuffle(max_files)

    print(len(max_files), "files found")

    batch_size = 50
    bins = [
        set(max_files[start : start + batch_size])
        for start in range(0, len(max_files), batch_size)
    ]

    # Apply displacements to avoid texture collisions
    displacements = {
        "curtain-fh": 2,
        "toaster_oven-kf": 7,
        "food_processor-ht": 8,
        "hockey_stick-ja": 6,
        "jewelry_cleaner-bh": 5,
        "tequilla-jo": 4,
        "baby_bottle-hv": 13,
        "coconut_milk-oq": 11,
        "log-da": 12,
    }
    for d, tgt in displacements.items():
        # Convert d to its path form
        p = (
            pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects")
            / d
            / "processed.max"
        )

        # Remove d from the bin it shows up in
        for bin in bins:
            if p in bin:
                bin.remove(p)
                break
        else:
            raise ValueError("Could not find displaced path" + str(p))

        # Add it to its new bin
        bins[tgt].add(p)

    # Identify any texture collisions
    if PRECHECK_COLLISION:
        any_collision = False
        texture_hashes = {}
        hash_to_file = {}
        for i, files in enumerate(tqdm.tqdm(bins)):
            texture_hashes[i] = defaultdict(set)
            hash_to_file[i] = defaultdict(set)
            for f in files:
                texture_dir = f.parent / "textures"
                if texture_dir.exists():
                    textures = list(texture_dir.rglob("*"))

                    # First check for collision within the same file
                    c = Counter([t.name.lower() for t in textures])
                    for k, v in c.items():
                        if v > 1:
                            print("Texture collision for", k, "within file", f)
                            any_collision = True

                    for t in textures:
                        # Skip directories & flatten them
                        if t.is_dir() or t.name == "Thumbs.db":
                            continue

                        with open(t, "rb") as f:
                            f_hash = hashlib.md5(f.read()).hexdigest()
                            texture_hashes[i][t.name.lower()].add(f_hash)
                            hash_to_file[i][(t.name.lower(), f_hash)].add(t)

            needs_removal = set()
            for basename, hashes in texture_hashes[i].items():
                if len(hashes) > 1:
                    providers = [
                        x for h in hashes for x in hash_to_file[i][(basename, h)]
                    ]
                    print(
                        "Texture collision for",
                        basename,
                        "between",
                        providers,
                        "in bin",
                        i,
                    )
                    for provider in providers[1:]:
                        relpath = provider.relative_to(
                            r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects"
                        )
                        needs_removal.add(relpath.parts[0])
                    any_collision = True

            for r in needs_removal:
                print("Remove", r, ":", i)

        if any_collision:
            return

    # Check if any of the files are empty
    if PRECHECK_EMPTY:
        any_empty = False
        for f in tqdm.tqdm(max_files):
            if len(rt.getMAXFileObjectNames(f, quiet=True)) == 0:
                print("Empty file", f)
                any_empty = True

        if any_empty:
            return

    for i, files in enumerate(bins):
        # Create an empty file
        rt.resetMaxFile(rt.name("noPrompt"))

        # Create the directory
        file_root = pathlib.Path(
            r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects\batch-%02d" % i
        )
        max_path = file_root / "processed.max"
        if max_path.exists():
            continue

        print("Starting file", i)
        file_root.mkdir(exist_ok=True, parents=True)

        textures_dir = file_root / "textures"
        textures_dir.mkdir(exist_ok=True, parents=True)

        # Merge in each file
        current_x_coordinate = 0
        for p in tqdm.tqdm(files):
            f = str(p)

            # Load everything in
            success, meshes = rt.mergeMaxFile(
                f,
                rt.Name("select"),
                rt.Name("autoRenameDups"),
                rt.Name("renameMtlDups"),
                quiet=True,
                mergedNodes=pymxs.byref(None),
            )
            assert success, f"Could not merge {f}"
            assert len(meshes) > 0, f"No objects found in file {f}"

            # Unhide everything
            for x in meshes:
                x.isHidden = False

            # Take everything in the selection and place them appropriately
            bb_min = np.min([x.min for x in meshes], axis=0)
            bb_max = np.max([x.max for x in meshes], axis=0)
            bb_size = bb_max - bb_min

            # Calculate the offset that everything needs to move by for the minimum to be at current_x_coordinate, 0, 0
            offset = np.array([current_x_coordinate, 0, 0]) - bb_min
            offset = offset.tolist()

            # Move everything by the offset amount
            for x in meshes:
                if x.parent:
                    continue
                x.position += rt.Point3(*offset)

            # Increment the current x position
            current_x_coordinate += bb_size[0] + MARGIN

            # Copy over the textures
            texture_dir = p.parent / "textures"
            if texture_dir.exists():
                for t in texture_dir.rglob("*"):
                    # Skip directories & flatten them
                    if t.is_dir() or t.name == "Thumbs.db":
                        continue

                    target_path = textures_dir / t.name
                    if target_path.exists():
                        # Check if the file contents are equal
                        with (
                            open(t, "rb") as t_file,
                            open(target_path, "rb") as target_path_file,
                        ):
                            t_hash = hashlib.md5(t_file.read()).hexdigest()
                            target_path_hash = hashlib.md5(
                                target_path_file.read()
                            ).hexdigest()
                        assert (
                            t_hash == target_path_hash
                        ), f"Two different texture files including {t} want to be copied to {target_path}"
                        continue

                    shutil.copy(t, target_path)

        # After loading everything, run a sanity check
        sc = SanityCheck().run()
        if sc["ERROR"]:
            raise ValueError(f"Sanity check failed for {i}:\n{sc['ERROR']}")

        # Save the output file.
        rt.saveMaxFile(str(max_path), quiet=True)

    print("Done!")


if __name__ == "__main__":
    bin_files()
