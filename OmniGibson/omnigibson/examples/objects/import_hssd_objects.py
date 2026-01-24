"""
Helper script to download OmniGibson dataset and assets.
Improved version that can import obj file and articulated file (glb, gltf).
"""

import hashlib
import pathlib
import traceback
import shutil
import tempfile
import pandas as pd
import os
import sys
from tqdm import tqdm

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_conversion_utils import (
    import_og_asset_from_urdf,
    generate_urdf_for_mesh,
)

gm.HEADLESS = True

DATASET_ROOT = pathlib.Path("/checkpoint/clear/cgokmen/behavior-data2/hssd")
DATASET_ROOT.mkdir(exist_ok=True)
ERRORS = DATASET_ROOT / "errors"
ERRORS.mkdir(exist_ok=True)
JOBS = DATASET_ROOT / "jobs"
JOBS.mkdir(exist_ok=True)
RESTART_EVERY = 128


def import_custom_object(
    orig_path: str,
    category: str,
    model: str,
):
    """
    Imports a custom-defined object asset into an OmniGibson-compatible USD format and saves the imported asset
    files to the custom dataset directory
    """

    model_root = DATASET_ROOT / "objects" / category / model
    success_file = model_root / "import.success"

    # Create a temporary working directory
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the GLB file to the temp dir first to prefix it with a letter-string.
            asset_path = os.path.join(temp_dir, f"hssd{model}.glb")
            shutil.copy2(orig_path, asset_path)

            # Try to generate URDF, may raise ValueError if too many submeshes
            urdf_path = generate_urdf_for_mesh(
                asset_path,
                temp_dir,
                category,
                model,
                collision_method="coacd",
                hull_count=128 if category == "stages" else 32,
                check_scale=False,
                rescale=False,
                overwrite=True,
            )
            assert urdf_path is not None, f"Failed to generate URDF for {asset_path}"

            # Convert to USD
            import_og_asset_from_urdf(
                category=category,
                model=model,
                urdf_path=str(urdf_path),
                collision_method=None,
                dataset_root=str(DATASET_ROOT),
                overwrite=False,
                use_usda=False,
            )

            # Touch a success file.
            success_file.touch()
    finally:
        if og.sim:
            og.clear()


def main():
    hssd_root = pathlib.Path("/checkpoint/clear/cgokmen/habitat-data/scene_datasets/hssd-hab")
    hssd_models_root = pathlib.Path("/checkpoint/clear/cgokmen/hssd-models")
    metadata = pd.read_csv(hssd_root / "metadata/hssd_obj_semantics_condensed.csv")

    models = sorted(
        [
            fn
            for fn in hssd_models_root.rglob("*.glb")
            if "filteredSupportSurface" not in fn.name and "collider" not in fn.name
        ]
    )

    # Re-sort jobs differently per run, so that if a previous array job failed it doesn't end up
    # with all the work again.
    models.sort(
        key=lambda x: hashlib.md5((str(x) + os.environ.get("SLURM_ARRAY_JOB_ID", default="")).encode()).hexdigest()
    )

    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])

    completed_count = 0
    for obj_path in tqdm(models[rank::world_size]):
        if obj_path.parts[-2] == "openings":
            category = "openings"
        elif obj_path.parts[-2] == "stages":
            category = "stages"
        else:
            # Check if it exists in the dataframe
            rows = metadata[metadata["Object Hash"] == obj_path.stem]
            if not rows.empty:
                category = rows.iloc[0][
                    "Semantic Category:\nCONDENSED\n\nThis is an effort to condense the semantic categories by a couple hundred"
                ]
            else:
                print(f"Warning: {obj_path.stem} not found in metadata, defaulting to 'object'")
                category = "object"

        # Sanitize both category and model names to contain only letters (and underscores for category)
        category = "".join(c if c.isalnum() or c == "_" else "_" for c in category.lower())
        model = "hssd" + "".join(c if c.isalnum() else "" for c in obj_path.stem)

        model_root = DATASET_ROOT / "objects" / category / model
        success_file = model_root / "import.success"
        if model_root.exists():
            # Check if we're fully done.
            if success_file.exists():
                continue

            # Otherwise nuke the directory
            shutil.rmtree(model_root)

        try:
            import_custom_object(
                orig_path=obj_path,
                category=category,
                model=model,
            )
            completed_count += 1
        except Exception as e:
            print(f"Error processing {obj_path}: {e}")
            # Log the error
            with open(ERRORS / obj_path.stem, "w") as f:
                f.write(traceback.format_exc())

        if completed_count >= RESTART_EVERY:
            return

    # If we reach here, we're done. Record the rank success.
    (JOBS / f"{rank}.success").touch()

    og.shutdown()


if __name__ == "__main__":
    main()
