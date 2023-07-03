"""
Helper script to download OmniGibson dataset and assets.
"""
import os
os.environ["OMNIGIBSON_NO_OMNIVERSE"] = "1"

from omnigibson.macros import gm
from omnigibson.utils.asset_utils import download_og_dataset, download_assets
import click


def main():
    # Ask user which dataset to install
    data_path = gm.DATA_PATH if os.path.isabs(gm.DATA_PATH) else os.path.abspath(f"omnigibson/{gm.DATA_PATH}")
    print(f"OmniGibson will now install the full dataset and assets (~20GB) under {data_path}.")
    print(f"If you want to install data under a different path, please change the DATA_PATH variable in omnigibson/macros.py and rerun scripts/download_dataset.py.")
    if click.confirm("Do you want to continue?"):
        # Only download if the dataset path doesn't exist
        if not os.path.exists(gm.DATASET_PATH):
            print("Downloading dataset...")
            download_og_dataset()

        # Only download if the asset path doesn't exist
        if not os.path.exists(gm.ASSET_PATH):
            print("Downloading assets...")
            download_assets()

        print("\nOmniGibson setup completed!\n")
    else:
        print("You chose not to install dataset for now. You can install it later by running python scripts/download_dataset.py.")


if __name__ == "__main__":
    main()
