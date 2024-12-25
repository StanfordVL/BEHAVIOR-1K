"""
Helper script to download OmniGibson dataset and assets.
"""

import os

os.environ["OMNIGIBSON_NO_OMNIVERSE"] = "1"

import click

from omnigibson.macros import gm
from omnigibson.utils.asset_utils import download_assets, download_og_dataset


def main():
    # Only execute if the dataset path or asset path does not exist
    dataset_exists, assets_exist = os.path.exists(gm.DATASET_PATH), os.path.exists(gm.ASSET_PATH)
    if not (dataset_exists and assets_exist):
        # Ask user which dataset to install
        print(f"OmniGibson will now install data under the following locations:")
        print(f"    dataset (~25GB): {gm.DATASET_PATH}")
        print(f"    assets (~2.5GB): {gm.ASSET_PATH}")
        print(
            f"If you want to install data under a different path, please change the DATA_PATH variable in omnigibson/macros.py and rerun omnigibson/download_datasets.py."
        )
        if click.confirm("Do you want to continue?", default=True):
            # Only download if the dataset path doesn't exist
            if not dataset_exists:
                print("Downloading dataset...")
                download_og_dataset()

            # Only download if the asset path doesn't exist
            if not assets_exist:
                print("Downloading assets...")
                download_assets()

            print("\nOmniGibson setup completed!\n")
        else:
            print(
                "You chose not to install dataset for now. You can install it later by running python omnigibson/download_datasets.py."
            )


if __name__ == "__main__":
    main()
