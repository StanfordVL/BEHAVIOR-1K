"""
Helper script to setup this OmniGibson repository. Configures environment and downloads assets
"""
import os
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import download_demo_data, download_og_dataset, download_assets
from omnigibson.utils.ui_utils import choose_from_options



def main():
    # Ask user which dataset to install
    print("Welcome to OmniGibson!")
    print()

    # Only download if the dataset path doesn't exist
    if not os.path.exists(gm.DATASET_PATH):
        print("Downloading dataset...")
        download_og_dataset()

    # Only download if the asset path doesn't exist
    if not os.path.exists(gm.ASSET_PATH):
        print("Downloading assets...")
        download_assets()

    print("\nOmniGibson setup completed!\n")


if __name__ == "__main__":
    main()
