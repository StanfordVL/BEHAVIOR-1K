"""
Helper script to setup this OmniGibson repository. Configures environment and downloads assets
"""
import os
import omnigibson as og
from omnigibson.utils.asset_utils import download_demo_data, download_og_dataset, download_assets
from omnigibson.utils.ui_utils import choose_from_options



def main():
    # Ask user which dataset to install
    print("Welcome to OmniGibson!")
    print()

    # Only download if the dataset path doesn't exist
    if not os.path.exists(og.og_dataset_path):
        print("Downloading dataset...")
        # dataset_options = {
        #     "Demo": "Download the demo OmniGibson dataset",
        #     "Full": "Download the full OmniGibson dataset",
        # }
        # dataset = choose_from_options(options=dataset_options, name="dataset")
        # if dataset == "Demo":
        download_demo_data()
        # else:
        #     download_og_dataset()

    print("Downloading assets...")
    download_assets()

    print("\nOmniGibson setup completed!\n")


if __name__ == "__main__":
    main()
