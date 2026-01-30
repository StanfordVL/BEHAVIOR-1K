import numpy as np
import pathlib
import sys

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm.auto import tqdm


combinations = {
    "bpv": ("behavior-1k-assets-train", "spoc-train", "vid2room-train"),
    "bp": ("behavior-1k-assets-train", "spoc-train"),
    "bv": ("behavior-1k-assets-train", "vid2room-train"),
    "b": ("behavior-1k-assets-train",),
}


def main():
    ds_root = pathlib.Path("/checkpoint/clear/cgokmen/merged_lerobot_datasets_2")

    for combination_name, combination in combinations.items():
        lerobot_datasets = []
        for child_dataset in combination:
            dataset = LeRobotDataset(root=ds_root/child_dataset, repo_id=child_dataset)
            lerobot_datasets.append(dataset)

        print("Merging", combination_name, "datasets", combination)
        merged = merge_datasets(lerobot_datasets, output_repo_id=combination_name, output_dir=ds_root/combination_name)
        print(f"Merged dataset: {merged.num_episodes} episodes")


if __name__ == "__main__":
    main()