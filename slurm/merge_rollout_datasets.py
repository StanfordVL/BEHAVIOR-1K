import numpy as np
import pathlib
import sys

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm.auto import tqdm



def main():
    input_root = pathlib.Path(sys.argv[1])
    output_root = pathlib.Path("/checkpoint/clear/cgokmen/merged_lerobot_datasets_2")
    max_videos = 1000000000

    dataset_type = input_root.name

    print("Processing", dataset_type, "datasets")

    lerobot_datasets = []
    subsets = list(input_root.iterdir())
    current_videos = 0
    for subset_root in tqdm(subsets, desc=f"Reading {dataset_type} datasets"):
        try:
            dataset = LeRobotDataset(root=subset_root, repo_id=subset_root.name)
            lerobot_datasets.append(dataset)
            current_videos += dataset.num_episodes
            if current_videos >= max_videos:
                break
        except Exception:
            continue

    print("Merging", len(lerobot_datasets), "datasets")
    output_root.mkdir(parents=True, exist_ok=True)
    merged = merge_datasets(lerobot_datasets, output_repo_id=dataset_type, output_dir=output_root/dataset_type)
    print(f"Merged dataset: {merged.num_episodes} episodes")


if __name__ == "__main__":
    main()