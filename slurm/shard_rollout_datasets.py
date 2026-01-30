import numpy as np
import pathlib
import sys
import hashlib

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm.auto import tqdm

def main():
    dataset_type_root = pathlib.Path(sys.argv[1])
    shard = int(sys.argv[2])
    total_shards = int(sys.argv[3])

    dataset_type = dataset_type_root.name
    output_dataset_name = f"{dataset_type}-{shard:04d}-{total_shards:04d}"

    print("Processing", dataset_type, "datasets")

    lerobot_datasets = []
    subsets = [
        x for x in dataset_type_root.iterdir()
        if int(hashlib.md5((x.name.rsplit("-", 1)[0] + "tomato").encode()).hexdigest(), 16) % total_shards == shard   
    ]
    for subset_root in tqdm(subsets, desc=f"Reading datasets"):
        try:
            dataset = LeRobotDataset(root=subset_root, repo_id=subset_root.name)
            lerobot_datasets.append(dataset)
        except Exception:
            continue

    print("Generating shard", shard, "of", total_shards, "for", dataset_type, "with", len(lerobot_datasets), "datasets")
    output_root = pathlib.Path("/checkpoint/clear/cgokmen/sharded_lerobot_datasets") / dataset_type
    output_root.mkdir(parents=True, exist_ok=True)
    merged = merge_datasets(lerobot_datasets, output_repo_id=output_dataset_name, output_dir=output_root/output_dataset_name)
    print(f"Generated shard {shard} of {total_shards} for {dataset_type} with {merged.num_episodes} episodes")


if __name__ == "__main__":
    main()