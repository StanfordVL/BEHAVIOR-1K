import pathlib
from tqdm.auto import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_episodes_in_subset(subset_root: pathlib.Path) -> int:
  try:
    lr_dataset = LeRobotDataset(root=subset_root, repo_id=subset_root.name)
    return lr_dataset.num_episodes
  except:
    return 0

def main():
  root = pathlib.Path("/checkpoint/clear/cgokmen/lerobot_datasets")
  for dataset in root.iterdir():
    dataset_episodes = 0
    subsets = list(dataset.iterdir())

    futures = {}
    with ProcessPoolExecutor() as executor:
      for subset_root in subsets:
        future = executor.submit(get_episodes_in_subset, subset_root)
        futures[future] = subset_root
      for future in tqdm(as_completed(futures), total=len(futures), desc=f"Counting episodes for {dataset.name}"):
        subset_root = futures[future]
        dataset_episodes += future.result()
    print(f"{dataset.name}: {dataset_episodes} videos")

if __name__ == "__main__":
  main()