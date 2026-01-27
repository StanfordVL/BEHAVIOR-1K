import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vid2scene_policy.data_collection.lerobot_datasets.datasets.lerobot_dataset import LeRobotDataset


def test_lerobot_dataset(dataset_path: str):
    """Test if LeRobotDataset can initialize from local path."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        sys.exit(1)

    try:
        print(f"Testing LeRobotDataset initialization for: {dataset_path}")

        absolute_dataset_path = dataset_path.resolve()
        dataset = LeRobotDataset(
            repo_id=absolute_dataset_path.name,
            root=absolute_dataset_path,
            force_cache_sync=False,
            revision="v3.0"
        )

        print("LeRobotDataset initialized successfully!")

        sample = dataset[0]

        print(f"Sample keys: {list(sample.keys())}")
        print(f"Sample length: {len(sample)}")

        for key, value in sample.items():
            if key.endswith("rgb") or key == "action":
                print(f"  {key}: {value.shape}")

        for key, value in sample.items():
            print(f"  {key}: {value}")

        print("LeRobotDataset test completed successfully.")

        return True

    except Exception as e:
        print(f"Failed to initialize LeRobotDataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_lerobot_dataset.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    success = test_lerobot_dataset(dataset_path)
    sys.exit(0 if success else 1)