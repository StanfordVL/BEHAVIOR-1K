import argparse

from vid2scene_policy.data_collection.config import DataCollectionConfig
from vid2scene_policy.data_collection.episode import run_data_collection


def main():
    parser = argparse.ArgumentParser(description="Collect pick-and-place data")
    parser.add_argument("--scene", default="Rs_int", help="Scene model name")
    parser.add_argument("--dataset", default="behavior-1k-assets", help="Dataset name (behavior-1k-assets or spoc)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--output", default="./lerobot_datasets", help="Output directory")
    parser.add_argument("--repo-id", default=None, help="Repository ID (default: {scene}_stretch_pick_place)")
    parser.add_argument("--max-nav-steps", type=int, default=1500, help="Max navigation steps per episode")
    args = parser.parse_args()

    repo_id = args.repo_id or f"{args.scene}_stretch_pick_place"

    config = DataCollectionConfig(
        scene_model=args.scene,
        dataset_name=args.dataset,
        num_episodes=args.episodes,
        output_dir=args.output,
        repo_id=repo_id,
        whitelist_graspable_path="configs/whitelist_graspable_objects.json",
        whitelist_support_path="configs/whitelist_support_objects.json",
        max_navigation_steps=args.max_nav_steps,
        max_grasp_steps=100,
        max_place_steps=100,
    )
    run_data_collection(config)


if __name__ == "__main__":
    main()
